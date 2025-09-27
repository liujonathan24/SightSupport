# live_transcribe_dual_cloud_balanced.py
# Balanced anti-hallucination setup (not too strict):
#  - [ME] mic and [SYS] loopback transcribed separately
#  - 3s windows, 1s overlap (more context than 0-overlap)
#  - moderate VAD + adaptive energy floor + light crosstalk guard
#  - temperature=0, language="en"
# Ctrl+C to stop. Appends to live_transcript.txt

import os, io, sys, time, queue, threading, warnings, collections, difflib
import numpy as np
import soundcard as sc
import soundfile as sf
import webrtcvad

# -------- CONFIG (tuned to be lenient but safer) --------
SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024

WIN_SECONDS = 3.0                # window length
HOP_SECONDS = 2.0                # 1s overlap
WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

# VAD (0..3). 1 is moderate; 2–3 are stricter.
VAD_AGGR_ME  = 1
VAD_AGGR_SYS = 1
SPEECH_RATIO_MIN = 0.20          # only 20% of frames need to be voiced

# Adaptive RMS floor: use 25th percentile of recent RMS * factor
RMS_RING_SIZE = 40               # ~80s of history at 2s hop per stream
RMS_FLOOR_FACTOR = 0.9           # a smidge below the quiet baseline

# Light crosstalk guard (small gap to decide dominance)
CROSSTALK_DB_GAP = 6.0           # dB difference to prefer one stream
EPS = 1e-12

# Cloud STT
OPENAI_MODEL = "gpt-4o-mini-transcribe"   # or "whisper-1"
OPENAI_TIMEOUT = 60
LANGUAGE = "en"
TEMPERATURE = 0.0

# Text sanity filter
MIN_CHARS = 4                    # drop tiny fragments
MIN_ALPHA_RATIO = 0.5            # require at least 50% alphabetic chars
SIMILARITY_DROP = 0.92           # drop if ≥92% similar to last line for that stream
# --------------------------------------------------------

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

try:
    from openai import OpenAI
except Exception as e:
    print(f"FATAL: openai package missing: {e}")
    sys.exit(1)

q_me  = queue.Queue()
q_sys = queue.Queue()
stop_flag = threading.Event()

def downmix_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2 and x.shape[1] > 1:
        return np.mean(x, axis=1, dtype=np.float32)
    return x.reshape(-1).astype(np.float32, copy=False)

def energy_rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

def db_from_rms(r: float) -> float:
    return 20.0 * np.log10(max(r, EPS))

def frame_iter_10ms(x_mono16k: np.ndarray):
    s = np.clip(x_mono16k, -1.0, 1.0).astype(np.float32, copy=False)
    s16 = (s * 32767.0).astype(np.int16)
    step = 160  # 10ms @ 16k
    for i in range(0, len(s16) - step + 1, step):
        yield s16[i:i+step].tobytes()

def speech_ratio(vad: webrtcvad.Vad, audio_block: np.ndarray) -> float:
    total = 0
    voiced = 0
    for fr in frame_iter_10ms(audio_block):
        total += 1
        if vad.is_speech(fr, SAMPLE_RATE):
            voiced += 1
    return (voiced / total) if total else 0.0

def wav_bytes_from_mono16k(audio: np.ndarray) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return bio.getvalue()

def check_cloud_init():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("FATAL: OPENAI_API_KEY not set.")
        return None
    try:
        client = OpenAI(timeout=OPENAI_TIMEOUT)
        # quick warmup
        silence = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        client.audio.transcriptions.create(
            file=("warmup.wav", wav_bytes_from_mono16k(silence)),
            model=OPENAI_MODEL, temperature=TEMPERATURE, language=LANGUAGE
        )
        return client
    except Exception as e:
        print(f"FATAL: cloud init failed: {e}")
        return None

def cloud_transcribe_fn(client):
    def transcribe(audio_block_16k: np.ndarray) -> str:
        resp = client.audio.transcriptions.create(
            file=("chunk.wav", wav_bytes_from_mono16k(audio_block_16k)),
            model=OPENAI_MODEL, temperature=TEMPERATURE, language=LANGUAGE
        )
        text = getattr(resp, "text", "") or (resp.get("text") if isinstance(resp, dict) else "")
        return (text or "").strip()
    return transcribe

def capture_loop():
    speaker = sc.default_speaker()
    microph = sc.default_microphone()
    print(f"default speaker: {speaker.name}")
    print(f"default microphone: {microph.name}")
    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)

    buf_me  = np.zeros(0, dtype=np.float32)
    buf_sys = np.zeros(0, dtype=np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:

        print("capture started; 3s windows / 1s overlap")
        while not stop_flag.is_set():
            sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
            me_frames  = mic_rec.record(numframes=BLOCK_FRAMES)

            buf_sys = np.concatenate([buf_sys, downmix_mono(sys_frames)]).astype(np.float32, copy=False)
            buf_me  = np.concatenate([buf_me,  downmix_mono(me_frames)]).astype(np.float32, copy=False)

            while len(buf_me) >= WIN_SAMPLES and len(buf_sys) >= WIN_SAMPLES:
                q_me.put(buf_me[:WIN_SAMPLES].copy())
                q_sys.put(buf_sys[:WIN_SAMPLES].copy())
                buf_me  = buf_me[HOP_SAMPLES:]
                buf_sys = buf_sys[HOP_SAMPLES:]

def worker(tag: str, q: queue.Queue, hop_seconds: float, transcribe_fn):
    """
    Per-stream worker with moderate gating:
      - VAD >= 0.20
      - adaptive RMS floor using running 25th percentile
      - light crosstalk drop using the other stream's latest RMS
    """
    t_accum = 0.0
    rms_hist = collections.deque(maxlen=RMS_RING_SIZE)
    last_text = ""

    # share the other stream's RMS via a simple global (updated by both workers)
    global last_rms_me, last_rms_sys
    if tag == "[ME]":
        other_rms_get = lambda: last_rms_sys
        set_my_rms    = lambda v: set_last_rms('me', v)
        vad = webrtcvad.Vad(VAD_AGGR_ME)
    else:
        other_rms_get = lambda: last_rms_me
        set_my_rms    = lambda v: set_last_rms('sys', v)
        vad = webrtcvad.Vad(VAD_AGGR_SYS)

    with open("live_transcript.txt", "a", encoding="utf-8") as f:
        while not stop_flag.is_set():
            try:
                audio = q.get(timeout=0.5)
            except queue.Empty:
                continue

            # measurements
            rms_val = energy_rms(audio)
            set_my_rms(rms_val)
            rms_hist.append(rms_val)
            if len(rms_hist) >= 5:
                floor = np.percentile(rms_hist, 25) * RMS_FLOOR_FACTOR
            else:
                floor = 0.0  # lenient at start

            ratio = speech_ratio(vad, audio)
            other_db = db_from_rms(other_rms_get() or 0.0)
            my_db    = db_from_rms(rms_val)

            # gating (moderate)
            pass_energy = (rms_val >= floor)
            pass_vad    = (ratio >= SPEECH_RATIO_MIN)
            pass_xtalk  = True
            if (my_db + CROSSTALK_DB_GAP) <= other_db:
                # other stream is clearly louder → skip this one
                pass_xtalk = False

            if not (pass_energy and pass_vad and pass_xtalk):
                t_accum += hop_seconds
                continue  # drop quietly

            # transcribe
            try:
                text = transcribe_fn(audio)
            except Exception as e:
                print(f"[warn] {tag} transcribe error: {e}", file=sys.stderr)
                text = ""

            # text sanity (soft)
            if text:
                if len(text) < MIN_CHARS:
                    text = ""
                else:
                    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
                    if alpha < MIN_ALPHA_RATIO:
                        text = ""
                    elif last_text:
                        sim = difflib.SequenceMatcher(None, last_text, text).ratio()
                        if sim >= SIMILARITY_DROP:
                            text = ""

            t_accum += hop_seconds
            if text:
                stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
                line = f"[{stamp}] {tag} {text}"
                print(line, flush=True)
                f.write(line + "\n"); f.flush()
                last_text = text

# simple globals for crosstalk check
last_rms_me  = 0.0
last_rms_sys = 0.0
def set_last_rms(which, val):
    global last_rms_me, last_rms_sys
    if which == 'me':
        last_rms_me = val
    else:
        last_rms_sys = val

def main():
    client = check_cloud_init()
    if client is None:
        print("diagnostics:")
        print(" - set OPENAI_API_KEY in this shell")
        print(" - try OPENAI_MODEL='whisper-1' if access is restricted")
        sys.exit(1)
    transcribe_fn = cloud_transcribe_fn(client)

    cap = threading.Thread(target=capture_loop, daemon=True)
    mew = threading.Thread(target=worker, args=("[ME]",  q_me,  HOP_SECONDS, transcribe_fn), daemon=True)
    sysw = threading.Thread(target=worker, args=("[SYS]", q_sys, HOP_SECONDS, transcribe_fn), daemon=True)

    print("starting… press Ctrl+C to stop")
    cap.start(); mew.start(); sysw.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        stop_flag.set()
        cap.join(timeout=1.0)
        mew.join(timeout=2.0)
        sysw.join(timeout=2.0)
        print("transcript appended to live_transcript.txt")

if __name__ == "__main__":
    main()
