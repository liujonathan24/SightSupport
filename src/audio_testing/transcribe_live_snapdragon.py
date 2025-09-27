# live_transcribe_dual.py
# Separate transcriptions for:
#   - [ME]  = Microphone (your speech)
#   - [SYS] = System loopback (other people / app audio)
# Works on Snapdragon PCs: tries local faster-whisper, falls back to OpenAI STT.
# Ctrl+C to stop. Appends to live_transcript.txt

import os, io, sys, time, queue, threading, warnings
import numpy as np
import soundcard as sc
import soundfile as sf

# Cloud STT
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Local STT (may fail on Windows-on-ARM)
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# ------------ Config ------------
SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024

# Windowing (context vs latency)
WIN_SECONDS = 6.0
HOP_SECONDS = 3.0

# Cross-talk gating (simple energy-based)
# If ME energy >> SYS, suppress SYS; if SYS >> ME, suppress ME
GATE_RATIO_DB = 8.0          # difference in dB to declare dominance (~6–10 dB works well)
FLOOR = 1e-8                 # numeric floor to avoid log issues

# Optional base gains (usually keep both 1.0 since we transcribe separately)
ME_GAIN  = 1.0
SYS_GAIN = 1.0

# Local model (if available)
LOCAL_MODEL_NAME = "base.en"
LOCAL_DEVICE = "cpu"
LOCAL_COMPUTE_TYPE = "int8"

# Cloud model (fallback)
OPENAI_MODEL = "gpt-4o-mini-transcribe"  # or "whisper-1"
OPENAI_TIMEOUT = 60
# --------------------------------

WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

speaker = sc.default_speaker()
microph = sc.default_microphone()

# Queues for windows of each stream
q_me  = queue.Queue()   # microphone windows
q_sys = queue.Queue()   # system windows
stop_flag = threading.Event()

def downmix_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2 and x.shape[1] > 1:
        return np.mean(x, axis=1, dtype=np.float32)
    return x.reshape(-1).astype(np.float32, copy=False)

def energy_rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

def db(x: float) -> float:
    x = max(x, FLOOR)
    return 10.0 * np.log10(x)

def capture_loop():
    """
    Capture loopback (system) + mic.
    Build overlapped windows for each stream *after* simple cross-talk gating:
      - If ME >> SYS by GATE_RATIO_DB, zero the system chunk for that frame slice.
      - If SYS >> ME by GATE_RATIO_DB, zero the mic chunk for that frame slice.
    This keeps each stream cleaner before transcription.
    """
    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)

    buf_me  = np.zeros(0, dtype=np.float32)
    buf_sys = np.zeros(0, dtype=np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:

        while not stop_flag.is_set():
            sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
            me_frames  = mic_rec.record(numframes=BLOCK_FRAMES)

            sys_mono = downmix_mono(sys_frames) * SYS_GAIN
            me_mono  = downmix_mono(me_frames)  * ME_GAIN

            # Energy-based mutual gating per block
            e_me  = energy_rms(me_mono)
            e_sys = energy_rms(sys_mono)

            # Compare in dB
            d_me  = db(e_me)
            d_sys = db(e_sys)

            if d_me - d_sys >= GATE_RATIO_DB:
                # Your voice dominates => gate system slice
                sys_mono[:] = 0.0
            elif d_sys - d_me >= GATE_RATIO_DB:
                # System dominates => gate mic slice
                me_mono[:] = 0.0
            # else: similar energies -> keep both (e.g., duet/overlap)

            # Accumulate into per-stream buffers
            buf_me  = np.concatenate([buf_me,  me_mono]).astype(np.float32, copy=False)
            buf_sys = np.concatenate([buf_sys, sys_mono]).astype(np.float32, copy=False)

            # Emit overlapped windows for each stream independently
            while len(buf_me) >= WIN_SAMPLES:
                q_me.put(buf_me[:WIN_SAMPLES].copy())
                buf_me = buf_me[HOP_SAMPLES:]

            while len(buf_sys) >= WIN_SAMPLES:
                q_sys.put(buf_sys[:WIN_SAMPLES].copy())
                buf_sys = buf_sys[HOP_SAMPLES:]

# ---------- Transcription backends ----------
def wav_bytes_from_mono16k(audio: np.ndarray, sr: int = 16000) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, audio, sr, subtype="PCM_16", format="WAV")
    return bio.getvalue()

def make_cloud_transcriber():
    if OpenAI is None:
        return None
    try:
        client = OpenAI(timeout=OPENAI_TIMEOUT)
        def transcribe(audio_block_16k: np.ndarray) -> str:
            wav_bytes = wav_bytes_from_mono16k(audio_block_16k)
            resp = client.audio.transcriptions.create(
                file=("chunk.wav", wav_bytes),
                model=OPENAI_MODEL,
            )
            text = getattr(resp, "text", "") or (resp.get("text") if isinstance(resp, dict) else "")
            return (text or "").strip()
        return transcribe
    except Exception:
        return None

def make_local_transcriber():
    if WhisperModel is None:
        return None
    try:
        model = WhisperModel(LOCAL_MODEL_NAME, device=LOCAL_DEVICE, compute_type=LOCAL_COMPUTE_TYPE)
        print(f"local model ready: {LOCAL_MODEL_NAME} on {LOCAL_DEVICE}/{LOCAL_COMPUTE_TYPE}")
        def transcribe(audio_block_16k: np.ndarray) -> str:
            segments, _ = model.transcribe(
                audio_block_16k,
                language="en",
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=1,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            return "".join(s.text for s in segments).strip()
        return transcribe
    except Exception as e:
        print(f"local faster-whisper unavailable: {e}")
        return None

# ---------- Dual transcribe loops ----------
def transcribe_worker(tag: str, q: queue.Queue, hop_seconds: float, outfile: str, transcribe_fn):
    """
    Worker that consumes one stream's windows and prints/appends labeled lines.
    tag: "[ME]" or "[SYS]"
    """
    t_accum = 0.0
    with open(outfile, "a", encoding="utf-8") as f:
        while not stop_flag.is_set():
            try:
                audio = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = transcribe_fn(audio)
            except Exception as e:
                text = ""
                print(f"[warn] {tag} transcribe error: {e}", file=sys.stderr)

            t_accum += hop_seconds
            if text:
                stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
                line = f"[{stamp}] {tag} {text}"
                print(line, flush=True)
                f.write(line + "\n"); f.flush()

def main():
    print("default speaker:", speaker.name)
    print("default microphone:", microph.name)
    print("initializing STT backends…")

    # Prefer local; if absent or fails, use cloud
    transcribe_fn = make_local_transcriber()
    using_cloud = False
    if transcribe_fn is None:
        transcribe_fn = make_cloud_transcriber()
        using_cloud = True
    if transcribe_fn is None:
        print("No transcription backend available. Install faster-whisper (if supported) or set OPENAI_API_KEY.")
        sys.exit(1)
    if using_cloud:
        print(f"using cloud STT: {OPENAI_MODEL}")

    print("starting capture + dual transcription… press Ctrl+C to stop.")
    cap = threading.Thread(target=capture_loop, daemon=True)

    # Two workers, one per stream
    me_worker  = threading.Thread(target=transcribe_worker,
                                  args=("[ME]", q_me, HOP_SECONDS, "live_transcript.txt", transcribe_fn),
                                  daemon=True)
    sys_worker = threading.Thread(target=transcribe_worker,
                                  args=("[SYS]", q_sys, HOP_SECONDS, "live_transcript.txt", transcribe_fn),
                                  daemon=True)

    cap.start(); me_worker.start(); sys_worker.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        stop_flag.set()
        cap.join(timeout=1.0)
        me_worker.join(timeout=2.0)
        sys_worker.join(timeout=2.0)
        print("transcript appended to live_transcript.txt")

if __name__ == "__main__":
    main()
