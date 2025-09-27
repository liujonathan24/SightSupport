# live_transcribe_dual_local_fw.py
# Fully local dual-stream transcription using faster-whisper:
#   [ME]  = Microphone (your speech)
#   [SYS] = System loopback (others/app audio)
# Mild anti-hallucination settings (not strict), no cloud, no keys.
# Ctrl+C to stop. Appends to live_transcript.txt

import sys, time, queue, threading, warnings, difflib, io
import numpy as np
import soundcard as sc
import soundfile as sf

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# ---------- Config ----------
SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024

# Windowing (balanced): 3s windows, 1s overlap
WIN_SECONDS = 3.0
HOP_SECONDS = 2.0
WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

# Light per-block cross-talk guard (prefer dominant stream)
GATE_RATIO_DB = 6.0
FLOOR = 1e-9  # numeric floor for dB calc

# Per-window light energy gates (lenient)
RMS_FLOOR_ME  = 0.0035
RMS_FLOOR_SYS = 0.0045

# faster-whisper model (local only)
MODEL_NAME = "base.en"     # "tiny.en" = faster, "small.en" = better/slower
DEVICE = "cpu"             # Snapdragon has no CUDA; CPU is correct
COMPUTE_TYPE = "int8"      # fastest on CPU; alternatives: "int8_float16", "float32"

# Text sanity checks (soft)
MIN_CHARS = 4              # drop tiny fragments
MIN_ALPHA_RATIO = 0.5      # at least 50% letters
SIMILARITY_DROP = 0.92     # drop if ≥92% similar to previous line (per stream)
# ---------------------------

# ---- Import local STT (no cloud) ----
try:
    from faster_whisper import WhisperModel
except Exception as e:
    print(f"FATAL: faster-whisper is not available or failed to import: {e}")
    print("Install with: pip install faster-whisper  (note: Windows-on-ARM wheels may be unavailable)")
    sys.exit(1)

# Init model up front so failures are obvious
try:
    _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"local model ready: {MODEL_NAME} on {DEVICE}/{COMPUTE_TYPE}")
except Exception as e:
    print(f"FATAL: failed to initialize faster-whisper: {e}")
    sys.exit(1)

# --------- Audio plumbing ---------
speaker = sc.default_speaker()
microph = sc.default_microphone()

q_me  = queue.Queue()   # mic windows
q_sys = queue.Queue()   # loopback windows
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
    Mild per-block cross-talk gating, then window each stream independently.
    """
    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)

    buf_me  = np.zeros(0, dtype=np.float32)
    buf_sys = np.zeros(0, dtype=np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:

        print("capture started; 3s windows / 1s overlap")
        while not stop_flag.is_set():
            sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
            me_frames  = mic_rec.record(numframes=BLOCK_FRAMES)

            sys_mono = downmix_mono(sys_frames)
            me_mono  = downmix_mono(me_frames)

            # Light cross-talk per block
            e_me, e_sys = energy_rms(me_mono), energy_rms(sys_mono)
            d_me, d_sys = db(e_me), db(e_sys)
            if d_me - d_sys >= GATE_RATIO_DB:
                sys_mono[:] = 0.0
            elif d_sys - d_me >= GATE_RATIO_DB:
                me_mono[:] = 0.0

            # Accumulate into per-stream buffers and window
            buf_me  = np.concatenate([buf_me,  me_mono]).astype(np.float32, copy=False)
            buf_sys = np.concatenate([buf_sys, sys_mono]).astype(np.float32, copy=False)

            while len(buf_me) >= WIN_SAMPLES:
                win = buf_me[:WIN_SAMPLES].copy()
                buf_me = buf_me[HOP_SAMPLES:]
                if energy_rms(win) >= RMS_FLOOR_ME:
                    q_me.put(win)

            while len(buf_sys) >= WIN_SAMPLES:
                win = buf_sys[:WIN_SAMPLES].copy()
                buf_sys = buf_sys[HOP_SAMPLES:]
                if energy_rms(win) >= RMS_FLOOR_SYS:
                    q_sys.put(win)

# ---------- Transcription (local) ----------
def transcribe_block(audio_block_16k: np.ndarray) -> str:
    # No context carry to avoid bleed/hallucinations across windows
    segments, _ = _model.transcribe(
        audio_block_16k,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
        beam_size=1,                 # faster, and reduces verbosity
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    return "".join(s.text for s in segments).strip()

def clean_and_emit(tag: str, text: str, t_accum: float, last_text_holder: dict, outfile: str):
    if not text:
        return False
    if len(text) < MIN_CHARS:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False
    last_text = last_text_holder.get(tag, "")
    if last_text:
        sim = difflib.SequenceMatcher(None, last_text, text).ratio()
        if sim >= SIMILARITY_DROP:
            return False

    stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
    line = f"[{stamp}] {tag} {text}"
    print(line, flush=True)
    with open("live_transcript.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()
    last_text_holder[tag] = text
    return True

# Shared RMS for light peer dominance check (window-level)
_last_rms_me  = 0.0
_last_rms_sys = 0.0

def transcribe_worker(tag: str, q_in: queue.Queue, hop_seconds: float, peer_rms_get):
    t_accum = 0.0
    last_text_holder = {}
    global _last_rms_me, _last_rms_sys

    while not stop_flag.is_set():
        try:
            audio = q_in.get(timeout=0.5)
        except queue.Empty:
            continue

        # update my window RMS
        r = energy_rms(audio)
        if tag == "[ME]":
            _last_rms_me = r
        else:
            _last_rms_sys = r

        # light peer dominance check (drop if peer >> me)
        my_db   = db(r)
        peer_db = db(peer_rms_get())
        if (peer_db - my_db) >= GATE_RATIO_DB:
            t_accum += hop_seconds
            continue

        # transcribe locally
        try:
            text = transcribe_block(audio)
        except Exception as e:
            print(f"[warn] {tag} transcribe error: {e}", file=sys.stderr)
            text = ""

        t_accum += hop_seconds
        clean_and_emit(tag, text, t_accum, last_text_holder, "live_transcript.txt")

def main():
    print("default speaker:", speaker.name)
    print("default microphone:", microph.name)

    cap = threading.Thread(target=capture_loop, daemon=True)

    peer_me  = lambda: _last_rms_me
    peer_sys = lambda: _last_rms_sys

    me_worker  = threading.Thread(target=transcribe_worker,
                                  args=("[ME]",  q_me,  HOP_SECONDS, peer_sys),
                                  daemon=True)
    sys_worker = threading.Thread(target=transcribe_worker,
                                  args=("[SYS]", q_sys, HOP_SECONDS, peer_me),
                                  daemon=True)

    print("starting… press Ctrl+C to stop")
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
