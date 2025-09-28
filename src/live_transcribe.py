# # live_transcribe.py
# # Separate transcriptions for:
# #   - [ME]  = Microphone (your speech)
# #   - [SYS] = System loopback (other people / app audio)
# # Works on Snapdragon PCs: tries local faster-whisper, falls back to OpenAI STT.
# # Appends to transcripts/live_transcript_{n}.txt

# import glob, re
# import os, io, sys, time, queue, threading, warnings
# import numpy as np
# import soundcard as sc
# import soundfile as sf

# # Cloud STT
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# # Local STT (may fail on Windows-on-ARM)
# try:
#     from faster_whisper import WhisperModel
# except Exception:
#     WhisperModel = None

# warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# # ------------ Config ------------
# SAMPLE_RATE = 16000
# BLOCK_FRAMES = 1024

# # Windowing (context vs latency)
# WIN_SECONDS = 6.0
# HOP_SECONDS = 3.0

# # Cross-talk gating (simple energy-based)
# # If ME energy >> SYS, suppress SYS; if SYS >> ME, suppress ME
# GATE_RATIO_DB = 12.0
# FLOOR = 1e-8

# # Optional base gains (usually keep both 1.0 since we transcribe separately)
# ME_GAIN  = 1.0
# SYS_GAIN = 1.0

# # Local model (if available)
# LOCAL_MODEL_NAME = "small.en"
# LOCAL_DEVICE = "cpu"
# LOCAL_COMPUTE_TYPE = "int8"

# # Cloud model (fallback)
# OPENAI_MODEL = "gpt-4o-mini-transcribe"  # or "whisper-1"
# OPENAI_TIMEOUT = 60
# # --------------------------------

# WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
# HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

# # Queues for windows of each stream
# q_me  = queue.Queue()   # microphone windows
# q_sys = queue.Queue()   # system windows

# # Control + I/O synchronization
# stop_flag = threading.Event()
# file_lock = threading.Lock()  # serialize writes across workers

# def next_transcript_path(prefix="live_transcript", ext=".txt", folder="transcripts"):
#     os.makedirs(folder, exist_ok=True)
#     # Match live_transcript.txt or live_transcript_<n>.txt
#     rx = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?{re.escape(ext)}$")
#     max_n = -1
#     for p in glob.glob(os.path.join(folder, f"{prefix}*{ext}")):
#         name = os.path.basename(p)
#         m = rx.match(name)
#         if not m:
#             continue
#         if m.group(1) is None:
#             # Bare file (live_transcript.txt) counts as index 0
#             max_n = max(max_n, 0)
#         else:
#             max_n = max(max_n, int(m.group(1)))
#     next_n = (max_n + 1) if max_n >= 0 else 0
#     return os.path.join(folder, f"{prefix}_{next_n}{ext}")

# def downmix_mono(x: np.ndarray) -> np.ndarray:
#     if x.ndim == 2 and x.shape[1] > 1:
#         return np.mean(x, axis=1, dtype=np.float32)
#     return x.reshape(-1).astype(np.float32, copy=False)

# def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
#     if x.size <= 1:
#         return x
#     y = np.empty_like(x)
#     y[0] = x[0]
#     y[1:] = x[1:] - coeff * x[:-1]
#     return y

# def normalize_rms(x: np.ndarray, target_db: float = -20.0, floor: float = 1e-9) -> np.ndarray:
#     rms = np.sqrt(np.mean(np.square(x), dtype=np.float32)) + floor
#     target = 10.0 ** (target_db / 20.0)
#     gain = target / rms
#     gain = float(np.clip(gain, 0.25, 8.0))
#     return (x * gain).astype(np.float32, copy=False)

# def energy_rms(x: np.ndarray) -> float:
#     if x.size == 0: return 0.0
#     return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

# def db(x: float) -> float:
#     x = max(x, FLOOR)
#     return 10.0 * np.log10(x)

# def capture_loop():
#     """
#     Capture loopback (system) + mic.
#     Build overlapped windows for each stream after simple cross-talk gating.
#     """
#     sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)

#     buf_me  = np.zeros(0, dtype=np.float32)
#     buf_sys = np.zeros(0, dtype=np.float32)

#     with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
#          microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:

#         while not stop_flag.is_set():
#             sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
#             me_frames  = mic_rec.record(numframes=BLOCK_FRAMES)

#             sys_mono = downmix_mono(sys_frames) * SYS_GAIN
#             me_mono  = downmix_mono(me_frames)  * ME_GAIN

#             # Energy-based mutual gating per block
#             e_me  = energy_rms(me_mono)
#             e_sys = energy_rms(sys_mono)

#             # Compare in dB
#             d_me  = db(e_me)
#             d_sys = db(e_sys)

#             if d_me - d_sys >= GATE_RATIO_DB:
#                 sys_mono[:] = 0.0
#             elif d_sys - d_me >= GATE_RATIO_DB:
#                 me_mono[:] = 0.0

#             # Accumulate into per-stream buffers
#             buf_me  = np.concatenate([buf_me,  me_mono]).astype(np.float32, copy=False)
#             buf_sys = np.concatenate([buf_sys, sys_mono]).astype(np.float32, copy=False)

#             # Emit overlapped windows for each stream independently
#             while len(buf_me) >= WIN_SAMPLES:
#                 q_me.put(buf_me[:WIN_SAMPLES].copy())
#                 buf_me = buf_me[HOP_SAMPLES:]

#             while len(buf_sys) >= WIN_SAMPLES:
#                 q_sys.put(buf_sys[:WIN_SAMPLES].copy())
#                 buf_sys = buf_sys[HOP_SAMPLES:]

# # ---------- Transcription backends ----------
# def wav_bytes_from_mono16k(audio: np.ndarray, sr: int = 16000) -> bytes:
#     bio = io.BytesIO()
#     sf.write(bio, audio, sr, subtype="PCM_16", format="WAV")
#     return bio.getvalue()

# def make_cloud_transcriber():
#     if OpenAI is None:
#         return None
#     try:
#         client = OpenAI(timeout=OPENAI_TIMEOUT)
#         def transcribe(audio_block_16k: np.ndarray) -> str:
#             wav_bytes = wav_bytes_from_mono16k(audio_block_16k)
#             kwargs = {"file": ("chunk.wav", wav_bytes), "model": OPENAI_MODEL}
#             if OPENAI_MODEL == "whisper-1":
#                 kwargs.update({
#                     "temperature": 0.0,
#                     "prompt": "Transcribe clearly sung English lyrics. No descriptions, only words.",
#                 })
#             resp = client.audio.transcriptions.create(**kwargs)
#             text = getattr(resp, "text", "") or (resp.get("text") if isinstance(resp, dict) else "")
#             return (text or "").strip()
#         return transcribe
#     except Exception:
#         return None

# def make_local_transcriber():
#     if WhisperModel is None:
#         return None
#     try:
#         model = WhisperModel(LOCAL_MODEL_NAME, device=LOCAL_DEVICE, compute_type=LOCAL_COMPUTE_TYPE)
#         print(f"local model ready: {LOCAL_MODEL_NAME} on {LOCAL_DEVICE}/{LOCAL_COMPUTE_TYPE}")
#         def transcribe(audio_block_16k: np.ndarray) -> str:
#             segments, _ = model.transcribe(
#                 audio_block_16k,
#                 language="en",
#                 vad_filter=True,
#                 condition_on_previous_text=False,
#                 beam_size=5,
#                 temperature=0.0,
#                 no_speech_threshold=0.8,
#                 compression_ratio_threshold=2.4,
#                 log_prob_threshold=-1.0,
#             )
#             return "".join(s.text for s in segments).strip()
#         return transcribe
#     except Exception as e:
#         print(f"local faster-whisper unavailable: {e}")
#         return None

# # ---------- Dual transcribe loops ----------
# def transcribe_worker(tag: str, q: queue.Queue, hop_seconds: float, outfile: str, transcribe_fn):
#     """
#     Worker that consumes one stream's windows and prints/appends labeled lines.
#     tag: "[ME]" or "[SYS]"
#     """
#     t_accum = 0.0
#     with open(outfile, "a", encoding="utf-8") as f:
#         while not stop_flag.is_set():
#             try:
#                 audio = q.get(timeout=0.5)
#             except queue.Empty:
#                 continue
#             try:
#                 proc = pre_emphasis(audio)
#                 proc = normalize_rms(proc, target_db=-20.0)
#                 text = transcribe_fn(proc)
#             except Exception as e:
#                 text = ""
#                 print(f"[warn] {tag} transcribe error: {e}", file=sys.stderr)

#             t_accum += hop_seconds
#             if text:
#                 stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
#                 line = f"[{stamp}] {tag} {text}"
#                 print(line, flush=True)
#                 # serialize writes across workers to avoid interleaving
#                 with file_lock:
#                     f.write(line + "\n")
#                     f.flush()

# _transcribe_thread = None

# def start_transcription():
#     global _transcribe_thread
#     if _transcribe_thread and _transcribe_thread.is_alive():
#         print("Transcription already running")
#         return
#     stop_flag.clear()  # allow loops to run
#     _transcribe_thread = threading.Thread(target=main, daemon=True)
#     _transcribe_thread.start()
#     print("Transcription started")

# def stop_transcription():
#     global _transcribe_thread
#     stop_flag.set()
#     th = _transcribe_thread
#     if th and th.is_alive():
#         th.join(timeout=5.0)  # wait for clean shutdown
#     _transcribe_thread = None
#     print("Stop signal sent")

# def main():
#     global speaker, microph

#     speaker = sc.default_speaker()
#     microph = sc.default_microphone()
#     print("default speaker:", speaker.name)
#     print("default microphone:", microph.name)
#     print("initializing STT backends…")

#     # Prefer local; if absent or fails, use cloud
#     transcribe_fn = make_local_transcriber()
#     using_cloud = False
#     if transcribe_fn is None:
#         transcribe_fn = make_cloud_transcriber()
#         using_cloud = True
#     if transcribe_fn is None:
#         print("No transcription backend available. Install faster-whisper (if supported) or set OPENAI_API_KEY.")
#         return  # don't kill Streamlit with sys.exit

#     if using_cloud:
#         print(f"using cloud STT: {OPENAI_MODEL}")

#     print("starting capture + dual transcription…")
#     cap = threading.Thread(target=capture_loop, daemon=True)

#     out_path = next_transcript_path()
#     print(f"writing transcript to: {out_path}")

#     # Two workers, one per stream
#     me_worker  = threading.Thread(target=transcribe_worker,
#                                   args=("[ME]", q_me, HOP_SECONDS, out_path, transcribe_fn),
#                                   daemon=True)
#     sys_worker = threading.Thread(target=transcribe_worker,
#                                   args=("[SYS]", q_sys, HOP_SECONDS, out_path, transcribe_fn),
#                                   daemon=True)

#     cap.start(); me_worker.start(); sys_worker.start()
#     try:
#         while not stop_flag.is_set():
#             time.sleep(0.2)
#     except KeyboardInterrupt:
#         print("\nstopping…")
#     finally:
#         stop_flag.set()
#         cap.join(timeout=1.0)
#         me_worker.join(timeout=2.0)
#         sys_worker.join(timeout=2.0)
#         print("transcript appended to", out_path)

# if __name__ == "__main__":
#     main()


# live_transcribe.py
# Separate transcriptions for:
#   - [ME]  = Microphone (your speech)
#   - [SYS] = System loopback (other people / app audio)
# Works on Snapdragon PCs: tries local faster-whisper, falls back to OpenAI STT.
# Appends to transcripts/live_transcript_{n}.txt
#
# Start/Stop from another script (e.g., Streamlit):
#   from live_transcribe import start_transcription, stop_transcription
#   start_transcription()
#   stop_transcription()

import glob, re
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
GATE_RATIO_DB = 12.0
FLOOR = 1e-8

# Optional base gains (usually keep both 1.0 since we transcribe separately)
ME_GAIN  = 1.0
SYS_GAIN = 1.0

# Local model (if available)
LOCAL_MODEL_NAME = "small.en"
LOCAL_DEVICE = "cpu"
LOCAL_COMPUTE_TYPE = "int8"

# Cloud model (fallback)
OPENAI_MODEL = "gpt-4o-mini-transcribe"  # or "whisper-1"
OPENAI_TIMEOUT = 60
# --------------------------------

WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

# Queues for windows of each stream
q_me  = queue.Queue()   # microphone windows
q_sys = queue.Queue()   # system windows

# Control + I/O synchronization
stop_flag = threading.Event()
file_lock = threading.Lock()  # serialize writes across workers


# ---------------- Windows COM helpers (needed by soundcard on Windows) ----------------
def _win_com_init():
    """Initialize COM in the current thread (no-op on non-Windows)."""
    if os.name == "nt":
        try:
            import ctypes
            COINIT_APARTMENTTHREADED = 0x2  # STA works best with Media Foundation
            ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        except Exception:
            # Already initialized or not available; ignore.
            pass

def _win_com_uninit():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass


# ---------------- File naming ----------------
def next_transcript_path(prefix="live_transcript", ext=".txt", folder="transcripts"):
    os.makedirs(folder, exist_ok=True)
    # Match live_transcript.txt or live_transcript_<n>.txt
    rx = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?{re.escape(ext)}$")
    max_n = -1
    for p in glob.glob(os.path.join(folder, f"{prefix}*{ext}")):
        name = os.path.basename(p)
        m = rx.match(name)
        if not m:
            continue
        if m.group(1) is None:
            max_n = max(max_n, 0)
        else:
            max_n = max(max_n, int(m.group(1)))
    next_n = (max_n + 1) if max_n >= 0 else 0
    return os.path.join(folder, f"{prefix}_{next_n}{ext}")


# ---------------- DSP helpers ----------------
def downmix_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2 and x.shape[1] > 1:
        return np.mean(x, axis=1, dtype=np.float32)
    return x.reshape(-1).astype(np.float32, copy=False)

def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if x.size <= 1:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y

def normalize_rms(x: np.ndarray, target_db: float = -20.0, floor: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), dtype=np.float32)) + floor
    target = 10.0 ** (target_db / 20.0)
    gain = target / rms
    gain = float(np.clip(gain, 0.25, 8.0))
    return (x * gain).astype(np.float32, copy=False)

def energy_rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

def db(x: float) -> float:
    x = max(x, FLOOR)
    return 10.0 * np.log10(x)


# ---------------- Capture loop (devices live inside this thread) ----------------
def capture_loop():
    """
    Capture loopback (system) + mic.
    Build overlapped windows for each stream after simple cross-talk gating.
    Devices are created and used within this thread to satisfy COM requirements.
    """
    _win_com_init()  # COM init in THIS thread

    try:
        speaker = sc.default_speaker()
        microph = sc.default_microphone()

        # On Windows, use the default speaker's loopback as a virtual "mic"
        sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)

        buf_me  = np.zeros(0, dtype=np.float32)
        buf_sys = np.zeros(0, dtype=np.float32)

        with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
             microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:

            # Optional prints; wrapped to avoid COM re-entry during .name access
            try:
                print("default speaker:", speaker.name)
            except Exception:
                pass
            try:
                print("default microphone:", microph.name)
            except Exception:
                pass

            while not stop_flag.is_set():
                sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
                me_frames  = mic_rec.record(numframes=BLOCK_FRAMES)

                sys_mono = downmix_mono(sys_frames) * SYS_GAIN
                me_mono  = downmix_mono(me_frames)  * ME_GAIN

                # Energy-based mutual gating per block
                e_me  = energy_rms(me_mono)
                e_sys = energy_rms(sys_mono)

                d_me  = db(e_me)
                d_sys = db(e_sys)

                if d_me - d_sys >= GATE_RATIO_DB:
                    sys_mono[:] = 0.0
                elif d_sys - d_me >= GATE_RATIO_DB:
                    me_mono[:] = 0.0

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
    finally:
        _win_com_uninit()  # tidy up COM for this thread


# ---------------- Transcription backends ----------------
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
            kwargs = {"file": ("chunk.wav", wav_bytes), "model": OPENAI_MODEL}
            if OPENAI_MODEL == "whisper-1":
                kwargs.update({
                    "temperature": 0.0,
                    "prompt": "Transcribe clearly sung English lyrics. No descriptions, only words.",
                })
            resp = client.audio.transcriptions.create(**kwargs)
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
                beam_size=5,                 # stronger search reduces random inserts
                temperature=0.0,             # deterministic decoding curbs hallucinations
                no_speech_threshold=0.8,     # conservative on near-silence
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
            return "".join(s.text for s in segments).strip()
        return transcribe
    except Exception as e:
        print(f"local faster-whisper unavailable: {e}")
        return None


# ---------------- Dual transcribe loops ----------------
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
                # Light speech-focused conditioning before STT
                proc = pre_emphasis(audio)
                proc = normalize_rms(proc, target_db=-20.0)
                text = transcribe_fn(proc)
            except Exception as e:
                text = ""
                print(f"[warn] {tag} transcribe error: {e}", file=sys.stderr)

            t_accum += hop_seconds
            if text:
                stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
                line = f"[{stamp}] {tag} {text}"
                print(line, flush=True)
                # serialize writes across workers to avoid interleaving
                with file_lock:
                    f.write(line + "\n")
                    f.flush()


# ---------------- Public controls for external UIs ----------------
_transcribe_thread = None

def start_transcription():
    """Start transcription in a background thread (safe to call from Streamlit)."""
    global _transcribe_thread
    if _transcribe_thread and _transcribe_thread.is_alive():
        print("Transcription already running")
        return

    stop_flag.clear()  # allow loops to run

    def _runner():
        # Initialize COM in the controller thread too (extra safety)
        _win_com_init()
        try:
            main()
        finally:
            _win_com_uninit()

    _transcribe_thread = threading.Thread(target=_runner, daemon=True)
    _transcribe_thread.start()
    print("Transcription started")

def stop_transcription():
    """Signal stop and wait briefly for a clean shutdown."""
    global _transcribe_thread
    stop_flag.set()
    th = _transcribe_thread
    if th and th.is_alive():
        th.join(timeout=5.0)
    _transcribe_thread = None
    print("Stop signal sent")


# ---------------- Main ----------------
def main():
    print("initializing STT backends…")

    # Prefer local; if absent or fails, use cloud
    transcribe_fn = make_local_transcriber()
    using_cloud = False
    if transcribe_fn is None:
        transcribe_fn = make_cloud_transcriber()
        using_cloud = True
    if transcribe_fn is None:
        print("No transcription backend available. Install faster-whisper (if supported) or set OPENAI_API_KEY.")
        return  # don't kill host process with sys.exit

    if using_cloud:
        print(f"using cloud STT: {OPENAI_MODEL}")

    print("starting capture + dual transcription…")
    cap = threading.Thread(target=capture_loop, daemon=True)

    out_path = next_transcript_path()
    print(f"writing transcript to: {out_path}")

    # Two workers, one per stream
    me_worker  = threading.Thread(target=transcribe_worker,
                                  args=("[ME]", q_me, HOP_SECONDS, out_path, transcribe_fn),
                                  daemon=True)
    sys_worker = threading.Thread(target=transcribe_worker,
                                  args=("[SYS]", q_sys, HOP_SECONDS, out_path, transcribe_fn),
                                  daemon=True)

    cap.start(); me_worker.start(); sys_worker.start()
    try:
        while not stop_flag.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        stop_flag.set()
        cap.join(timeout=1.0)
        me_worker.join(timeout=2.0)
        sys_worker.join(timeout=2.0)
        print("transcript appended to", out_path)


if __name__ == "__main__":
    main()

