# live_transcribe_dual_local_whispercpp.py
# Fully-local dual captions on Snapdragon:
#   [ME]  = mic
#   [SYS] = system loopback
# Uses whisper.cpp CLI (no internet, no keys). Mild anti-hallucination.

import os, sys, time, queue, threading, warnings, tempfile, pathlib, subprocess, io
import numpy as np
import soundcard as sc
import soundfile as sf

# ---- EDIT THESE PATHS IF NEEDED ----
WHISPER_EXE   = r"C:\whispercpp\main.exe"
WHISPER_MODEL = r"C:\whispercpp\models\ggml-base.en.bin"
LANGUAGE      = "en"
# ------------------------------------

SAMPLE_RATE  = 16000
BLOCK_FRAMES = 1024

# Balanced context/latency
WIN_SECONDS  = 3.0
HOP_SECONDS  = 2.0
WIN_SAMPLES  = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES  = int(HOP_SECONDS * SAMPLE_RATE)

# Mild anti-hallucination (lenient)
RMS_FLOOR_ME  = 0.0035
RMS_FLOOR_SYS = 0.0045
CROSSTALK_DB_GAP = 6.0
EPS = 1e-12

# whisper.cpp flags: -nt (no timestamps) to print just text to stdout, -np (no progress)
WHISPER_FLAGS = ["-nt", "-np", "-l", LANGUAGE]

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

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

def run_whispercpp_block(audio: np.ndarray) -> str:
    """Write a temp WAV, call whisper.cpp, return stdout text."""
    tmp_dir = pathlib.Path(tempfile.gettempdir()) / "whisper_live"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / f"chunk_{int(time.time()*1000)}.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
    try:
        cmd = [WHISPER_EXE, "-m", WHISPER_MODEL, "-f", str(wav_path)] + WHISPER_FLAGS
        out = subprocess.check_output(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = out.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError:
        text = ""
    finally:
        try: wav_path.unlink(missing_ok=True)
        except Exception: pass
    return text

def capture_loop():
    speaker = sc.default_speaker()
    microph = sc.default_microphone()
    print("default speaker:", speaker.name)
    print("default microphone:", microph.name)

    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)
    buf_me, buf_sys = np.zeros(0, np.float32), np.zeros(0, np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:
        print("capture started; 3s windows / 1s overlap")
        while not stop_flag.is_set():
            sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
            me_frames  = mic_rec.record(numframes=BLOCK_FRAMES)

            sys_mono = downmix_mono(sys_frames)
            me_mono  = downmix_mono(me_frames)

            # light per-block crosstalk guard
            d_me  = db_from_rms(energy_rms(me_mono))
            d_sys = db_from_rms(energy_rms(sys_mono))
            if d_me - d_sys >= CROSSTALK_DB_GAP:
                sys_mono[:] = 0.0
            elif d_sys - d_me >= CROSSTALK_DB_GAP:
                me_mono[:] = 0.0

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

def worker(tag: str, q_in: queue.Queue, hop_seconds: float):
    t_accum = 0.0
    with open("live_transcript.txt", "a", encoding="utf-8") as f:
        while not stop_flag.is_set():
            try:
                audio = q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = run_whispercpp_block(audio)
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
    # upfront checks so failures are obvious
    if not os.path.exists(WHISPER_EXE):
        print(f"FATAL: whisper.cpp executable not found at {WHISPER_EXE}")
        sys.exit(1)
    if not os.path.exists(WHISPER_MODEL):
        print(f"FATAL: model file not found at {WHISPER_MODEL}")
        sys.exit(1)

    cap = threading.Thread(target=capture_loop, daemon=True)
    mew = threading.Thread(target=worker, args=("[ME]",  q_me,  HOP_SECONDS), daemon=True)
    sysw = threading.Thread(target=worker, args=("[SYS]", q_sys, HOP_SECONDS), daemon=True)

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
