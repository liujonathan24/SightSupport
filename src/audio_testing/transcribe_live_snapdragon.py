# live_transcribe_whispercpp.py
# Snapdragon-friendly live captions:
#   - Capture with soundcard (loopback + mic)
#   - Transcribe by shelling out to whisper.cpp CLI (ARM/NEON works on Snapdragon)
# Ctrl+C to stop. Appends to live_transcript.txt

import os, sys, time, queue, threading, warnings, subprocess, tempfile, pathlib
import numpy as np
import soundcard as sc
import soundfile as sf

# ---- CONFIG: set these paths to your whisper.cpp install ----
WHISPER_EXE   = r"C:\whispercpp\main.exe"
WHISPER_MODEL = r"C:\whispercpp\models\ggml-base.en.bin"
LANGUAGE      = "en"

# Capture / mix
SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024
MIX_SYS_GAIN = 0.35
MIX_MIC_GAIN = 0.65
DUCK_THRESHOLD = 0.005
DUCK_ATTENUATION = 0.25
PEAK_LIMIT = 1.2

# Windowing (context vs. latency)
WIN_SECONDS = 6.0           # analyze 6s windows
HOP_SECONDS = 3.0           # stride 3s (50% overlap)

# Whisper.cpp flags:
# -nt (no timestamps) prints just text; remove if you want timestamps and parse .srt instead
WHISPER_FLAGS = ["-nt", "-np", "-l", LANGUAGE]  # no progress bar, no timestamps, force lang

# ------------------------------------------------------------

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

speaker = sc.default_speaker()
microph = sc.default_microphone()

q_audio = queue.Queue()
stop_flag = threading.Event()

def downmix_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2 and x.shape[1] > 1:
        return np.mean(x, axis=1, dtype=np.float32)
    return x.reshape(-1).astype(np.float32, copy=False)

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

def capture_loop():
    """Capture system loopback + mic, duck system under voice, build overlapped windows."""
    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)
    buf = np.zeros(0, dtype=np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:
        while not stop_flag.is_set():
            sys_frames = sys_rec.record(numframes=BLOCK_FRAMES)
            mic_frames = mic_rec.record(numframes=BLOCK_FRAMES)

            sys_mono = downmix_mono(sys_frames)
            mic_mono = downmix_mono(mic_frames)

            # Simple VAD/energy for ducking
            mic_energy = rms(mic_mono)
            sys_gain = MIX_SYS_GAIN * (DUCK_ATTENUATION if mic_energy > DUCK_THRESHOLD else 1.0)

            mixed = (sys_gain * sys_mono) + (MIX_MIC_GAIN * mic_mono)

            # Light peak protection
            peak = np.max(np.abs(mixed)) if mixed.size else 0.0
            if peak > PEAK_LIMIT:
                mixed = mixed / peak

            # Assemble overlapped windows
            buf = np.concatenate([buf, mixed]).astype(np.float32, copy=False)
            while len(buf) >= WIN_SAMPLES:
                q_audio.put(buf[:WIN_SAMPLES].copy())
                buf = buf[HOP_SAMPLES:]

def run_whispercpp(wav_path: str) -> str:
    """
    Call whisper.cpp CLI on wav_path and return the transcribed text (no timestamps).
    We use -otxt to force text output; but -nt prints to stdout already.
    To minimize file I/O, read stdout directly.
    """
    cmd = [WHISPER_EXE, "-m", WHISPER_MODEL, "-f", wav_path] + WHISPER_FLAGS
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        # whisper.cpp prints utf-8; ensure decode handles CRLF
        return out.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError as e:
        return ""

def transcribe_loop():
    print("whisper.cpp:", WHISPER_EXE)
    print("model:", WHISPER_MODEL)
    t_accum = 0.0
    tmp_dir = pathlib.Path(tempfile.gettempdir()) / "whisper_live"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open("live_transcript.txt", "a", encoding="utf-8") as f:
        while not stop_flag.is_set():
            try:
                audio = q_audio.get(timeout=0.5)  # 6s window
            except queue.Empty:
                continue

            # Write a temp mono 16k wav
            ts = int(time.time() * 1000)
            wav_path = tmp_dir / f"chunk_{ts}.wav"
            sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")

            # Call whisper.cpp
            text = run_whispercpp(str(wav_path))

            # Advance "clock" by hop, matching overlap notion
            t_accum += HOP_SECONDS
            stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))

            if text:
                line = f"[{stamp}] {text}"
                print(line, flush=True)
                f.write(line + "\n"); f.flush()

            # Clean up temp file
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

def main():
    # Basic sanity checks up front
    if not pathlib.Path(WHISPER_EXE).exists():
        print(f"ERROR: whisper.cpp executable not found at {WHISPER_EXE}")
        sys.exit(1)
    if not pathlib.Path(WHISPER_MODEL).exists():
        print(f"ERROR: model file not found at {WHISPER_MODEL}")
        sys.exit(1)

    print("Default speaker:", speaker.name)
    print("Default microphone:", microph.name)
    print("Starting live capture… press Ctrl+C to stop.")

    cap = threading.Thread(target=capture_loop, daemon=True)
    trn = threading.Thread(target=transcribe_loop, daemon=True)
    cap.start(); trn.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop_flag.set()
        cap.join(timeout=1.0)
        trn.join(timeout=2.0)
        print("Transcript appended to live_transcript.txt")

if __name__ == "__main__":
    main()
