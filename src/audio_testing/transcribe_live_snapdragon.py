# transribe_live.py
# Live captions using soundcard (loopback + mic) + faster-whisper (CPU)
# Ctrl+C to stop. Appends to live_transcript.txt

import warnings
import time, queue, threading, sys
import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel

# suppress soundcard runtime warnings (optional)
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# ---------- config ----------
SAMPLE_RATE = 16000           # whisper prefers 16k mono
LANGUAGE = "en"               # force english
MODEL_NAME = "base.en"        # try "tiny.en" for faster / "small.en" for better
DEVICE = "cpu"                # cpu only
COMPUTE_TYPE = "int8"         # fast on cpu; alt: "int8_float16" or "float32" (slower)

# capture/mix
BLOCK_CAP_FRAMES = 1024       # small read size from devices
MIX_SYS_GAIN = 0.35           # lower system audio in the mix
MIX_MIC_GAIN = 0.65           # prioritize mic
DUCK_THRESHOLD = 0.005        # if mic rms > this, duck system more
DUCK_ATTENUATION = 0.25       # additional attenuation when speaking
PEAK_LIMIT = 1.2              # soft peak clamp

# windowing for whisper (context vs latency)
WIN_SECONDS = 6.0             # analyze 6s windows
HOP_SECONDS = 3.0             # stride 3s (50% overlap)
PRINT_EMPTY = False           # print empty lines when VAD produces nothing
NO_SPEECH_THRESHOLD = 0.6     # be slightly stricter
LOG_PROB_THRESHOLD = -1.0     # discard very low-confidence text
BEAM_SIZE = 1                 # keep it fast; 2–4 slows but can help
ROLLING_PROMPT_WORDS = 20     # carry last N words forward
# ----------------------------

# derived
WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

# devices
speaker = sc.default_speaker()     # output device
microph = sc.default_microphone()  # input device

# queues/flags
q_audio = queue.Queue()
stop_flag = threading.Event()

def downmix_mono(x: np.ndarray) -> np.ndarray:
    # x shape: (frames, channels) or (frames,)
    if x.ndim == 2 and x.shape[1] > 1:
        x = np.mean(x, axis=1, dtype=np.float32)
    else:
        x = x.reshape(-1).astype(np.float32, copy=False)
    return x

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

def capture_loop():
    """Capture system loopback + mic, duck system under voice, build overlapped windows."""
    # create a loopback 'microphone' for the speaker (system audio)
    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)

    buf = np.zeros(0, dtype=np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:
        while not stop_flag.is_set():
            # read small frames from both
            sys_frames = sys_rec.record(numframes=BLOCK_CAP_FRAMES)  # (N, 2)
            mic_frames = mic_rec.record(numframes=BLOCK_CAP_FRAMES)  # (N, 1) or (N,)

            sys_mono = downmix_mono(sys_frames)
            mic_mono = downmix_mono(mic_frames)

            # simple voice activity energy
            mic_energy = rms(mic_mono)

            # base mix with optional ducking
            sys_gain = MIX_SYS_GAIN
            if mic_energy > DUCK_THRESHOLD:
                sys_gain *= DUCK_ATTENUATION

            mixed = (sys_gain * sys_mono) + (MIX_MIC_GAIN * mic_mono)

            # light peak protection
            peak = np.max(np.abs(mixed)) if mixed.size else 0.0
            if peak > PEAK_LIMIT:
                mixed = mixed / peak

            # accumulate and push fixed-size windows with overlap
            buf = np.concatenate([buf, mixed]).astype(np.float32, copy=False)
            while len(buf) >= WIN_SAMPLES:
                q_audio.put(buf[:WIN_SAMPLES].copy())  # 6s chunk
                buf = buf[HOP_SAMPLES:]                # advance by hop (3s)

def transcribe_loop():
    # model init
    try:
        model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"model ready: {MODEL_NAME} on {DEVICE}/{COMPUTE_TYPE}")
    except Exception as e:
        print(f"{DEVICE} init failed ({e}); falling back to cpu/int8")
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        print(f"model ready: {MODEL_NAME} on cpu/int8")

    rolling_prompt = ""
    t_accum = 0.0

    with open("live_transcript.txt", "a", encoding="utf-8") as f:
        while not stop_flag.is_set():
            try:
                audio = q_audio.get(timeout=0.5)  # float32 mono @ 16k, window size
            except queue.Empty:
                continue

            segments, info = model.transcribe(
                audio,
                language=LANGUAGE,
                vad_filter=True,
                condition_on_previous_text=True,
                initial_prompt=rolling_prompt,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                log_prob_threshold=LOG_PROB_THRESHOLD,
                beam_size=BEAM_SIZE,
            )

            text = "".join(s.text for s in segments).strip()

            # advance clock by hop to align with overlap notion
            t_accum += HOP_SECONDS

            if text or PRINT_EMPTY:
                stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
                line = f"[{stamp}] {text}"
                print(line, flush=True)
                if text:
                    f.write(line + "\n")
                    f.flush()

            # update rolling prompt with last N words for continuity
            if text:
                words = (rolling_prompt + " " + text).split()
                rolling_prompt = " ".join(words[-ROLLING_PROMPT_WORDS:])

def main():
    print("default speaker:", speaker.name)
    print("default microphone:", microph.name)
    print("starting live capture… press Ctrl+C to stop.")
    cap = threading.Thread(target=capture_loop, daemon=True)
    trn = threading.Thread(target=transcribe_loop, daemon=True)
    cap.start()
    trn.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        stop_flag.set()
        cap.join(timeout=1.0)
        trn.join(timeout=1.0)
        print("transcript appended to live_transcript.txt")

if __name__ == "__main__":
    main()
