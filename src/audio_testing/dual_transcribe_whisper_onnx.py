# dual_transcribe_whisper_onnx.py
# Local ONNX/DirectML transcription for two sources:
#   [ME]  = default microphone
#   [SYS] = default speaker loopback (system output)
#
# Backend: sherpa-onnx Whisper (encoder.onnx, decoder.onnx, tokens.txt)
# Reference: standalone ONNX approach + 16k/mono chunking from the repo you shared.
# Prints and appends to live_transcript.txt

import argparse, time, sys, queue, threading, warnings
import numpy as np
import soundcard as sc
import soundfile as sf

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# -------- CLI args --------
p = argparse.ArgumentParser()
p.add_argument("--encoder", required=True, help="Path to whisper encoder.onnx")
p.add_argument("--decoder", required=True, help="Path to whisper decoder.onnx")
p.add_argument("--tokens",  required=True, help="Path to tokens.txt")
p.add_argument("--lang", default="en", help="Language code, e.g. en")
p.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz)")
p.add_argument("--chunk", type=float, default=4.0, help="Chunk size (sec), like the reference config")
p.add_argument("--hop", type=float, default=2.0, help="Hop (sec) for slight overlap")
p.add_argument("--silence-thresh", type=float, default=0.001, help="Drop windows under this RMS")
p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8)//2), help="Inference threads")
args = p.parse_args()

SAMPLE_RATE = args.sr
WIN_SECONDS = float(args.chunk)   # 4 s by default (per the reference README’s config) :contentReference[oaicite:4]{index=4}
HOP_SECONDS = float(args.hop)     # 2 s (50% overlap)
WIN_SAMPLES = int(WIN_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)
SILENCE_RMS = float(args.silence_thresh)

# ---- sherpa-onnx (Whisper) ----
try:
    import sherpa_onnx
except Exception as e:
    print(f"FATAL: sherpa-onnx not installed or failed to import: {e}")
    print("Install with: pip install sherpa-onnx onnxruntime-directml")
    sys.exit(1)

# Configure offline Whisper recognizer (non-streaming) as in sherpa-onnx docs
# We keep it simple: English, transcribe task, greedy (beam_size=1), and let ORT use DirectML provider automatically on Windows.
whisper_cfg = sherpa_onnx.OfflineWhisperModelConfig(
    encoder=args.encoder,
    decoder=args.decoder,
    tokens=args.tokens,
    num_threads=args.threads,
    debug=False,
    language=args.lang,
    task="transcribe",
    tail_paddings=-1,  # default padding behavior (see docs)
)

recognizer_cfg = sherpa_onnx.OfflineRecognizerConfig(
    whisper=whisper_cfg,
    # you can adjust provider priority if you want CPU fallback explicitly:
    # provider_config=sherpa_onnx.ProviderConfig(provider="dml"),
)

recognizer = sherpa_onnx.OfflineRecognizer(recognizer_cfg)

# ---- audio capture plumbing ----
speaker = sc.default_speaker()
microph = sc.default_microphone()

q_me  = queue.Queue()
q_sys = queue.Queue()
stop_flag = threading.Event()

def downmix_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2 and x.shape[1] > 1:
        return np.mean(x, axis=1, dtype=np.float32)
    return x.reshape(-1).astype(np.float32, copy=False)

def rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

def capture_loop():
    sys_mic = sc.get_microphone(id=speaker.name, include_loopback=True)
    buf_me  = np.zeros(0, dtype=np.float32)
    buf_sys = np.zeros(0, dtype=np.float32)

    with sys_mic.recorder(samplerate=SAMPLE_RATE, channels=2) as sys_rec, \
         microph.recorder(samplerate=SAMPLE_RATE, channels=1) as mic_rec:
        print("capture started; 4s windows / 2s hop (slight overlap)")
        while not stop_flag.is_set():
            sys_frames = sys_rec.record(numframes=1024)
            me_frames  = mic_rec.record(numframes=1024)

            sys_mono = downmix_mono(sys_frames)
            me_mono  = downmix_mono(me_frames)

            buf_sys = np.concatenate([buf_sys, sys_mono]).astype(np.float32, copy=False)
            buf_me  = np.concatenate([buf_me,  me_mono]).astype(np.float32, copy=False)

            while len(buf_me) >= WIN_SAMPLES:
                win = buf_me[:WIN_SAMPLES].copy()
                buf_me = buf_me[HOP_SAMPLES:]
                if rms(win) >= SILENCE_RMS:
                    q_me.put(win)

            while len(buf_sys) >= WIN_SAMPLES:
                win = buf_sys[:WIN_SAMPLES].copy()
                buf_sys = buf_sys[HOP_SAMPLES:]
                if rms(win) >= SILENCE_RMS:
                    q_sys.put(win)

def decode_block(audio_16k: np.ndarray) -> str:
    # Write to a temp WAV buffer for the recognizer convenience API
    # (sherpa-onnx also supports from arrays, but WaveReader gives us robust resampling if needed)
    wav_path = None
    try:
        import tempfile, pathlib
        wav_path = pathlib.Path(tempfile.gettempdir()) / f"chunk_{int(time.time()*1000)}.wav"
        sf.write(str(wav_path), audio_16k, SAMPLE_RATE, subtype="PCM_16")

        # Create per-chunk stream and run non-streaming decode (as in sherpa offline examples)
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio_16k)  # already 16k mono
        recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()
    finally:
        if wav_path:
            try: wav_path.unlink(missing_ok=True)
            except Exception: pass

def worker(tag: str, q_in: queue.Queue, hop_seconds: float):
    t_accum = 0.0
    with open("live_transcript.txt", "a", encoding="utf-8") as f:
        while not stop_flag.is_set():
            try:
                audio = q_in.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                text = decode_block(audio)
            except Exception as e:
                print(f"[warn] {tag} decode error: {e}", file=sys.stderr)
                text = ""

            t_accum += hop_seconds
            if text:
                stamp = time.strftime("%H:%M:%S", time.gmtime(t_accum))
                line = f"[{stamp}] {tag} {text}"
                print(line, flush=True)
                f.write(line + "\n"); f.flush()

def main():
    print("default speaker:", speaker.name)
    print("default microphone:", microph.name)
    cap = threading.Thread(target=capture_loop, daemon=True)
    mew = threading.Thread(target=worker, args=("[ME]",  q_me,  HOP_SECONDS), daemon=True)
    sysw= threading.Thread(target=worker, args=("[SYS]", q_sys, HOP_SECONDS), daemon=True)
    print("starting… Ctrl+C to stop")
    cap.start(); mew.start(); sysw.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        stop_flag.set()
        cap.join(timeout=1.0); mew.join(timeout=2.0); sysw.join(timeout=2.0)
        print("transcript appended to live_transcript.txt")

if __name__ == "__main__":
    import os
    main()
