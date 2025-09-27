import soundcard as sc
import soundfile as sf
import numpy as np
import time

SAMPLE_RATE = 48000
BLOCK = 1024
DURATION = 10  # seconds; set None to run until Ctrl+C

# Get default system devices
spk = sc.default_speaker()
mic = sc.default_microphone()

sys_frames, mic_frames = [], []

def record_blocks(seconds=None):
    t_end = None if seconds is None else time.time() + seconds
    with sc.get_microphone(id=spk.name, include_loopback=True).recorder(samplerate=SAMPLE_RATE) as sys_rec, \
         sc.get_microphone(id=mic.name).recorder(samplerate=SAMPLE_RATE) as mic_rec:
        while t_end is None or time.time() < t_end:
            sys_frames.append(sys_rec.record(numframes=BLOCK))
            mic_frames.append(mic_rec.record(numframes=BLOCK))

record_blocks(DURATION)

system = np.concatenate(sys_frames, axis=0)
voice  = np.concatenate(mic_frames, axis=0)

# Mix with headroom
mixed = 0.5*system + 0.5*voice
peak = np.max(np.abs(mixed)) if mixed.size else 0.0
if peak > 1.0:
    mixed = mixed / peak

sf.write("system.wav", system, SAMPLE_RATE, subtype="PCM_16")
sf.write("mic.wav",    voice,  SAMPLE_RATE, subtype="PCM_16")
sf.write("mixed.wav",  mixed,  SAMPLE_RATE, subtype="PCM_16")

print("Saved system.wav, mic.wav, mixed.wav")
