# SightSupport: Empowering Blind Individuals with a Visually Aware Meeting Assistant

Individuals who are blind or low vision face major struggles in digital communication, including video calls and online meetings. Whereas sighted people are able to “read the room” from facial reactions, physical cues, and body language, blind individuals lack this mode of information. So, while people with vision can easily tell when others are engaged, distracted, or even not present, people with visual impairment are left in the metaphorical dark.

In fact, studies show that a significant portion of conversations are actually conveyed through nonverbal body language such as facial expressions and gestures, such as head shaking or nodding. Especially when online work reduces social interaction between employees, which limits the chance to ask about coworkers’ reactions, an automated detection measure would provide blind people the awareness that they deserve. 

To solve this problem, we built an intelligent gesture recognition system that detects and interprets body language and visual cues on digital communication sites, such as Zoom meetings or Google meet. Furthermore, we integrated an assistant that a user can query, making it possible for someone to learn about key team members’ reactions to their ideas, or summarize the tacit sentiment regarding a specific idea. Cues and relevant information will then be relayed to visually impaired people in real-time through haptic feedback and audio prompts, helping them to participate fully in online communication. 

# Features

## Key Highlights
- **Real-time body language detection** using multimodal AI (Qwen2.5-VL)
- **Haptic feedback** for sentiment-aware notifications on mobile
- **Dual-stream audio transcription** with cross-talk detection
- **RAG-powered assistant** for querying meeting context

## Core Capabilities

### Audio Processing
- Dual-stream transcription separating microphone [ME] and system audio [SYS]
- Local transcription via faster-whisper or cloud fallback via OpenAI Whisper API
- Advanced DSP: pre-emphasis filtering (0.97), RMS normalization (-20dB target)
- Overlapped 6-second windows with 3-second hop for low-latency analysis

### Visual Analysis
- Multimodal VLM (Qwen2.5-VL-7B-Instruct) running on LM Studio (4 bit quantized).
- Continuous frame capture at 0.5s intervals from Zoom/Google Meet
- 4x4 grid storyboard generation for gesture analysis
- Real-time sentiment classification (positive/negative expressions)

### Meeting Assistant (RAG)
- Context-aware Q&A using meeting transcripts as knowledge base
- Streaming responses for real-time token-by-token feedback

### User Interfaces
- **Web Dashboard (Streamlit)**: Live transcript viewer, process control, assistant chat
- **Desktop HUD (PyQt5)**: Always-on-top frameless display with translucent background
- Animated settings panel with toggleable features
- Dark/light theme support with high-contrast accessibility

### Accessibility
- Haptic feedback via Pushbullet API (single pulse = positive, double = negative)
- System-level hotkeys: Alt+D (data collect), Alt+X (reset), Alt+Z (settings)
- Audio-first design for screen reader compatibility
- Keyboard-only navigation support

### Platform Integration
- Native window capture for Zoom and Google Meet via Windows API
- Windows Media Foundation for audio device enumeration
- Multi-threaded architecture with synchronized file I/O


# Setup:
First, we set up a python environment using venv.
``` 
python -m venv SightSupport

# Activate the environment
source SightSupport/bin/activate # On macOS/Linux:
SightSupport\Scripts\Activate.ps1 # On Windows (PowerShell):
SightSupport\Scripts\activate.bat # On Windows (cmd.exe):

# Install dependencies
pip install -r requirements.txt
```

Required to run Multimodal LLM inference on Snapdragon X NPUs, we use LM Studio:
```
lms get qwen2.5-vl-7b-instruct
- select lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF
- select Q3_k_L model (5.44 GB).
lms load qwen2.5-vl-7b-instruct
```

A secret file, .env, should be added to the main SightSupport/ directory with the following keys: CIRRASCALE_API_KEY, CIRRASCALE_BASE_URL, CIRRASCALE_MODEL, and PUSH_BULLET_API. To receive haptic notifications on the phone, we use the app Pushbullet. 
