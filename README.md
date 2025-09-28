# SightSupport: Empowering Blind Individuals with a Visually Aware Meeting Assistant

Individuals who are blind or low vision face major struggles in digital communication, including video calls and online meetings. Whereas sighted people are able to “read the room” from facial reactions, physical cues, and body language, blind individuals lack this mode of information. So, while people with vision can easily tell when others are engaged, distracted, or even not present, people with visual impairment are left in the metaphorical dark.

In fact, studies show that a significant portion of conversations are actually conveyed through nonverbal body language such as facial expressions and gestures, such as head shaking or nodding. Especially when online work reduces social interaction between employees, which limits the chance to ask about coworkers’ reactions, an automated detection measure would provide blind people the awareness that they deserve. 

We plan to build an intelligent gesture recognition system that can detect and interpret body language and visual cues on digital communication sites, such as Zoom meetings or phone calls. Furthermore, we will add an integrated assistant that a user can query, making it possible for someone to learn about key team members’ reactions to their ideas, or summarize the tacit sentiment regarding a specific idea. Cues and relevant information will then be relayed to visually impaired people in real-time through haptic feedback and audio prompts, helping them to participate fully in online communication. 


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