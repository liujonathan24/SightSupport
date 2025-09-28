# SightSupport


# Setup:
``` 
python -m venv SightSupport

# Activate the environment
source SightSupport/bin/activate # On macOS/Linux:
SightSupport\Scripts\Activate.ps1 # On Windows (PowerShell):
SightSupport\Scripts\activate.bat # On Windows (cmd.exe):

# Install dependencies
pip install -r requirements.txt
```



Required for MLLM inference:
Download LM Studio

lms get qwen2.5-vl-7b-instruct
- select lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF
- select Q3_k_L model (5.44 GB).
lms load qwen2.5-vl-7b-instruct


(SightSupport) PS C:\Users\QCWorkshop13\Documents\GitHub\SightSupport> lms status
Server: ON (port: 1234)

Loaded Models
  · qwen2.5-vl-7b-instruct@q3_k_l - 5.44 GB