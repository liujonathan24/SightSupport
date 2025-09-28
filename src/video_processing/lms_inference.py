# pip install openai pillow  (openai>=1.43 works great)
from openai import OpenAI
import base64
from pathlib import Path
import numpy as np
from PIL import Image

# lms load lms get qwen2.5-vl-7b-instruct-i1@q2_k_l, choose IQ1_S

# LM Studio runs an OpenAI-compatible server on localhost:1234 by default
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")  # api_key can be any non-empty string

# helper to load an image as base64 data URL
def to_data_url(fp):
    p = Path(fp)
    mime = "image/png" if p.suffix.lower() in [".png"] else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# choose your local model id as shown in the list above
model = "Qwen2.5-VL-7B-Instruct" 

# 1) text + local image (base64)
print(np.array(Image.open('./test2.png')).shape)
img_url = to_data_url("./test2.png")

resp = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image? Give 3 bullet points."},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }
    ],
    temperature=0.2,
)
print(resp.choices[0].message.content)
