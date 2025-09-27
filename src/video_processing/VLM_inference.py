# Windows on ARM64 only. Python 3.11 recommended.
# pip install onnx onnxruntime-qnn transformers>=4.49.0 qwen-vl-utils pillow requests numpy

import os
import tempfile
import numpy as np
from PIL import Image
import onnx
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import onnxruntime as ort

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# 1) Load HF model (PyTorch) and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="cpu", trust_remote_code=True
).eval()

# 2) Prepare sample inputs (text + one image)
text = "Describe this image."
img = Image.new("RGB", (560, 420), color=(200, 180, 160))  # replace with a real image path
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": ""}, 
        ] 
    }
]
pixel_values, image_grid_thw = process_vision_info(messages)  # returns pixel_values, image_sizes etc.
image_grid_thw = torch.tensor([[1, 15, 20]])

# Manually add original image size if needed
image_sizes = torch.tensor([[img.height, img.width]])
print(len(pixel_values))
pixel_values = torch.tensor(np.array(pixel_values[0]), dtype=torch.uint8)
print(type(pixel_values), type(image_grid_thw))

# Qwen2.5-VL uses special tokens and projector; build inputs the same way as LMDeploy docs
inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=True,
)

# 3) Get model’s internal prepared inputs (vision -> projector -> embeds) using forward pass hooks
# For clarity, do a single forward to obtain shapes and sample tensors; then export subgraphs.
with torch.no_grad():
    # Many VLMs expose helpers via trust_remote_code; following LMDeploy structure.
    # The model usually takes:
    #   input_ids, attention_mask, pixel_values (vision), image_grid_thw or image_sizes
    pt_inputs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
     
        # image_sizes=image_sizes,
    )
    pt_inputs = {k: v for k, v in pt_inputs.items() if v is not None}
    print(pt_inputs)
    out = model(**pt_inputs)

# 4) Export two ONNX graphs:
# 4a) Vision encoder+projector -> vision_embeds
#    Depending on model internals, export a function that maps pixel_values (+meta) to projected embeddings.
#    Not all repos expose a clean module; for demo, assume model.vision_project() exists.
#    Replace the lambda targets with the correct submodules for Qwen2.5-VL.
vision_module = model.get_vision_tower()  # may be model.vision_tower or similar
projector = model.get_vision_projector()  # projector that maps vision features to LLM space

class VisionToEmbeds(torch.nn.Module):
    def __init__(self, vt, proj):
        super().__init__()
        self.vt = vt
        self.proj = proj
    def forward(self, pixel_values, image_grid_thw=None, image_sizes=None):
        feats = self.vt(pixel_values=pixel_values, image_grid_thw=image_grid_thw, image_sizes=image_sizes)
        embeds = self.proj(feats)
        return embeds

vision_to_embeds = VisionToEmbeds(vision_module, projector).eval()

tmpdir = tempfile.mkdtemp()
vision_onnx = os.path.join(tmpdir, "qwen2p5vl_vision.onnx")
dummy_pix = pixel_values
dummy_grid = image_grid_thw
dummy_sizes = image_sizes
dynamic_axes = {
    "pixel_values": {0: "batch", 2: "h", 3: "w"},
}
input_names = ["pixel_values"]
input_tensors = [dummy_pix]
if dummy_grid is not None:
    input_names.append("image_grid_thw")
    input_tensors.append(dummy_grid)
if dummy_sizes is not None:
    input_names.append("image_sizes")
    input_tensors.append(dummy_sizes)

torch.onnx.export(
    vision_to_embeds,
    tuple(input_tensors),
    vision_onnx,
    input_names=input_names,
    output_names=["vision_embeds"],
    opset_version=17,
    dynamic_axes=dynamic_axes,
)

onnx.checker.check_model(onnx.load(vision_onnx))

# 4b) LLM decode graph: (input_ids, attention_mask, vision_embeds) -> logits
#    Many implementations accept vision_embeds via cross-attention or by concatenation tokens.
#    For a practical export, target a generation subgraph (prefill) that returns logits for next token.

class VLDecode(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask, vision_embeds):
        return self.m.generate_logits(input_ids=input_ids, attention_mask=attention_mask, vision_embeds=vision_embeds)

vl_decode = VLDecode(model).eval()

llm_onnx = os.path.join(tmpdir, "qwen2p5vl_decode.onnx")
dummy_ids = inputs["input_ids"]
dummy_mask = inputs["attention_mask"]
# Generate a dummy embeds by running vision_to_embeds once
with torch.no_grad():
    dummy_embeds = vision_to_embeds(
        dummy_pix,
        image_grid_thw=dummy_grid if dummy_grid is not None else None,
        image_sizes=dummy_sizes if dummy_sizes is not None else None,
    )

torch.onnx.export(
    vl_decode,
    (dummy_ids, dummy_mask, dummy_embeds),
    llm_onnx,
    input_names=["input_ids", "attention_mask", "vision_embeds"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
)

onnx.checker.check_model(onnx.load(llm_onnx))

# 5) Create ONNX Runtime sessions with QNN EP (HTP backend/NPU)
#    You need Windows on ARM64 and onnxruntime-qnn installed.
qnn_provider = (
    "QNNExecutionProvider",
    {
        "backend_type": "htp",                 # NPU
        "htp_performance_mode": "sustained_high_performance",
        "vtcm_mb": "64",                       # adjust per memory limits
        "profiling_level": "basic",
    },
)

# Fallback CPU if some ops are unsupported; ORT will partition automatically.
providers = [qnn_provider, "CPUExecutionProvider"]

sess_vision = ort.InferenceSession(vision_onnx, providers=providers)
sess_llm = ort.InferenceSession(llm_onnx, providers=providers)

# 6) Run inference
vision_out = sess_vision.run(["vision_embeds"], {
    "pixel_values": dummy_pix.numpy(),
    **({"image_grid_thw": dummy_grid.numpy()} if dummy_grid is not None else {}),
    **({"image_sizes": dummy_sizes.numpy()} if dummy_sizes is not None else {}),
})[0]

logits = sess_llm.run(["logits"], {
    "input_ids": dummy_ids.numpy(),
    "attention_mask": dummy_mask.numpy(),
    "vision_embeds": vision_out,
})[0]

print("Logits shape:", logits.shape)
