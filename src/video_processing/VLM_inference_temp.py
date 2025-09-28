# Windows on ARM64 only. Python 3.11 recommended.
# pip install onnx onnxruntime-qnn "transformers>=4.49.0" qwen-vl-utils pillow requests numpy torch

import os
import tempfile
import numpy as np
from PIL import Image
import onnx
import torch
import onnxruntime as ort

# CHANGED: use the official processor + model class
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# 1) Load HF model and processor
# CHANGED: processor instead of tokenizer; model class specific to Qwen2.5-VL
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # CHANGED: safe on CPU
    device_map="cpu",
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2) Prepare sample inputs (text + one image)
text = "Describe this image."
img = Image.new("RGB", (560, 420), color=(200, 180, 160))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": text},
        ]
    }
]

# CHANGED: let the processor build the chat with image placeholders
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# CHANGED: let the helper gather visual inputs
image_inputs, video_inputs = process_vision_info(messages)

# CHANGED: one call that returns consistent pixel_values + image_grid_thw (+ sizes if needed)
inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Optional sanity check: placeholder tokens vs grid area
if hasattr(model.config, "image_token_id") and "input_ids" in inputs:
    n_img_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()
    H = inputs["image_grid_thw"][0, 1].item()
    W = inputs["image_grid_thw"][0, 2].item()
    assert n_img_tokens == H * W, f"Mismatch: tokens={n_img_tokens}, grid={H}x{W}"

# 3) Forward once to confirm things run
with torch.no_grad():
    out = model(**inputs)

# 4) Export two ONNX graphs
# 4a) Vision encoder+projector -> vision_embeds

# CHANGED: get the actual vision tower and projector from the model
# Accessors are provided by trust_remote_code in HF Qwen repos
vision_module = model.get_vision_tower()
projector = model.get_vision_projector()

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

# CHANGED: take dummies directly from processor outputs (correct dtypes/shapes)
dummy_pix = inputs["pixel_values"]                        # float tensor as expected by the tower
dummy_grid = inputs.get("image_grid_thw", None)
dummy_sizes = inputs.get("image_sizes", None)

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
# CHANGED: avoid nonstandard generate_logits; just call the model and return .logits

class VLDecode(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask, vision_embeds, image_grid_thw=None, image_sizes=None):
        # The model expects image info through kwargs. We pass embeds explicitly and skip pixel_values.
        out = self.m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_embeds=vision_embeds,
            image_grid_thw=image_grid_thw,
            image_sizes=image_sizes,
        )
        return out.logits

vl_decode = VLDecode(model).eval()

llm_onnx = os.path.join(tmpdir, "qwen2p5vl_decode.onnx")
dummy_ids = inputs["input_ids"]
dummy_mask = inputs["attention_mask"]

with torch.no_grad():
    dummy_embeds = vision_to_embeds(
        dummy_pix,
        image_grid_thw=dummy_grid if dummy_grid is not None else None,
        image_sizes=dummy_sizes if dummy_sizes is not None else None,
    )

torch.onnx.export(
    vl_decode,
    # CHANGED: also thread grid/sizes through decode so KV planning matches, though some builds may not need it
    (dummy_ids, dummy_mask, dummy_embeds, dummy_grid if dummy_grid is not None else torch.zeros((1,3), dtype=torch.long), dummy_sizes if dummy_sizes is not None else torch.zeros((1,2), dtype=torch.long)),
    llm_onnx,
    input_names=["input_ids", "attention_mask", "vision_embeds", "image_grid_thw", "image_sizes"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        # vision_embeds can be dynamic over sequence length; omit unless needed by your runtime
    },
)

onnx.checker.check_model(onnx.load(llm_onnx))

# 5) ONNX Runtime with QNN EP (HTP backend/NPU)
qnn_provider = (
    "QNNExecutionProvider",
    {
        "backend_type": "htp",
        "htp_performance_mode": "sustained_high_performance",
        "vtcm_mb": "64",
        "profiling_level": "basic",
    },
)
providers = [qnn_provider, "CPUExecutionProvider"]

sess_vision = ort.InferenceSession(vision_onnx, providers=providers)
sess_llm = ort.InferenceSession(llm_onnx, providers=providers)

# 6) Run inference
feeds_vision = {"pixel_values": dummy_pix.numpy()}
if dummy_grid is not None:
    feeds_vision["image_grid_thw"] = dummy_grid.numpy()
if dummy_sizes is not None:
    feeds_vision["image_sizes"] = dummy_sizes.numpy()

vision_out = sess_vision.run(["vision_embeds"], feeds_vision)[0]

feeds_llm = {
    "input_ids": dummy_ids.numpy(),
    "attention_mask": dummy_mask.numpy(),
    "vision_embeds": vision_out,
}
# keep API symmetry in decode session even if some builds ignore these two:
if dummy_grid is not None:
    feeds_llm["image_grid_thw"] = dummy_grid.numpy()
if dummy_sizes is not None:
    feeds_llm["image_sizes"] = dummy_sizes.numpy()

logits = sess_llm.run(["logits"], feeds_llm)[0]
print("Logits shape:", logits.shape)
