# Windows on ARM64 only. Python 3.11 recommended.
# pip install --upgrade "transformers>=4.49.0" qwen-vl-utils pillow numpy onnx onnxruntime onnxruntime-qnn torch

import os
import math
import tempfile
import numpy as np
from PIL import Image

import onnx
import torch
import onnxruntime as ort

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# 1) Load model + processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

print([n for n, _ in model.named_modules()][:200])

# 2) Prepare inputs (messages -> processor -> tensors)
text = "Describe this image."
img = Image.new("RGB", (560, 420), color=(200, 180, 160))  # replace with a real image if you like

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": text},
        ]
    }
]

# Build prompt with the image placeholder tokens
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# Gather vision I/O from messages (works for images and videos)
image_inputs, video_inputs = process_vision_info(messages)

# Create the final, consistent tensor bundle
inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# 3) Quick forward to validate and to surface any dtype/shape issues immediately
with torch.no_grad():
    _out = model(**inputs)

# 4) Sanity readout: grid vs tokens (Qwen2.5-VL compresses patches; tokens != H*W)
if "image_grid_thw" in inputs:
    T, H, W = inputs["image_grid_thw"][0].tolist()
    grid_area = T * H * W
    if hasattr(model.config, "image_token_id"):
        n_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()
        if n_image_tokens > 0:
            pack = grid_area / n_image_tokens
            print(f"grid={T}x{H}x{W}={grid_area}, image_tokens={n_image_tokens}, pack_factor≈{pack:.2f}")
        else:
            print("No image placeholder tokens found in input_ids.")

# 5) Build exportable submodules

# Accessors for the visual tower and projector; trust_remote_code models provide helpers.
def get_vision_tower(m):
    try:
        return getattr(model.model, "visual", None)
    except:
        print(":(")
    if hasattr(m, "get_vision_tower"):
        return m.get_vision_tower()
    if hasattr(m, "vision_tower"):
        return m.vision_tower
    raise RuntimeError("Vision tower accessor not found on model.")

def get_vision_projector(m):
    try:
        return getattr(model.model, "projector", None)
    except:
        print(":(")
    if hasattr(m, "get_vision_projector"):
        return m.get_vision_projector()
    if hasattr(m, "vision_proj") or hasattr(m, "multi_modal_projector"):
        return getattr(m, "vision_proj", getattr(m, "multi_modal_projector"))
    raise RuntimeError("Vision projector accessor not found on model.")

vision_module = get_vision_tower(model)
projector = get_vision_projector(model)

class VisionToEmbeds(torch.nn.Module):
    def __init__(self, vt, proj):
        super().__init__()
        self.vt = vt
        self.proj = proj
    def forward(self, pixel_values, image_grid_thw=None, image_sizes=None):
        # The vision tower returns per-frame features using grid meta
        feats = self.vt(pixel_values=pixel_values, image_grid_thw=image_grid_thw, image_sizes=image_sizes)
        embeds = self.proj(feats)  # compressed/packed visual tokens projected into LLM space
        return embeds

vision_to_embeds = VisionToEmbeds(vision_module, projector).eval()

# 6) Collect dummy tensors (exact ones produced by processor) for ONNX tracing
dummy_pix   = inputs["pixel_values"]                         # float32
dummy_grid  = inputs.get("image_grid_thw", None)             # Long[int64], shape (N, 3) with [T, H, W]
dummy_sizes = inputs.get("image_sizes", None)                # Long[int64], shape (N, 2) with [orig_h, orig_w]
dummy_ids   = inputs["input_ids"]
dummy_mask  = inputs["attention_mask"]

# Precompute a sample vision_embeds for decoding export
with torch.no_grad():
    dummy_embeds = vision_to_embeds(
        dummy_pix,
        image_grid_thw=dummy_grid if dummy_grid is not None else None,
        image_sizes=dummy_sizes if dummy_sizes is not None else None,
    )

# 7) Export ONNX models
tmpdir = tempfile.mkdtemp()
vision_onnx = os.path.join(tmpdir, "qwen2p5vl_vision.onnx")
llm_onnx    = os.path.join(tmpdir, "qwen2p5vl_decode.onnx")

# 7a) Vision ONNX: (pixel_values, image_grid_thw?, image_sizes?) -> vision_embeds
vision_input_names = ["pixel_values"]
vision_inputs = [dummy_pix]
vision_dynamic_axes = {
    "pixel_values": {0: "batch", 2: "h", 3: "w"},
}
if dummy_grid is not None:
    vision_input_names.append("image_grid_thw")
    vision_inputs.append(dummy_grid)
if dummy_sizes is not None:
    vision_input_names.append("image_sizes")
    vision_inputs.append(dummy_sizes)

torch.onnx.export(
    vision_to_embeds,
    tuple(vision_inputs),
    vision_onnx,
    input_names=vision_input_names,
    output_names=["vision_embeds"],
    opset_version=17,
    dynamic_axes=vision_dynamic_axes,
)
onnx.checker.check_model(onnx.load(vision_onnx))
print(f"Exported: {vision_onnx}")

# 7b) Decode ONNX: (input_ids, attention_mask, vision_embeds, image_grid_thw?, image_sizes?) -> logits
# Many trust_remote_code builds accept vision_embeds directly; we call model(...) and return .logits
class VLDecode(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask, vision_embeds, image_grid_thw=None, image_sizes=None):
        out = self.m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_embeds=vision_embeds,
            image_grid_thw=image_grid_thw,
            image_sizes=image_sizes,
        )
        return out.logits

vl_decode = VLDecode(model).eval()

# For export, some runtimes want non-None dummies for optional inputs; supply minimal placeholders if absent
grid_for_decode  = dummy_grid  if dummy_grid  is not None else torch.tensor([[1, 1, 1]], dtype=torch.long)
sizes_for_decode = dummy_sizes if dummy_sizes is not None else torch.tensor([[img.height, img.width]], dtype=torch.long)

torch.onnx.export(
    vl_decode,
    (dummy_ids, dummy_mask, dummy_embeds, grid_for_decode, sizes_for_decode),
    llm_onnx,
    input_names=["input_ids", "attention_mask", "vision_embeds", "image_grid_thw", "image_sizes"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        # vision_embeds is already sequence-like after packing; keep static unless you need it dynamic.
    },
)
onnx.checker.check_model(onnx.load(llm_onnx))
print(f"Exported: {llm_onnx}")

# 8) Create ONNX Runtime sessions (QNN EP on Windows/ARM64 if available; fallback to CPU)
qnn_provider = (
    "QNNExecutionProvider",
    {
        "backend_type": "htp",
        "htp_performance_mode": "sustained_high_performance",
        "vtcm_mb": "64",
        "profiling_level": "basic",
    },
)
providers = []
try:
    # QNN EP will only load on Windows/ARM64 with onnxruntime-qnn installed
    _test = ort.get_available_providers()
    if "QNNExecutionProvider" in _test:
        providers = [qnn_provider, "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception:
    providers = ["CPUExecutionProvider"]

sess_vision = ort.InferenceSession(vision_onnx, providers=providers)
sess_llm = ort.InferenceSession(llm_onnx, providers=providers)

# 9) Run inference with ORT
feeds_vision = {"pixel_values": dummy_pix.numpy()}
if "image_grid_thw" in inputs:
    feeds_vision["image_grid_thw"] = dummy_grid.numpy()
if "image_sizes" in inputs:
    feeds_vision["image_sizes"] = dummy_sizes.numpy()

vision_out = sess_vision.run(["vision_embeds"], feeds_vision)[0]

feeds_llm = {
    "input_ids": dummy_ids.numpy(),
    "attention_mask": dummy_mask.numpy(),
    "vision_embeds": vision_out,
}
# Keep API symmetry; safe to pass along even if the runtime path ignores them
feeds_llm["image_grid_thw"] = grid_for_decode.numpy()
feeds_llm["image_sizes"] = sizes_for_decode.numpy()

logits = sess_llm.run(["logits"], feeds_llm)[0]
print("Logits shape:", logits.shape)

# 10) Optional: demonstrate a token-vs-grid readout after ORT
if "image_grid_thw" in inputs and hasattr(model.config, "image_token_id"):
    T, H, W = inputs["image_grid_thw"][0].tolist()
    grid_area = T * H * W
    n_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()
    print(f"After ORT: grid_area={grid_area}, image_tokens={n_image_tokens}, pack≈{grid_area / max(1, n_image_tokens):.2f}")
