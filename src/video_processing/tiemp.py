
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

model_id = "Qwen/Qwen3-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cpu"
)

print([n for n, _ in model.named_modules()])

# 1) Access the projector module
projector = model.visual  # <-- this is the vision projector

# 2) (Optional) Save just the projector weights
torch.save(projector.state_dict(), "qwen3vl_7b_projector.pt")

# 3) (Optional) Export the projector to ONNX
#    The projector expects vision features of shape [B, T, C_vit].
#    Use the ViT hidden size for C_vit, and any token length T (e.g., 64).
vit_dim = getattr(model.config, "vision_config", getattr(model, "visual").config).hidden_size
print(vit_dim)
dummy = torch.randn(1, 64, vit_dim, dtype=torch.float32)  # ONNX wants float32 by default

torch.onnx.export(
    projector,
    (dummy,),
    "qwen25vl_7b_projector.onnx",
    input_names=["vision_features"],
    output_names=["projected_features"],
    dynamic_axes={"vision_features": {0: "batch", 1: "tokens"},
                  "projected_features": {0: "batch", 1: "tokens"}},
    opset_version=17,
    grid_thw = 
)
