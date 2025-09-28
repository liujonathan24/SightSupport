from huggingface_hub import snapshot_download
import os

def fix_config_name(model_dir: str):
    cfg_old = os.path.join(model_dir, "config.json")
    cfg_new = os.path.join(model_dir, "genai_config.json")
    if os.path.exists(cfg_old) and not os.path.exists(cfg_new):
        os.rename(cfg_old, cfg_new)

def download_phi3_vision_onnx(model_id: str = "microsoft/Phi-3.5-vision-instruct-onnx",
                              local_dir: str = "./phi3v-onnx"):
    """Downloads the ONNX model pack (includes genai_config.json, tokenizer, *.onnx)."""
    path = snapshot_download(repo_id=model_id, local_dir=local_dir, repo_type="model", ignore_patterns=["*.md"]) + "\onnx"
    fix_config_name(path)
    return path

# Example:
model_dir = download_phi3_vision_onnx()
