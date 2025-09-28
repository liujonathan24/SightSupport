import onnxruntime as ort


from huggingface_hub import snapshot_download

local_dir = "models"
repo_id = "marcusmi4n/abeja-qwen2.5-7b-japanese-qnn"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    # allow_patterns=["*.onnx", "onnx/*"],
    # possibly also include config files if needed:
    # allow_patterns=["*.onnx", "onnx/*", "*.json", "config.yaml"]
    # ignore everything else
)


session = ort.InferenceSession("models/onnx/prefill/model.onnx")