import os
import onnxruntime_genai as og
from PIL import Image
from src.video_processing.model_helpers import download_phi3_vision_onnx

def load_phi3_vision_npu(model_dir: str):
    """
    Loads Phi-3 Vision Instruct ONNX using ONNX Runtime GenAI and the QNN EP (HTP).
    Returns (model, processor).
    """
    # Load the GenAI config shipped with the repo
    # cfg_path = os.path.join(model_dir, "genai_config.json")
    config = og.Config(model_dir)

    # Ensure we select only QNN EP (Qualcomm NPU / HTP)
    config.clear_providers()
    config.append_provider("QNNExecutionProvider")              # EP name
    config.set_provider_option("backend_type", "htp")           # NPU backend
    # Optional perf hint (values: default, burst, sustained_high_performance, etc.)
    config.set_provider_option("htp_performance_mode", "burst")

    # Create the model
    model = og.Model(config)

    # Create multimodal pre/post processor for prompts + images
    processor = model.create_multimodal_processor()
    return model, processor

def run_phi3v_inference_npu(model, processor, image_path: str, prompt: str,
                            max_tokens: int = 128, temperature: float = 0.2, top_p: float = 0.9):
    """
    Simple image→text chat turn using Phi-3 Vision on QNN. Returns the generated text.
    """
    # Prepare inputs
    image = Image.open(image_path).convert("RGB")

    # Build generation params
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_tokens, temperature=temperature, top_p=top_p)

    # Processor packs inputs for the model (tokenizes + encodes image)
    inputs = processor(prompt=prompt, images=og.Images.open(image_path))
    params.set_inputs(inputs)

    # Run the generate() loop
    generator = og.Generator(model, params)
    while not generator.is_done():
        generator.generate_next_token()

    # Decode tokens back to text
    output_ids = generator.get_sequence(0)
    return processor.decode(output_ids)

# Example usage:
model_dir = download_phi3_vision_onnx()
print(model_dir)
model, processor = load_phi3_vision_npu(model_dir)
text = run_phi3v_inference_npu(model, processor, "test.png", "Describe this image in one sentence.")
print(text)
