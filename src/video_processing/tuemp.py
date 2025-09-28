import os
import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    DynamicCache,
)


class PatchedQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def forward(self, *args):
        inputs_embeds, attention_mask, position_ids, *past_key_values_args = args

        # Convert past_key_values list to DynamicCache
        if len(past_key_values_args) == 0:
            past_key_values = None
        else:
            past_key_values = DynamicCache(self.config.num_hidden_layers)
            for i in range(self.config.num_hidden_layers):
                key = past_key_values_args.pop(0)
                value = past_key_values_args.pop(0)
                past_key_values.update(key_states=key, value_states=value, layer_idx=i)

        o = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        flattened_past_key_values_outputs = {
            "logits": o.logits,
        }
        output_past_key_values: DynamicCache = o.past_key_values
        for i, (key, value) in enumerate(
            zip(output_past_key_values.key_cache, output_past_key_values.value_cache)
        ):
            flattened_past_key_values_outputs[f"present.{i}.key"] = key
            flattened_past_key_values_outputs[f"present.{i}.value"] = value

        return flattened_past_key_values_outputs


# Constants
OUTPUT_FOLDER = "output"
EMBEDDING_MODEL_NAME = "embed_tokens.onnx"
TEXT_MODEL_NAME = "decoder_model_merged.onnx"
VISION_MODEL_NAME = "vision_encoder.onnx"
TEMP_MODEL_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "temp")
FINAL_MODEL_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "onnx")


# Load model and processor
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = PatchedQwen2VLForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)


# Save model configs and processor
model.config.save_pretrained(OUTPUT_FOLDER)
model.generation_config.save_pretrained(OUTPUT_FOLDER)
processor.save_pretrained(OUTPUT_FOLDER)
os.makedirs(TEMP_MODEL_OUTPUT_FOLDER, exist_ok=True)


# Configuration values
## Text model
text_config = model.config
num_heads = text_config.num_attention_heads
num_key_value_heads = text_config.num_key_value_heads
head_dim = text_config.hidden_size // num_heads
num_layers = text_config.num_hidden_layers
hidden_size = text_config.hidden_size

## Vision model
vision_config = model.config.vision_config
channel = vision_config.in_chans
temporal_patch_size = vision_config.temporal_patch_size
patch_size = vision_config.spatial_patch_size


# Dummy input sizes
grid_t, grid_h, grid_w = [1, 16, 16]
batch_size = 1
sequence_length = 16
num_channels = 3
past_sequence_length = 0

image_batch_size = 1  # TODO: Add support for > 1 images
assert image_batch_size == 1


# Dummy inputs
## Embedding inputs
input_ids = torch.randint(
    0, model.config.vocab_size, (batch_size, sequence_length), dtype=torch.int64
)

## Text inputs
dummy_past_key_values_kwargs = {
    f"past_key_values.{i}.{key}": torch.zeros(
        batch_size,
        num_key_value_heads,
        past_sequence_length,
        head_dim,
        dtype=torch.float32,
    )
    for i in range(num_layers)
    for key in ["key", "value"]
}
inputs_embeds = torch.ones(
    batch_size, sequence_length, hidden_size, dtype=torch.float32
)
attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.int64)
position_ids = torch.ones(3, batch_size, sequence_length, dtype=torch.int64)

## Vision inputs
grid_thw = torch.tensor(
    [[grid_t, grid_h, grid_w]] * image_batch_size, dtype=torch.int64
)
pixel_values = torch.randn(
    image_batch_size * grid_t * grid_h * grid_w,
    channel * temporal_patch_size * patch_size * patch_size,
    dtype=torch.float32,
)


# ONNX Exports
## Embedding model
embedding_inputs = dict(input_ids=input_ids)
embedding_inputs_positional = tuple(embedding_inputs.values())
# model.model.embed_tokens(*embedding_inputs_positional)  # Test forward pass
model.model.language_model.get_input_embeddings()
EMBED_TOKENS_OUTPUT_PATH = os.path.join(TEMP_MODEL_OUTPUT_FOLDER, EMBEDDING_MODEL_NAME)
torch.onnx.export(
    model.model.language_model.get_input_embeddings(),
    args=embedding_inputs_positional,
    f=EMBED_TOKENS_OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=list(embedding_inputs.keys()),
    output_names=["inputs_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
    },
)

## Text model
text_inputs = dict(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    position_ids=position_ids,
    **dummy_past_key_values_kwargs,
)
text_inputs_positional = tuple(text_inputs.values())
text_outputs = model.forward(*text_inputs_positional)  # Test forward pass
TEXT_MODEL_OUTPUT_PATH=os.path.join(TEMP_MODEL_OUTPUT_FOLDER, TEXT_MODEL_NAME)
torch.onnx.export(
    model,
    args=text_inputs_positional,
    f=TEXT_MODEL_OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=list(text_inputs.keys()),
    output_names=["logits"]
    + [f"present.{i}.{key}" for i in range(num_layers) for key in ["key", "value"]],
    dynamic_axes={
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "position_ids": {1: "batch_size", 2: "sequence_length"},
        **{
            f"past_key_values.{i}.{key}": {0: "batch_size", 2: "past_sequence_length"}
            for i in range(num_layers)
            for key in ["key", "value"]
        },
        "logits": {0: "batch_size", 1: "sequence_length"},
        **{
            f"present.{i}.{key}": {0: "batch_size", 2: "past_sequence_length + 1"}
            for i in range(num_layers)
            for key in ["key", "value"]
        },
    },
)

## Vision model
vision_inputs = dict(
    pixel_values=pixel_values,
    grid_thw=grid_thw,
)
vision_inputs_positional = tuple(vision_inputs.values())
vision_outputs = model.visual.forward(*vision_inputs_positional)  # Test forward pass
VISION_ENCODER_OUTPUT_PATH = os.path.join(TEMP_MODEL_OUTPUT_FOLDER, VISION_MODEL_NAME)
torch.onnx.export(
    model.visual,
    args=vision_inputs_positional,
    f=VISION_ENCODER_OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=list(vision_inputs.keys()),
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values": {
            0: "batch_size * grid_t * grid_h * grid_w",
            1: "channel * temporal_patch_size * patch_size * patch_size",
        },
        "grid_thw": {0: "batch_size"},
        "image_features": {0: "batch_size * grid_t * grid_h * grid_w"},
    },
)


# Post-processing
import onnx
import onnxslim
from optimum.onnx.graph_transformations import check_and_save_model

os.makedirs(FINAL_MODEL_OUTPUT_FOLDER, exist_ok=True)
for name in (EMBEDDING_MODEL_NAME, TEXT_MODEL_NAME, VISION_MODEL_NAME):
    temp_model_path = os.path.join(TEMP_MODEL_OUTPUT_FOLDER, name)
    
    ## Shape inference (especially needed by the vision encoder)
    onnx.shape_inference.infer_shapes_path(temp_model_path, check_type=True, strict_mode=True)
    
    ## Attempt to optimize the model with onnxslim
    try:
        model = onnxslim.slim(temp_model_path)
    except Exception as e:
        print(f"Failed to slim {model}: {e}")
        model = onnx.load(temp_model_path)

    ## Save model
    final_model_path = os.path.join(FINAL_MODEL_OUTPUT_FOLDER, name)
    check_and_save_model(model, final_model_path)

## Cleanup
import shutil
shutil.rmtree(TEMP_MODEL_OUTPUT_FOLDER)
