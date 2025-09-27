#!/usr/bin/env bash
set -e

# directory to store ONNX models
MODEL_DIR="whisper_onnx_models"
mkdir -p "$MODEL_DIR"

# list of model names you want
MODELS=("tiny.en" "base.en" "small.en" "medium.en")

# base URL for Sherpa ONNX releases
SHERPA_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

for model in "${MODELS[@]}"; do
  tarball="${SHERPA_BASE}/sherpa-onnx-whisper-${model}.tar.bz2"
  echo "Downloading $model from $tarball"
  wget -O "${MODEL_DIR}/${model}.tar.bz2" "$tarball"

  echo "Extracting $model"
  tar -xjf "${MODEL_DIR}/${model}.tar.bz2" -C "$MODEL_DIR"
done

echo "Done. Models in $MODEL_DIR"
