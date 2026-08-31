#!/bin/bash

# Ensure we are in the vlm_math_distillation directory
cd "$(dirname "$0")/.."

# Activate the virtual environment
source .venv/bin/activate

echo "Starting mlx-vlm LoRA fine-tuning for CoT Distillation (Qwen2-VL-2B)..."

# Run the training
python3 -m mlx_vlm.lora \
    --model-path mlx-community/Qwen2-VL-2B-Instruct-4bit \
    --dataset data_splits_synthetic/ \
    --val-batches 1 \
    --steps-per-eval 50 \
    --iters 250 \
    --train-on-completions \
    --batch-size 1 \
    --learning-rate 5e-6 \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --grad-checkpoint \
    --output-path lora_adapters_2b

echo "Training complete! Adapters saved to lora_adapters_2b/"
