# Distilling Mathematical Reasoning into a 2B Vision Model

This project demonstrates how to teach a small, local Vision-Language Model (`Qwen2-VL-2B`) to solve complex geometric reasoning problems using **Synthetic Chain-of-Thought (CoT) Distillation**.

Instead of fine-tuning the 2B model directly on generic datasets (which can cause catastrophic forgetting and hallucinations), we use a smarter 7B model to generate synthetic step-by-step reasoning traces. We then fine-tune the 2B model on these highly structured XML reasoning paths. 

The result? The 2B model successfully absorbs the reasoning capabilities of the 7B model and learns to solve complex MathVista geometry problems flawlessly while running locally on a Mac.

## Pipeline Steps

The `/src` folder contains the clean pipeline to reproduce this experiment:

1. **`01_generate_synthetic_data.py`**
   - Uses `Qwen2.5-VL-7B` to iterate through the MathVista dataset and solve complex problems correctly.
   - Formats the 7B model's reasoning into strict `<thought_process>` and `<answer>` XML tags.
   - Saves the dataset to `/data_splits_synthetic`.

2. **`02_train.sh`**
   - Uses `mlx_vlm.lora` to fine-tune `Qwen2-VL-2B` on the synthetic data.
   - Teaches the 2B model to replicate the XML-structured reasoning paths.
   - Outputs the adapter weights to `/lora_adapters_2b`.

3. **`03_test_reasoning.py`**
   - Benchmarks the 2B Base model against the 2B CoT-Distilled model on the training data.
   - Demonstrates how the base model falls into infinite hallucination loops, while the distilled model outputs clean, correct XML logic.

4. **`04_eval_unseen.py`**
   - Evaluates the CoT-Distilled model on *unseen* MathVista validation examples to prove the reasoning and formatting generalize out-of-distribution.

