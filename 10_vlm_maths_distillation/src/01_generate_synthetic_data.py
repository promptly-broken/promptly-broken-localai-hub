import json
import os
import random
from tqdm import tqdm
from datasets import load_dataset
import mlx.core as mx
from mlx_vlm import load, generate

def main():
    print("Loading Base Model (no adapter) for Synthetic Data Generation...")
    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    model, processor = load(model_path)
    
    print("Loading MathVista dataset...")
    # Load a small subset of MathVista
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    
    os.makedirs("data_splits_synthetic", exist_ok=True)
    images_dir = "data_splits_synthetic/images"
    os.makedirs(images_dir, exist_ok=True)
    
    synthetic_records = []
    max_examples = 50 # 50 is plenty for formatting distillation
    
    print(f"Generating {max_examples} synthetic reasoning traces...")
    
    for i, example in enumerate(tqdm(dataset)):
        if len(synthetic_records) >= max_examples:
            break
            
        img = example.get('decoded_image')
        if img is None:
            continue
            
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_filename = f"mathvista_{i}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        
        question = example['query']
        
        # Prompt the base model to output reasoning and answer explicitly
        prompt_text = (
            f"{question}\n\n"
            "Solve this step-by-step. You must separate your final answer from your reasoning. "
            "Output EXACTLY in this format:\n"
            "REASONING: [your step by step reasoning]\n"
            "ANSWER: [your final short answer]"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": img_path},
                ],
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Generate the natural reasoning using the base model
        try:
            res = generate(model, processor, prompt=prompt, image=[img_path], max_tokens=512, verbose=False)
            output = res.text.strip()
        except Exception as e:
            print(f"Error generating for {i}: {e}")
            continue
            
        # Parse the output
        if "REASONING:" in output and "ANSWER:" in output:
            try:
                reasoning_part = output.split("REASONING:")[1].split("ANSWER:")[0].strip()
                answer_part = output.split("ANSWER:")[1].strip()
                
                # Construct the desired XML format for the training target
                assistant_text = f"<thought_process>\n{reasoning_part}\n</thought_process>\n<answer>{answer_part}</answer>"
                
                # The user prompt should just be the original question, without our prompt engineering hacks
                target_user_content = [
                    {"type": "text", "text": question},
                    {"type": "image", "image": img_path}
                ]
                
                record = {
                    "messages": [
                        {"role": "user", "content": target_user_content},
                        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
                    ],
                    "images": [f"data_splits_synthetic/images/{img_filename}"]
                }
                synthetic_records.append(record)
                
            except IndexError:
                continue
    
    # Save the synthetic dataset
    train_path = "data_splits_synthetic/train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for r in synthetic_records:
            f.write(json.dumps(r) + "\n")
            
    # Copy some to valid.jsonl just to satisfy MLX
    valid_path = "data_splits_synthetic/valid.jsonl"
    with open(valid_path, "w", encoding="utf-8") as f:
        for r in synthetic_records[:5]:
            f.write(json.dumps(r) + "\n")
            
    print(f"Successfully generated {len(synthetic_records)} synthetic examples and saved to {train_path}.")

if __name__ == "__main__":
    main()
