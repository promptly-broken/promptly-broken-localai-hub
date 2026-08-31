import os
import json
import re
import mlx.core as mx
from mlx_vlm import load, generate

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def map_choice_to_value(ans, choices):
    if not choices:
        return ans
    ans_clean = ans.strip().upper()
    
    # Match single letter or last letter (e.g. "Answer:B")
    match = re.search(r'([A-Z])\s*$', ans_clean)
    if match:
        idx = ord(match.group(1)) - ord('A')
        if 0 <= idx < len(choices):
            return choices[idx]
            
    return ans

def main():
    base_model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    adapter_path = "lora_adapters_2b"
    sample_dir = "evaluation_samples"
    
    # ---------------------------------------------------------
    # 1. Evaluate Base Model
    # ---------------------------------------------------------
    print("Loading Base Model...")
    model, processor = load(base_model_path)
    
    base_preds = []
    print("Running Base Model evaluation...")
    for i in range(10):
        json_path = os.path.join(sample_dir, f"{i}.json")
        img_path = os.path.join(sample_dir, f"{i}.jpg")
        if not os.path.exists(json_path) or not os.path.exists(img_path):
            base_preds.append("N/A")
            continue
            
        with open(json_path, "r") as f:
            example = json.load(f)
            
        messages = [{"role": "user", "content": [{"type": "text", "text": example['query']}, {"type": "image", "image": img_path}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        try:
            res = generate(model, processor, prompt=prompt, image=[img_path], max_tokens=1024, verbose=False)
            ans = extract_answer(res.text)
            ans = map_choice_to_value(ans, example.get('choices', []))
            base_preds.append(ans.replace('\n', ' '))
        except:
            base_preds.append("Error")
            
    # Free memory
    del model
    del processor
    mx.metal.clear_cache()
    
    # ---------------------------------------------------------
    # 2. Evaluate CoT-Distilled Model
    # ---------------------------------------------------------
    print("\nLoading CoT-Distilled Model...")
    model, processor = load(base_model_path, adapter_path=adapter_path)
    
    cot_preds = []
    gts = []
    
    results_md = "# 🧪 2B CoT-Distilled Model Evaluation (Demo Subset)\n\n"
    print("Running CoT-Distilled Model evaluation...")
    
    for i in range(10):
        json_path = os.path.join(sample_dir, f"{i}.json")
        img_path = os.path.join(sample_dir, f"{i}.jpg")
        if not os.path.exists(json_path) or not os.path.exists(img_path):
            cot_preds.append("N/A")
            gts.append("N/A")
            continue
            
        with open(json_path, "r") as f:
            example = json.load(f)
            
        gts.append(str(example['answer']))
        question = example['query']
        
        messages = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": img_path}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        print(f"\n--- Testing Example {i} ---")
        print(f"Question: {question}")
        print("CoT Output:")
        
        try:
            res = generate(model, processor, prompt=prompt, image=[img_path], max_tokens=1024, verbose=True)
            print("\n") # Add a newline after the verbose stream finishes
            output = res.text
            ans = extract_answer(output)
            ans = map_choice_to_value(ans, example.get('choices', []))
            cot_preds.append(ans.replace('\n', ' '))
        except Exception as e:
            output = f"Error: {str(e)}"
            cot_preds.append("Error")
            
        results_md += f"## Example {i}\n"
        results_md += f"**Question:** {question}\n\n"
        results_md += f"**Ground Truth Answer:** `{example['answer']}`\n\n"
        results_md += f"**Model Output:**\n```xml\n{output}\n```\n\n---\n\n"
        
    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        f.write(results_md)
        
    # ---------------------------------------------------------
    # 3. Print CLI Summary Table
    # ---------------------------------------------------------
    print("\n\n" + "="*85)
    print("🎓 EVALUATION SUMMARY: Base Model vs CoT-Distilled Model")
    print("="*85)
    print(f"{'ID':<4} | {'Ground Truth':<15} | {'Base Model Output':<25} | {'CoT Model Output':<25}")
    print("-" * 85)
    
    for i in range(10):
        if i >= len(gts): break
        gt = gts[i][:15]
        bp = (base_preds[i][:22] + "...") if len(base_preds[i]) > 25 else base_preds[i]
        cp = (cot_preds[i][:22] + "...") if len(cot_preds[i]) > 25 else cot_preds[i]
        print(f"{i:<4} | {gt:<15} | {bp:<25} | {cp:<25}")
    print("="*85)
    print("\nEvaluation complete! Full CoT traces saved to evaluation_results.md")

if __name__ == "__main__":
    main()
