import mlx.core as mx
from mlx_vlm import load, generate

IMAGE = "data/images/mathvista_sample_5.jpg"
QUESTION = "Hint: Please answer the question requiring an integer answer and provide the final value.\nQuestion: Solve for X."
BASE_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
ADAPTER = "lora_adapters_2b"

def run_inference(model_path, adapter_path=None):
    model, processor = load(model_path, adapter_path=adapter_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION},
                {"type": "image", "image": IMAGE},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    print("\n" + "="*60)
    if adapter_path:
        print("  COT-DISTILLED MODEL (with LoRA adapter)")
    else:
        print("  BASE MODEL (no adapter)")
    print("="*60)
    
    res = generate(model, processor, prompt=prompt, image=[IMAGE], max_tokens=512, verbose=True)
    
    print("\n--- RAW OUTPUT ---")
    print(res.text)
    print("--- END ---\n")
    return res.text

if __name__ == "__main__":
    base_out = run_inference(BASE_MODEL)
    import gc
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()
        
    cot_out = run_inference(BASE_MODEL, ADAPTER)
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"Base output length : {len(base_out)} chars")
    print(f"CoT output length  : {len(cot_out)} chars")
    print(f"CoT has <thought_process>: {'<thought_process>' in cot_out}")
    print(f"CoT has <answer>: {'<answer>' in cot_out}")
