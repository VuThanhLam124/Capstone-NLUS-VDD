#!/usr/bin/env python3
"""
Quick test for Qwen3-Coder-30B-A3B-Instruct on A5000 (24GB)

Model specs:
- 30.5B total params, 3.3B active (MoE)
- 128 experts, 8 activated per token
- 256K context native

Memory estimation (4-bit NF4):
- Model weights: ~8-10GB
- KV cache: ~2-4GB (depends on context)
- Total: ~12-14GB VRAM

Should fit comfortably on A5000!
"""

import torch
import gc

def get_gpu_memory():
    """Get GPU memory info in GB"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return {
            "total": f"{total:.1f}GB",
            "allocated": f"{allocated:.1f}GB", 
            "reserved": f"{reserved:.1f}GB",
            "free": f"{total - reserved:.1f}GB"
        }
    return {"error": "No CUDA device"}

def test_qwen_coder():
    print("="*60)
    print("Testing Qwen3-Coder-30B-A3B-Instruct")
    print("="*60)
    
    # Check GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory before: {get_gpu_memory()}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    
    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"\nLoading {model_name}...")
    print("(This may take a few minutes on first run)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    print(f"\n✅ Model loaded!")
    print(f"Memory after loading: {get_gpu_memory()}")
    
    # Test SQL generation
    print("\n" + "="*60)
    print("Testing SQL generation...")
    print("="*60)
    
    test_prompt = """Bạn là một chuyên gia SQL. Tạo câu lệnh SQL cho câu hỏi sau.

Schema:
- customer (c_customer_sk INT, c_first_name VARCHAR, c_last_name VARCHAR)
- store_sales (ss_customer_sk INT, ss_sales_price DECIMAL, ss_sold_date_sk INT)
- date_dim (d_date_sk INT, d_year INT, d_month_seq INT)

Câu hỏi: Tìm top 10 khách hàng có tổng doanh số cao nhất năm 2022

SQL:"""

    messages = [
        {"role": "user", "content": test_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("\nGenerating...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("Generated SQL:")
    print("="*60)
    print(response)
    
    print(f"\n✅ Memory after inference: {get_gpu_memory()}")
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_qwen_coder()
