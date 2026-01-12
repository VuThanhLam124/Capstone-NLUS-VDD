# Finetune & Benchmark Commands

## Qwen3-Coder-30B-A3B-Instruct

### Finetune
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --lora-r 8 \
    --output ./qwen_coder_finetuned
```

### Benchmark (base model - vLLM, fast)
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --skip-train --use-vllm \
    --easy --schema-linking --few-shot 3
```

### Benchmark (finetuned - vLLM)
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --skip-train --use-vllm \
    --adapter ./qwen_coder_finetuned \
    --easy --schema-linking --few-shot 3
```

---

## DeepSeek-Coder-V2-Lite-Instruct

### Finetune
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 2 \
    --lora-r 8 \
    --output ./deepseek_coder_finetuned
```

### Benchmark (base model - vLLM, fast)
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --skip-train --use-vllm \
    --easy --schema-linking --few-shot 3
```

### Benchmark (finetuned - vLLM)
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --skip-train --use-vllm \
    --adapter ./deepseek_coder_finetuned \
    --easy --schema-linking --few-shot 3
```

---

## So sanh VRAM

| Model | Inference | Finetune (QLoRA) |
|-------|-----------|------------------|
| Qwen3-Coder-30B | ~24GB | ~40GB |
| DeepSeek-V2-Lite | ~16GB | ~20GB |
