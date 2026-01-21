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

---

## Augmented Training (Recommended)

### 1. Generate augmented data
```bash
python research_pipeline/augment_training_data.py
# Output: train_augmented.jsonl (10k+ samples from 3.7k original)
```

### 2. Finetune DeepSeek with augmented data (5 epochs, lower lr)
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --train-data research_pipeline/datasets/train_augmented.jsonl \
    --epochs 5 \
    --lr 5e-6 \
    --batch-size 8 \
    --grad-accum 2 \
    --lora-r 8 \
    --output ./deepseek_coder_finetuned
```

### 3. Benchmark finetuned model
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --skip-train --use-vllm \
    --adapter ./deepseek_coder_finetuned \
    --easy --schema-linking --few-shot 3
```

### Training Parameters Comparison

| Version | Data | Epochs | LR | Expected |
|---------|------|--------|-----|----------|
| v1 (original) | 3.7k | 3 | 2e-5 | 53% |
| v2 (augmented) | 10k+ | 5 | 5e-6 | 60%+ |

---

## Enhanced Mode (NEW)

Sử dụng `--enhance` để bật 4 tính năng cải tiến:

1. **Dynamic Few-shot**: Chọn examples phù hợp với loại câu hỏi
2. **Post-processing SQL**: Tự động sửa lỗi cột/bảng không tồn tại
3. **Self-Correction**: Retry 1 lần khi gặp lỗi syntax
4. **Enhanced Business Rules**: Rules cụ thể hơn cho các pattern hay sai

### Benchmark với Enhanced Mode
```bash
python research_pipeline/finetune_qwen_coder.py \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --skip-train --use-vllm \
    --adapter ./deepseek_coder_finetuned_v2 \
    --easy --schema-linking \
    --enhance
```
