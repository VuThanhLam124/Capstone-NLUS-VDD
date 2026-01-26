# Voice-to-SQL Vietnamese Pipeline: A Case Study on Decision Support Systems With TPC-DS Data Warehouse

This repository contains the official implementation and data for the paper **"Voice-to-SQL Vietnamese Pipeline: A Case Study on Decision Support Systems With TPC-DS Data Warehouse"**.

## Abstract

We present an end-to-end Voice-to-SQL pipeline designed to democratize access to complex data warehouses for non-technical users. Our system integrates state-of-the-art Vietnamese Automatic Speech Recognition (ASR) with a finetuned Large Language Model (Qwen3-Coder-30B) for SQL generation. We benchmark our system on the TPC-DS dataset, a rigorous decision support benchmark. Our experiments show that using **Gemini Flash 3 Pro** for ASR achieves a Word Error Rate (WER) of **0.05**, while our finetuned Text-to-SQL model achieves an **Exact Match Accuracy (EMA) of 68%** on schema-linked queries, outperforming generic baselines (GPT-o3) in end-to-end performance.

## Key Features

- **End-to-End Pipeline**: Modular integration of ASR (Speech-to-Text) and Text-to-SQL.
- **Data Warehouse Focus**: Targeted optimization for **TPC-DS** snowflake schemas (24 tables) vs standard academic Spider datasets.
- **Dynamic Schema Linking**: Vector-based retrieval (RAG) to handle massive schemas within context windows.
- **Vietnamese Support**: Specialized optimization for Vietnamese logic and domain-specific terminology.
- **Low-Latency**: End-to-end execution in ~2.3 seconds.

## Repository Structure

```
├── research_pipeline/       # Training and Benchmarking Scripts
│   ├── finetune_qwen_coder.py      # Main QLoRA finetuning script
│   ├── sql_enhancement.py          # Post-processing & Dynamic Few-shot logic
│   ├── benchmark_api_models.py     # Evaluation scripts
│   └── business_rule.txt           # Defined business rules for generating SQL
├── paper/                   # LaTeX Source of the Paper
│   ├── samples.tex                 # Main paper file
│   └── 2407.19517v1.pdf            # Reference material
├── speech_to_text_pipeline/ # ASR Components
├── notebooks/               # Jupyter Notebooks for EDA
└── requirements.txt         # Python dependencies
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/VuThanhLam124/Capstone-NLUS-VDD.git
cd Capstone-NLUS-VDD

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.1+
- `vllm` for efficient inference
- `duckdb` for SQL execution
- `peft`, `transformers`, `trl` for finetuning

## ⚡ Usage

### 1. Fine-tuning the Model
To reproduce our 5-epoch QLoRA fine-tuning on Qwen3-Coder-30B:

```bash
python research_pipeline/finetune_qwen_coder.py \
    --model Qwen/Qwen3-Coder-30B-Instruct \
    --train-data ./your_dataset \
    --output ./checkpoints \
    --epochs 5 \
    --lr 2e-4
```

### 2. Running Benchmarks
Evaluate the model with Dynamic Schema Linking and Enhancement enabled:

```bash
python research_pipeline/finetune_qwen_coder.py \
    --skip-train \
    --enhance \
    --adapter ./checkpoints/final_adapter \
    --use_vllm
    --short 5
    --schema-linking
```


## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@article
```

## 👥 Authors

- **Vu Thanh Lam** - *FPT University* - [lamvthe180779@fpt.edu.vn](mailto:lamvthe180779@fpt.edu.vn)

## 🙏 Acknowledgements

This research uses the TPC-DS benchmark by the Transaction Processing Performance Council. We thank the open-source community for libraries like `transformers`, `vllm`, and `duckdb`.

## 📈 Reproducibility & Logs

To ensure transparency and reproducibility of our results, we provide full execution logs:
- **Dataset benchmark for text to sql**: research_pipeline/datasets/test_easy.csv & research_pipeline/datasets/test.csv
- **DeepSeek-V2 Case Study Logs**:
    - [logl.json](./logl.json): Finetuned Model (58% accuracy, 1163ms latency).
    - [logl2.json](./logl2.json): Baseline Model (56% accuracy, 1298ms latency).
- **API Benchmark Logs**: Full JSON logs for Gemini Flash 3, GPT-4o, and other API-based models can be found in the [log_benchmark_text_to_sql](./log_benchmark_text_to_sql/) directory.
- **ASR Evaluation**: We will public the metadata for our **custom evaluation on 800 samples** used to calculate WER in `speech_to_text_pipeline/eval_metadata.csv` (if available) or detailed in the paper.