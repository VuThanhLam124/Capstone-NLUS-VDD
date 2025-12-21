import os
import sys
import argparse
import time
import gc
import torch
import duckdb
import jiwer
import pandas as pd
import librosa
from transformers import (
    pipeline, 
    AutoProcessor, 
    AutoModelForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC
)

# ==========================================
# 1. SETUP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE.upper()}")

def install_dependencies():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckdb", "openai-whisper", "jiwer", "bitsandbytes", "transformers", "peft", "accelerate", "scipy", "soundfile", "librosa", "pandas", "tabulate", "sentencepiece", "torchaudio"])
    print("Dependencies installed.")

# ==========================================
# 2. ASR ENGINE WRAPPERS
# ==========================================
class ASRWrapper:
    def transcribe(self, input_data):
        raise NotImplementedError

class PhoWhisperWrapper(ASRWrapper):
    def __init__(self):
        print("Loading PhoWhisper-large...")
        self.pipe = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device=0 if DEVICE == "cuda" else -1)
        
    def transcribe(self, input_data):
        # input_data is numpy array
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        return self.pipe({"raw": input_data, "sampling_rate": 16000})["text"]

class Wav2Vec2Wrapper(ASRWrapper):
    def __init__(self, model_id="nguyenvulebinh/wav2vec2-base-vi-vlsp2020"):
        print(f"Loading {model_id}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)

    def transcribe(self, input_data):
        if isinstance(input_data, torch.Tensor):
            speech = input_data.numpy()
        else:
            speech = input_data
        
        input_values = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]

class ChunkformerWrapper(ASRWrapper):
    def __init__(self):
        print("Loading Chunkformer...")
        model_id = "khanhld/chunkformer-ctc-large-vie"
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForCTC.from_pretrained(model_id).to(DEVICE)
        except:
            print("Fallback: treating Chunkformer as Wav2Vec2 compatible...")
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)

    def transcribe(self, input_data):
        if isinstance(input_data, torch.Tensor):
            speech = input_data.numpy()
        else:
            speech = input_data
            
        input_values = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]

# ==========================================
# 3. BENCHMARK RUNNER
# ==========================================
def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def run_benchmark(data_path):
    print(f"Loading data from: {data_path}")
    
    metadata_path = os.path.join(data_path, "metadata.csv")
    tensors_dir = os.path.join(data_path, "tensors")
    
    if not os.path.exists(metadata_path):
        print(f"FATAL: metadata.csv not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} samples.")
    
    results = []
    models_to_test = [
        ("PhoWhisper-Large", PhoWhisperWrapper),
        ("Wav2Vec2-Base-Vi", Wav2Vec2Wrapper),
        ("Chunkformer-Large", ChunkformerWrapper)
    ]
    
    for model_name, ModelClass in models_to_test:
        print(f"\nEvaluating Model: {model_name}")
        cleanup_gpu()
        
        try:
            start_load = time.time()
            asr_engine = ModelClass()
            print(f"Model loaded in {time.time() - start_load:.2f}s")
            
            for _, row in df.iterrows():
                tensor_path = os.path.join(tensors_dir, row['tensor_filename'])
                if not os.path.exists(tensor_path):
                    print(f"Missing tensor: {tensor_path}")
                    continue
                    
                input_data = torch.load(tensor_path)
                
                start_infer = time.time()
                try:
                    hyp_text = asr_engine.transcribe(input_data)
                except Exception as e:
                    print(f"Error: {e}")
                    hyp_text = ""
                infer_time = time.time() - start_infer
                
                wer = jiwer.wer(str(row['text']).lower(), hyp_text.lower())
                
                results.append({
                    "Model": model_name,
                    "ID": row['id'],
                    "GroundTruth": row['text'],
                    "Transcript": hyp_text,
                    "WER": wer,
                    "Time": infer_time
                })
            
            del asr_engine
            
        except Exception as e:
            print(f"Failed {model_name}: {e}")
            
    # Save Results
    res_df = pd.DataFrame(results)
    res_path = os.path.join(data_path, "benchmark_results.csv")
    res_df.to_csv(res_path, index=False)
    
    print("\nSUMMARY:")
    print(res_df.groupby("Model")[["WER", "Time"]].mean())
    print(f"Saved results to {res_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path containing metadata.csv and tensors/ folder")
    args = parser.parse_args()
    
    # Optional: Install if needed
    # install_dependencies()
    
    run_benchmark(args.data_path)
