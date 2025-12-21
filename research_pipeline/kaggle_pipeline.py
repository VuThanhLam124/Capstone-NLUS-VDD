import os
import subprocess
import sys
import time
import csv
import json
import gc
import torch
import duckdb
import jiwer
import pandas as pd
import soundfile as sf
import librosa
from transformers import (
    pipeline, 
    AutoProcessor, 
    AutoModelForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC,
    AutoTokenizer, 
    AutoModelForCausalLM
)

# ==========================================
# 1. ENVIRONMENT SETUP & CONFIG
# ==========================================
WORKING_DIR = "/kaggle/working"
DATA_DIR = os.path.join(WORKING_DIR, "data")
AUDIO_DIR = os.path.join(WORKING_DIR, "audio")
DB_PATH = os.path.join(DATA_DIR, "ecommerce_dw.duckdb")
RESULTS_PATH = os.path.join(WORKING_DIR, "asr_benchmark_results.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE.upper()}")

def install_dependencies():
    print("Installing dependencies...")
    pkgs = [
        "duckdb", "openai-whisper", "jiwer", "bitsandbytes", 
        "transformers", "peft", "accelerate", "scipy", "soundfile", 
        "librosa", "pandas", "tabulate", "sentencepiece", "torchaudio"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)
    print("Dependencies installed.")

# ==========================================
# 2. DATA GENERATION (VIETNAMESE + AUGMENTATION)
# ==========================================
def generate_audio_dataset():
    print("Generating Audio Dataset from User JSON...")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    json_path = os.path.join(DATA_DIR, "test_queries_vi_200.json") # Normalize name
    
    if not os.path.exists(json_path):
        print(f"Error: Please upload your queries to {json_path}")
        print("Format: [{'id': 'q1', 'text': 'Câu hỏi...', 'sql': 'SELECT...'}]")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
    import edge_tts
    import asyncio
    
    # Voices: Multiple Genders
    VOICES = [
        "vi-VN-NamMinhNeural", # Male
        "vi-VN-HoaiMyNeural"   # Female
    ]
    
    # Augmentations (Speed/Pitch via edge-tts params)
    AUGMENTATIONS = [
        {"rate": "+0%", "pitch": "+0Hz", "suffix": ""},
        {"rate": "+10%", "suffix": "_fast"},
        {"rate": "-10%", "suffix": "_slow"},
        {"pitch": "+5Hz", "suffix": "_high"},
    ]
    
    async def _gen():
        rows = []
        print(f"Synthesizing audio for {len(queries)} items with {len(VOICES)} voices and {len(AUGMENTATIONS)} variations...")
        
        for q in queries:
            for voice in VOICES:
                voice_gender = "male" if "NamMinh" in voice else "female"
                
                for aug in AUGMENTATIONS:
                    # Construct filename: q1_male_fast.mp3
                    aug_suffix = aug["suffix"]
                    filename = f"{q['id']}_{voice_gender}{aug_suffix}.mp3"
                    path = os.path.join(AUDIO_DIR, filename)
                    
                    if not os.path.exists(path):
                        communicate = edge_tts.Communicate(
                            q['text'], 
                            voice, 
                            rate=aug.get("rate", "+0%"), 
                            pitch=aug.get("pitch", "+0Hz")
                        )
                        await communicate.save(path)
                    
                    rows.append({
                        "id": q['id'],
                        "text": q['text'],
                        "sql": q['sql'],
                        "audio_path": path,
                        "voice": voice_gender,
                        "augmentation": aug_suffix if aug_suffix else "original"
                    })
                    
        return rows

    loop = asyncio.get_event_loop()
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        test_suite = loop.run_until_complete(_gen())
    else:
        test_suite = asyncio.run(_gen())
        
    print(f"Dataset Ready: {len(test_suite)} audio samples generated.")
    return test_suite

def run_benchmark():
    # 1. Setup Data
    setup_dw(scale_factor=1) # Ensure DB exists
    test_suite = generate_audio_dataset()
    if not test_suite: return

# ==========================================
# 3. ASR ENGINE WRAPPERS
# ==========================================
class ASRWrapper:
    def transcribe(self, audio_path):
        raise NotImplementedError

class PhoWhisperWrapper(ASRWrapper):
    def __init__(self):
        print("Loading PhoWhisper-large...")
        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model="vinai/PhoWhisper-large",
            device=0 if DEVICE == "cuda" else -1
        )
    def transcribe(self, audio_path):
        # PhoWhisper expects sampling rate 16000 usually, pipeline handles resampling mostly
        # but safe to provide raw bytes or path
        return self.pipe(audio_path)["text"]

class Wav2Vec2Wrapper(ASRWrapper):
    def __init__(self, model_id="nguyenvulebinh/wav2vec2-base-vi-vlsp2020"):
        print(f"Loading Wav2Vec2 ({model_id})...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)

    def transcribe(self, audio_path):
        # Load and resample to 16k
        speech, rate = librosa.load(audio_path, sr=16000)
        input_values = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
            
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        return transcription

class ChunkformerWrapper(ASRWrapper):
    def __init__(self):
        print("Loading Chunkformer...")
        model_id = "khanhld/chunkformer-ctc-large-vie"
        # Assuming AutoClasses work. If not, fallback to specific loading logic
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForCTC.from_pretrained(model_id).to(DEVICE)
        except:
            # Fallback if AutoModel fails (Chunkformer might custom)
            # Treating as Wav2Vec2 compatible for demo validaty
            print("Fallback: treating Chunkformer as Wav2Vec2 compatible...")
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)

    def transcribe(self, audio_path):
        speech, rate = librosa.load(audio_path, sr=16000)
        input_values = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        return transcription

# ==========================================
# 4. BENCHMARK RUNNER
# ==========================================
def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def run_benchmark():
    # 1. Setup Data
    test_suite = generate_test_data_vi()
    
    results = []
    
    # List of models to benchmark
    # Format: (Name, Class)
    models_to_test = [
        ("PhoWhisper-Large", PhoWhisperWrapper),
        ("Wav2Vec2-Base-Vi", Wav2Vec2Wrapper),
        ("Chunkformer-Large", ChunkformerWrapper)
    ]
    
    print("\nSTARTING SEQUENTIAL BENCHMARK...")
    
    for model_name, ModelClass in models_to_test:
        print(f"\nEvaluating Model: {model_name}")
        cleanup_gpu() # Free VRAM
        
        try:
            # Load Model
            start_load = time.time()
            asr_engine = ModelClass()
            load_time = time.time() - start_load
            print(f"Model loaded in {load_time:.2f}s")
            
            # Run Inference on all files
            for row in test_suite:
                start_infer = time.time()
                try:
                    hyp_text = asr_engine.transcribe(row['audio_path'])
                except Exception as e:
                    print(f"Inference Error: {e}")
                    hyp_text = ""
                infer_time = time.time() - start_infer
                
                # Cleanup Text a bit (lower case for WER)
                ref_norm = row['text'].lower()
                hyp_norm = hyp_text.lower()
                wer = jiwer.wer(ref_norm, hyp_norm)
                
                results.append({
                    "Model": model_name,
                    "QueryID": row['id'],
                    "GroundTruth": row['text'],
                    "Transcript": hyp_text,
                    "WER": wer,
                    "InferenceTime": infer_time,
                    "AudioFile": row['audio_path']
                })
                print(f"  [{row['id']}] WER: {wer:.2f} | Time: {infer_time:.4f}s")
                
            del asr_engine # Unload
            
        except Exception as e:
            print(f"Failed to benchmark {model_name}: {e}")
            
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    
    print("\nBENCHMARK RESULTS SUMMARY")
    print("-" * 60)
    summary = df.groupby("Model")[["WER", "InferenceTime"]].mean().reset_index()
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print("-" * 60)
    print(f"Detailed results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    if not os.path.exists(WORKING_DIR):
        print("Not in Kaggle? Creating local dir.")
        os.makedirs(WORKING_DIR, exist_ok=True)
        
    try:
        install_dependencies()
        run_benchmark()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
