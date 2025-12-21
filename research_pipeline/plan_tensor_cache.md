# Plan: Tensor Caching for Faster Benchmarking

## 1. Pre-processing Step
Add a function `preprocess_to_tensors()`:
```python
def preprocess_to_tensors(test_suite):
    print("Preprocessing audio to Tensors (16kHz)...")
    os.makedirs(TENSOR_DIR, exist_ok=True)
    
    for row in test_suite:
        # Load and Resample once
        arr, _ = librosa.load(row['audio_path'], sr=16000)
        tensor_path = row['audio_path'].replace(".mp3", ".pt").replace("/audio/", "/tensors/")
        
        # Save as Torch Tensor
        torch.save(torch.from_numpy(arr), tensor_path)
        row['tensor_path'] = tensor_path # Update row meta
```

## 2. Model Wrapper Updates
Modify `transcribe(audio_input)` to handle both Path (str) and Signal (Tensor/Numpy).
- **PhoWhisper (Pipeline)**: Accepts numpy array directly `{"raw": arr, "sampling_rate": 16000}`.
- **Wav2Vec2/Chunkformer**: Accepts numpy array directly in `processor(arr, ...)`.

## 3. Storage
- Zip `tensors/` folder as `tensors_dataset.zip`.
- This is much faster for the user to load next time (no MP3 decoding overhead).
