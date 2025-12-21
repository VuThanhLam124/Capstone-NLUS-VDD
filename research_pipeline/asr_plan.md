# ASR Benchmark Plan
## Models
1. **PhoWhisper-large**: `vinai/PhoWhisper-large`
   - Use `pipeline("automatic-speech-recognition")` from transformers.
2. **Wav2Vec2-base-vi**: `nguyenvulebinh/wav2vec2-base-vi-vlsp2020`
   - Requires `Wav2Vec2Processor` and `Wav2Vec2ForCTC`.
3. **Chunkformer**: `khanhld/chunkformer-ctc-large-vie` (Need verification, fallback to `nguyenvulebinh/w2v-bert-base` or similar if not found, but assumming exists).

## Data Update
- Change test queries to Vietnamese (e.g., "Tìm top 10 sản phẩm...").
- Use `edge-tts` voice `vi-VN-NamMinhNeural`.

## Execution Flow
For each audio sample:
  For each Model:
    Start Timer
    Transcribe
    End Timer
    Calculate WER
    Store Result

Display Table using `pandas` or `tabulate`.
