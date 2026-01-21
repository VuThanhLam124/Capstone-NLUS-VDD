---
title: SQL ASR Test
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
---

# SQL ASR Test

This Space transcribes audio and applies SQL-aware keyword correction.

## Features
- SQL prompt to bias ASR toward keywords
- Post-processing to fix common SQL keyword errors (SELECT, JOIN, TOP, ...)
- Model + language selection for multilingual use (Whisper, PhoWhisper, Chunkformer)

## Environment variables
- `ASR_MODEL_ID`: default `openai/whisper-medium` (supports Whisper, PhoWhisper, Chunkformer)
- `ASR_LANGUAGE`: default `auto` (options: `auto`, `vietnamese`, `english`)
- `ASR_CHUNK_LENGTH_S`: default `30` (Whisper/PhoWhisper only)
- `SQL_PROMPT_EN`: English prompt text
- `SQL_PROMPT_VI`: Vietnamese prompt text
- `SQL_PROMPT_AUTO`: prompt text used for `auto` language (defaults to EN)
- `USE_SQL_PROMPT`: set to `0` to disable prompt
- `APPLY_SQL_CORRECTION`: set to `0` to disable correction
- `CHUNKFORMER_CHUNK_SIZE`: default `64`
- `CHUNKFORMER_LEFT_CONTEXT`: default `128`
- `CHUNKFORMER_RIGHT_CONTEXT`: default `128`

## Notes
- Chunkformer is Vietnamese-only; use Whisper/PhoWhisper for English.
- For best accuracy, use a GPU Space or a smaller model.
