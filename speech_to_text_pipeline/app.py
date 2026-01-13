import os
import sys
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

sys.path.append(os.path.dirname(__file__))

from sql_correction import correct_sql_keywords, SQL_KEYWORDS

CHUNKFORMER_MODEL_ID = "khanhld/chunkformer-ctc-large-vie"

MODEL_CHOICES = [
    ("Whisper (medium)", "openai/whisper-medium"),
    ("Whisper (large-v3)", "openai/whisper-large-v3"),
    ("PhoWhisper (base)", "vinai/PhoWhisper-base"),
    ("PhoWhisper (large)", "vinai/PhoWhisper-large"),
    ("Chunkformer (large-vie)", CHUNKFORMER_MODEL_ID),
]

MODEL_ID_DEFAULT = os.getenv("ASR_MODEL_ID", "openai/whisper-medium").strip()
MODEL_VALUES = [value for _, value in MODEL_CHOICES]
if MODEL_ID_DEFAULT not in MODEL_VALUES:
    MODEL_CHOICES = [("Custom model", MODEL_ID_DEFAULT)] + MODEL_CHOICES

LANGUAGE_ALIASES = {
    "": "auto",
    "auto": "auto",
    "vi": "vietnamese",
    "vietnamese": "vietnamese",
    "en": "english",
    "english": "english",
}

LANGUAGE_DEFAULT_RAW = os.getenv("ASR_LANGUAGE", "auto").strip().lower()
LANGUAGE_DEFAULT = LANGUAGE_ALIASES.get(LANGUAGE_DEFAULT_RAW, "auto")
LANGUAGE_OPTIONS = ["auto", "vietnamese", "english"]

CHUNK_LENGTH_S = int(os.getenv("ASR_CHUNK_LENGTH_S", "30"))

DEFAULT_PROMPT_EN = (
    "SQL keywords: SELECT, FROM, WHERE, JOIN, LEFT JOIN, RIGHT JOIN, "
    "INNER JOIN, OUTER JOIN, GROUP BY, ORDER BY, LIMIT, TOP, DISTINCT, "
    "INSERT, UPDATE, DELETE, CREATE TABLE, ALTER TABLE, DROP TABLE, VALUES, "
    "INTO, ON, AS, HAVING, COUNT, AVG, MIN, MAX, SUM."
)
DEFAULT_PROMPT_VI = (
    "Tu khoa SQL: SELECT, FROM, WHERE, JOIN, LEFT JOIN, RIGHT JOIN, "
    "INNER JOIN, OUTER JOIN, GROUP BY, ORDER BY, LIMIT, TOP, DISTINCT, "
    "INSERT, UPDATE, DELETE, CREATE TABLE, ALTER TABLE, DROP TABLE, VALUES, "
    "INTO, ON, AS, HAVING, COUNT, AVG, MIN, MAX, SUM."
)

PROMPT_TEXT_EN = os.getenv("SQL_PROMPT_EN", DEFAULT_PROMPT_EN)
PROMPT_TEXT_VI = os.getenv("SQL_PROMPT_VI", DEFAULT_PROMPT_VI)
PROMPT_TEXT_AUTO = os.getenv("SQL_PROMPT_AUTO", PROMPT_TEXT_EN)

USE_SQL_PROMPT_DEFAULT = os.getenv("USE_SQL_PROMPT", "1") != "0"
APPLY_SQL_CORRECTION_DEFAULT = os.getenv("APPLY_SQL_CORRECTION", "1") != "0"

CHUNKFORMER_CHUNK_SIZE = int(os.getenv("CHUNKFORMER_CHUNK_SIZE", "64"))
CHUNKFORMER_LEFT_CONTEXT = int(os.getenv("CHUNKFORMER_LEFT_CONTEXT", "128"))
CHUNKFORMER_RIGHT_CONTEXT = int(os.getenv("CHUNKFORMER_RIGHT_CONTEXT", "128"))

DEVICE_INDEX = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

PIPELINES = {}
PROMPT_IDS_CACHE = {}
CHUNKFORMER_MODELS = {}


def is_chunkformer(model_id):
    return model_id == CHUNKFORMER_MODEL_ID


def get_pipeline(model_id):
    pipe = PIPELINES.get(model_id)
    if pipe is None:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=DEVICE_INDEX,
            torch_dtype=DTYPE,
            chunk_length_s=CHUNK_LENGTH_S,
        )
        PIPELINES[model_id] = pipe
    return pipe


def get_prompt_text(language):
    if language == "vietnamese":
        return PROMPT_TEXT_VI
    if language == "english":
        return PROMPT_TEXT_EN
    return PROMPT_TEXT_AUTO


def get_prompt_ids(model_id, prompt_text):
    if not prompt_text:
        return None
    cache_key = (model_id, prompt_text)
    if cache_key in PROMPT_IDS_CACHE:
        return PROMPT_IDS_CACHE[cache_key]

    pipe = get_pipeline(model_id)
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "get_prompt_ids"):
        PROMPT_IDS_CACHE[cache_key] = None
        return None

    try:
        prompt_ids = tokenizer.get_prompt_ids(prompt_text, return_tensors="pt")
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
        if prompt_ids.ndim > 1:
            prompt_ids = prompt_ids.squeeze(0)
        prompt_ids = prompt_ids.to(pipe.model.device, dtype=torch.long)
    except Exception:
        prompt_ids = None
    PROMPT_IDS_CACHE[cache_key] = prompt_ids
    return prompt_ids


def build_generate_kwargs(model_id, language, use_prompt):
    kwargs = {"task": "transcribe"}
    if language and language != "auto":
        kwargs["language"] = language
    if use_prompt:
        prompt_text = get_prompt_text(language)
        prompt_ids = get_prompt_ids(model_id, prompt_text)
        if prompt_ids is not None:
            kwargs["prompt_ids"] = prompt_ids
    return kwargs


def get_chunkformer_model(model_id):
    model = CHUNKFORMER_MODELS.get(model_id)
    if model is None:
        from chunkformer import ChunkFormerModel

        model = ChunkFormerModel.from_pretrained(model_id)
        try:
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            pass
        CHUNKFORMER_MODELS[model_id] = model
    return model


def prepare_chunkformer_audio(audio_path):
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    return audio, sample_rate


def transcribe_chunkformer(audio_path, model_id):
    model = get_chunkformer_model(model_id)
    audio, sample_rate = prepare_chunkformer_audio(audio_path)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name

        hyp = model.endless_decode(
            audio_path=tmp_path,
            chunk_size=CHUNKFORMER_CHUNK_SIZE,
            left_context_size=CHUNKFORMER_LEFT_CONTEXT,
            right_context_size=CHUNKFORMER_RIGHT_CONTEXT,
            return_timestamps=False,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return hyp if isinstance(hyp, str) else str(hyp)


def transcribe(audio_path, model_id, language, use_prompt, apply_correction):
    if not audio_path:
        return "", ""

    try:
        if is_chunkformer(model_id):
            text = transcribe_chunkformer(audio_path, model_id)
        else:
            pipe = get_pipeline(model_id)
            generate_kwargs = build_generate_kwargs(model_id, language, use_prompt)
            with torch.inference_mode():
                result = pipe(audio_path, generate_kwargs=generate_kwargs)
            text = result["text"] if isinstance(result, dict) else str(result)
    except Exception as exc:
        error_message = f"Error: {exc}"
        return error_message, error_message

    corrected = (
        correct_sql_keywords(text, enable_fuzzy=True)
        if apply_correction
        else text
    )
    return text, corrected


with gr.Blocks(title="SQL ASR Test") as demo:
    gr.Markdown(
        "# SQL ASR Test\n"
        "Upload audio and compare raw vs SQL-corrected transcripts."
    )

    with gr.Row():
        audio = gr.Audio(type="filepath", label="Audio")

    with gr.Row():
        model_id = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=MODEL_ID_DEFAULT,
            label="ASR model",
        )
        language = gr.Dropdown(
            choices=LANGUAGE_OPTIONS,
            value=LANGUAGE_DEFAULT,
            label="Language",
        )

    with gr.Row():
        use_prompt = gr.Checkbox(
            value=USE_SQL_PROMPT_DEFAULT,
            label="Use SQL prompt",
        )
        apply_correction = gr.Checkbox(
            value=APPLY_SQL_CORRECTION_DEFAULT,
            label="Apply SQL keyword correction",
        )

    run_btn = gr.Button("Transcribe")

    raw_text = gr.Textbox(label="Raw transcript", lines=6)
    corrected_text = gr.Textbox(label="SQL-corrected transcript", lines=6)

    run_btn.click(
        transcribe,
        inputs=[audio, model_id, language, use_prompt, apply_correction],
        outputs=[raw_text, corrected_text],
    )

    keyword_list = ", ".join(SQL_KEYWORDS)
    gr.Markdown(f"**SQL keywords:** {keyword_list}")


if __name__ == "__main__":
    demo.launch()
