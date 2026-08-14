import asyncio
from pathlib import Path
import tempfile
import threading

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from app import (
    load_phowhisper_model,
    text_to_speech,
    transcribe_audio,
    translate_text_to_english,
)


BASE_DIR = Path(__file__).resolve().parent
MAX_TEXT_LENGTH = 5_000
MAX_AUDIO_SIZE_MB = 20

web_app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
web_app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_SIZE_MB * 1024 * 1024

_model = None
_model_device = None
_model_lock = threading.Lock()


def get_phowhisper_model():
    """Load the model lazily and share it between requests."""
    global _model, _model_device
    if _model is None:
        with _model_lock:
            if _model is None:
                _model, _model_device = load_phowhisper_model()
    return _model, _model_device


def get_json_text() -> str:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("Vui lòng nhập nội dung cần xử lý.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Nội dung không được vượt quá {MAX_TEXT_LENGTH:,} ký tự.")
    return text


@web_app.get("/")
def index():
    return render_template("index.html", max_text_length=MAX_TEXT_LENGTH)


@web_app.post("/api/translate")
def translate():
    text = get_json_text()
    return jsonify(
        {
            "source_text": text,
            "translated_text": translate_text_to_english(text),
        }
    )


@web_app.post("/api/transcribe")
def transcribe():
    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        raise ValueError("Không tìm thấy dữ liệu ghi âm.")

    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in {".webm", ".ogg", ".mp4", ".wav", ".m4a"}:
        suffix = ".webm"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            audio_file.save(temp_file)

        model, device = get_phowhisper_model()
        transcript = transcribe_audio(temp_path, model, device)
        if not transcript:
            raise ValueError("Không nhận diện được giọng nói trong bản ghi âm.")

        return jsonify(
            {
                "source_text": transcript,
                "translated_text": translate_text_to_english(transcript),
            }
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@web_app.post("/api/speech")
def speech():
    text = get_json_text()
    temp_path = Path(tempfile.gettempdir()) / f"photranslate-{threading.get_ident()}.mp3"
    try:
        asyncio.run(text_to_speech(text, temp_path))
        audio_bytes = temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)

    response_file = tempfile.SpooledTemporaryFile(max_size=2 * 1024 * 1024)
    response_file.write(audio_bytes)
    response_file.seek(0)
    return send_file(
        response_file,
        mimetype="audio/mpeg",
        download_name="english-speech.mp3",
    )


@web_app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, HTTPException):
        status_code = error.code or 500
        message = (
            f"Bản ghi âm không được vượt quá {MAX_AUDIO_SIZE_MB} MB."
            if status_code == 413
            else error.description
        )
    elif isinstance(error, ValueError):
        status_code = 400
        message = str(error)
    else:
        web_app.logger.exception("Unhandled request error")
        status_code = 500
        message = "Có lỗi xảy ra khi xử lý. Vui lòng thử lại."

    if request.path.startswith("/api/"):
        return jsonify({"error": message}), status_code
    return message, status_code


if __name__ == "__main__":
    web_app.run(host="127.0.0.1", port=5000, debug=False)
