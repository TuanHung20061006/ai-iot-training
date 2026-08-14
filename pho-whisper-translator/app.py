import asyncio
import os
from pathlib import Path
import struct
import time
import wave

import edge_tts
import psutil
import torch
from deep_translator import GoogleTranslator
from pvrecorder import PvRecorder
from transformers import pipeline

# Save recordings in the project's audio_file directory.
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_file"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_PATH = AUDIO_DIR / "audio_recording.wav"
ENGLISH_SPEECH_PATH = AUDIO_DIR / "english_speech.mp3"
PHOWHISPER_MODEL_NAME = "vinai/PhoWhisper-small"
EDGE_TTS_VOICE = "en-US-EmmaMultilingualNeural"
RECORDING_DURATION_SECONDS = 20


def print_resource_usage(label: str) -> None:
    """Print CPU, RAM, and CUDA memory used by this process."""
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    ram_mb = process.memory_info().rss / (1024 ** 2)

    print(f"\n--- {label} ---")
    print(f"CPU usage: {cpu_percent:.1f}%")
    print(f"RAM usage: {ram_mb:.2f} MB")

    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU memory allocated: {allocated_mb:.2f} MB")
        print(f"GPU memory reserved: {reserved_mb:.2f} MB")


def load_phowhisper_model(model_name: str = PHOWHISPER_MODEL_NAME):
    """Load PhoWhisper once and return the pipeline together with its device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading PhoWhisper model '{model_name}' on {device}...")

    pipeline_kwargs = {
        "task": "automatic-speech-recognition",
        "model": model_name,
        "device": 0 if device == "cuda" else -1,
    }
    if device == "cuda":
        pipeline_kwargs["dtype"] = torch.float16

    return pipeline(**pipeline_kwargs), device


def transcribe_audio(audio_path: Path, model, device: str) -> str:
    """Convert a saved audio file to Vietnamese text."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("Transcribing audio...")
    result = model(
        str(audio_path),
        generate_kwargs={"language": "vi", "task": "transcribe"},
    )
    return result["text"].strip()


def translate_text_to_english(text: str) -> str:
    """Translate Vietnamese text to English text."""
    if not text.strip():
        raise ValueError("Cannot translate empty text.")

    print("Translating Vietnamese text to English...")
    translated_text = GoogleTranslator(source="vi", target="en").translate(text)
    return translated_text.strip()


async def text_to_speech(
    text: str,
    output_path: Path,
    voice: str = EDGE_TTS_VOICE,
) -> Path:
    """Convert English text to speech and save it as an MP3 file."""
    if not text.strip():
        raise ValueError("Cannot create speech from empty text.")

    print(f"Generating English speech with voice '{voice}'...")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))
    return output_path

def main() -> None:
    """Run the original command-line recording and translation workflow."""
    # Load all audio input devices.
    for index, device in enumerate(PvRecorder.get_available_devices()):
        print(f"[{index}] {device}")

    input_device_index = int(input("Select input device index: "))
    # A frame contains 32 milliseconds of 16 kHz audio.
    recorder = PvRecorder(device_index=input_device_index, frame_length=512)
    audio = []
    target_samples = RECORDING_DURATION_SECONDS * 16000

    try:
        recorder.start()
        print(f"Recording for {RECORDING_DURATION_SECONDS} seconds...")
        while len(audio) < target_samples:
            frame = recorder.read()
            audio.extend(frame)
    except KeyboardInterrupt:
        print("\nRecording stopped early.")
    finally:
        recorder.stop()
        recorder.delete()

    # The recorder returns fixed-size frames, so trim the final frame to 20 seconds.
    audio = audio[:target_samples]
    with wave.open(str(AUDIO_PATH), "wb") as audio_file:
        audio_file.setparams((1, 2, 16000, len(audio), "NONE", "NONE"))
        audio_file.writeframes(struct.pack("h" * len(audio), *audio))
    print(f"Audio saved to: {AUDIO_PATH}")

    print_resource_usage("Before loading PhoWhisper")
    phowhisper_model, phowhisper_device = load_phowhisper_model()
    print_resource_usage("After loading PhoWhisper")

    transcription_start = time.perf_counter()
    transcript = transcribe_audio(
        AUDIO_PATH,
        phowhisper_model,
        phowhisper_device,
    )
    transcription_time = time.perf_counter() - transcription_start

    print_resource_usage("After transcription")
    print(f"\nTranscription time: {transcription_time:.2f} seconds")

    english_translation = translate_text_to_english(transcript)
    print(f"Vietnamese transcript: {transcript}")
    print(f"English translation: {english_translation}")

    speech_path = asyncio.run(
        text_to_speech(english_translation, ENGLISH_SPEECH_PATH)
    )
    print(f"English speech saved to: {speech_path}")

    print("CUDA available:", torch.cuda.is_available())
    print("PhoWhisper device:", phowhisper_device)


if __name__ == "__main__":
    main()
