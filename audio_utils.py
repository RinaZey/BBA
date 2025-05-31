# audio_utils.py

import os
import warnings
from pathlib import Path
import wave
from pydub import AudioSegment
import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import json as _json

# ────────────────────────────────────────────────────────────
# директория проекта (чтобы найти модель Vosk)
# ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ────────────────────────────────────────────────────────────
# Vosk: загрузка оффлайн-модели
# ────────────────────────────────────────────────────────────
VOSK_MODEL = Model(str(BASE_DIR / "models" / "vosk-small-ru"))

print("model_dir =", BASE_DIR / "models" / "vosk-small-ru")
print("contains =", os.listdir(BASE_DIR / "models" / "vosk-small-ru"))

# ────────────────────────────────────────────────────────────
# ffmpeg (для pydub)
# ────────────────────────────────────────────────────────────
FFMPEG_BIN = r"C:\Users\kolya\ffmpeg-2025-05-26-git-43a69886b2-full_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
warnings.filterwarnings("ignore", ".*ffmpeg or avconv.*", category=RuntimeWarning)
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# ────────────────────────────────────────────────────────────
# инициализация распознавания и синтеза речи
# ────────────────────────────────────────────────────────────
recognizer = sr.Recognizer()

engine = pyttsx3.init()
voices = engine.getProperty('voices')
for v in voices:
    # ищем русский мужской голос
    if "pavel" in v.id.lower() or ("russian" in v.name.lower() and "male" in v.name.lower()):
        engine.setProperty('voice', v.id)
        break
engine.setProperty('rate', 140)  # чуть медленнее среднего

# ────────────────────────────────────────────────────────────
def ogg_to_wav(ogg_path: str, wav_path: str):
    """Конвертирует .ogg → .wav (16 kHz, моно)."""
    audio = AudioSegment.from_file(ogg_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format='wav')

def wav_to_ogg(wav_path: str, ogg_path: str):
    """Конвертирует .wav → .ogg."""
    audio = AudioSegment.from_file(wav_path)
    audio.export(ogg_path, format='ogg')

def stt_from_wav(wav_path: str) -> str:
    """Распознаёт русскую речь офлайн через Vosk."""
    import wave, json as _json
    with wave.open(wav_path, "rb") as wf:
        rec = KaldiRecognizer(VOSK_MODEL, wf.getframerate())
        rec.SetWords(True)
        # читаем покрупнее: 8000 фреймов ≈ 0.5 c при 16 kHz
        while True:
            data = wf.readframes(8000)
            if not data:
                break
            rec.AcceptWaveform(data)
        result = _json.loads(rec.FinalResult())
        return result.get("text", "").strip()

def tts_to_ogg(text: str, ogg_path: str):
    """
    Синтезирует текст в mp3 через pyttsx3, конвертирует в ogg и сохраняет.
    """
    tmp_mp3 = ogg_path.replace(".ogg", ".mp3")
    engine.save_to_file(text, tmp_mp3)
    engine.runAndWait()
    # конвертируем mp3 → ogg
    audio = AudioSegment.from_file(tmp_mp3)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(ogg_path, format='ogg')
    os.remove(tmp_mp3)
