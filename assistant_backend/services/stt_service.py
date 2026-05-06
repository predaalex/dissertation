from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import wave

import numpy as np
import torch


class STTUnavailableError(RuntimeError):
    """Raised when speech-to-text is not configured."""


class SpeechToTextService:
    def __init__(self, model_size: str = "base", language: str | None = None):
        self.model_size = model_size
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.backend_name = f"openai-whisper:{model_size}"

    def healthcheck(self) -> bool:
        try:
            import whisper  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str | None = None,
        task: str = "transcribe",
        language: str | None = None,
    ) -> str:
        self._ensure_model_loaded()
        if task not in {"transcribe", "translate"}:
            raise STTUnavailableError("Whisper task must be either 'transcribe' or 'translate'.")

        audio_input = self._prepare_audio_input(audio_bytes, filename=filename)
        options = {
            "fp16": self.device == "cuda",
            "verbose": False,
            "task": task,
        }
        effective_language = language if language is not None else self.language
        if effective_language:
            options["language"] = effective_language

        temp_path = None
        try:
            if isinstance(audio_input, Path):
                temp_path = audio_input
                result = self.model.transcribe(str(audio_input), **options)
            else:
                result = self.model.transcribe(audio_input, **options)
        except Exception as exc:  # noqa: BLE001
            raise STTUnavailableError(f"Whisper failed to transcribe audio: {exc}") from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        return str(result.get("text", "")).strip()

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        try:
            import whisper
        except Exception as exc:  # noqa: BLE001
            raise STTUnavailableError(
                "OpenAI Whisper is not installed. Install backend requirements before using speech-to-text."
            ) from exc

        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
        except Exception as exc:  # noqa: BLE001
            raise STTUnavailableError(
                f"Failed to load Whisper model '{self.model_size}': {exc}"
            ) from exc

    def _prepare_audio_input(self, audio_bytes: bytes, filename: str | None = None):
        try:
            waveform, sample_rate = self._read_wav_bytes(audio_bytes)
            if sample_rate == 16000:
                return waveform
        except STTUnavailableError:
            raise
        except Exception:  # noqa: BLE001
            pass

        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            return Path(temp_file.name)

    @staticmethod
    def _read_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw_audio = wav_file.readframes(frame_count)

        if sample_width == 2:
            waveform = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            waveform = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sample_width == 1:
            waveform = (np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise STTUnavailableError(f"Unsupported WAV sample width: {sample_width} bytes.")

        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1)

        return waveform.astype(np.float32), sample_rate
