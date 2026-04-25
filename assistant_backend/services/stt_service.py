from __future__ import annotations


class STTUnavailableError(RuntimeError):
    """Raised when speech-to-text is not configured."""


class SpeechToTextService:
    backend_name = "unconfigured"

    def healthcheck(self) -> bool:
        return False

    def transcribe(self, audio_bytes: bytes, filename: str | None = None) -> str:
        raise STTUnavailableError(
            "Speech-to-text is not configured yet. The endpoint is available, but no STT backend is attached."
        )
