import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    whisper_language: str | None = os.getenv("WHISPER_LANGUAGE")
    session_history_limit: int = int(os.getenv("SESSION_HISTORY_LIMIT", "10"))
    emotion_vote_window: int = int(os.getenv("EMOTION_VOTE_WINDOW", "5"))
    emotion_confidence_threshold: float = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.0"))
    emotion_face_cropping_enabled: bool = os.getenv(
        "EMOTION_FACE_CROPPING_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    project_root: Path = Path(__file__).resolve().parent.parent
    emotion_model_path: str = os.getenv(
        "EMOTION_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "facial_classifier" / "models" / "20260426_183732_finetune_mobilenetv2_finetune_classweighted_strong_fkvjn8eq_bs32_lr3e-04_frac1.00_f10p5369.pt"),
    )
