import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    session_history_limit: int = int(os.getenv("SESSION_HISTORY_LIMIT", "10"))
    emotion_vote_window: int = int(os.getenv("EMOTION_VOTE_WINDOW", "5"))
    emotion_confidence_threshold: float = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.0"))
    project_root: Path = Path(__file__).resolve().parent.parent
    emotion_model_path: str = os.getenv(
        "EMOTION_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "facial_classifier" / "models" / "latest_baseline.pt"),
    )
