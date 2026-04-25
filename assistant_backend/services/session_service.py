from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


class SessionNotFoundError(KeyError):
    """Raised when a session id does not exist."""


class SessionService:
    def __init__(self, history_limit: int = 10):
        self.history_limit = history_limit
        self._sessions: dict[str, dict] = {}

    def create_session(self) -> dict:
        session_id = str(uuid4())
        session = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "latest_emotion": None,
            "history": [],
        }
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        del self._sessions[session_id]

    def update_emotion(self, session_id: str, emotion_label: str, confidence: float | None) -> dict:
        session = self.get_session(session_id)
        session["latest_emotion"] = {
            "emotion_label": emotion_label,
            "confidence": confidence,
            "updated_at": datetime.now(timezone.utc),
        }
        return session["latest_emotion"]

    def get_latest_emotion(self, session_id: str) -> dict | None:
        return self.get_session(session_id)["latest_emotion"]

    def append_message(self, session_id: str, role: str, content: str) -> dict:
        session = self.get_session(session_id)
        message = {
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }
        session["history"].append(message)
        session["history"] = session["history"][-self.history_limit :]
        return message
