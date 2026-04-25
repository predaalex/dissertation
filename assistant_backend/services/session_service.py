from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4


class SessionNotFoundError(KeyError):
    """Raised when a session id does not exist."""


class SessionService:
    def __init__(
        self,
        history_limit: int = 10,
        emotion_vote_window: int = 5,
        emotion_confidence_threshold: float = 0.0,
    ):
        self.history_limit = history_limit
        self.emotion_vote_window = emotion_vote_window
        self.emotion_confidence_threshold = emotion_confidence_threshold
        self._sessions: dict[str, dict] = {}

    def create_session(self) -> dict:
        session_id = str(uuid4())
        session = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "latest_emotion": None,
            "emotion_history": [],
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
        accepted = confidence is None or confidence >= self.emotion_confidence_threshold
        emotion_event = {
            "emotion_label": emotion_label,
            "confidence": confidence,
            "updated_at": datetime.now(timezone.utc),
            "accepted": accepted,
        }

        if accepted:
            session["emotion_history"].append(emotion_event)
            session["emotion_history"] = session["emotion_history"][-self.emotion_vote_window :]
            session["latest_emotion"] = self._compute_majority_emotion(session["emotion_history"])
        elif session["latest_emotion"] is None:
            session["latest_emotion"] = {
                "emotion_label": emotion_label,
                "confidence": confidence,
                "updated_at": emotion_event["updated_at"],
                "vote_count": 0,
                "window_size": 0,
                "accepted": False,
            }

        latest = dict(session["latest_emotion"]) if session["latest_emotion"] is not None else None
        if latest is None:
            latest = {
                "emotion_label": emotion_label,
                "confidence": confidence,
                "updated_at": emotion_event["updated_at"],
                "vote_count": 0,
                "window_size": 0,
            }
        latest["accepted"] = accepted
        latest["raw_emotion_label"] = emotion_label
        latest["raw_confidence"] = confidence
        latest["confidence_threshold"] = self.emotion_confidence_threshold
        session["latest_emotion"] = latest
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

    @staticmethod
    def _compute_majority_emotion(emotion_history: list[dict]) -> dict | None:
        if not emotion_history:
            return None

        counts = Counter(item["emotion_label"] for item in emotion_history)
        highest_vote_count = max(counts.values())
        candidates = {label for label, count in counts.items() if count == highest_vote_count}

        majority_label = None
        majority_event = None
        for event in reversed(emotion_history):
            if event["emotion_label"] in candidates:
                majority_label = event["emotion_label"]
                majority_event = event
                break

        majority_confidences = [
            item["confidence"]
            for item in emotion_history
            if item["emotion_label"] == majority_label and item["confidence"] is not None
        ]
        majority_confidence = (
            sum(majority_confidences) / len(majority_confidences)
            if majority_confidences
            else majority_event["confidence"]
        )

        return {
            "emotion_label": majority_label,
            "confidence": majority_confidence,
            "updated_at": majority_event["updated_at"],
            "vote_count": highest_vote_count,
            "window_size": len(emotion_history),
            "accepted": True,
        }
