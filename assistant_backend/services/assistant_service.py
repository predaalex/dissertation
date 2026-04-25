from __future__ import annotations

from typing import Any


class AssistantService:
    def __init__(self, history_limit: int = 10):
        self.history_limit = history_limit

    def build_prompt(
        self,
        user_text: str,
        history: list[dict[str, Any]],
        latest_emotion: dict[str, Any] | None,
    ) -> str:
        lines = [
            "You are an emotion-aware desktop AI assistant.",
            "Use the user's detected facial emotion as soft context, not certainty.",
            "Be supportive, concise, and helpful.",
        ]

        if latest_emotion is None:
            lines.append("No recent emotion prediction is available.")
        else:
            confidence = latest_emotion.get("confidence")
            if confidence is None:
                lines.append(
                    f"The user's latest detected facial emotion is {latest_emotion['emotion_label']}. "
                    "Treat this as soft context, not certainty."
                )
            else:
                lines.append(
                    f"The user's latest detected facial emotion is {latest_emotion['emotion_label']} "
                    f"with confidence {confidence:.2f}. Treat this as soft context, not certainty."
                )

        lines.append("")
        lines.append("Recent conversation:")
        trimmed_history = history[-self.history_limit :]
        if not trimmed_history:
            lines.append("(no previous conversation)")
        else:
            for item in trimmed_history:
                lines.append(f"{item['role'].upper()}: {item['content']}")

        lines.append("")
        lines.append(f"USER: {user_text}")
        lines.append("ASSISTANT:")

        return "\n".join(lines)
