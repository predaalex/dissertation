from __future__ import annotations

import json
from typing import Iterator

import requests

from assistant_backend.schemas import OllamaStreamEvent


class OllamaServiceError(RuntimeError):
    """Raised when the local Ollama service fails."""


class OllamaService:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def healthcheck(self, timeout: float = 2.0) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            return response.ok
        except requests.RequestException:
            return False

    def stream_generate(self, prompt: str) -> Iterator[OllamaStreamEvent]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=(5, 300),
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        raise OllamaServiceError("Invalid JSON chunk received from Ollama.") from exc

                    yield OllamaStreamEvent(
                        response=data.get("response", ""),
                        done=bool(data.get("done", False)),
                        total_duration=data.get("total_duration"),
                        eval_count=data.get("eval_count"),
                        eval_duration=data.get("eval_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        prompt_eval_duration=data.get("prompt_eval_duration"),
                        done_reason=data.get("done_reason"),
                        raw=data,
                    )
        except requests.RequestException as exc:
            raise OllamaServiceError(f"Failed to reach Ollama: {exc}") from exc
