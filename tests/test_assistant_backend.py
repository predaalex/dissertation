from __future__ import annotations

from fastapi.testclient import TestClient

from assistant_backend.main import create_app
from assistant_backend.schemas import OllamaStreamEvent


class FakeLLMService:
    def __init__(self, events=None, healthy=True, should_fail=False):
        self.events = events or [
            OllamaStreamEvent(response="Hello", done=False, raw={}),
            OllamaStreamEvent(response=" world", done=True, eval_count=2, raw={}),
        ]
        self.healthy = healthy
        self.should_fail = should_fail

    def healthcheck(self, timeout: float = 2.0) -> bool:
        return self.healthy

    def stream_generate(self, prompt: str):
        if self.should_fail:
            from assistant_backend.services.llm_service import OllamaServiceError

            raise OllamaServiceError("Ollama unavailable")
        yield from self.events


class FakeSTTService:
    backend_name = "fake-stt"

    def __init__(self, transcript="transcribed audio", healthy=True):
        self.transcript = transcript
        self.healthy = healthy

    def healthcheck(self) -> bool:
        return self.healthy

    def transcribe(self, audio_bytes: bytes, filename: str | None = None) -> str:
        return self.transcript


def build_client() -> TestClient:
    app = create_app()
    app.state.llm_service = FakeLLMService()
    app.state.stt_service = FakeSTTService()
    return TestClient(app)


def test_health_reports_services():
    client = build_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["services"]["ollama"] is True
    assert payload["services"]["speech_to_text"] is True


def test_streamed_assistant_response_text_only():
    client = build_client()
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/emotion",
        json={"emotion_label": "Happy", "confidence": 0.88},
    )

    response = client.post(
        "/assistant/respond",
        data={"session_id": session_id, "text": "Hello there", "use_cached_emotion": "true"},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: metadata" in body
    assert "event: token" in body
    assert "event: done" in body
    assert '"emotion_label": "Happy"' in body
    assert '"response_text": "Hello world"' in body

    session_state = client.get(f"/sessions/{session_id}").json()
    assert len(session_state["history"]) == 2
    assert session_state["history"][0]["role"] == "user"
    assert session_state["history"][1]["role"] == "assistant"


def test_streamed_assistant_response_with_audio():
    client = build_client()
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(
        "/assistant/respond",
        data={"session_id": session_id},
        files={"audio": ("sample.wav", b"fake-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.text
    assert '"transcript_used": "transcribed audio"' in body
    assert "event: done" in body


def test_streamed_assistant_response_missing_session():
    client = build_client()
    response = client.post(
        "/assistant/respond",
        data={"session_id": "missing", "text": "Hello"},
    )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"stage": "session"' in response.text


def test_streamed_assistant_response_missing_input():
    client = build_client()
    session_id = client.post("/sessions").json()["session_id"]
    response = client.post("/assistant/respond", data={"session_id": session_id})
    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"stage": "validation"' in response.text


def test_streamed_assistant_response_ollama_failure_does_not_persist_partial():
    app = create_app()
    app.state.llm_service = FakeLLMService(should_fail=True)
    app.state.stt_service = FakeSTTService()
    client = TestClient(app)

    session_id = client.post("/sessions").json()["session_id"]
    response = client.post(
        "/assistant/respond",
        data={"session_id": session_id, "text": "Hello"},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"stage": "ollama"' in response.text
    session_state = client.get(f"/sessions/{session_id}").json()
    assert session_state["history"] == []
