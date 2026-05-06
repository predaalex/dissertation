from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from assistant_backend.config import Settings
from assistant_backend.schemas import (
    CreateSessionResponse,
    EmotionPredictionResponse,
    EmotionState,
    EmotionUpdateRequest,
    HealthResponse,
    SessionEmotionPredictionResponse,
    SessionStateResponse,
    TranscriptionResponse,
)
from assistant_backend.services.assistant_service import AssistantService
from assistant_backend.services.emotion_service import EmotionService, EmotionServiceError
from assistant_backend.services.llm_service import OllamaService, OllamaServiceError
from assistant_backend.services.session_service import SessionNotFoundError, SessionService
from assistant_backend.services.stt_service import STTUnavailableError, SpeechToTextService


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="Emotion-Aware Assistant Backend", version="0.1.0")

    app.state.settings = settings
    app.state.session_service = SessionService(
        history_limit=settings.session_history_limit,
        emotion_vote_window=settings.emotion_vote_window,
        emotion_confidence_threshold=settings.emotion_confidence_threshold,
    )
    app.state.assistant_service = AssistantService(history_limit=settings.session_history_limit)
    app.state.llm_service = OllamaService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    app.state.stt_service = SpeechToTextService(
        model_size=settings.whisper_model_size,
        language=settings.whisper_language,
    )
    app.state.emotion_service = EmotionService(
        checkpoint_path=settings.emotion_model_path,
        face_cropping_enabled=settings.emotion_face_cropping_enabled,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        llm_ready = app.state.llm_service.healthcheck()
        stt_ready = app.state.stt_service.healthcheck()
        emotion_ready = app.state.emotion_service.healthcheck()
        overall = "ok" if llm_ready and emotion_ready else "degraded"
        return HealthResponse(
            status=overall,
            services={
                "ollama": llm_ready,
                "speech_to_text": stt_ready,
                "emotion_model": emotion_ready,
            },
            model=app.state.settings.ollama_model,
        )

    @app.post("/sessions", response_model=CreateSessionResponse)
    def create_session() -> CreateSessionResponse:
        session = app.state.session_service.create_session()
        return CreateSessionResponse(
            session_id=session["session_id"],
            created_at=session["created_at"],
        )

    @app.get("/sessions/{session_id}", response_model=SessionStateResponse)
    def get_session(session_id: str) -> SessionStateResponse:
        try:
            session = app.state.session_service.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found.")

        latest_emotion = session["latest_emotion"]
        emotion_obj = EmotionState(**latest_emotion) if latest_emotion is not None else None
        return SessionStateResponse(
            session_id=session["session_id"],
            created_at=session["created_at"],
            latest_emotion=emotion_obj,
            history=session["history"],
        )

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> JSONResponse:
        try:
            app.state.session_service.delete_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found.")
        return JSONResponse({"deleted": True, "session_id": session_id})

    @app.post("/sessions/{session_id}/emotion", response_model=EmotionState)
    def update_session_emotion(session_id: str, payload: EmotionUpdateRequest) -> EmotionState:
        try:
            emotion = app.state.session_service.update_emotion(
                session_id=session_id,
                emotion_label=payload.emotion_label,
                confidence=payload.confidence,
            )
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found.")
        return EmotionState(**emotion)

    @app.post("/emotion/predict", response_model=EmotionPredictionResponse)
    async def predict_emotion(
        image: UploadFile = File(...),
        disable_face_cropping: bool = Form(default=False),
    ) -> EmotionPredictionResponse:
        image_bytes = await image.read()
        try:
            prediction = app.state.emotion_service.predict(
                image_bytes,
                disable_face_cropping=disable_face_cropping,
            )
        except EmotionServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return EmotionPredictionResponse(**prediction)

    @app.post("/sessions/{session_id}/emotion/predict", response_model=SessionEmotionPredictionResponse)
    async def predict_and_cache_emotion(
        session_id: str,
        image: UploadFile = File(...),
        disable_face_cropping: bool = Form(default=False),
    ) -> SessionEmotionPredictionResponse:
        try:
            app.state.session_service.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found.")

        image_bytes = await image.read()
        try:
            prediction = app.state.emotion_service.predict(
                image_bytes,
                disable_face_cropping=disable_face_cropping,
            )
        except EmotionServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        emotion_state = app.state.session_service.update_emotion(
            session_id=session_id,
            emotion_label=prediction["emotion_label"],
            confidence=prediction["confidence"],
        )

        return SessionEmotionPredictionResponse(
            session_id=session_id,
            emotion_label=emotion_state["emotion_label"],
            emotion_index=prediction["emotion_index"],
            confidence=emotion_state["confidence"],
            probabilities=prediction["probabilities"],
            updated_at=emotion_state["updated_at"],
            vote_count=emotion_state.get("vote_count"),
            window_size=emotion_state.get("window_size"),
            accepted=emotion_state.get("accepted"),
            raw_emotion_label=emotion_state.get("raw_emotion_label"),
            raw_confidence=emotion_state.get("raw_confidence"),
            confidence_threshold=emotion_state.get("confidence_threshold"),
            face_detected=prediction["face_detected"],
            face_crop_applied=prediction["face_crop_applied"],
            face_bbox=prediction["face_bbox"],
        )

    @app.post("/speech-to-text/transcribe", response_model=TranscriptionResponse)
    async def transcribe_audio(
        audio: UploadFile = File(...),
        task: str = Form(default="transcribe"),
        language: str | None = Form(default=None),
    ) -> TranscriptionResponse:
        audio_bytes = await audio.read()
        try:
            transcript = app.state.stt_service.transcribe(
                audio_bytes,
                filename=audio.filename,
                task=task,
                language=language,
            )
        except STTUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return TranscriptionResponse(transcript=transcript, backend=app.state.stt_service.backend_name)

    @app.post("/assistant/respond")
    async def respond(
        session_id: str = Form(...),
        text: str | None = Form(default=None),
        use_cached_emotion: bool = Form(default=True),
        audio: UploadFile | None = File(default=None),
    ) -> StreamingResponse:
        def stream() -> Iterator[str]:
            started_at = time.perf_counter()
            try:
                session = app.state.session_service.get_session(session_id)
            except SessionNotFoundError:
                yield sse_event(
                    "error",
                    {"stage": "session", "message": "Session not found."},
                )
                return

            transcript_used = None
            final_user_text = (text or "").strip()

            if audio is not None:
                try:
                    audio_bytes = audio.file.read()
                    transcript_used = app.state.stt_service.transcribe(
                        audio_bytes,
                        filename=audio.filename,
                        task="transcribe",
                    )
                    final_user_text = transcript_used.strip()
                except STTUnavailableError as exc:
                    yield sse_event(
                        "error",
                        {"stage": "speech_to_text", "message": str(exc)},
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    yield sse_event(
                        "error",
                        {"stage": "speech_to_text", "message": f"Failed to transcribe audio: {exc}"},
                    )
                    return

            if not final_user_text:
                yield sse_event(
                    "error",
                    {"stage": "validation", "message": "Either text or audio must be provided."},
                )
                return

            latest_emotion = None
            if use_cached_emotion:
                latest_emotion = session["latest_emotion"]

            prompt = app.state.assistant_service.build_prompt(
                user_text=final_user_text,
                history=session["history"],
                latest_emotion=latest_emotion,
            )

            yield sse_event(
                "metadata",
                {
                    "session_id": session_id,
                    "emotion_label": None if latest_emotion is None else latest_emotion["emotion_label"],
                    "emotion_confidence": None if latest_emotion is None else latest_emotion.get("confidence"),
                    "transcript_used": transcript_used,
                    "model": app.state.settings.ollama_model,
                },
            )

            full_response_parts: list[str] = []
            final_event = None

            try:
                for event in app.state.llm_service.stream_generate(prompt):
                    if event.response:
                        full_response_parts.append(event.response)
                        yield sse_event("token", {"text": event.response})
                    if event.done:
                        final_event = event
            except OllamaServiceError as exc:
                yield sse_event(
                    "error",
                    {"stage": "ollama", "message": str(exc)},
                )
                return

            full_response = "".join(full_response_parts)
            app.state.session_service.append_message(session_id, "user", final_user_text)
            app.state.session_service.append_message(session_id, "assistant", full_response)

            latency_ms = int((time.perf_counter() - started_at) * 1000)
            yield sse_event(
                "done",
                {
                    "session_id": session_id,
                    "response_text": full_response,
                    "user_text_used": final_user_text,
                    "emotion_label": None if latest_emotion is None else latest_emotion["emotion_label"],
                    "emotion_confidence": None if latest_emotion is None else latest_emotion.get("confidence"),
                    "latency_ms": latency_ms,
                    "done_reason": None if final_event is None else final_event.done_reason,
                    "eval_count": None if final_event is None else final_event.eval_count,
                    "eval_duration": None if final_event is None else final_event.eval_duration,
                    "prompt_eval_count": None if final_event is None else final_event.prompt_eval_count,
                    "prompt_eval_duration": None if final_event is None else final_event.prompt_eval_duration,
                },
            )

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
