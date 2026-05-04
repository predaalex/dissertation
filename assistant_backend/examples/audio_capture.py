from __future__ import annotations

from io import BytesIO
import os
import time
import wave
from threading import Lock

import numpy as np
import sounddevice as sd


def parse_audio_device(device: str | None):
    if device is None or device == "":
        return None
    if device.isdigit():
        return int(device)
    return device


def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    input_devices = []
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            input_devices.append(
                {
                    "index": index,
                    "name": device.get("name"),
                    "max_input_channels": int(device.get("max_input_channels", 0)),
                    "default_samplerate": float(device.get("default_samplerate", 0.0)),
                }
            )
    return input_devices


def clear_pending_keypresses() -> None:
    if os.name != "nt":
        return

    import msvcrt

    while msvcrt.kbhit():
        msvcrt.getch()


def key_was_pressed() -> bool:
    if os.name != "nt":
        return False

    import msvcrt

    if not msvcrt.kbhit():
        return False
    msvcrt.getch()
    return True


def pcm16_wav_bytes_from_float32(
    recording: np.ndarray,
    sample_rate: int = 16000,
) -> bytes:
    clipped = np.clip(recording.reshape(-1), -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_audio.tobytes())

    return buffer.getvalue()


def record_wav_bytes(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    device: str | None = None,
) -> bytes:
    input_device = parse_audio_device(device)
    frame_count = int(duration_seconds * sample_rate)
    recording = sd.rec(
        frame_count,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=input_device,
    )
    sd.wait()

    return pcm16_wav_bytes_from_float32(recording, sample_rate=sample_rate)


def record_wav_bytes_until_keypress(
    max_duration_seconds: float = 30.0,
    sample_rate: int = 16000,
    device: str | None = None,
    poll_interval_seconds: float = 0.05,
) -> bytes:
    input_device = parse_audio_device(device)
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[audio warning] {status}")
        chunks.append(indata.copy())

    clear_pending_keypresses()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=input_device,
        callback=callback,
    ):
        started_at = time.perf_counter()
        while time.perf_counter() - started_at < max_duration_seconds:
            if key_was_pressed():
                break
            time.sleep(poll_interval_seconds)

    if not chunks:
        return pcm16_wav_bytes_from_float32(
            np.zeros((1, 1), dtype=np.float32),
            sample_rate=sample_rate,
        )

    recording = np.concatenate(chunks, axis=0)
    return pcm16_wav_bytes_from_float32(recording, sample_rate=sample_rate)


def iter_cumulative_wav_snapshots_until_keypress(
    max_duration_seconds: float = 30.0,
    sample_rate: int = 16000,
    device: str | None = None,
    snapshot_interval_seconds: float = 4.0,
    poll_interval_seconds: float = 0.05,
):
    input_device = parse_audio_device(device)
    chunks: list[np.ndarray] = []
    chunks_lock = Lock()
    last_yielded_sample_count = 0

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[audio warning] {status}")
        with chunks_lock:
            chunks.append(indata.copy())

    clear_pending_keypresses()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=input_device,
        callback=callback,
    ):
        started_at = time.perf_counter()
        last_snapshot_at = started_at
        stop_requested = False

        while time.perf_counter() - started_at < max_duration_seconds:
            now = time.perf_counter()
            if key_was_pressed():
                stop_requested = True
                break

            if now - last_snapshot_at >= snapshot_interval_seconds:
                snapshot = _copy_recording_chunks(chunks, chunks_lock)
                sample_count = int(snapshot.shape[0]) if snapshot is not None else 0
                if snapshot is not None and sample_count > last_yielded_sample_count:
                    last_yielded_sample_count = sample_count
                    yield (
                        pcm16_wav_bytes_from_float32(snapshot, sample_rate=sample_rate),
                        now - started_at,
                        False,
                    )
                last_snapshot_at = now

            time.sleep(poll_interval_seconds)

        final_snapshot = _copy_recording_chunks(chunks, chunks_lock)
        final_sample_count = int(final_snapshot.shape[0]) if final_snapshot is not None else 0
        if final_snapshot is not None and (
            final_sample_count > last_yielded_sample_count or stop_requested
        ):
            yield (
                pcm16_wav_bytes_from_float32(final_snapshot, sample_rate=sample_rate),
                time.perf_counter() - started_at,
                True,
            )


def _copy_recording_chunks(chunks: list[np.ndarray], chunks_lock: Lock) -> np.ndarray | None:
    with chunks_lock:
        if not chunks:
            return None
        return np.concatenate([chunk.copy() for chunk in chunks], axis=0)
