import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from audio_capture import (
    iter_cumulative_wav_snapshots_until_keypress,
    list_input_devices,
)


BASE_URL = "http://127.0.0.1:8000"
GLOW_PATH = shutil.which("glow")


def create_session():
    response = requests.post(f"{BASE_URL}/sessions", timeout=10)
    response.raise_for_status()
    return response.json()


def start_camera_poller(
    session_id,
    camera_index,
    poll_interval,
    no_preview,
    manual_only,
    show_camera_logs,
    disable_face_cropping,
):
    script_path = Path(__file__).resolve().parent / "predict_emotion_from_camera.py"
    command = [
        sys.executable,
        str(script_path),
        "--session-id",
        session_id,
        "--camera-index",
        str(camera_index),
        "--poll-interval",
        str(poll_interval),
    ]

    if no_preview:
        command.append("--no-preview")
    if manual_only:
        command.append("--manual-only")
    if disable_face_cropping:
        command.append("--disable-face-cropping")

    stdout = None if show_camera_logs else subprocess.DEVNULL
    stderr = None if show_camera_logs else subprocess.DEVNULL
    return subprocess.Popen(command, stdout=stdout, stderr=stderr)


def render_markdown_with_glow(markdown_text: str) -> bool:
    if GLOW_PATH is None:
        return False

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False,
        ) as temp_file:
            temp_file.write(markdown_text)
            temp_path = temp_file.name

        subprocess.run([GLOW_PATH, temp_path], check=False)
        return True
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def stream_assistant_reply(session_id, prompt, use_cached_emotion=True, render_markdown=True):
    data = {
        "session_id": session_id,
        "text": prompt,
        "use_cached_emotion": str(use_cached_emotion).lower(),
    }

    with requests.post(
        f"{BASE_URL}/assistant/respond",
        data=data,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        metadata = None
        done_payload = None
        current_event = None
        streamed_chunks = []

        if render_markdown and GLOW_PATH is not None:
            print("\nassistant> [streaming response, rendering markdown when complete...]", flush=True)
        else:
            print("\nassistant> ", end="", flush=True)

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue

            if not line.startswith("data: "):
                continue

            payload = json.loads(line[len("data: ") :])

            if current_event == "metadata":
                metadata = payload
            elif current_event == "token":
                token = payload.get("text", "")
                streamed_chunks.append(token)
                if not render_markdown or GLOW_PATH is None:
                    print(token, end="", flush=True)
            elif current_event == "error":
                print(f"\n[error] {payload.get('stage')}: {payload.get('message')}")
                return metadata, None
            elif current_event == "done":
                done_payload = payload

        full_response = (
            done_payload.get("response_text")
            if done_payload is not None and done_payload.get("response_text") is not None
            else "".join(streamed_chunks)
        )

        if render_markdown and full_response and GLOW_PATH is not None:
            rendered = render_markdown_with_glow(full_response)
            if not rendered:
                print(full_response)
        else:
            print()
        return metadata, done_payload


def transcribe_audio_bytes(audio_bytes, task="transcribe", timeout=300):
    files = {
        "audio": ("microphone_recording.wav", audio_bytes, "audio/wav"),
    }
    response = requests.post(
        f"{BASE_URL}/speech-to-text/transcribe",
        files=files,
        data={"task": task},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def transcribe_microphone_live(max_duration_seconds, sample_rate, device, chunk_seconds):
    print(
        f"Recording from microphone. Press any key to stop, "
        f"or wait {max_duration_seconds:.1f}s."
    )
    latest_transcript = ""

    for audio_bytes, elapsed_seconds, is_final in iter_cumulative_wav_snapshots_until_keypress(
        max_duration_seconds=max_duration_seconds,
        sample_rate=sample_rate,
        device=device,
        snapshot_interval_seconds=chunk_seconds,
    ):
        label = "final" if is_final else f"{elapsed_seconds:.1f}s"
        print(f"[voice {label}] transcribing...", flush=True)
        result = transcribe_audio_bytes(audio_bytes)
        transcript = result.get("transcript", "").strip()
        if transcript:
            latest_transcript = transcript
            print(f"[voice transcript] {latest_transcript}", flush=True)

    return latest_transcript


def transcribe_microphone_once(max_duration_seconds, sample_rate, device):
    from audio_capture import record_wav_bytes_until_keypress

    print(
        f"Recording from microphone. Press any key to stop, "
        f"or wait {max_duration_seconds:.1f}s."
    )
    audio_bytes = record_wav_bytes_until_keypress(
        max_duration_seconds=max_duration_seconds,
        sample_rate=sample_rate,
        device=device,
    )
    result = transcribe_audio_bytes(audio_bytes)
    return result.get("transcript", "").strip()


def transcribe_microphone(
    max_duration_seconds,
    sample_rate,
    device,
    live_transcript,
    chunk_seconds,
):
    if live_transcript:
        return transcribe_microphone_live(
            max_duration_seconds=max_duration_seconds,
            sample_rate=sample_rate,
            device=device,
            chunk_seconds=chunk_seconds,
        )

    return transcribe_microphone_once(
        max_duration_seconds=max_duration_seconds,
        sample_rate=sample_rate,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument(
        "--show-video-preview",
        action="store_true",
        help="Show the OpenCV camera preview window. Disabled by default.",
    )
    parser.add_argument(
        "--show-camera-logs",
        action="store_true",
        help="Show background camera poller logs in the terminal. Disabled by default.",
    )
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after starting the camera poller before the first chat prompt.",
    )
    parser.add_argument(
        "--no-markdown-render",
        action="store_true",
        help="Do not render completed assistant replies with glow; print raw streamed text instead.",
    )
    parser.add_argument(
        "--disable-face-cropping",
        action="store_true",
        help="Disable local and backend face cropping in the camera poller.",
    )
    parser.add_argument(
        "--voice-duration",
        type=float,
        default=30.0,
        help="Maximum seconds to record when using /VoiceToText.",
    )
    parser.add_argument(
        "--voice-sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate used by /VoiceToText.",
    )
    parser.add_argument(
        "--voice-device",
        default=None,
        help="Optional sounddevice input device id or name. Omit to use the Windows default microphone.",
    )
    parser.add_argument(
        "--voice-live-chunk-seconds",
        type=float,
        default=4.0,
        help="Seconds between near-live Whisper transcript updates during /VoiceToText.",
    )
    parser.add_argument(
        "--disable-live-voice-transcript",
        action="store_true",
        help="Only transcribe after recording finishes instead of showing near-live transcript updates.",
    )
    parser.add_argument(
        "--voice-transcript-only",
        action="store_true",
        help="Print the /VoiceToText transcript without sending it to the assistant.",
    )
    parser.add_argument(
        "--list-voice-devices",
        action="store_true",
        help="List available microphone input devices and exit.",
    )
    args = parser.parse_args()

    if args.list_voice_devices:
        for device in list_input_devices():
            print(device)
        return

    session = create_session()
    session_id = session["session_id"]

    print("Created session:")
    print(session)
    print(f"\nSession id: {session_id}")
    print("Starting camera emotion poller...")

    camera_process = start_camera_poller(
        session_id=session_id,
        camera_index=args.camera_index,
        poll_interval=args.poll_interval,
        no_preview=not args.show_video_preview,
        manual_only=args.manual_only,
        show_camera_logs=args.show_camera_logs,
        disable_face_cropping=args.disable_face_cropping,
    )

    print(
        f"Waiting {args.startup_wait:.1f}s for initial emotion polling before chat starts..."
    )
    time.sleep(args.startup_wait)

    print("\nInteractive chat started.")
    print("Type your message and press Enter.")
    print("Commands:")
    print("  /session  -> show session id")
    print("  /state    -> fetch current session state")
    print("  /VoiceToText -> record microphone audio, print transcript, and send it to the assistant")
    print("  /quit     -> stop chat and camera poller")

    try:
        while True:
            user_text = input("\nyou> ").strip()

            if not user_text:
                continue

            command_text = user_text.lower()

            if command_text == "/quit":
                break

            if command_text == "/session":
                print(f"session_id={session_id}")
                continue

            if command_text == "/state":
                response = requests.get(f"{BASE_URL}/sessions/{session_id}", timeout=10)
                response.raise_for_status()
                print(response.json())
                continue

            if command_text in {"/voicetotext", "/voice", "/vtt"}:
                try:
                    transcript = transcribe_microphone(
                        max_duration_seconds=args.voice_duration,
                        sample_rate=args.voice_sample_rate,
                        device=args.voice_device,
                        live_transcript=not args.disable_live_voice_transcript,
                        chunk_seconds=args.voice_live_chunk_seconds,
                    )
                    print("Transcript:")
                    print(transcript)
                except requests.HTTPError as exc:
                    print(f"[voice-to-text error] backend returned {exc.response.status_code}")
                    print(exc.response.text)
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"[voice-to-text error] {exc}")
                    continue

                if not transcript:
                    print("[voice-to-text] No transcript was produced.")
                    continue

                if args.voice_transcript_only:
                    continue

                metadata, done_payload = stream_assistant_reply(
                    session_id=session_id,
                    prompt=transcript,
                    use_cached_emotion=True,
                    render_markdown=not args.no_markdown_render,
                )

                if metadata is not None:
                    print(
                        f"[emotion used] label={metadata.get('emotion_label')} "
                        f"confidence={metadata.get('emotion_confidence')}"
                    )

                if done_payload is not None:
                    print(
                        f"[latency] {done_payload.get('latency_ms')} ms"
                    )
                continue

            metadata, done_payload = stream_assistant_reply(
                session_id=session_id,
                prompt=user_text,
                use_cached_emotion=True,
                render_markdown=not args.no_markdown_render,
            )

            if metadata is not None:
                print(
                    f"[emotion used] label={metadata.get('emotion_label')} "
                    f"confidence={metadata.get('emotion_confidence')}"
                )

            if done_payload is not None:
                print(
                    f"[latency] {done_payload.get('latency_ms')} ms"
                )
    finally:
        if camera_process.poll() is None:
            camera_process.terminate()
            try:
                camera_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                camera_process.kill()


if __name__ == "__main__":
    main()
