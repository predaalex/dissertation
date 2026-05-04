import argparse
from pathlib import Path

import requests

from audio_capture import list_input_devices, record_wav_bytes


BASE_URL = "http://127.0.0.1:8000"


def transcribe_audio_bytes(audio_bytes: bytes, task="transcribe", filename="microphone_recording.wav", timeout=300):
    files = {
        "audio": (filename, audio_bytes, "audio/wav"),
    }
    response = requests.post(
        f"{BASE_URL}/speech-to-text/transcribe",
        files=files,
        data={"task": task},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to record from the default microphone.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate. Whisper expects 16000 Hz for direct waveform input.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sounddevice input device id or name. Omit to use the Windows default microphone.",
    )
    parser.add_argument(
        "--save-audio-path",
        default=None,
        help="Optional path where the captured WAV should be saved for debugging.",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Optional existing audio file to transcribe or translate instead of recording from the microphone.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task to run. Use translate to translate speech into English.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Shortcut for --task translate.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        for device in list_input_devices():
            print(device)
        return

    task = "translate" if args.translate else args.task

    if args.audio_path:
        input_path = Path(args.audio_path)
        print(f"Using existing audio file: {input_path}")
        audio_bytes = input_path.read_bytes()
        filename = input_path.name
    else:
        print(f"Recording {args.duration:.1f}s from microphone...")
        audio_bytes = record_wav_bytes(
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
            device=args.device,
        )
        filename = "microphone_recording.wav"

    if args.save_audio_path and not args.audio_path:
        output_path = Path(args.save_audio_path)
        output_path.write_bytes(audio_bytes)
        print(f"Saved recording to {output_path}")

    print("Sending audio to /speech-to-text/transcribe...")
    result = transcribe_audio_bytes(audio_bytes, task=task, filename=filename)
    print("Text:")
    print(result.get("transcript", ""))
    print("Raw response:")
    print(result)


if __name__ == "__main__":
    main()
