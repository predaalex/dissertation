import argparse
from pathlib import Path

from audio_capture import list_input_devices, record_wav_bytes, record_wav_bytes_until_keypress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the captured 16 kHz mono PCM WAV file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to record in fixed-duration mode.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum seconds to record in keypress-stop mode.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate. Keep 16000 for Whisper-friendly WAV output.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sounddevice input device id or name. Omit to use the Windows default microphone.",
    )
    parser.add_argument(
        "--stop-on-keypress",
        action="store_true",
        help="Record until any key is pressed, or until --max-duration is reached.",
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

    if not args.output_path:
        parser.error("--output-path is required unless --list-devices is used.")

    if args.stop_on_keypress:
        print(
            f"Recording from microphone. Press any key to stop, "
            f"or wait {args.max_duration:.1f}s."
        )
        audio_bytes = record_wav_bytes_until_keypress(
            max_duration_seconds=args.max_duration,
            sample_rate=args.sample_rate,
            device=args.device,
        )
    else:
        print(f"Recording {args.duration:.1f}s from microphone...")
        audio_bytes = record_wav_bytes(
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
            device=args.device,
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    print(f"Saved WAV to {output_path}")


if __name__ == "__main__":
    main()
