"""Optional utility to record one-channel EEG values from a serial port.

Install the optional dependency first:
    pip install -e ".[serial]"

Example:
    python scripts/stream_serial_to_txt.py --port COM3 --baud 115200 --seconds 60 --out data/raw/session_01.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record numeric serial values to a text file.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM3 or /dev/ttyUSB0.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required. Install with: pip install -e '.[serial]'") from exc

    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.seconds
    count = 0

    with serial.Serial(args.port, args.baud, timeout=1) as ser, args.out.open("w", encoding="utf-8") as handle:
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                float(line)
            except ValueError:
                continue
            handle.write(f"{line}\n")
            count += 1

    print(f"Saved {count} samples to {args.out}")


if __name__ == "__main__":
    main()
