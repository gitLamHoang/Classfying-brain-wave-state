# Hardware and Data-Collection Protocol

This project originally used a portable EEG device connected to an Arduino-style serial pipeline. The repository intentionally separates the **hardware collection layer** from the **machine-learning layer** so the classifier can be reviewed and tested without physical hardware.

## Expected raw data format

The Python prediction script expects a plain text file with one numeric EEG sample per line:

```text
512
498
505
...
```

Recommended metadata to record outside the raw signal file:

| Field | Example |
|---|---|
| participant_id | P01 |
| session_id | 2026-06-20-awake-01 |
| sampling_rate_hz | 512 |
| label | awake or sleepy |
| condition_notes | eyes open, seated, no talking |
| duration_seconds | 180 |

Do not commit personally identifiable participant data to GitHub. Keep raw recordings in `data/raw/` or `data/private/`, which are ignored by Git.

## Reproducible collection checklist

1. Confirm the serial sampling rate and the EEG sampling rate.
2. Run a 10-second test recording and check that the file contains numeric values only.
3. Use the same window length during training and prediction.
4. Keep labels simple and consistent: for example, `awake` and `sleepy`.
5. Save model metrics with the same commit used to train the model.

## Optional serial recording

Install the optional serial dependency:

```bash
pip install -e ".[serial]"
```

Then record a session:

```bash
python scripts/stream_serial_to_txt.py --port COM3 --baud 115200 --seconds 60 --out data/raw/session_01.txt
```
