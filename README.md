# Deepfake Detection Studio

A Gradio application for screening images and videos with a trained binary image classifier.

## Requirements
- Python 3.12
- FFmpeg
- `models/final_deepfake_model.keras`

## Setup (local)

Linux/macOS:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place `final_deepfake_model.keras` in the `models/` directory. Install FFmpeg if it is not already installed.

Verify model loading and inference:
```bash
python check_model.py
```

Run:
```bash
python app.py
```

Open the local URL printed in the terminal, normally http://127.0.0.1:7860.

## Limits and limitations
- Images: JPG/JPEG/PNG/WEBP, maximum 20 MB and 40 megapixels.
- Videos: MP4/MOV/WEBM, maximum 100 MB and 60 seconds; sampled at approximately 1 FPS.
- CPU inference, one queued job at a time, sequential video frame processing.
- Model output is a fake score, not a calibrated forensic probability.
- Video screening applies an image classifier to sampled frames; it does not analyze temporal consistency, audio, lip-sync, or motion. Results are not definitive forensic proof.
