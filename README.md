# Deepfake Detection

Python project for exploring, preparing, and training a deepfake detection workflow.

## Project Structure

- `01_explore_dataset.py` - dataset exploration utilities
- `02_prepare_dataset.py` - dataset preparation pipeline
- `03_extract_frames.py` - video frame extraction
- `final_training.py` - model training entry point
- `config.py` - project configuration
- `requirements.txt` - Python dependencies

## Setup

```powershell
python -m venv deepfake_env
.\deepfake_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notes

Large local folders such as datasets, processed data, logs, model outputs, and virtual environments are intentionally ignored by Git.
