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
python -m pip install -r requirements.txt
```

## Notes

Large local folders such as datasets, processed data, logs, model outputs, and virtual environments are intentionally ignored by Git.

## Mobile Prediction App

The Expo app in `mobile_app/` uploads a video to the FastAPI server in
`app_api/`. The server extracts MediaPipe facial landmarks and applies the
trained ensemble model from `data/processed/enhanced_ensemble_model.pkl`.

Install and start the API:

```powershell
python -m pip install -r app_api/requirements.txt
python -m uvicorn app_api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal, start the mobile app:

```powershell
cd mobile_app
npm install
npm start
```

The mobile app requires Node.js LTS and npm. Install Node.js first if the
`node --version` command is not available.

Use Expo Go on a phone connected to the same Wi-Fi network. Enter the laptop's
IPv4 API address in the app, such as `http://192.168.1.5:8000`.

## Desktop Prediction App

The same FastAPI server includes a desktop browser interface. Start the API and
open `http://127.0.0.1:8000` in a browser:

```powershell
python -m uvicorn app_api.main:app --host 0.0.0.0 --port 8000
```

The deployment model uses a conservative decision policy. Ambiguous landmark
scores are shown as `Needs manual review` instead of forcing a real/fake answer.
To refit the production model on the train and validation splits:

```powershell
python -m app_api.rebuild_deployment_model
```

## Online Deployment

The repository includes a `Dockerfile` for deploying the API and desktop page
to Railway. The container uses the compact model bundle in `deployment_models/`
instead of the ignored local dataset directory.

1. Push the repository to GitHub.
2. Create a Railway project from the GitHub repository.
3. Let Railway build the included `Dockerfile`.
4. In the Railway service settings, generate a public domain for port `8080`.
5. Open the generated HTTPS URL to use the desktop app.

For the mobile app, replace the local API address with the generated Railway
HTTPS URL.
