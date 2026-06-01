# Deployment Model Bundle

This directory contains only the inference artifacts required by the online
API container:

- `enhanced_ensemble_model.pkl`
- `scaler.pkl`

Regenerate the production model locally with:

```powershell
python -m app_api.rebuild_deployment_model
```

Then copy the refreshed artifacts from `data/processed/` into this directory
before deploying.
