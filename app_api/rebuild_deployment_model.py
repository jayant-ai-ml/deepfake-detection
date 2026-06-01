"""Refit the production ensemble and save a conservative decision policy."""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from final_training import create_ensemble, extract_enhanced_temporal_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = DATA_DIR / "enhanced_ensemble_model.pkl"
REPORT_PATH = DATA_DIR / "deployment_model_report.json"
SELECT_K_BEST = 5000

# Landmark-only predictions in the middle range are not reliable enough for a
# forced answer. The app should surface these cases for manual verification.
REAL_MAX_PROBABILITY = 0.20
FAKE_MIN_PROBABILITY = 0.75


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    landmarks = np.load(DATA_DIR / split / "landmarks.npy")
    labels = pd.read_csv(DATA_DIR / split / "labels.csv")["label"].to_numpy()
    return landmarks, labels


def summarize_policy(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    reviewed = (probabilities > REAL_MAX_PROBABILITY) & (
        probabilities < FAKE_MIN_PROBABILITY
    )
    covered = ~reviewed
    predictions = (probabilities >= FAKE_MIN_PROBABILITY).astype(int)
    covered_labels = labels[covered]
    covered_predictions = predictions[covered]

    return {
        "real_max_probability": REAL_MAX_PROBABILITY,
        "fake_min_probability": FAKE_MIN_PROBABILITY,
        "total_samples": int(len(labels)),
        "auto_classified_samples": int(covered.sum()),
        "manual_review_samples": int(reviewed.sum()),
        "coverage": float(covered.mean()),
        "auto_classified_accuracy": float(
            accuracy_score(covered_labels, covered_predictions)
        ),
        "auto_classified_confusion_matrix": confusion_matrix(
            covered_labels, covered_predictions, labels=[0, 1]
        ).tolist(),
    }


def main() -> None:
    print("Loading train, validation, and test landmark sequences...")
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    x_deploy = np.concatenate([x_train, x_val], axis=0)
    y_deploy = np.concatenate([y_train, y_val], axis=0)

    print("Extracting enhanced temporal features...")
    deploy_features = extract_enhanced_temporal_features(x_deploy)
    test_features = extract_enhanced_temporal_features(x_test)

    scaler = StandardScaler()
    deploy_scaled = scaler.fit_transform(deploy_features)
    test_scaled = scaler.transform(test_features)

    selector = SelectKBest(score_func=f_classif, k=min(SELECT_K_BEST, deploy_scaled.shape[1]))
    deploy_scaled = selector.fit_transform(deploy_scaled, y_deploy)
    test_scaled = selector.transform(test_scaled)

    print("Fitting production ensemble on train + validation data...")
    ensemble, _ = create_ensemble(y_deploy)
    ensemble.fit(deploy_scaled, y_deploy)

    test_probabilities = ensemble.predict_proba(test_scaled)[:, 1]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_samples": int(len(y_deploy)),
        "test_policy": summarize_policy(y_test, test_probabilities),
    }

    if MODEL_PATH.exists():
        backup_path = MODEL_PATH.with_suffix(".pre_deployment.pkl")
        shutil.copy2(MODEL_PATH, backup_path)
        print(f"Backed up previous model to {backup_path}")

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "ensemble": ensemble,
                "scaler": scaler,
                "selector": selector,
                "threshold": FAKE_MIN_PROBABILITY,
                "decision_policy": {
                    "real_max_probability": REAL_MAX_PROBABILITY,
                    "fake_min_probability": FAKE_MIN_PROBABILITY,
                },
                "feature_names": [f"f_{index}" for index in range(deploy_scaled.shape[1])],
                "training_mode": "train_plus_validation_deployment_refit",
            },
            model_file,
        )

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved production model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
