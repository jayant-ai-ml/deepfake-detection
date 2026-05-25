# train_enhanced_model.py
"""
ENHANCED Deepfake Detection Model with 5 Improvement Strategies
Strategy 1: More Training Data (2000+ videos)
Strategy 2: Temporal Features (Velocity + Acceleration)
Strategy 3: K-Fold Cross Validation
Strategy 4: Optimized Threshold Finding
Strategy 5: Ensemble with Multiple Models
"""

import json
import pickle
import sys
import warnings
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================
# ENHANCED CONFIGURATION
# ============================================
class Config:
    DATA_PATH = Path(__file__).resolve().parent / "data" / "processed"
    RANDOM_SEED = 42
    
    # Strategy 1: Use train+val for final fitting.
    # Keep this False when checking overfit, because threshold tuning needs
    # a validation split that the model has not seen.
    USE_ALL_DATA = False
    
    # Strategy 2: Enhanced temporal features
    INCLUDE_VELOCITY = True
    INCLUDE_ACCELERATION = True
    INCLUDE_STATISTICAL_MOMENTS = True  # Skewness, kurtosis
    
    # Strategy 3: K-Fold cross validation
    N_FOLDS = 5
    
    # Strategy 4: Optimized threshold
    FIND_OPTIMAL_THRESHOLD = True
    THRESHOLD_METRIC = "accuracy"  # Options: 'accuracy', 'f1'

    # Reduce noisy high-dimensional features before fitting tree ensembles.
    USE_FEATURE_SELECTION = True
    SELECT_K_BEST = 5000
    
    # Strategy 5: Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'xgb': 2,
        'lgb': 2,
        'hgb': 1,
        'rf': 1
    }
    
    # Advanced feature engineering
    INCLUDE_LANDMARK_PAIRS = True  # Relative distances between landmarks

config = Config()

# ============================================
# STRATEGY 1: LOAD ALL AVAILABLE DATA
# ============================================
def load_all_splits():
    """Load train, val, test splits and optionally combine train+val for more data."""
    train_landmarks = np.load(config.DATA_PATH / "train" / "landmarks.npy")
    train_labels = pd.read_csv(config.DATA_PATH / "train" / "labels.csv")["label"].values
    
    val_landmarks = np.load(config.DATA_PATH / "val" / "landmarks.npy")
    val_labels = pd.read_csv(config.DATA_PATH / "val" / "labels.csv")["label"].values
    
    test_landmarks = np.load(config.DATA_PATH / "test" / "landmarks.npy")
    test_labels = pd.read_csv(config.DATA_PATH / "test" / "labels.csv")["label"].values
    
    total_processed = len(train_landmarks) + len(val_landmarks) + len(test_landmarks)
    print(f"   Processed samples found: {total_processed}")
    print(
        f"   Split sizes - Train: {len(train_landmarks)}, "
        f"Val: {len(val_landmarks)}, Test: {len(test_landmarks)}"
    )

    # Combine train and val only for a final fit. For honest model selection,
    # keep validation separate and reserve test for final reporting.
    if config.USE_ALL_DATA:
        x_train = np.concatenate([train_landmarks, val_landmarks], axis=0)
        y_train = np.concatenate([train_labels, val_labels], axis=0)
        x_val = test_landmarks
        y_val = test_labels
        x_test = test_landmarks
        y_test = test_labels
        
        print(f"\n📊 DATA SUMMARY (Strategy 1 - More Data):")
        print(f"   Training: train + val = {len(x_train)} samples")
        print(f"   Threshold validation: test split = {len(x_val)} samples")
        print(f"   Test: {len(x_test)} samples")
    else:
        x_train, y_train = train_landmarks, train_labels
        x_val, y_val = val_landmarks, val_labels
        x_test, y_test = test_landmarks, test_labels
    
    print(f"   Class distribution - Real: {np.sum(y_train==0)}, Fake: {np.sum(y_train==1)}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# ============================================
# STRATEGY 2: ENHANCED TEMPORAL FEATURES
# ============================================
def extract_enhanced_temporal_features(x):
    """
    Extract comprehensive temporal features including:
    - Mean, std, min, max per frame
    - Velocity (frame-to-frame differences)
    - Acceleration (velocity differences)
    - Statistical moments (skewness, kurtosis)
    - Landmark pair distances (if enabled)
    """
    # x shape: (videos, frames, features)
    
    features_list = []
    
    # Base statistics
    features_list.append(x.mean(axis=1))           # Mean per feature
    features_list.append(x.std(axis=1))            # Std per feature
    features_list.append(x.min(axis=1))            # Min per feature
    features_list.append(x.max(axis=1))            # Max per feature
    features_list.append(np.percentile(x, 25, axis=1))  # Q1
    features_list.append(np.percentile(x, 75, axis=1))  # Q3
    features_list.append(np.median(x, axis=1))     # Median
    
    # First frame and last frame (captures start/end state)
    features_list.append(x[:, 0, :])               # First frame
    features_list.append(x[:, -1, :])              # Last frame
    
    # Range (max - min)
    features_list.append(x.max(axis=1) - x.min(axis=1))
    
    # STRATEGY 2: Velocity (frame-to-frame differences)
    if config.INCLUDE_VELOCITY:
        diff = np.diff(x, axis=1)  # (N, frames-1, features)
        features_list.append(diff.mean(axis=1))
        features_list.append(diff.std(axis=1))
        features_list.append(np.abs(diff).max(axis=1))
    
    # STRATEGY 2: Acceleration (velocity differences)
    if config.INCLUDE_ACCELERATION:
        acc = np.diff(np.diff(x, axis=1), axis=1)  # (N, frames-2, features)
        if acc.shape[1] > 0:
            features_list.append(acc.mean(axis=1))
            features_list.append(acc.std(axis=1))
    
    # STRATEGY 2: Statistical moments (skewness, kurtosis)
    if config.INCLUDE_STATISTICAL_MOMENTS:
        # Skewness
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-8
        skew = np.mean(((x - mean) / std) ** 3, axis=1)
        features_list.append(skew)
        
        # Kurtosis
        kurt = np.mean(((x - mean) / std) ** 4, axis=1) - 3
        features_list.append(kurt)
    
    # Combine all features
    enhanced_features = np.concatenate(features_list, axis=1)
    
    print(f"\n📊 ENHANCED FEATURES (Strategy 2):")
    print(f"   Original shape: {x.shape}")
    print(f"   Enhanced shape: {enhanced_features.shape}")
    print(f"   Feature breakdown:")
    print(f"      - Base stats: {features_list[0].shape[1] * 7} features")
    if config.INCLUDE_VELOCITY:
        print(f"      - Velocity: {diff.shape[2] * 3} features")
    if config.INCLUDE_ACCELERATION:
        print(f"      - Acceleration: {acc.shape[2] * 2} features")
    if config.INCLUDE_STATISTICAL_MOMENTS:
        print(f"      - Moments: {skew.shape[1] * 2} features")
    
    return enhanced_features.astype(np.float32)

# ============================================
# STRATEGY 3: K-FOLD CROSS VALIDATION
# ============================================
def cross_validate_models(X, y, models_dict, n_folds=5):
    """Perform cross-validation for model selection"""
    print("\n" + "="*60)
    print("STRATEGY 3: K-FOLD CROSS VALIDATION")
    print("="*60)
    
    results = {}
    
    for name, model in models_dict.items():
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        results[name] = {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std(),
            'scores': cv_scores
        }
        print(f"\n   {name}:")
        print(f"      CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['mean_f1'])
    print(f"\n   🏆 Best model: {best_model} (F1: {results[best_model]['mean_f1']:.4f})")
    
    return results, best_model

# ============================================
# STRATEGY 4: OPTIMAL THRESHOLD FINDING
# ============================================
def find_optimal_threshold(y_true, probabilities):
    """Find threshold that maximizes F1 score with balanced precision/recall"""
    thresholds = np.linspace(0.2, 0.8, 121)
    best_t = 0.5
    best_score = -1.0
    best_metrics = {}
    
    f1_scores = []
    accuracy_scores = []
    
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        accuracy = accuracy_score(y_true, preds)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        score = accuracy if config.THRESHOLD_METRIC == "accuracy" else f1
        
        if score > best_score:
            best_score = score
            best_t = float(threshold)
            best_metrics = {
                'accuracy': accuracy,
                'precision': precision_score(y_true, preds, zero_division=0),
                'recall': recall_score(y_true, preds, zero_division=0),
                'f1': f1
            }
    
    print(f"\n📊 STRATEGY 4: Optimal Threshold Analysis")
    print(f"   Optimized metric: {config.THRESHOLD_METRIC}")
    print(f"   Best threshold: {best_t:.3f}")
    print(f"   Best validation score: {best_score:.4f}")
    print(f"   Corresponding Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   F1 at threshold: {best_metrics['f1']:.4f}")
    print(f"   Precision/Recall at threshold: {best_metrics['precision']:.4f}/{best_metrics['recall']:.4f}")
    
    # Plot threshold analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1')
    ax.plot(thresholds, accuracy_scores, 'g-', linewidth=2, label='Accuracy')
    ax.axvline(best_t, color='r', linestyle='--', label=f'Optimal threshold = {best_t:.3f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Validation Score vs Classification Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.DATA_PATH / 'threshold_analysis.png', dpi=150)
    plt.close()
    
    return best_t, best_metrics

# ============================================
# STRATEGY 5: ENSEMBLE WITH MULTIPLE MODELS
# ============================================
def create_ensemble(y_train=None):
    """Create voting ensemble with multiple models"""
    print("\n" + "="*60)
    print("STRATEGY 5: ENSEMBLE MODEL CREATION")
    print("="*60)
    
    scale_pos_weight = 1.0
    if y_train is not None and np.sum(y_train == 1) > 0:
        scale_pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))

    # Individual models
    models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.65,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.65,
            num_leaves=24,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=2.0,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            verbose=-1,
            n_jobs=-1
        ),
        'hgb': HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            l2_regularization=0.1,
            max_leaf_nodes=16,
            random_state=config.RANDOM_SEED
        ),
        'rf': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
    }
    
    # Create voting ensemble with custom weights
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft',
        weights=[config.ENSEMBLE_WEIGHTS.get(name, 1) for name in models.keys()]
    )
    
    print(f"\n   Ensemble members:")
    for name, weight in config.ENSEMBLE_WEIGHTS.items():
        print(f"      - {name.upper()}: weight {weight}")
    
    print(f"\n   Total models in ensemble: {len(models)}")
    
    return ensemble, models

# ============================================
# ADVANCED METRICS & VISUALIZATION
# ============================================
def plot_roc_curves(y_true, probabilities_dict, save_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, probs in probabilities_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curves.png', dpi=150)
    plt.close()

def save_confusion_matrix(y_true, y_pred, title, save_path):
    """Save confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    plt.savefig(save_path / f'confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

def print_classification_report(y_true, y_pred, y_probs, title):
    """Print comprehensive classification report"""
    print(f"\n📊 {title} CLASSIFICATION REPORT")
    print("="*40)
    print(f"   Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"   AUC:       {roc_auc_score(y_true, y_probs):.4f}")
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"   Specificity: {specificity:.4f}")

# ============================================
# MAIN TRAINING PIPELINE
# ============================================
def main():
    print("="*60)
    print("🚀 ENHANCED DEEPFAKE DETECTION - 5 STRATEGIES")
    print("="*60)
    
    print("\n🎯 Active Strategies:")
    if config.USE_ALL_DATA:
        print("   1. ✅ More Training Data (Combined Train+Val)")
    else:
        print("   1. ⚠️ Honest Evaluation Mode (Train/Val/Test kept separate)")
    print("   2. ✅ Enhanced Temporal Features (Velocity + Acceleration + Moments)")
    print("   3. ✅ K-Fold Cross Validation")
    print("   4. ✅ Optimized Threshold Selection")
    print("   5. ✅ Multi-Model Ensemble (XGBoost + LightGBM + HGB + RF)")
    
    # Load data
    print("\n" + "="*60)
    print("📂 LOADING DATASET")
    print("="*60)
    x_train, y_train, x_val, y_val, x_test, y_test = load_all_splits()
    
    # Extract enhanced features
    print("\n" + "="*60)
    print("🔧 FEATURE ENGINEERING")
    print("="*60)
    X_train = extract_enhanced_temporal_features(x_train)
    X_val = extract_enhanced_temporal_features(x_val)
    X_test = extract_enhanced_temporal_features(x_test)
    
    # Scale features
    print("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    selector = None
    if config.USE_FEATURE_SELECTION:
        k = min(config.SELECT_K_BEST, X_train_scaled.shape[1])
        print(f"\nSelecting top {k} features...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
        X_val_scaled = selector.transform(X_val_scaled)
        X_test_scaled = selector.transform(X_test_scaled)
        print(f"   Selected feature shape: {X_train_scaled.shape}")
    
    # Create ensemble
    ensemble, base_models = create_ensemble(y_train)
    
    # Cross-validation for model selection
    cv_results, best_model_name = cross_validate_models(
        X_train_scaled, y_train, base_models, n_folds=config.N_FOLDS
    )
    
    # Train ensemble on all training data
    print("\n" + "="*60)
    print("🏋️ TRAINING ENSEMBLE MODEL")
    print("="*60)
    ensemble.fit(X_train_scaled, y_train)
    
    # Get predictions from ensemble
    train_probs = ensemble.predict_proba(X_train_scaled)[:, 1]
    val_probs = ensemble.predict_proba(X_val_scaled)[:, 1]
    test_probs = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold
    if config.FIND_OPTIMAL_THRESHOLD:
        optimal_threshold, threshold_metrics = find_optimal_threshold(y_val, val_probs)
    else:
        optimal_threshold = 0.5
    
    # Final predictions
    train_preds = (train_probs >= optimal_threshold).astype(int)
    val_preds = (val_probs >= optimal_threshold).astype(int)
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    # Print results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print_classification_report(y_train, train_preds, train_probs, "TRAIN")
    print_classification_report(y_val, val_preds, val_probs, "VALIDATION")
    print_classification_report(y_test, test_preds, test_probs, "TEST")
    
    # Save confusion matrices
    save_confusion_matrix(y_test, test_preds, "TEST", config.DATA_PATH)
    
    # Get predictions from individual models for ROC curves
    print("\n📈 Generating model comparison plots...")
    individual_probs = {}
    for name, model in base_models.items():
        model.fit(X_train_scaled, y_train)
        individual_probs[name.upper()] = model.predict_proba(X_test_scaled)[:, 1]
    
    # Add ensemble predictions
    individual_probs['ENSEMBLE'] = test_probs
    
    # Plot ROC curves
    plot_roc_curves(y_test, individual_probs, config.DATA_PATH)
    
    # Save model, scaler, and threshold
    model_path = config.DATA_PATH / "enhanced_ensemble_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            'ensemble': ensemble,
            'scaler': scaler,
            'selector': selector,
            'threshold': optimal_threshold,
            'cv_results': cv_results,
            'feature_names': [f'f_{i}' for i in range(X_train_scaled.shape[1])]
        }, f)
    
    # Save results
    results = {
        'train_metrics': {
            'accuracy': float(accuracy_score(y_train, train_preds)),
            'precision': float(precision_score(y_train, train_preds)),
            'recall': float(recall_score(y_train, train_preds)),
            'f1_score': float(f1_score(y_train, train_preds)),
            'auc': float(roc_auc_score(y_train, train_probs))
        },
        'validation_metrics': {
            'accuracy': float(accuracy_score(y_val, val_preds)),
            'precision': float(precision_score(y_val, val_preds)),
            'recall': float(recall_score(y_val, val_preds)),
            'f1_score': float(f1_score(y_val, val_preds)),
            'auc': float(roc_auc_score(y_val, val_probs))
        },
        'test_metrics': {
            'accuracy': float(accuracy_score(y_test, test_preds)),
            'precision': float(precision_score(y_test, test_preds)),
            'recall': float(recall_score(y_test, test_preds)),
            'f1_score': float(f1_score(y_test, test_preds)),
            'auc': float(roc_auc_score(y_test, test_probs))
        },
        'optimal_threshold': float(optimal_threshold),
        'threshold_metric': config.THRESHOLD_METRIC,
        'cv_results': {k: {'mean_f1': float(v['mean_f1']), 'std_f1': float(v['std_f1'])} 
                      for k, v in cv_results.items()},
        'best_model': best_model_name,
        'feature_dim': int(X_train_scaled.shape[1]),
        'raw_feature_dim': int(X_train.shape[1]),
        'training_samples': len(X_train_scaled),
        'strategies_applied': [
            'honest_train_val_test_split',
            'enhanced_temporal_features',
            'feature_selection',
            'kfold_cross_validation', 
            'optimal_threshold',
            'multi_model_ensemble'
        ]
    }
    
    with open(config.DATA_PATH / "enhanced_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'ensemble_prediction': test_preds,
        'ensemble_probability': test_probs,
        'correct': (test_preds == y_test).astype(int)
    })
    
    # Add individual model predictions
    for name, probs in individual_probs.items():
        predictions_df[f'{name}_probability'] = probs
        predictions_df[f'{name}_prediction'] = (probs >= optimal_threshold).astype(int)
    
    predictions_df.to_csv(config.DATA_PATH / "enhanced_predictions.csv", index=False)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Saved files:")
    print(f"   - Model: {model_path}")
    print(f"   - Results: {config.DATA_PATH / 'enhanced_results.json'}")
    print(f"   - Predictions: {config.DATA_PATH / 'enhanced_predictions.csv'}")
    print(f"   - ROC Curves: {config.DATA_PATH / 'roc_curves.png'}")
    print(f"   - Confusion Matrix: {config.DATA_PATH / 'confusion_matrix_TEST.png'}")
    print(f"   - Threshold Analysis: {config.DATA_PATH / 'threshold_analysis.png'}")
    
    # Final summary
    print("\n📊 FINAL PERFORMANCE SUMMARY:")
    print(f"   🎯 Test Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")
    print(f"   🎯 Test F1 Score: {results['test_metrics']['f1_score']:.4f}")
    print(f"   🎯 Test AUC: {results['test_metrics']['auc']:.4f}")
    
    if results['test_metrics']['accuracy'] >= 0.85:
        print("\n🎉 EXCELLENT! Target accuracy (85%+) achieved!")
    elif results['test_metrics']['accuracy'] >= 0.75:
        print("\n👍 GOOD! Close to 85% target!")
    else:
        print("\n⚠️ Needs improvement. Consider adding more data or tuning hyperparameters.")
    
    return ensemble, results

if __name__ == "__main__":
    np.random.seed(config.RANDOM_SEED)
    model, results = main()
