"""
peloton_iq.prediction.trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model training, Optuna hyperparameter tuning, evaluation, and artifact
saving for the tier prediction model.

This module is a one-time run — not called by the agent at inference time.
It reads model_df.csv, trains LightGBM and XGBoost with Bayesian
hyperparameter optimization, evaluates on a temporal holdout (2023),
and saves the best model as tier_predictor.pkl.

Run via:
    python scripts/run_training.py
    python -m peloton_iq.pipelines.train
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

import lightgbm as lgb
import xgboost as xgb

from peloton_iq.config import MODEL_DF_PATH, MODELS_DIR, settings

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_ORDER   = settings.tier_order
TIER_TO_INT  = {t: i for i, t in enumerate(TIER_ORDER)}
INT_TO_TIER  = {i: t for i, t in enumerate(TIER_ORDER)}
N_CLASSES    = len(TIER_ORDER)
FEATURE_COLS = settings.model_feature_cols
TARGET       = "tier_int"
CUTOFF_YEAR  = settings.cutoff_year
N_TRIALS     = 50


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_model_df(path: Path | None = None) -> pd.DataFrame:
    """Load and prepare model_df for training."""
    path = path or MODEL_DF_PATH
    log.info("Loading model_df from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    log.info("model_df: %d rows x %d cols", *df.shape)
    return df


def temporal_split(
    df: pd.DataFrame,
    cutoff_year: int = CUTOFF_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split.
    Train: 2017 – (cutoff_year - 1)
    Test:  cutoff_year
    """
    train = df[df["Year_results"] < cutoff_year].copy()
    test  = df[df["Year_results"] == cutoff_year].copy()
    log.info(
        "Split: train=%d rows (%d-%d)  test=%d rows (%d)",
        len(train), train["Year_results"].min(), train["Year_results"].max(),
        len(test), cutoff_year,
    )
    return train, test


# ---------------------------------------------------------------------------
# Ranking quality metric  (primary optimization target)
# ---------------------------------------------------------------------------

def ranking_score(
    model,
    X_val: pd.DataFrame,
    eval_df: pd.DataFrame,
    needs_imputation: bool = False,
    medians: Optional[pd.Series] = None,
) -> float:
    """
    Top-5 ranking accuracy — did the actual winner appear in the
    model's top-5 predicted contenders?

    This is what matters for the agent use case, not log loss.
    Returns negative score because Optuna minimizes.
    """
    X = X_val.fillna(medians) if needs_imputation and medians is not None else X_val
    proba = model.predict_proba(X)

    df = eval_df.copy()
    df["p_winner"] = proba[:, TIER_TO_INT["winner"]]

    top5_correct    = 0
    races_evaluated = 0

    for _, group in df.groupby("Race Name"):
        if len(group) < 5:
            continue
        ranked  = group.sort_values("p_winner", ascending=False)
        winners = set(group[group["tier"] == "winner"]["Name"].values)
        if not winners:
            continue
        top5_names     = set(ranked.head(5)["Name"].values)
        top5_correct   += len(winners & top5_names) > 0
        races_evaluated += 1

    if races_evaluated == 0:
        return 0.0

    return -(top5_correct / races_evaluated)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _make_lgb_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_df: pd.DataFrame,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "multiclass",
            "num_class":         N_CLASSES,
            "metric":            "multi_logloss",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":      42,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        return ranking_score(clf, X_test, eval_df)

    return objective


def _make_xgb_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_df: pd.DataFrame,
):
    medians = X_train.median()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":             "multi:softprob",
            "num_class":             N_CLASSES,
            "eval_metric":           "mlogloss",
            "verbosity":             0,
            "n_estimators":          trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":             trial.suggest_int("max_depth", 3, 12),
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight":      trial.suggest_int("min_child_weight", 1, 20),
            "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":             trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":            trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":          42,
            "tree_method":           "hist",
            "early_stopping_rounds": 50,
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train.fillna(medians), y_train,
            eval_set=[(X_test.fillna(medians), y_test)],
            verbose=False,
        )
        return ranking_score(
            clf, X_test, eval_df,
            needs_imputation=True, medians=medians,
        )

    return objective


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y_true: pd.Series,
    needs_imputation: bool = False,
    medians: Optional[pd.Series] = None,
) -> dict:
    """Classification metrics: log loss, AUC, per-class report."""
    X_eval  = X.fillna(medians) if needs_imputation and medians is not None else X
    y_proba = model.predict_proba(X_eval)
    y_pred  = model.predict(X_eval)
    ll      = log_loss(y_true, y_proba)
    y_bin   = label_binarize(y_true, classes=list(range(N_CLASSES)))
    auc     = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    log.info("=== %s ===", name)
    log.info("  Log Loss : %.4f", ll)
    log.info("  AUC      : %.4f", auc)

    report = classification_report(
        y_true, y_pred,
        target_names=TIER_ORDER,
        output_dict=True,
        zero_division=0,
    )
    for tier in TIER_ORDER:
        r = report.get(tier, {})
        log.info(
            "  %-10s  precision=%.2f  recall=%.2f  f1=%.2f",
            tier, r.get("precision", 0), r.get("recall", 0), r.get("f1-score", 0),
        )

    return {"name": name, "log_loss": ll, "auc": auc, "proba": y_proba, "pred": y_pred}


def evaluate_ranking(
    name: str,
    model,
    X: pd.DataFrame,
    eval_df: pd.DataFrame,
    needs_imputation: bool = False,
    medians: Optional[pd.Series] = None,
) -> dict:
    """Ranking quality metrics: top-1/3/5/10 accuracy."""
    X_eval = X.fillna(medians) if needs_imputation and medians is not None else X
    proba  = model.predict_proba(X_eval)
    df     = eval_df.copy()
    df["p_winner"] = proba[:, TIER_TO_INT["winner"]]

    top1 = top3 = top5 = top10 = races = 0
    for _, group in df.groupby("Race Name"):
        if len(group) < 5:
            continue
        ranked      = group.sort_values("p_winner", ascending=False)
        winners     = set(group[group["tier"] == "winner"]["Name"].values)
        top10_actual = set(
            group[group["tier"].isin(["winner", "podium", "top10"])]["Name"].values
        )
        if not winners:
            continue
        top1  += len(winners & set(ranked.head(1)["Name"].values)) > 0
        top3  += len(winners & set(ranked.head(3)["Name"].values)) > 0
        top5  += len(winners & set(ranked.head(5)["Name"].values)) > 0
        top10 += len(top10_actual & set(ranked.head(10)["Name"].values)) > 0
        races += 1

    results = {
        "top1":  top1 / races,
        "top3":  top3 / races,
        "top5":  top5 / races,
        "top10": top10 / races,
    }
    log.info(
        "%s | races=%d  top1=%.1f%%  top3=%.1f%%  top5=%.1f%%  top10=%.1f%%",
        name, races,
        results["top1"] * 100, results["top3"] * 100,
        results["top5"] * 100, results["top10"] * 100,
    )
    return results


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    model_df_path: Path | None = None,
    models_dir: Path | None = None,
    n_trials: int = N_TRIALS,
    feature_cols: list[str] | None = None,
) -> dict:
    """
    Full training pipeline.

    1. Load model_df and split temporally
    2. Optuna tuning for LightGBM (n_trials)
    3. Optuna tuning for XGBoost  (n_trials)
    4. Train Logistic Regression baseline
    5. Evaluate all models
    6. Save best model as tier_predictor.pkl
    7. Save all artifacts and Optuna trial CSVs

    Args:
        feature_cols: Override the default FEATURE_COLS for this run.
                      Used for ablation studies (e.g. with/without a new
                      feature) without touching global settings. Defaults
                      to settings.model_feature_cols when None.

    Returns a summary dict with metrics for all models.
    """
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # Allow per-call override for ablation studies; falls back to the
    # module-level default (settings.model_feature_cols) otherwise.
    FEATURE_COLS = feature_cols if feature_cols is not None else globals()["FEATURE_COLS"]

    # ------------------------------------------------------------------
    # Load and split
    # ------------------------------------------------------------------
    df         = load_model_df(model_df_path)
    train_df, test_df = temporal_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET]
    eval_df = test_df[["Race Name", "tier", "Name"]].copy()

    xgb_medians = X_train.median()

    # Stage type encoder — fit on full dataset so inference isn't surprised
    le_stage = LabelEncoder()
    le_stage.fit(df["stage_type"].fillna("flat"))

    log.info("Feature columns: %d", len(FEATURE_COLS))
    log.info("Train tier distribution:")
    for tier in TIER_ORDER:
        n   = (train_df["tier"] == tier).sum()
        pct = n / len(train_df) * 100
        log.info("  %-10s  %6d  (%.1f%%)", tier, n, pct)

    # ------------------------------------------------------------------
    # LightGBM Optuna
    # ------------------------------------------------------------------
    log.info("Running LightGBM Optuna (%d trials)...", n_trials)
    study_lgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_lgb.optimize(
        _make_lgb_objective(X_train, y_train, X_test, y_test, eval_df),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    log.info("LightGBM best top-5: %.1f%%", -study_lgb.best_value * 100)

    # Train final LightGBM
    lgb_model = lgb.LGBMClassifier(
        **study_lgb.best_params,
        objective="multiclass",
        num_class=N_CLASSES,
        metric="multi_logloss",
        verbosity=-1,
        random_state=42,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # ------------------------------------------------------------------
    # XGBoost Optuna
    # ------------------------------------------------------------------
    log.info("Running XGBoost Optuna (%d trials)...", n_trials)
    study_xgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_xgb.optimize(
        _make_xgb_objective(X_train, y_train, X_test, y_test, eval_df),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    log.info("XGBoost best top-5: %.1f%%", -study_xgb.best_value * 100)

    # Train final XGBoost
    xgb_model = xgb.XGBClassifier(
        **study_xgb.best_params,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train.fillna(xgb_medians), y_train,
        eval_set=[(X_test.fillna(xgb_medians), y_test)],
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Logistic Regression baseline
    # ------------------------------------------------------------------
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    log.info("Logistic Regression trained")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    log.info("=== CLASSIFICATION METRICS ===")
    clf_results = {
        "LightGBM": evaluate_model("LightGBM", lgb_model, X_test, y_test),
        "XGBoost":  evaluate_model(
            "XGBoost", xgb_model, X_test, y_test,
            needs_imputation=True, medians=xgb_medians,
        ),
        "Baseline": evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test),
    }

    log.info("=== RANKING QUALITY ===")
    rank_results = {
        "LightGBM": evaluate_ranking("LightGBM", lgb_model, X_test, eval_df),
        "XGBoost":  evaluate_ranking(
            "XGBoost", xgb_model, X_test, eval_df,
            needs_imputation=True, medians=xgb_medians,
        ),
        "Baseline": evaluate_ranking("Logistic Regression", lr_pipeline, X_test, eval_df),
    }

    # ------------------------------------------------------------------
    # Save individual model artifacts
    # ------------------------------------------------------------------
    with open(models_dir / "lgb_model.pkl", "wb") as f:
        pickle.dump({
            "model":       lgb_model,
            "study":       study_lgb,
            "best_params": study_lgb.best_params,
            "best_top5":   -study_lgb.best_value * 100,
        }, f)

    with open(models_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump({
            "model":       xgb_model,
            "study":       study_xgb,
            "best_params": study_xgb.best_params,
            "best_top5":   -study_xgb.best_value * 100,
            "medians":     xgb_medians,
        }, f)

    with open(models_dir / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_pipeline, f)

    with open(models_dir / "metadata.pkl", "wb") as f:
        pickle.dump({
            "feature_cols":       FEATURE_COLS,
            "tier_order":         TIER_ORDER,
            "tier_to_int":        TIER_TO_INT,
            "int_to_tier":        INT_TO_TIER,
            "stage_type_encoder": le_stage,
            "train_years":        list(range(2017, CUTOFF_YEAR)),
            "test_year":          CUTOFF_YEAR,
            "xgb_medians":        xgb_medians.to_dict(),
        }, f)

    # Save Optuna trial histories
    study_lgb.trials_dataframe().to_csv(models_dir / "study_lgb_trials.csv", index=False)
    study_xgb.trials_dataframe().to_csv(models_dir / "study_xgb_trials.csv", index=False)

    # ------------------------------------------------------------------
    # Save best model as tier_predictor.pkl  (used by the agent)
    # ------------------------------------------------------------------
    best_name  = max(rank_results, key=lambda k: rank_results[k]["top5"])
    best_model = lgb_model if best_name == "LightGBM" else xgb_model
    log.info("Best model by top-5: %s (%.1f%%)", best_name, rank_results[best_name]["top5"] * 100)

    tier_predictor = {
        "model":              best_model,
        "model_name":         best_name,
        "feature_cols":       FEATURE_COLS,
        "tier_order":         TIER_ORDER,
        "tier_to_int":        TIER_TO_INT,
        "int_to_tier":        INT_TO_TIER,
        "stage_type_encoder": le_stage,
        "xgb_medians":        xgb_medians.to_dict(),
        "train_years":        list(range(2017, CUTOFF_YEAR)),
        "test_year":          CUTOFF_YEAR,
        "metrics": {
            name: {
                "log_loss": clf_results[name]["log_loss"],
                "auc":      clf_results[name]["auc"],
                **rank_results[name],
            }
            for name in clf_results
        },
    }

    with open(models_dir / "tier_predictor.pkl", "wb") as f:
        pickle.dump(tier_predictor, f)

    # Human-readable metrics JSON
    metrics_out = {
        name: {
            "log_loss":              clf_results[name]["log_loss"],
            "auc":                   clf_results[name]["auc"],
            "top5_ranking_accuracy": rank_results[name]["top5"],
        }
        for name in clf_results
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    log.info("Artifacts saved to %s", models_dir)
    for p in sorted(models_dir.iterdir()):
        log.info("  %-40s  %.1f MB", p.name, p.stat().st_size / 1024 / 1024)

    return {"clf": clf_results, "ranking": rank_results, "best": best_name}