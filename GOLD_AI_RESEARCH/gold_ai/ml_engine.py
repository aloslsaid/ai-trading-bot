"""Chronological, leakage-aware classification on explicitly labeled real history."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtesting.engine import Signals
from .indicators.classic import compute_classic


def make_features(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.0005):
    """Create causal indicators and a future-return target; raw price levels are not features."""
    if horizon < 1:
        raise ValueError("horizon must be at least one bar")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("input bars must have a unique, chronological index")
    f = compute_classic(df)
    f["return_1"] = f["close"].pct_change()
    f["return_3"] = f["close"].pct_change(3)
    f["return_12"] = f["close"].pct_change(12)
    f["hour_sin"] = np.sin(2 * np.pi * f.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.index.hour / 24)
    f["weekday_sin"] = np.sin(2 * np.pi * f.index.dayofweek / 7)
    f["weekday_cos"] = np.cos(2 * np.pi * f.index.dayofweek / 7)
    future_return = f["close"].shift(-horizon) / f["close"] - 1
    future_return = future_return.where(_contiguous_forward_mask(df.index, horizon))
    y = pd.Series(
        np.where(future_return > threshold, 1, np.where(future_return < -threshold, 0, np.nan)),
        index=f.index,
        name="target",
    )
    excluded = {"open", "high", "low", "close", "volume", "spread"}
    columns = [
        col for col in f.columns
        if col not in excluded and not col.startswith(("bid_", "ask_"))
    ]
    X = f[columns].replace([np.inf, -np.inf], np.nan)
    return X, y, columns


def _contiguous_forward_mask(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    intervals = index.to_series().diff()
    observed = intervals[intervals > pd.Timedelta(0)]
    if observed.empty:
        return pd.Series(False, index=index)
    expected = observed.value_counts().index[0]
    contiguous = pd.Series(True, index=index)
    for step in range(1, horizon + 1):
        contiguous &= intervals.shift(-step).eq(expected)
    return contiguous


def _chronological_sets(X: pd.DataFrame, y: pd.Series, horizon: int, train_frac: float, val_frac: float):
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than one")
    labeled = X.join(y)
    n = len(labeled)
    train_boundary = int(n * train_frac)
    test_boundary = int(n * (train_frac + val_frac))
    train_end = train_boundary - horizon
    validation_end = test_boundary - horizon
    if train_end <= 0 or validation_end <= train_boundary or test_boundary >= n:
        raise ValueError("not enough labeled rows for chronological train/validation/test splits and embargoes")
    train = labeled.iloc[:train_end].dropna(subset=["target"])
    validation = labeled.iloc[train_boundary:validation_end].dropna(subset=["target"])
    test = labeled.iloc[test_boundary:].dropna(subset=["target"])
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("chronological split produced an empty partition")
    return train, validation, test


def _pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def _model_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=8,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": _pipeline(
            HistGradientBoostingClassifier(
                max_iter=160, max_leaf_nodes=15, l2_regularization=2, random_state=42,
            )
        ),
    }


def _classification_metrics(y_true: pd.Series, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    result = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }
    result["auc"] = float(roc_auc_score(y_true, probability)) if y_true.nunique() == 2 else None
    return result


def _score_key(metrics: dict) -> tuple[float, float]:
    auc = metrics["auc"]
    return (auc if auc is not None and np.isfinite(auc) else metrics["accuracy"], metrics["f1"])


def train_model(
    df: pd.DataFrame,
    out_path: str | Path,
    horizon: int = 3,
    threshold: float = 0.0005,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    *,
    data_label: str,
) -> dict:
    """Compare classifiers on validation history, refit on train+validation, and test once."""
    if data_label != "real":
        raise ValueError("ML training accepts only data explicitly labeled 'real', never synthetic data")
    X, y, columns = make_features(df, horizon, threshold)
    if int(y.notna().sum()) < 500:
        raise ValueError(
            f"Need at least 500 labeled real bars after feature/target preparation; found {int(y.notna().sum())}"
        )
    train, validation, test = _chronological_sets(X, y, horizon, train_frac, val_frac)
    if train["target"].nunique() != 2 or test["target"].nunique() != 2:
        raise ValueError("train and test partitions must each contain both classes; adjust history or threshold")

    comparisons = {}
    candidates = _model_candidates()
    for name, model in candidates.items():
        model.fit(train[columns], train["target"].astype(int))
        probabilities = model.predict_proba(validation[columns])[:, 1]
        comparisons[name] = _classification_metrics(validation["target"].astype(int), probabilities)
    selected_name = max(comparisons, key=lambda name: _score_key(comparisons[name]))

    training = pd.concat([train, validation]).sort_index()
    if training["target"].nunique() != 2:
        raise ValueError("combined training partition must contain both classes")
    model = _model_candidates()[selected_name]
    model.fit(training[columns], training["target"].astype(int))
    test_probability = model.predict_proba(test[columns])[:, 1]
    metrics = {
        "data_label": data_label,
        "selected_model": selected_name,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "horizon_bars": horizon,
        "neutral_threshold": threshold,
        "embargo_bars": horizon,
        "train_start": str(train.index[0]),
        "train_end": str(train.index[-1]),
        "validation_start": str(validation.index[0]),
        "validation_end": str(validation.index[-1]),
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        "validation_comparison": comparisons,
        "test": _classification_metrics(test["target"].astype(int), test_probability),
        "warning": "Classification metrics are not trading returns; use a cost-aware out-of-sample backtest.",
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": columns,
        "metrics": metrics,
        "horizon": horizon,
        "threshold": threshold,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "data_label": data_label,
        "selected_model": selected_name,
    }, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def evaluate_model(df: pd.DataFrame, model_path: str | Path) -> dict:
    """Recreate the saved chronological held-out test partition and evaluate the frozen model."""
    pack = joblib.load(model_path)
    if pack.get("data_label") != "real":
        raise ValueError("saved model is not labeled as trained on real data")
    X, y, _ = make_features(df, pack["horizon"], pack["threshold"])
    _, _, test = _chronological_sets(
        X, y, pack["horizon"], pack["train_frac"], pack["val_frac"]
    )
    probability = pack["model"].predict_proba(test[pack["features"]])[:, 1]
    metrics = {
        "selected_model": pack["selected_model"],
        "data_label": pack["data_label"],
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        **_classification_metrics(test["target"].astype(int), probability),
    }
    return metrics


def model_signals(df: pd.DataFrame, model_path: str | Path, buy_threshold: float = 0.6, sell_threshold: float = 0.4) -> Signals:
    """Generate close-time model signals with causal ATR-based stops and targets."""
    if not (0.5 < buy_threshold <= 1 and 0 <= sell_threshold < 0.5):
        raise ValueError("buy_threshold must exceed 0.5 and sell_threshold must be below 0.5")
    pack = joblib.load(model_path)
    if pack.get("data_label") != "real":
        raise ValueError("saved model is not labeled as trained on real data")
    X, _, _ = make_features(df, pack["horizon"], pack["threshold"])
    probabilities = pd.Series(
        pack["model"].predict_proba(X[pack["features"]])[:, 1], index=df.index
    )
    direction = pd.Series(0, index=df.index, dtype=int)
    direction = direction.mask(probabilities >= buy_threshold, 1)
    direction = direction.mask(probabilities <= sell_threshold, -1)
    atr = compute_classic(df)["atr_14"]
    return Signals(direction, atr * 1.5, atr * 3.0)


def predict_latest(df: pd.DataFrame, model_path: str | Path) -> dict:
    pack = joblib.load(model_path)
    if pack.get("data_label") != "real":
        raise ValueError("saved model is not labeled as trained on real data")
    X, _, _ = make_features(df, pack["horizon"], pack["threshold"])
    row = X[pack["features"]].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
    probability = float(pack["model"].predict_proba(row)[0, 1])
    return {
        "probability_buy": probability,
        "probability_sell": 1 - probability,
        "signal": "BUY" if probability >= 0.60 else ("SELL" if probability <= 0.40 else "HOLD"),
        "model_metrics": pack["metrics"],
    }
