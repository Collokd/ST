"""Chronological splitting, model definitions, tuning and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2, gap: int = 1) -> Split:
    """Split by trading date (never shuffled).

    ``gap`` trading dates are dropped at the end of train and validation: the
    label of the last row in a block depends on the next session's close, which
    belongs to the following block.
    """
    dates = np.sort(df["date"].unique())
    n = len(dates)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    if i_tr <= gap or i_va - i_tr <= gap or n - i_va < 1:
        raise ValueError("Not enough trading dates for a chronological train/val/test split.")
    tr_dates, va_dates, te_dates = dates[:i_tr - gap], dates[i_tr:i_va - gap], dates[i_va:]
    pick = lambda ds: df[df["date"].isin(ds)].reset_index(drop=True)
    return Split(pick(tr_dates), pick(va_dates), pick(te_dates))


def model_specs(random_state: int = 42) -> dict[str, tuple[object, dict[str, list]]]:
    """Estimator plus a small hyperparameter grid for each classifier."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000),
            {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [2, 3, 5, 8], "min_samples_leaf": [20, 50, 100], "class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50], "max_features": ["sqrt", 0.5], "class_weight": [None, "balanced"]},
        ),
        "svm": (
            SVC(kernel="rbf"),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01], "class_weight": [None, "balanced"]},
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    # The scaler is fit inside the pipeline, i.e. on training rows only.
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def _grid(params: dict[str, list]):
    keys = list(params)
    for values in product(*(params[k] for k in keys)):
        yield dict(zip(keys, values))


def _scores(model, X) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    s = _scores(model, X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, s) if s is not None and len(np.unique(y)) == 2 else np.nan,
        "confusion": confusion_matrix(y, pred, labels=[0, 1]),
    }


def tune_on_validation(estimator, grid: dict[str, list], Xtr, ytr, Xva, yva, metric: str = "f1_macro"):
    best, best_params, best_score = None, None, -np.inf
    for params in _grid(grid):
        model = make_pipeline(clone(estimator).set_params(**params)).fit(Xtr, ytr)
        score = evaluate(model, Xva, yva)[metric]
        if score > best_score:
            best, best_params, best_score = model, params, score
    return best, best_params, best_score


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    confusions: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    models: dict[tuple[str, str], Pipeline] = field(default_factory=dict)


def run_experiment(split: Split, feature_sets: dict[str, list[str]], random_state: int = 42, metric: str = "f1_macro") -> ExperimentResult:
    """Tune each model on validation, refit on train+val, report on held-out test."""
    rows, confusions, models = [], {}, {}
    trval = pd.concat([split.train, split.val], ignore_index=True)
    yte = split.test["target"].to_numpy()

    majority = DummyClassifier(strategy="most_frequent").fit(trval[["ret_1"]], trval["target"])
    m = evaluate(majority, split.test[["ret_1"]], yte)
    confusions[("baseline", "majority_class")] = m.pop("confusion")
    rows.append({"feature_set": "baseline", "model": "majority_class", "best_params": "", "val_score": np.nan, **m})

    for fs_name, cols in feature_sets.items():
        Xtr, ytr = split.train[cols], split.train["target"]
        Xva, yva = split.val[cols], split.val["target"]
        for name, (est, grid) in model_specs(random_state).items():
            _, params, val_score = tune_on_validation(est, grid, Xtr, ytr, Xva, yva, metric)
            final = make_pipeline(clone(est).set_params(**params)).fit(trval[cols], trval["target"])
            m = evaluate(final, split.test[cols], yte)
            confusions[(fs_name, name)] = m.pop("confusion")
            models[(fs_name, name)] = final
            rows.append({"feature_set": fs_name, "model": name, "best_params": str(params), "val_score": val_score, **m})
    return ExperimentResult(pd.DataFrame(rows), confusions, models)
