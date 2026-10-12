"""Splitting the data by time, training the classifiers, and measuring how well they do.

Plan:
  1. Split by date into train / validation / test (no shuffling - we can't
     train on the future and test on the past).
  2. For each model, try a few settings, train on the training set and keep
     the setting that scores best on the validation set.
  3. Retrain that setting on train + validation and score it once on the test set.
  4. Do all of this twice: with market features only, and with market +
     sentiment features. The difference answers our research question.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Split:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test


def chronological_split(data, train_frac=0.6, val_frac=0.2, gap=1):
    """Split by trading date: oldest dates for training, newest for testing.

    We leave a 1-day gap after the training and validation periods. The label
    of the last training day uses the next day's close, and that day belongs
    to the validation period, so without the gap a little bit of validation
    information would leak into training.
    """
    dates = np.sort(data["date"].unique())
    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    if train_end <= gap or val_end - train_end <= gap or val_end >= n:
        raise ValueError("Not enough trading days to make a train/validation/test split.")

    train_dates = dates[:train_end - gap]
    val_dates = dates[train_end:val_end - gap]
    test_dates = dates[val_end:]

    train = data[data["date"].isin(train_dates)].reset_index(drop=True)
    val = data[data["date"].isin(val_dates)].reset_index(drop=True)
    test = data[data["date"].isin(test_dates)].reset_index(drop=True)
    return Split(train, val, test)


def get_models(random_state=42):
    """The four classifiers from our proposal, each with a small grid of settings to try.

    class_weight="balanced" is in every grid in case up and down days are not 50/50.
    The grids are kept small on purpose so the models can't overfit the validation set too much.
    """
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
            {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50], "max_features": ["sqrt", 0.5],
             "class_weight": [None, "balanced"]},
        ),
        "svm": (
            SVC(kernel="rbf"),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01], "class_weight": [None, "balanced"]},
        ),
    }


def make_pipeline(model):
    # Putting the scaler inside the pipeline means it only ever learns the
    # mean/std from the data the model is trained on, never from test data.
    return Pipeline([("scale", StandardScaler()), ("clf", model)])


def all_settings(grid):
    """Turn {"C": [1, 10], "gamma": [a, b]} into every combination as a dict."""
    names = list(grid)
    for values in itertools.product(*grid.values()):
        yield dict(zip(names, values))


def model_scores(model, X):
    """A continuous score for ROC-AUC (probability if available, else decision value)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass  # e.g. SVC without probability=True
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate(model, X, y):
    predictions = model.predict(X)
    scores = model_scores(model, X)
    both_classes_present = len(np.unique(y)) == 2

    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, scores) if scores is not None and both_classes_present else np.nan,
        "confusion": confusion_matrix(y, predictions, labels=[0, 1]),
    }


def tune_on_validation(model, grid, X_train, y_train, X_val, y_val, metric="f1_macro"):
    """Try every setting in the grid and return the one with the best validation score.

    We pick by macro F1 instead of accuracy so a model that just predicts
    "up" every day doesn't win by default.
    """
    best_settings = None
    best_score = -np.inf
    for settings in all_settings(grid):
        candidate = make_pipeline(clone(model).set_params(**settings))
        candidate.fit(X_train, y_train)
        score = evaluate(candidate, X_val, y_val)[metric]
        if score > best_score:
            best_settings, best_score = settings, score
    return best_settings, best_score


class ExperimentResult:
    def __init__(self, metrics, confusions, models):
        self.metrics = metrics        # one row per (feature set, model)
        self.confusions = confusions  # confusion matrix for each (feature set, model)
        self.models = models          # the final trained models


def run_experiment(split, feature_sets, random_state=42, metric="f1_macro"):
    rows = []
    confusions = {}
    trained_models = {}

    train_and_val = pd.concat([split.train, split.val], ignore_index=True)
    y_test = split.test["target"].to_numpy()

    # Baseline: always predict whichever direction was most common in training.
    # Any useful model should beat this.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_and_val[["ret_1"]], train_and_val["target"])
    result = evaluate(baseline, split.test[["ret_1"]], y_test)
    confusions[("baseline", "majority_class")] = result.pop("confusion")
    rows.append({"feature_set": "baseline", "model": "majority_class", "best_params": "",
                 "val_score": np.nan, **result})

    for set_name, columns in feature_sets.items():
        for model_name, (model, grid) in get_models(random_state).items():
            settings, val_score = tune_on_validation(
                model, grid,
                split.train[columns], split.train["target"],
                split.val[columns], split.val["target"],
                metric,
            )

            final_model = make_pipeline(clone(model).set_params(**settings))
            final_model.fit(train_and_val[columns], train_and_val["target"])

            result = evaluate(final_model, split.test[columns], y_test)
            confusions[(set_name, model_name)] = result.pop("confusion")
            trained_models[(set_name, model_name)] = final_model
            rows.append({"feature_set": set_name, "model": model_name, "best_params": str(settings),
                         "val_score": val_score, **result})

    return ExperimentResult(pd.DataFrame(rows), confusions, trained_models)
