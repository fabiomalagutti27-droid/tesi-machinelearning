"""Model construction, training pipelines, and cross-validation routines."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from src.models.evaluate import compute_metrics


def get_model(model_name: str, random_state: int = 42, **kwargs: Any) -> BaseEstimator:
    """Instantiate a regression estimator according to thesis specification.

    Supported models:
      - 'linear_regression' / 'linear'
      - 'random_forest' / 'rf'
      - 'gradient_boosting' / 'gb'

    Parameters
    ----------
    model_name : str
        Canonical name or shorthand for estimator architecture.
    random_state : int, default=42
        Seed for pseudo-random number generator reproducibility.
    **kwargs : Any
        Additional keyword overrides passed to estimator constructor.

    Returns
    -------
    BaseEstimator
        Configured Scikit-Learn regressor instance.
    """
    normalized = model_name.lower().replace(" ", "_").replace("-", "_")

    if normalized in ("linear_regression", "linear", "lr"):
        return LinearRegression(**kwargs)

    if normalized in ("random_forest", "randomforest", "rf"):
        rf_kwargs = {
            "n_estimators": kwargs.pop("n_estimators", 200),
            "random_state": random_state,
            "n_jobs": kwargs.pop("n_jobs", -1),
        }
        rf_kwargs.update(kwargs)
        return RandomForestRegressor(**rf_kwargs)

    if normalized in ("gradient_boosting", "gradientboosting", "gb", "gbr"):
        gb_kwargs = {
            "random_state": random_state,
        }
        gb_kwargs.update(kwargs)
        return GradientBoostingRegressor(**gb_kwargs)

    raise ValueError(
        f"Unsupported model '{model_name}'. Choose from 'linear_regression', 'random_forest', or 'gradient_boosting'."
    )


def train_and_evaluate_split(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = 42,
    model_name: Optional[str] = None,
) -> Tuple[BaseEstimator, Dict[str, Union[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """Train estimator on train split (default 70/30) and compute test set evaluation metrics.

    Parameters
    ----------
    model : BaseEstimator
        Unfitted or pre-configured Scikit-Learn regressor.
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Target response series.
    test_size : float, default=0.30
        Holdout proportion (e.g. 0.30 for 70/30 split).
    random_state : int, default=42
        Random split seed.
    model_name : Optional[str], default=None
        Identifier label.

    Returns
    -------
    fitted_model : BaseEstimator
        Model fitted on training data.
    metrics : Dict[str, Union[str, float]]
        Test set performance metrics.
    y_test : np.ndarray
        True target values in test set.
    y_pred : np.ndarray
        Predictions for test set.
    errors : np.ndarray
        Residuals (y_test - y_pred).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_arr = np.asarray(y_test, dtype=float)
    errors = y_test_arr - y_pred

    metrics = compute_metrics(y_test_arr, y_pred, model_name=model_name or model.__class__.__name__)
    return model, metrics, y_test_arr, y_pred, errors


def run_10fold_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    random_state: int = 42,
    model_name: Optional[str] = None,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Union[str, float]], np.ndarray, np.ndarray]:
    """Perform k-fold cross-validation and compute out-of-fold generalization metrics.

    Parameters
    ----------
    model : BaseEstimator
        Regressor architecture to evaluate.
    X : pd.DataFrame
        Full predictor matrix.
    y : pd.Series
        Full target response vector.
    n_splits : int, default=10
        Number of cross-validation partitions.
    random_state : int, default=42
        Cross-validation shuffling seed.
    model_name : Optional[str], default=None
        Descriptive label.
    n_jobs : int, default=-1
        Parallel workers for CV prediction.

    Returns
    -------
    metrics : Dict[str, Union[str, float]]
        Consolidated metrics across all out-of-fold predictions.
    y_pred_cv : np.ndarray
        Out-of-fold predictions.
    errors_cv : np.ndarray
        Out-of-fold residuals (y_true - y_pred_cv).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_arr = np.asarray(y, dtype=float)

    y_pred_cv = cross_val_predict(model, X, y_arr, cv=kf, n_jobs=n_jobs)
    errors_cv = y_arr - y_pred_cv

    metrics = compute_metrics(
        y_arr,
        y_pred_cv,
        model_name=(model_name or model.__class__.__name__) + f" ({n_splits}-Fold CV)",
    )
    return metrics, y_pred_cv, errors_cv


def save_trained_model(
    model: BaseEstimator,
    features: List[str],
    target: str = "z_score",
    filepath: Union[str, Path] = "models/rf_model.joblib",
    description: str = "",
) -> Path:
    """Serialize model artifact with associated input schema metadata via joblib.

    Parameters
    ----------
    model : BaseEstimator
        Trained Scikit-Learn model.
    features : List[str]
        Ordered feature column names used during training.
    target : str, default="z_score"
        Name of target response variable.
    filepath : Union[str, Path]
        Destination joblib file path.
    description : str
        Contextual description of the model configuration.

    Returns
    -------
    Path
        Destination path of saved model.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "features": features,
        "target": target,
        "description": description or f"{model.__class__.__name__} trained on {features}",
    }
    joblib.dump(payload, out_path)
    return out_path


def load_trained_model(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load serialized model artifact and metadata from joblib file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to .joblib file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys 'model', 'features', 'target', and 'description'.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model file '{filepath}' not found.")
    return joblib.load(path)
