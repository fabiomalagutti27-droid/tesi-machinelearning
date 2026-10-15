"""Model evaluation metrics and diagnostic utilities for urban heat prediction."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Compute Coefficient of Determination (R^2)."""
    return float(r2_score(y_true, y_pred))


def mape_percent(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    eps: float = 1e-8,
) -> float:
    """Compute Mean Absolute Percentage Error (MAPE) with small-epsilon safeguard.

    Note: Because LST z-scores cross zero, small denominator values can produce
    disproportionate percentage metrics. RMSE, MAE, and R2 provide primary
    objective validation.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_t), eps)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: Optional[str] = None,
) -> Dict[str, Union[str, float]]:
    """Compute consolidated statistical evaluation metrics.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground truth observed values.
    y_pred : Union[pd.Series, np.ndarray]
        Model estimated values.
    model_name : Optional[str], default=None
        Label identifying model architecture.

    Returns
    -------
    Dict[str, Union[str, float]]
        Dictionary containing model label, R2, RMSE, MAE, and MAPE.
    """
    res = {
        "R2": r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE(%)": mape_percent(y_true, y_pred),
    }
    if model_name:
        res["Model"] = model_name
    return res


def compute_feature_importances(
    model: Any,
    feature_names: List[str],
) -> pd.Series:
    """Extract relative predictor importances from tree ensembles or linear regressors.

    Parameters
    ----------
    model : Any
        Fitted Scikit-Learn estimator.
    feature_names : List[str]
        List of feature column names corresponding to training matrix X.

    Returns
    -------
    pd.Series
        Sorted Series with feature names as index and importance weights as values.
    """
    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    elif hasattr(model, "coef_"):
        weights = np.abs(model.coef_)
        if weights.sum() > 0:
            weights = weights / weights.sum()
    else:
        weights = np.zeros(len(feature_names))

    return pd.Series(weights, index=feature_names).sort_values(ascending=False)


def format_metrics_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format and sort a collection of model metrics dictionaries into a clean summary table."""
    df = pd.DataFrame(results)
    preferred_order = ["Model", "R2", "RMSE", "MAE", "MAPE(%)"]
    existing_cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[existing_cols]
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    return df
