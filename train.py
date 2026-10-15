#!/usr/bin/env python3
"""CLI entrypoint for training and benchmarking urban surface temperature models.

Usage example:
    python train.py --data data.zip --feature-set all --cv
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_talea_dataset
from src.features.engineering import prepare_feature_matrices
from src.models.evaluate import compute_feature_importances, format_metrics_table
from src.models.trainer import (
    get_model,
    run_10fold_cv,
    save_trained_model,
    train_and_evaluate_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train machine learning models for Land Surface Temperature (LST) z-score estimation."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.zip",
        help="Path to data.zip archive or extracted data/ directory.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["base", "extended", "all"],
        default="all",
        help="Feature set to train on: 'base' (NDVI, Albedo), 'extended' (+ composite indices), or 'all' (benchmarks both).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model list ('linear', 'rf', 'gb') or 'all'.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Execute 10-fold cross validation on the top-performing model.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def get_model_instances(model_arg: str, seed: int):
    arg = model_arg.strip().lower()
    if arg == "all":
        return [
            ("Linear Regression", get_model("linear", random_state=seed)),
            ("Random Forest", get_model("rf", random_state=seed)),
            ("Gradient Boosting", get_model("gb", random_state=seed)),
        ]
    models = []
    for item in arg.split(","):
        item = item.strip()
        if item in ("linear", "lr", "linear_regression"):
            models.append(("Linear Regression", get_model("linear", random_state=seed)))
        elif item in ("rf", "random_forest", "randomforest"):
            models.append(("Random Forest", get_model("rf", random_state=seed)))
        elif item in ("gb", "gbr", "gradient_boosting", "gradientboosting"):
            models.append(("Gradient Boosting", get_model("gb", random_state=seed)))
        else:
            raise ValueError(f"Unknown model identifier: {item}")
    return models


def run_experiment_for_feature_set(
    set_name: str,
    df_geo,
    model_list,
    seed: int,
    save_dir: Path,
    run_cv: bool,
    cv_splits: int,
):
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Feature Set '{set_name.upper()}' (70/30 Train/Test Split)")
    print(f"{'='*70}")

    X, y, feature_cols = prepare_feature_matrices(df_geo, feature_set=set_name)
    print(f"Dataset Dimensions: {X.shape[0]} samples, {X.shape[1]} features: {feature_cols}")

    results = []
    fitted_models = {}

    for name, model in model_list:
        fitted, metrics, _, _, _ = train_and_evaluate_split(
            model=model,
            X=X,
            y=y,
            test_size=0.30,
            random_state=seed,
            model_name=name,
        )
        results.append(metrics)
        fitted_models[name] = fitted

    df_metrics = format_metrics_table(results)
    print("\nModel Performance Comparison:")
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Feature Importance for Random Forest
    if "Random Forest" in fitted_models:
        rf = fitted_models["Random Forest"]
        importances = compute_feature_importances(rf, feature_cols)
        print("\nFeature Importances (Random Forest):")
        for feat, imp in importances.items():
            print(f"  - {feat:<24}: {imp:.5f} ({imp*100:.2f}%)")

        # Save model
        save_path = save_dir / f"rf_{set_name}_model.joblib"
        save_trained_model(
            model=rf,
            features=feature_cols,
            target="z_score",
            filepath=save_path,
            description=f"RandomForestRegressor trained on {set_name} features: {feature_cols}",
        )
        print(f"Saved trained Random Forest to: {save_path}")

    # Optional K-Fold Cross Validation
    if run_cv:
        print(f"\n--- Running {cv_splits}-Fold Cross Validation on Random Forest ---")
        rf_cv = get_model("rf", random_state=seed)
        cv_metrics, _, _ = run_10fold_cv(
            model=rf_cv,
            X=X,
            y=y,
            n_splits=cv_splits,
            random_state=seed,
            model_name=f"Random Forest ({set_name})",
        )
        print(f"  R²   ({cv_splits}-fold): {cv_metrics['R2']:.5f}")
        print(f"  RMSE ({cv_splits}-fold): {cv_metrics['RMSE']:.5f}")
        print(f"  MAE  ({cv_splits}-fold): {cv_metrics['MAE']:.5f}")
        print(f"  MAPE ({cv_splits}-fold): {cv_metrics['MAPE(%)']:.5f} %")


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TALEA dataset from: {args.data}")
    df_geo, base_feats, ext_feats = load_talea_dataset(args.data)
    print(f"Successfully loaded {len(df_geo)} urban polygons.")
    print(f"Base Features: {base_feats}")
    print(f"Available Extended Features: {ext_feats}")

    feature_sets_to_run = []
    if args.feature_set == "all":
        feature_sets_to_run.append("base")
        if len(ext_feats) > len(base_feats):
            feature_sets_to_run.append("extended")
    else:
        feature_sets_to_run.append(args.feature_set)

    for fset in feature_sets_to_run:
        models = get_model_instances(args.models, args.seed)
        run_experiment_for_feature_set(
            set_name=fset,
            df_geo=df_geo,
            model_list=models,
            seed=args.seed,
            save_dir=save_dir,
            run_cv=args.cv,
            cv_splits=args.cv_splits,
        )

    print("\nTraining workflow completed successfully.")


if __name__ == "__main__":
    main()
