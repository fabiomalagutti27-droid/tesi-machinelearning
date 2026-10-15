#!/usr/bin/env python3
"""CLI entrypoint for evaluating models, loading saved checkpoints, or running 10-fold CV.

Usage examples:
    python evaluate.py --data data.zip --10fold
    python evaluate.py --model-path models/rf_base_model.joblib
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_talea_dataset
from src.features.engineering import prepare_feature_matrices
from src.models.evaluate import (
    compute_feature_importances,
    compute_metrics,
    format_metrics_table,
)
from src.models.trainer import (
    get_model,
    load_trained_model,
    run_10fold_cv,
    train_and_evaluate_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models, inspect saved checkpoints, or run 10-fold cross validation."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.zip",
        help="Path to data.zip archive or extracted data/ directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a serialized .joblib model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["base", "extended"],
        default="extended",
        help="Feature set to use if evaluating from scratch (default: 'extended').",
    )
    parser.add_argument(
        "--10fold",
        dest="run_10fold",
        action="store_true",
        help="Run rigorous 10-fold cross validation across all candidate models.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def evaluate_saved_model(model_path: str, df_geo):
    print(f"\nLoading saved model artifact from: {model_path}")
    checkpoint = load_trained_model(model_path)
    model = checkpoint["model"]
    features = checkpoint["features"]
    target = checkpoint.get("target", "z_score")
    description = checkpoint.get("description", "")

    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Features: {features}")
    print(f"Description: {description}")

    # Prepare data
    missing_feats = [f for f in features if f not in df_geo.columns]
    if missing_feats:
        print(f"Error: Missing required features in dataset: {missing_feats}", file=sys.stderr)
        sys.exit(1)

    clean_df = df_geo.dropna(subset=features + [target])
    X = clean_df[features].astype(float)
    y = clean_df[target].astype(float)

    # Evaluate full set predictions
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, model_name=f"{model.__class__.__name__} (Inference)")

    print("\nEvaluation Metrics on Target Dataset:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.5f}")
        else:
            print(f"  {k:<12}: {v}")

    importances = compute_feature_importances(model, features)
    print("\nFeature Importances:")
    for feat, imp in importances.items():
        print(f"  - {feat:<24}: {imp:.5f} ({imp*100:.2f}%)")


def run_benchmark_and_cv(df_geo, feature_set: str, seed: int, run_cv: bool, cv_splits: int):
    X, y, feature_cols = prepare_feature_matrices(df_geo, feature_set=feature_set)
    print(f"\nEvaluating models on feature set '{feature_set}' ({X.shape[0]} samples, {X.shape[1]} features)")

    model_candidates = [
        ("Linear Regression", get_model("linear", random_state=seed)),
        ("Random Forest", get_model("rf", random_state=seed)),
        ("Gradient Boosting", get_model("gb", random_state=seed)),
    ]

    split_results = []
    for name, model in model_candidates:
        _, metrics, _, _, _ = train_and_evaluate_split(
            model=model,
            X=X,
            y=y,
            test_size=0.30,
            random_state=seed,
            model_name=name,
        )
        split_results.append(metrics)

    print("\n--- 70/30 Train/Test Holdout Results ---")
    df_split = format_metrics_table(split_results)
    print(df_split.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if run_cv:
        print(f"\n--- {cv_splits}-Fold Cross-Validation Comparison ---")
        cv_results = []
        for name, _ in model_candidates:
            # Create fresh instance
            model_cv = get_model(name, random_state=seed)
            cv_metrics, _, _ = run_10fold_cv(
                model=model_cv,
                X=X,
                y=y,
                n_splits=cv_splits,
                random_state=seed,
                model_name=name,
            )
            cv_results.append(cv_metrics)

        df_cv = format_metrics_table(cv_results)
        print(df_cv.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def main():
    args = parse_args()

    print(f"Loading TALEA dataset from: {args.data}")
    df_geo, base_feats, ext_feats = load_talea_dataset(args.data)
    print(f"Successfully loaded {len(df_geo)} urban polygons.")

    if args.model_path:
        evaluate_saved_model(args.model_path, df_geo)
    else:
        run_benchmark_and_cv(
            df_geo=df_geo,
            feature_set=args.feature_set,
            seed=args.seed,
            run_cv=args.run_10fold,
            cv_splits=args.cv_splits,
        )


if __name__ == "__main__":
    main()
