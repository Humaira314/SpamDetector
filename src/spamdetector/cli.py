from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

from .rules import apply_rules, load_whitelist
from .train import (
    evaluate_model,
    load_model,
    save_metrics,
    save_model,
    train_and_select,
    train_model,
)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def predict_with_confidence(model: object, texts: List[str]) -> List[dict]:
    labels = model.predict(texts)
    confidences: List[Optional[float]] = [None] * len(texts)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        for idx, row in enumerate(probs):
            confidences[idx] = float(max(row))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        if hasattr(scores, "ndim") and scores.ndim > 1:
            max_scores = scores.max(axis=1)
        else:
            max_scores = scores
        confidences = [float(sigmoid(score)) for score in max_scores]

    return [
        {"label": label, "confidence": confidences[idx]}
        for idx, label in enumerate(labels)
    ]


def read_lines(file_path: str) -> List[str]:
    return [line.strip() for line in Path(file_path).read_text(encoding="utf-8").splitlines()]


def handle_train(args: argparse.Namespace) -> None:
    if args.model == "both":
        best_model, results, models = train_and_select(
            data_path=args.data,
            test_size=args.test_size,
            random_state=args.random_state,
            use_stemming=args.stem,
            use_stopwords=not args.no_stopwords,
            use_nltk_stopwords=not args.no_nltk_stopwords,
            calibrate_svm=not args.no_calibrate_svm,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_model(models["nb"], str(output_dir / "nb_model.joblib"))
        save_model(models["svm"], str(output_dir / "svm_model.joblib"))
        save_model(models[best_model], str(output_dir / "best_model.joblib"))
        save_metrics(results, str(output_dir / "metrics.json"))

        print(f"Best model: {best_model}")
        print(f"Saved models and metrics to {output_dir}")
        return

    model, metrics = train_model(
        data_path=args.data,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        use_stemming=args.stem,
        use_stopwords=not args.no_stopwords,
        use_nltk_stopwords=not args.no_nltk_stopwords,
        calibrate_svm=not args.no_calibrate_svm,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{args.model}_model.joblib"
    save_model(model, str(model_path))
    save_metrics(metrics, str(output_dir / "metrics.json"))

    print(f"Saved model to {model_path}")


def handle_evaluate(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    metrics = evaluate_model(model, args.data)
    print(metrics)


def handle_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    whitelist_path = None if args.no_whitelist else args.whitelist
    whitelist = load_whitelist(whitelist_path)

    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [line for line in read_lines(args.file) if line]
    else:
        raise ValueError("Provide --text or --file")

    results = predict_with_confidence(model, texts)
    for text, result in zip(texts, results):
        label, confidence, rule = apply_rules(
            label=result["label"],
            confidence=result["confidence"],
            text=text,
            whitelist=whitelist,
            threshold=args.threshold,
            low_confidence_label=args.low_confidence_label,
        )
        confidence_str = f"{confidence:.4f}" if confidence is not None else "n/a"
        rule_str = rule or "-"
        print(f"{label}\t{confidence_str}\t{rule_str}\t{text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spam detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True, help="Path to CSV dataset")
    train_parser.add_argument(
        "--model",
        choices=["nb", "svm", "both"],
        default="both",
        help="Model type to train",
    )
    train_parser.add_argument("--output-dir", default="models", help="Output directory")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--stem", action="store_true", help="Enable stemming")
    train_parser.add_argument("--no-stopwords", action="store_true")
    train_parser.add_argument("--no-nltk-stopwords", action="store_true")
    train_parser.add_argument("--no-calibrate-svm", action="store_true")
    train_parser.set_defaults(func=handle_train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    eval_parser.add_argument("--data", required=True, help="Path to CSV dataset")
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.set_defaults(func=handle_evaluate)

    predict_parser = subparsers.add_parser("predict", help="Predict spam or ham")
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--text", help="Single message text")
    predict_parser.add_argument("--file", help="Path to file with one message per line")
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Low-confidence threshold for labeling",
    )
    predict_parser.add_argument(
        "--low-confidence-label",
        choices=["ham", "spam", "uncertain"],
        default="uncertain",
        help="Label used when confidence is below the threshold",
    )
    predict_parser.add_argument(
        "--whitelist",
        default="config/whitelist.txt",
        help="Path to a domain allowlist file",
    )
    predict_parser.add_argument(
        "--no-whitelist",
        action="store_true",
        help="Disable allowlist checks",
    )
    predict_parser.set_defaults(func=handle_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
