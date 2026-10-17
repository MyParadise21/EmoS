"""Dependency-free metrics for 1-4 quality-score predictions."""

import argparse
import json
import math
import re
from pathlib import Path


def parse_score(value):
    """Accept a score or one unambiguous answer tag; never guess from reasoning."""
    if type(value) is int:
        return value if 1 <= value <= 4 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if "<answer>" in text or "</answer>" in text:
        if len(answers) != 1 or text.count("<answer>") != 1 or text.count("</answer>") != 1:
            return None
        text = answers[0].strip()
    if len(text) == 3 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return int(text) if re.fullmatch(r"[1-4]", text) else None


def read_results(path):
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Input must contain a non-empty JSON array or JSONL records")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not an object")
        if "id" in record:
            identifier = record["id"]
            if not isinstance(identifier, str) or identifier in seen:
                raise ValueError(f"Invalid or duplicate sample ID at record {index}")
            seen.add(identifier)
    return records


def calculate_metrics(records):
    if not records:
        raise ValueError("Cannot calculate metrics for empty input")
    confusion = [[0] * 4 for _ in range(4)]
    errors = []
    exact = adjacent = failed = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not an object")
        truth = record.get("true_score")
        if type(truth) is not int or truth not in range(1, 5):
            raise ValueError(f"Record {index}: true_score must be an integer from 1 to 4")
        if record.get("error"):
            failed += 1
            continue
        prediction = parse_score(record.get("model_output_score"))
        if prediction is None:
            continue
        error = abs(truth - prediction)
        errors.append(error)
        exact += error == 0
        adjacent += error <= 1
        confusion[truth - 1][prediction - 1] += 1
    total, valid = len(records), len(errors)
    mse = sum(error ** 2 for error in errors) / valid if valid else None
    return {
        "total_samples": total,
        "valid_samples": valid,
        "invalid_predictions": total - valid,
        "inference_errors": failed,
        "coverage": valid / total,
        "exact_accuracy": exact / total,
        "adjacent_accuracy": adjacent / total,
        "valid_exact_accuracy": exact / valid if valid else None,
        "valid_adjacent_accuracy": adjacent / valid if valid else None,
        "mae": sum(errors) / valid if valid else None,
        "mse": mse,
        "rmse": math.sqrt(mse) if mse is not None else None,
        "confusion_matrix": confusion,
    }


def summarize(records):
    overall = calculate_metrics(records)
    groups = {}
    for record in records:
        category = record.get("category", "unspecified")
        if not isinstance(category, str) or not category:
            raise ValueError("Category must be a non-empty string")
        groups.setdefault(category, []).append(record)
    return {
        "overall": overall,
        "by_category": {category: calculate_metrics(items)
                        for category, items in sorted(groups.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(read_results(args.input))
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
