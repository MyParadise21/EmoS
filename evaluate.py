"""Run selected EmoS categories with a Qwen2.5-Omni-compatible evaluator."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from data import RUBRICS, build_message, load_tasks
from metrics import parse_score, summarize


def evaluate_batches(tasks, predict, batch_size=1):
    """Keep one result per input, including failed batches, in input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        try:
            outputs = predict(batch)
            if (not isinstance(outputs, list) or len(outputs) != len(batch)
                    or any(not isinstance(output, str) for output in outputs)):
                raise ValueError("Predictor must return one string per input")
        except Exception as error:
            logging.exception("Inference failed for batch starting at %d", start)
            for task in batch:
                yield {**task, "model_output_score": None,
                       "model_output_original": None, "error": type(error).__name__}
            continue
        for task, output in zip(batch, outputs):
            yield {**task, "model_output_score": parse_score(output),
                   "model_output_original": output, "error": None}


def make_predictor(args):
    try:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("Install requirements.txt in the inference environment first") from error

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, revision=args.revision)
    processor.tokenizer.padding_side = "left"
    dtype = getattr(torch, args.dtype)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype, device_map="auto",
    ).eval()

    def predict(batch):
        messages = [build_message(task, args.data_root, args.think, args.think_max_len)
                    for task in batch]
        audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audios, sampling_rate=16000,
                           return_tensors="pt", padding=True).to(model.device).to(dtype)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.size(1):]
        return processor.batch_decode(generated, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Extracted standard-test audio for ONE language/split")
    parser.add_argument("--categories", required=True, nargs="+", choices=tuple(RUBRICS),
                        help="Explicit category selection, e.g. BAP IAA EST")
    parser.add_argument("--model", help="Local evaluator checkpoint or Hugging Face model ID")
    parser.add_argument("--revision", default="main", help="Model revision; prefer a commit ID")
    parser.add_argument("--dataset-revision", default="unspecified",
                        help="Record the downloaded dataset revision for reproducibility")
    parser.add_argument("--output", type=Path, help="New prediction .jsonl file; never overwritten")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--think-max-len", type=int, default=50,
                        help="Word limit in the optional reasoning prompt; <=0 means no limit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate file layout and count tasks without loading models or audio")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    if not args.dry_run and (not args.model or not args.output):
        parser.error("--model and --output are required unless --dry-run is used")
    if args.output and args.output.suffix.lower() != ".jsonl":
        parser.error("--output must end in .jsonl")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        tasks = load_tasks(args.data_root, args.categories)
        counts = Counter(task["category"] for task in tasks)
        print(json.dumps({"total_samples": len(tasks), "by_category": dict(sorted(counts.items()))}, indent=2))
        if args.dry_run:
            return 0
        output = args.output
        metadata_path = output.with_suffix(".run.json")
        metrics_path = output.with_suffix(".metrics.json")
        for path in (output, metadata_path, metrics_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}; choose a new output name")
        predict = make_predictor(args)
        metadata = {
            "dataset": "https://huggingface.co/datasets/zsy814/EmoS",
            "dataset_revision": args.dataset_revision,
            "model": Path(args.model).name if Path(args.model).exists() else args.model,
            "model_revision": args.revision,
            "categories": sorted(set(args.categories)),
            "total_samples": len(tasks),
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "padding_side": "left",
            "think": args.think,
            "think_max_len": args.think_max_len,
            "rubrics_sha256": hashlib.sha256(Path(__file__).with_name("rubrics.json").read_bytes()).hexdigest(),
            "versions": {name: version(name) for name in
                         ("torch", "transformers", "qwen-omni-utils", "accelerate", "huggingface-hub")},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        results = []
        with output.open("x", encoding="utf-8") as stream:
            for result in evaluate_batches(tasks, predict, args.batch_size):
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                results.append(result)
        report = summarize(results)
        with metrics_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        logging.info("Saved predictions, run configuration and metrics to %s", output.parent)
        return 1 if report["overall"]["inference_errors"] else 0
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
