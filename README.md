# EmoS — ACM MM 2026

Evaluation code and scoring rubrics for emotional intelligence in spoken dialogue. Given user and response audio, the evaluator assigns a quality score from **1 to 4** and compares it with the reference score.

**Dataset: [zsy814/EmoS](https://huggingface.co/datasets/zsy814/EmoS)**

## Tasks

| Dimension | Code | Task |
| --- | --- | --- |
| Perception | BAP | Basic Acoustic Information Perception |
| Perception | IAA | Implicit Attitude Analysis |
| Understanding | EST | Emotional State Tracking |
| Understanding | ECA | Emotional Cause Analysis |
| Utilization | ECM | Emotion-Cognition Matching |
| Utilization | EPA | Emotion-Driven Plan Adjustment |
| Management | SSE | Social Strategy Execution |
| Management | PMB | Proactive Mitigation and Emotional Buffering |
| Management | CRD | Conflict Resolution and De-escalation |
| Management | VAR | Value Alignment |

EST and ECA use multi-turn dialogue history; the other tasks use a single user–response pair. Scoring criteria are provided in `rubrics.json`.

## Setup

Use Python 3.10+ and a CUDA-compatible PyTorch installation.

```bash
pip install -r requirements.txt
```

Provide a Qwen2.5-Omni-compatible evaluator checkpoint through `--model`. Model weights are not included; a base Qwen model can be used as a baseline but is not the trained EmoS evaluator.

## Dataset

Download the standard test set from Hugging Face. For Chinese:

```bash
huggingface-cli download zsy814/EmoS archives/test/cn.tar --repo-type dataset --local-dir data/EmoS
mkdir -p data/EmoS/cn
tar --no-same-owner -xf data/EmoS/archives/test/cn.tar -C data/EmoS/cn
```

For English, replace `cn` with `en`. Evaluate the two languages separately. The `test50_human` archives are for human evaluation and are not inputs to this script.

Keep the category and scene folders intact. Each scene should contain `Model_1.wav` through `Model_4.wav`, where the number is the reference score, plus one `user_*.wav`. For EST/ECA, the context instead uses numbered `user_speechN.wav` and `model_speechN.wav` files, ending with a user turn.

## Evaluation

Run commands from the repository root. First check the data layout and sample counts:

```bash
python evaluate.py --data-root data/EmoS/cn --categories BAP IAA EST --dry-run
```

Then run evaluation with your checkpoint:

```bash
python evaluate.py \
  --data-root data/EmoS/cn \
  --categories BAP IAA EST \
  --model /path/to/evaluator \
  --batch-size 1 \
  --output outputs/cn_selected.jsonl
```

Choose any subset with `--categories`. To run all tasks, use `BAP IAA EST ECA ECM EPA SSE PMB CRD VAR`. Add `--think` for reasoning before scoring. See `python evaluate.py --help` for other options.

The run saves predictions (`.jsonl`), settings (`.run.json`), and overall/per-category metrics (`.metrics.json`). Existing files are not overwritten; use a new output name for each run.

## Metrics

Results include exact accuracy, adjacent accuracy (within one point), MAE, MSE, RMSE, and a confusion matrix. Accuracies are reported on a 0–1 scale. Invalid predictions count as incorrect for accuracy; error metrics use valid predictions only. Coverage and valid-only accuracies are also included.

To calculate metrics from an existing prediction file:

```bash
python metrics.py -i outputs/cn_selected.jsonl -o outputs/recomputed.metrics.json
```

## Tests

The offline tests do not require a GPU or model weights:

```bash
python -B -m unittest discover -s tests -v
```
