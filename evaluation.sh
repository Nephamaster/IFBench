#!/usr/bin/env bash
set -u
set -o pipefail

# Batch-evaluate existing response files for:
#   Llama/Qwen x Kmeans/Random x 01-12 = 48 models.
# The per-example evaluation files and the final comparison CSV are written
# into one directory instead of one directory per model.
#
# Usage:
#   bash evaluation.sh
#
# Optional overrides:
#   PYTHON_BIN=python INPUT_FILE=data/IFBench_test.jsonl bash evaluation.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_FILE="${INPUT_FILE:-data/IFBench_test.jsonl}"
DATA_ROOT="${DATA_ROOT:-data}"
EVAL_DIR="${EVAL_DIR:-eval}"
SUMMARY_FILE="${SUMMARY_FILE:-${EVAL_DIR}/evaluation_scores.csv}"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input file does not exist: $INPUT_FILE" >&2
    exit 1
fi

mkdir -p "$EVAL_DIR"
mkdir -p "$(dirname "$SUMMARY_FILE")"

total=0
success=0
missing=0
failed=0
success_models=()
missing_models=()
failed_models=()

for MODEL_KIND in llama31 qwen3; do
    for DATA_KIND in Kmeans Random; do
        for number in $(seq 1 12); do
            ID=$(printf '%02d' "$number")

            if [[ "$MODEL_KIND" == "llama31" ]]; then
                MODEL="llama-3.1-8b-${DATA_KIND,,}-${ID}"
            else
                MODEL="qwen3-8b-${DATA_KIND,,}-${ID}"
            fi

            total=$((total + 1))
            RESPONSE_FILE="${DATA_ROOT}/${MODEL}-response.jsonl"

            if [[ ! -f "$RESPONSE_FILE" ]]; then
                echo "[MISSING] $MODEL: $RESPONSE_FILE"
                missing=$((missing + 1))
                missing_models+=("$MODEL")
                continue
            fi

            echo "[EVALUATING] $MODEL"
            if "$PYTHON_BIN" -m run_eval \
                --input_data="$INPUT_FILE" \
                --input_response_data="$RESPONSE_FILE" \
                --output_dir="$EVAL_DIR"; then
                echo "[OK] $MODEL"
                success=$((success + 1))
                success_models+=("$MODEL")
            else
                echo "[FAILED] $MODEL" >&2
                failed=$((failed + 1))
                failed_models+=("$MODEL")
            fi
        done
    done
done

# Collect the score JSON files written by run_eval.py into one comparison file.
if ! "$PYTHON_BIN" - "$SUMMARY_FILE" "$EVAL_DIR" "${success_models[@]}" <<'PY'
import csv
import json
import os
import sys


summary_file = sys.argv[1]
eval_dir = os.path.abspath(sys.argv[2])
models = sys.argv[3:]
fieldnames = [
    "model",
    "strict_prompt_correct",
    "strict_prompt_total",
    "strict_prompt_accuracy",
    "strict_instruction_correct",
    "strict_instruction_total",
    "strict_instruction_accuracy",
    "loose_prompt_correct",
    "loose_prompt_total",
    "loose_prompt_accuracy",
    "loose_instruction_correct",
    "loose_instruction_total",
    "loose_instruction_accuracy",
]

rows = []
for model in models:
    score_file = os.path.join(eval_dir, f"{model}-response-eval_scores.json")
    with open(score_file, "r", encoding="utf-8") as file:
        score = json.load(file)

    row = {"model": model}
    for mode in ("strict", "loose"):
        prefix = f"{mode}_"
        for level in ("prompt_level", "instruction_level"):
            report = score[mode][level]
            level_name = level.removesuffix("_level")
            row[f"{prefix}{level_name}_correct"] = report["correct"]
            row[f"{prefix}{level_name}_total"] = report["total"]
            row[f"{prefix}{level_name}_accuracy"] = report["accuracy"]
    rows.append(row)

with open(summary_file, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
then
    echo "Failed to write score summary: $SUMMARY_FILE" >&2
    failed=$((failed + 1))
fi

echo
echo "Summary: total=$total success=$success missing=$missing failed=$failed"
echo "Score comparison file: $SUMMARY_FILE"

if (( ${#missing_models[@]} > 0 )); then
    echo "Missing models: ${missing_models[*]}" >&2
fi
if (( ${#failed_models[@]} > 0 )); then
    echo "Failed models: ${failed_models[*]}" >&2
fi

if (( missing > 0 || failed > 0 )); then
    exit 1
fi
exit 0
