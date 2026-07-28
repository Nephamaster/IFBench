#!/usr/bin/env bash
set -euo pipefail

# Run from the IFBench repository directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_KIND="${1:-llama31}"
DATA_KIND="${2:-Random}"
ID="${3:-01}"
PORT="${PORT:-8002}"

if [[ "$ID" =~ ^[0-9]+$ ]]; then
    ID=$(printf '%02d' "$((10#$ID))")
else
    echo "ID must be an integer from 01 to 12: $ID" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-32}"
# Non-thinking IFBench responses normally do not need 4096 generated tokens.
# Override with MAX_TOKENS=4096 if a task needs longer answers.
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
# Keep standard greedy IFBench decoding. Repetition penalties can hide a
# train/inference mismatch and should only be enabled for diagnostics.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SEED="${SEED:-42}"

case "$DATA_KIND" in
    Kmeans|kmeans) DATA_KIND="Kmeans" ;;
    Random|random) DATA_KIND="Random" ;;
    *)
        echo "Unsupported data kind: $DATA_KIND; use Kmeans or Random." >&2
        exit 2
        ;;
esac

case "$MODEL_KIND" in
    llama31|llama)
        MODEL_KIND="llama31"
        MODEL="llama-3.1-8b-${DATA_KIND,,}-${ID}"
        # Llama 3.1 uses EOT and end-of-text as terminal tokens.
        STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-128001,128009}"
        ;;
    qwen3|qwen)
        MODEL_KIND="qwen3"
        MODEL="qwen3-8b-${DATA_KIND,,}-${ID}"
        # Qwen3 uses <|im_end|> as the assistant-turn terminator.
        STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-151645}"
        ;;
    *)
        echo "Unsupported model kind: $MODEL_KIND; use llama31 or qwen3." >&2
        exit 2
        ;;
esac

API_BASE="${API_BASE:-http://127.0.0.1:${PORT}/v1}"
INPUT_FILE="${INPUT_FILE:-data/IFBench_test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-data/${MODEL}-responses.jsonl}"
EVAL_DIR="${EVAL_DIR:-eval/${MODEL}}"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input file does not exist: $INPUT_FILE" >&2
    exit 1
fi

# Wait for vLLM to expose the OpenAI-compatible endpoint.
READY=0
for _ in $(seq 1 60); do
    if curl -fsS "${API_BASE}/models" | "$PYTHON_BIN" -c \
        'import json, sys; expected = sys.argv[1]; data = json.load(sys.stdin); raise SystemExit(0 if any(item.get("id") == expected for item in data.get("data", [])) else 1)' \
        "$MODEL" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [[ "$READY" != "1" ]]; then
    echo "The expected served model is not ready: ${MODEL} at ${API_BASE}" >&2
    echo "Check that serve_vllm.sh uses the same model kind, data kind, ID, and port." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")" "$EVAL_DIR"

echo "Model       : $MODEL"
echo "API         : $API_BASE"
echo "Input       : $INPUT_FILE"
echo "Output      : $OUTPUT_FILE"
echo "Max tokens  : $MAX_TOKENS"
echo "Temperature : $TEMPERATURE"
echo "Repetition  : $REPETITION_PENALTY"

# Do not use --resume by default: old response files may contain the previous
# non-stopping/repeating outputs. Set RESUME=1 only for an intentional resume.
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_ARGS+=(--resume)
fi

"$PYTHON_BIN" generate_responses.py \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --stop-token-ids "$STOP_TOKEN_IDS" \
    "${RESUME_ARGS[@]}"

"$PYTHON_BIN" -m run_eval \
    --input_data="$INPUT_FILE" \
    --input_response_data="$OUTPUT_FILE" \
    --output_dir="$EVAL_DIR"