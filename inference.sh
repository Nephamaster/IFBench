#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference.sh llama31 Random 01
#   bash inference.sh qwen3   Kmeans 01
#
# Optional overrides:
#   PORT=8002 WORKERS=16 MAX_TOKENS=2048 RESUME=0 bash inference.sh ...

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
MAX_TOKENS="${MAX_TOKENS:-2048}"
# IFBench paper generation setting: greedy decoding.
TEMPERATURE="${TEMPERATURE:-0.0}"
# Keep 1.0 for official/comparable evaluation. Use >1 only for diagnosis.
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
        # Llama-3.1 chat-turn terminator. It is not the tokenizer EOS in this model.
        DEFAULT_STOP_TOKEN_IDS="128009"
        DEFAULT_CHAT_TEMPLATE_KWARGS='{}'
        ;;
    qwen3|qwen)
        MODEL_KIND="qwen3"
        MODEL="qwen3-8b-${DATA_KIND,,}-${ID}"
        # Qwen3 ChatML turn terminator; tokenizer EOS is <|endoftext|> (151643).
        DEFAULT_STOP_TOKEN_IDS="151645"
        DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
        ;;
    *)
        echo "Unsupported model kind: $MODEL_KIND; use llama31 or qwen3." >&2
        exit 2
        ;;
esac

STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-$DEFAULT_STOP_TOKEN_IDS}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-$DEFAULT_CHAT_TEMPLATE_KWARGS}"

API_BASE="${API_BASE:-http://127.0.0.1:${PORT}/v1}"
INPUT_FILE="${INPUT_FILE:-data/IFBench_test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-data/${MODEL}-response.jsonl}"
EVAL_DIR="${EVAL_DIR:-eval/${MODEL}}"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input file does not exist: $INPUT_FILE" >&2
    exit 1
fi

# Wait for the vLLM OpenAI-compatible endpoint.
READY=0
for _ in $(seq 1 60); do
    if curl -fsS "${API_BASE}/models" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [[ "$READY" != "1" ]]; then
    echo "vLLM is not ready: ${API_BASE}" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")" "$EVAL_DIR"

echo "Model                : $MODEL"
echo "API                  : $API_BASE"
echo "Input                : $INPUT_FILE"
echo "Output               : $OUTPUT_FILE"
echo "Max tokens           : $MAX_TOKENS"
echo "Temperature          : $TEMPERATURE"
echo "Repetition penalty   : $REPETITION_PENALTY"
echo "Stop token IDs       : $STOP_TOKEN_IDS"
echo "Chat template kwargs : $CHAT_TEMPLATE_KWARGS"

# Old files may contain non-stopping/repeating responses. Resume only when
# explicitly requested after confirming that the existing file uses this setup.
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
    --stop-token-ids "$STOP_TOKEN_IDS" \
    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    "${RESUME_ARGS[@]}"

"$PYTHON_BIN" -m run_eval \
    --input_data="$INPUT_FILE" \
    --input_response_data="$OUTPUT_FILE" \
    --output_dir="$EVAL_DIR"
