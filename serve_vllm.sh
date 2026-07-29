#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash serve_vllm.sh llama31 Random 01
#   bash serve_vllm.sh qwen3   Kmeans 01

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_KIND="${1:-llama31}"
DATA_KIND="${2:-Random}"
ID="${3:-01}"

if [[ "$ID" =~ ^[0-9]+$ ]]; then
    ID=$(printf '%02d' "$((10#$ID))")
else
    echo "ID must be an integer from 01 to 12: $ID" >&2
    exit 2
fi

MODEL_ROOT="${MODEL_ROOT:-/share/project/wuhaiming/spaces/scs/output/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

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
        MODEL_PATH="${MODEL_ROOT}/Llama-3.1-8B-SFT-${DATA_KIND}-${ID}"
        # Compatibility with the old directory spelling used by this repo.
        LEGACY_MODEL_PATH="${MODEL_ROOT}/LLama-3.1-8B-SFT-${DATA_KIND}-${ID}"
        SERVED_MODEL_NAME="llama-3.1-8b-${DATA_KIND,,}-${ID}"
        EXPECTED_STOP_TOKEN="<|eot_id|>"
        EXPECTED_STOP_TOKEN_ID="128009"
        ;;
    qwen3|qwen)
        MODEL_KIND="qwen3"
        MODEL_PATH="${MODEL_ROOT}/Qwen3-8B-Base-SFT-${DATA_KIND}-${ID}"
        LEGACY_MODEL_PATH=""
        SERVED_MODEL_NAME="qwen3-8b-${DATA_KIND,,}-${ID}"
        EXPECTED_STOP_TOKEN="<|im_end|>"
        EXPECTED_STOP_TOKEN_ID="151645"
        ;;
    *)
        echo "Unsupported model kind: $MODEL_KIND; use llama31 or qwen3." >&2
        exit 2
        ;;
esac

if [[ ! -d "$MODEL_PATH" && -n "$LEGACY_MODEL_PATH" && -d "$LEGACY_MODEL_PATH" ]]; then
    echo "[WARN] Using legacy directory spelling: $LEGACY_MODEL_PATH" >&2
    MODEL_PATH="$LEGACY_MODEL_PATH"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Merged model directory does not exist: $MODEL_PATH" >&2
    exit 1
fi

# Llama Base tokenizers often have no chat_template. Use the bundled Llama3
# template unless the caller explicitly provides another one.
if [[ "$MODEL_KIND" == "llama31" && -z "${CHAT_TEMPLATE:-}" ]]; then
    BUNDLED_CHAT_TEMPLATE="${SCRIPT_DIR}/llama3_chat_template.jinja"
    if [[ -f "$BUNDLED_CHAT_TEMPLATE" ]]; then
        CHAT_TEMPLATE="$BUNDLED_CHAT_TEMPLATE"
    fi
fi

# Chat Completions requires a valid chat template. Do not silently deploy a
# merged Base directory with a missing or malformed tokenizer template.
if [[ -z "${CHAT_TEMPLATE:-}" && "${CHECK_CHAT_TEMPLATE:-1}" == "1" ]]; then
    if ! "$PYTHON_BIN" - "$MODEL_PATH" <<'PY'
import sys
from transformers import AutoTokenizer

model_path = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if not tokenizer.chat_template:
    raise SystemExit("tokenizer.chat_template is empty")
print("chat_template: ok")
PY
    then
        echo "No usable chat_template found in: $MODEL_PATH" >&2
        echo "Either fix the merged tokenizer files or set:" >&2
        echo "  CHAT_TEMPLATE=/absolute/path/to/template.jinja bash serve_vllm.sh ..." >&2
        exit 1
    fi
fi

# Validate the assistant-turn terminator used by training. inference.sh passes
# this token through vLLM stop_token_ids because it can differ from EOS.
"$PYTHON_BIN" - \
    "$MODEL_PATH" \
    "$EXPECTED_STOP_TOKEN" \
    "$EXPECTED_STOP_TOKEN_ID" <<'PY'
import sys
from transformers import AutoTokenizer

model_path, stop_token, expected_id_text = sys.argv[1:]
expected_id = int(expected_id_text)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
stop_id = tokenizer.convert_tokens_to_ids(stop_token)

print(f"tokenizer.eos_token       : {tokenizer.eos_token}")
print(f"tokenizer.eos_token_id    : {tokenizer.eos_token_id}")
print(f"{stop_token} token id   : {stop_id}")

if stop_id != expected_id:
    raise SystemExit(
        f"Unexpected {stop_token} ID: got {stop_id}, expected {expected_id}. "
        "Update inference.sh before running evaluation."
    )
PY

VLLM_ARGS=(
    "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TP_SIZE"
    --dtype auto
    --max-model-len "$MAX_MODEL_LEN"
    # Do not import model-specific sampling defaults. inference.sh passes
    # temperature and chat-turn stop IDs explicitly for reproducibility.
    --generation-config vllm
    --disable-log-stats
    --host "$HOST"
    --port "$PORT"
)

# Qwen3 enables thinking by default in its chat template. Disable it at the
# server level; inference.sh also sends the same request-level setting.
if [[ "$MODEL_KIND" == "qwen3" ]]; then
    VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
    VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

echo "Model path              : $MODEL_PATH"
echo "Served model name       : $SERVED_MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES    : $CUDA_DEVICES"
echo "Tensor parallel size    : $TP_SIZE"
echo "Port                    : $PORT"
echo "Max model length        : $MAX_MODEL_LEN"
echo "Expected stop token     : $EXPECTED_STOP_TOKEN ($EXPECTED_STOP_TOKEN_ID)"

exec env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "${VLLM_ARGS[@]}"
