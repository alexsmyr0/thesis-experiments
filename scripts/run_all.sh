#!/bin/sh
# Run the experiment pipeline from the repo root or anywhere else.
#   sh scripts/run_all.sh --quick     # split + TF-IDF only (seconds)
#   sh scripts/run_all.sh             # + CodeBERT full and LoRA
#   sh scripts/run_all.sh --qwen      # also Qwen-7B prompting and MLX LoRA (Apple Silicon only, hours)
#   Add --codebert=full or --codebert=lora to run only one CodeBERT configuration (default: both).
#
# THESIS_CSV / THESIS_OUT (absolute paths) redirect input and all outputs; see src/common.py.
#   THESIS_CSV=$PWD/data/v2/neutral_all.csv THESIS_OUT=$PWD/experiments/neutral_all sh scripts/run_all.sh
# CODEBERT_ARGS passes extra flags to codebert_finetune.py (e.g. CODEBERT_ARGS=--fp16 on a CUDA GPU).
# PY overrides the interpreter (default: .venv/bin/python in the repo if present, else python3).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/src"
if [ -z "$PY" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"; else PY=python3; fi
fi
LOGS="${THESIS_OUT:-$REPO/experiments/reproduction_v1}/logs"
mkdir -p "$LOGS"

# Plain sh has no pipefail, so write logs with redirection to keep failures fatal.
step() { log=$1; shift; echo "== $*"; "$PY" "$@" > "$LOGS/$log.log" 2>&1 || { tail -20 "$LOGS/$log.log"; exit 1; }; }

QUICK=0; QWEN=0; CB="full lora"
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --qwen) QWEN=1 ;;
    --codebert=full) CB="full" ;;
    --codebert=lora) CB="lora" ;;
    --codebert=both) CB="full lora" ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

step prepare_data prepare_data.py
step tfidf tfidf_baselines.py
if [ "$QUICK" = 0 ]; then
  for m in $CB; do step codebert_$m codebert_finetune.py --mode $m $CODEBERT_ARGS; done
fi
if [ "$QWEN" = 1 ]; then
  step prompting_zeroshot qwen_prompting.py --mode zeroshot
  step prompting_fewshot qwen_prompting.py --mode fewshot
  step qwen_lora_prepare qwen_lora.py prepare
  step qwen_lora_run qwen_lora.py train
  step qwen_lora_eval qwen_lora.py evaluate
fi
"$PY" summarize.py | tee "$LOGS/summary.txt"
