#!/usr/bin/env bash
# Full pipeline for the multi-height experiments: smoke test -> MultiviewX -> Wildtrack -> collect results.
#
#   conda activate earlybird
#   cd EarlyBird && git pull
#   tmux new -s all
#   bash run_all.sh                       # data in ../data/MultiviewX and ../data/Wildtrack
#   MVX_DIR=/path/MultiviewX WT_DIR=/path/Wildtrack bash run_all.sh
#
# Arms: none (EarlyBird), m4 (M1 gate + pooled Re-ID + multi-height), h2 (multi-height only, ablation of m4).
# Seeds 42 2024 1337, each with the conf-threshold sweep and the tracker sweep.
# Safe to re-run after an interruption: finished steps are skipped, training resumes from last.ckpt.
# Everything is logged to ../experiments/run_all.log; all result zips end up in ../experiments/all_results.tar.gz.
set -uo pipefail
cd "$(dirname "$0")"

MVX_DIR="${MVX_DIR:-../data/MultiviewX}"
WT_DIR="${WT_DIR:-../data/Wildtrack}"
ARMS="${ARMS:-none m4 h2}"
SEEDS="${SEEDS:-42 2024 1337}"
OUT="$(cd .. && pwd)/experiments"
mkdir -p "$OUT"
LOG="$OUT/run_all.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
FAILED=()

say "===== run_all: arms [$ARMS] seeds [$SEEDS] | commit $(git log -1 --format='%h %s')"
python -c "import torch, lightning; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, '| lightning', lightning.__version__, '|', torch.cuda.get_device_name(0))" \
  || { say "Python env lỗi – chưa 'conda activate earlybird'?"; exit 1; }

# 1) smoke: the new h2 arm on MultiviewX (1 epoch x 20 batches), fail early before days of training
if [ -d "$MVX_DIR" ]; then
  say "===== 1/4 smoke h2 (MultiviewX)"
  python run_experiments.py --data_dir "$MVX_DIR" --smoke --arms h2 --out "$OUT" || { say "SMOKE LỖI – dừng"; exit 1; }
fi

run_dataset() {  # $1 = name, $2 = data dir
  if [ ! -d "$2" ]; then
    say "!!!!! bỏ qua $1: không thấy $2 (đặt biến ${1^^}_DIR=...)"
    FAILED+=("$1: missing data dir")
    return
  fi
  say "===== $1: $2"
  # unit tests run once per dataset inside run_experiments.py
  if ! python run_experiments.py --data_dir "$2" --arms $ARMS --seeds $SEEDS --tracker_sweep --out "$OUT"; then
    say "!!!!! $1 có lỗi – xem log trên, chuyển sang bước tiếp theo"
    FAILED+=("$1")
  fi
}

say "===== 2/4 MultiviewX"
run_dataset mvx "$MVX_DIR"
say "===== 3/4 Wildtrack"
run_dataset wt "$WT_DIR"

say "===== 4/4 gom kết quả"
( cd "$OUT" && tar czf all_results.tar.gz results_*.zip run_all.log 2>/dev/null )
ls -la "$OUT"/results_*.zip | awk '{print $5, $9}'
say "📦 $OUT/all_results.tar.gz"
if [ ${#FAILED[@]} -gt 0 ]; then
  say "XONG NHƯNG CÓ LỖI: ${FAILED[*]}"
  exit 1
fi
say "XONG TẤT CẢ"
