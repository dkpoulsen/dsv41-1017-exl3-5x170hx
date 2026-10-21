#!/bin/bash
# Start DeepSeek-V4.1-Flash-1017 Mia 2.9bpw EXL3 on RAM-Monster (5x CMP 170HX).
# Usage: run-mia.sh <tp5|pp5>
set -euo pipefail

LAYOUT=${1:-}
if [[ "$LAYOUT" != "tp5" && "$LAYOUT" != "pp5" ]]; then
  echo "Usage: $0 <tp5|pp5>" >&2
  exit 1
fi

HERE=/home/dkp/serve-mia
CFG="$HERE/ds41-mia-${LAYOUT}.yml"
TUNE="$HERE/autotune-${LAYOUT}"
P="$HERE/engine-patch"
E=/engine/src/exllamav3
DEF_CHUNK=$([[ "$LAYOUT" == "pp5" ]] && echo 65536 || echo 16384)
mkdir -p "$TUNE"
chown -R dkp:users "$TUNE"

run_podman() {
  sudo -u dkp env HOME=/home/dkp XDG_RUNTIME_DIR=/run/user/1002 PATH=/run/wrappers/bin:/run/current-system/sw/bin podman "$@"
}

for c in ds41-mia-tp5 ds41-mia-pp5 ds41-mia-fg; do
  run_podman rm -f "$c" >/dev/null 2>&1 || true
done
sleep 2

NAME="ds41-mia-${LAYOUT}"
echo "Starting $NAME with $CFG ..."
run_podman run -d --name "$NAME" \
  --device nvidia.com/gpu=all \
  --network=host \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=65536:65536 \
  -w /tabby \
  --entrypoint python3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_HOME=/usr/local/cuda \
  -e PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e PYTHONPATH=/engine/src:/engine/native \
  -e HF_HUB_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e TABBY_EXL3_GENERATOR_CHUNK=${GEN_CHUNK:-$DEF_CHUNK} \
  -e EXLLAMAV3_TUNE_CACHE=/autotune \
  -e EXLLAMA_NO_P2P_COPY=${NO_P2P_COPY:-0} \
  -e EXL3_DSA_INDEXER_DTYPE=sm_80:int8 \
  -e EXL3_INT8_GEMV=0 \
  -e EXL3_NGRAM_STREAM=${NGRAM_STREAM:-0} \
  -e EXL3_NGRAM_LOCK=${NGRAM_LOCK:-0} \
  -e EXL3_LAYERS_PER_DEVICE=${LAYERS_PER_DEVICE:-8,6,10,8,8} \
  -e EXL3_DSPARK_CONF=${EXL3_DSPARK_CONF:-0.4} \
  -v /home/dkp/serve-mia/native-opt:/engine/native:ro \
  -v "$P/generator.py:$E/generator/generator.py:ro" \
  -v "$P/row_table.py:$E/modules/row_table.py:ro" \
  -v "$P/ngram_embedding.py:$E/modules/ngram_embedding.py:ro" \
  -v "$P/deepseek_v41_mod.py:$E/modules/arch_specific/deepseek_v41.py:ro" \
  -v "$P/dsv4.py:$E/modules/dsv4.py:ro" \
  -v "$P/dspark.py:$E/modules/arch_specific/dspark.py:ro" \
  -v "$P/dsa.py:$E/cache/dsa.py:ro" \
  -v "$P/model_ls.py:$E/model/model_ls.py:ro" \
  -v "$P/device_copy.py:$E/util/device_copy.py:ro" \
  -v /home/dkp/models/1017-mia-2p9:/home/dkp/models/1017-mia-2p9 \
  -v "$HERE:/serve:ro" \
  -v "$TUNE:/autotune" \
  localhost/ds41-mia-exl3:latest \
  /tabby/main.py --config "/serve/ds41-mia-${LAYOUT}.yml"
