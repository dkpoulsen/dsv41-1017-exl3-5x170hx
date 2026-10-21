#!/usr/bin/env bash
# Launch the Mia 2.9 EXL3 model in one layout, in a container, on all five CMP 170HX.
#   run-mia.sh <tp5|pp5> [stop]
# Stops any running ds41-mia container first. Logs: podman logs -f ds41-mia
set -euo pipefail
L="${1:?usage: run-mia.sh <tp5|pp5> [stop]}"
if [[ "${2:-}" == "stop" || -n "${2:-}" ]]; then :; fi
case "$L" in tp5|pp5) ;; *) echo "layout must be tp5 or pp5" >&2; exit 2;; esac

pod() { sudo -u dkp env HOME=/home/dkp XDG_RUNTIME_DIR=/run/user/1002 USER=dkp \
        PATH=/run/wrappers/bin:/run/current-system/sw/bin "$@"; }

pod podman stop --ignore -t 30 ds41-mia >/dev/null 2>&1 || true
pod podman rm -f --ignore ds41-mia >/dev/null 2>&1 || true
mkdir -p /home/dkp/serve-mia/autotune-$L

pod podman run -d --replace \
  --name ds41-mia \
  --device nvidia.com/gpu=all \
  --network=host --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
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
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e TABBY_EXL3_GENERATOR_CHUNK=16384 \
  -e EXLLAMAV3_TUNE_CACHE=/autotune \
  -e EXLLAMA_NO_P2P_COPY=1 \
  -e EXL3_DSA_INDEXER_DTYPE=sm_80:int8 \
  -e EXL3_INT8_GEMV=0 \
  -e EXL3_NGRAM_STREAM=${NGRAM_STREAM:-0} \
  -e EXL3_NGRAM_LOCK=${NGRAM_LOCK:-0} \
  -v /home/dkp/serve-mia/engine-patch/ngram_embedding.py:/engine/src/exllamav3/modules/ngram_embedding.py:ro \
  -v /home/dkp/models/1017-mia-2p9:/home/dkp/models/1017-mia-2p9 \
  -v /home/dkp/serve-mia:/serve:ro \
  -v /home/dkp/serve-mia/autotune-$L:/autotune \
  localhost/ds41-mia-exl3:latest \
  /tabby/main.py --config /serve/ds41-mia-$L.yml

echo "started layout $L (container ds41-mia); waiting for :1920 ..."
for i in $(seq 1 240); do
  if curl -s -o /dev/null http://127.0.0.1:1920/v1/models; then echo "port is up"; exit 0; fi
  if ! pod podman ps --filter name=ds41-mia --format "{{.Names}}" 2>/dev/null | grep -q ds41-mia; then
    echo "container exited during load; last logs:" >&2
    pod podman logs --tail 80 ds41-mia >&2 2>&1 || true
    exit 1
  fi
  sleep 10
done
echo "port did not come up in 40 min; last logs:" >&2
pod podman logs --tail 80 ds41-mia >&2 2>&1 || true
exit 1
