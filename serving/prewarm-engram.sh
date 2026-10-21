#!/bin/bash
# Read the engram table file(s) of the served view into the page cache
# (read-ahead off on purpose, FADV_RANDOM). Prefers engram-fp4.safetensors (97.28 GiB)
# when present in the view, else falls back to model-00047/00048 (189 GiB fp8).
set -euo pipefail
VIEW=/home/dkp/models/1017-mia-2p9/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
if [[ -e "$VIEW/engram-fp4.safetensors" ]]; then
  FILES=("$VIEW/engram-fp4.safetensors")
else
  for f in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    [[ -e "$VIEW/$f" ]] || { echo "no $f under $VIEW" >&2; exit 1; }
  done
  FILES=("$VIEW/model-00047-of-00048.safetensors" "$VIEW/model-00048-of-00048.safetensors")
fi
if [[ ${PREWARM_DROP:-0} == 1 ]]; then
  sync; echo 3 > /proc/sys/vm/drop_caches
fi
echo "before: $(free -g | awk '/Mem:/ {print "used " $3 " GiB, cache " $6 " GiB, available " $7 " GiB"}')"
python3 - "${FILES[@]}" <<'PY'
import os, sys, time
t0, total = time.time(), 0
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            total += len(block)
    finally:
        os.close(fd)
print(f"read {total} bytes ({total/2**30:.2f} GiB) in {time.time() - t0:.0f} s")
PY
echo "after:  $(free -g | awk '/Mem:/ {print "used " $3 " GiB, cache " $6 " GiB, available " $7 " GiB"}')"
