{ config, lib, pkgs, ... }:

# ---------------------------------------------------------------------------
# DeepSeek-V4.1-Flash-1017 on 4x CMP 170HX (GNmac-lab vLLM PP4 stack).
# Serves on http://127.0.0.1:8000/v1 as "deepseek-v4.1-flash"
# with pipeline-parallel-size 4, layer partition [10,11,11,8],
# and PIECEWISE CUDA graphs.
# ---------------------------------------------------------------------------

let
  cfg = config.services.ds41-flash;

  uid = "1002";

  podmanEnv = [
    "XDG_RUNTIME_DIR=/run/user/${uid}"
    "HOME=/home/dkp"
    "USER=dkp"
    "PATH=/run/wrappers/bin:/run/current-system/sw/bin"
  ];

  podman = "${pkgs.podman}/bin/podman";

  launcher = pkgs.writeShellScript "ds41-flash-run" ''
    set -eu
    ${podman} rm -f --ignore ds41-flash ds41-flash-test ds41-flash-pp2tp2 ds41-debug >/dev/null 2>&1 || true
    exec ${podman} run --rm --replace \
      --name ds41-flash \
      --device nvidia.com/gpu=all --ipc=host \
      --cap-add SYS_NICE \
      --ulimit memlock=-1:-1 --ulimit stack=67108864 \
      -e PYTHONUNBUFFERED=1 \
      -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
      -e VLLM_PLUGINS=vllm_exl3 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXL3_MODEL_DIR="/models/DeepSeek-V4.1-Flash-1017-hybrid" \
      -e VLLM_USE_V2_MODEL_RUNNER=0 -e DSV41_DSPARK_V1=1 -e VLLM_EXL3_COOP=1 \
      -e VLLM_EXL3_TC_MOE_V2=1 -e VLLM_EXL3_TC_MOE_PREFILL=1 -e VLLM_EXL3_TC_PREFILL_ROWS=2048 \
      -e VLLM_EXL3_TC_MOE_SO=/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so -e VLLM_EXL3_TC_HOOK=/opt/dsv41-tc-moe/exl3_tc_hook.py \
      -e VLLM_EXL3_MOE_TP_ALIGN=128 -e VLLM_EXL3_MOE_TP_ROTATE=1 \
      -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_P2P_LEVEL=SYS \
      -e VLLM_PP_LAYER_PARTITION="10,11,11,8" \
      -e DSV41_ENGRAM_STORAGE=ssd -e DSV41_ENGRAM_DIR="/models/DeepSeek-V4.1-Flash-1017-engram" \
      -e DSV41_EXPERTS_ONLY_FILES_RE='^model-[0-9]+-of-000(43|39)\.safetensors$' \
      -e TORCH_CUDA_ARCH_LIST=8.0 -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor \
      -e VLLM_DSV41_DECODE_SPLITK=1 -e DSV41_SHARED_HEAVY_BLOCKS=1 \
      -e DSV41_PREFILL_CHUNK_REQS=1 -e DSV41_INDEXER_BUF_MULT=4 \
      -e DSV41_DRAFT_HEAD_FP8=1 \
      -e DSV41_VARIANTS=dsv41_topk_meta,dsv41_h2d_lean,dsv41_ragged_lean,dsv41_gate \
      -e DSV41_GATE_TAU=0 \
      -e VLLM_DSV4_INDEXER_DTYPE=int8 \
      -p 127.0.0.1:8000:8000 \
      -v /home/dkp/models:/models:ro \
      -v /home/dkp/ds41-flash-patches/mqa_logits_triton.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/mqa_logits_triton.py:ro \
      -v /home/dkp/gnmac-lab/cache/root-cache:/root/.cache \
      -v /home/dkp/gnmac-lab/cache/triton:/root/.triton \
      -v /home/dkp/gnmac-lab/cache/tilelang:/root/.tilelang \
      --entrypoint vllm \
      localhost/ds41-cmp170hx:1.1.0 \
      serve /models/DeepSeek-V4.1-Flash-1017-hybrid \
      --quantization exl3 --load-format safetensors --safetensors-load-strategy lazy \
      --tensor-parallel-size 1 --pipeline-parallel-size 4 \
      --served-model-name deepseek-v4.1-flash \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.95 \
      --kv-cache-dtype fp8_ds_mla --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1,2,3,4,5,6,8,12,18,24,30,36,42,48]}' \
      --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}' \
      --tokenizer-mode deepseek_v41 \
      --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
  '';
in
{
  options.services.ds41-flash = {
    enable = lib.mkEnableOption "DeepSeek-V4.1-Flash GNmac-lab vLLM PP4 serving engine";
  };

  config = lib.mkIf cfg.enable {
    # Keep the GPUs initialised between requests (avoids clock-ramp and
    # driver-unload latency on the first token after an idle period).
    systemd.services.nvidia-persistence = {
      description = "Enable NVIDIA persistence mode";
      wantedBy = [ "multi-user.target" ];
      after = [ "systemd-modules-load.service" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${config.boot.kernelPackages.nvidia_x11_latest.bin}/bin/nvidia-smi -pm 1";
      };
    };

    systemd.services.ds41-flash = {
      description = "DeepSeek-V4.1-Flash GNmac-lab vLLM PP4 serving engine";
      wants = [ "network-online.target" ];
      after = [ "network.target" "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "exec";
        User = "dkp";
        Group = "users";
        TimeoutStartSec = "15m";
        TimeoutStopSec = "60s";
        Environment = podmanEnv;
        ExecStart = "${launcher}";
        ExecStartPost = "${pkgs.python3}/bin/python3 /etc/nixos/prewarm-agent.py";
        Restart = "on-failure";
        RestartSec = 5;
      };
    };
  };
}
