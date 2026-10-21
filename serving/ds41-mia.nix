{ config, lib, pkgs, ... }:

# ---------------------------------------------------------------------------
# DeepSeek-V4.1-Flash-1017 (Mia-AiLab EAQ-v1 2.9bpw EXL3 + 97.3 GiB MXFP4
# Engram in RAM + DSpark MTP speculative decoding) on 5x CMP 170HX (PP5).
#
#   https://github.com/dkpoulsen/dsv41-1017-exl3-5x170hx
#
# Serves on http://127.0.0.1:1920/v1 as "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
# at 1,048,576 context (FP16 KV cache), using:
#   - EXL3_LAYERS_PER_DEVICE=8,6,10,8,8 (pool/indexer-aligned 5-stage split,
#     co-locating DSpark tap layers 36..38 on GPU 4 with the MTP drafter)
#   - EXLLAMA_NO_P2P_COPY=0 (direct async P2P copies across all 5 CMP 170HX)
#   - EXL3_NGRAM_STREAM=0 (97.28 GiB packed e2m1 FP4 Engram held in host RAM)
#   - TABBY_EXL3_GENERATOR_CHUNK=65536 (16x 4096-token pipelined prefill)
#   - Engine patches from /home/dkp/serve-mia/engine-patch/
# ---------------------------------------------------------------------------

let
  cfg = config.services.ds41-mia;

  uid = "1002";

  podmanEnv = [
    "XDG_RUNTIME_DIR=/run/user/${uid}"
    "HOME=/home/dkp"
    "USER=dkp"
    "PATH=/run/wrappers/bin:/run/current-system/sw/bin"
  ];

  podman = "${pkgs.podman}/bin/podman";

  serveDir = "/home/dkp/serve-mia";
  patchDir = "${serveDir}/engine-patch";
  engineDir = "/engine/src/exllamav3";
  tuneDir = "${serveDir}/autotune-${cfg.layout}";
  genChunk = if cfg.layout == "pp5" then "65536" else "16384";

  launcher = pkgs.writeShellScript "ds41-mia-run" ''
    set -eu
    mkdir -p "${tuneDir}"
    ${podman} rm -f --ignore ds41-mia-tp5 ds41-mia-pp5 ds41-mia-fg >/dev/null 2>&1 || true
    exec ${podman} run --rm --replace \
      --name ${cfg.containerName} \
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
      -e TABBY_EXL3_GENERATOR_CHUNK=${genChunk} \
      -e EXLLAMAV3_TUNE_CACHE=/autotune \
      -e EXLLAMA_NO_P2P_COPY=0 \
      -e EXL3_DSA_INDEXER_DTYPE=sm_80:int8 \
      -e EXL3_INT8_GEMV=0 \
      -e EXL3_NGRAM_STREAM=0 \
      -e EXL3_NGRAM_LOCK=0 \
      -e EXL3_LAYERS_PER_DEVICE=${cfg.layersPerDevice} \
      -e EXL3_PROF=${cfg.prof} \
      -v ${serveDir}/native-opt:/engine/native:ro \
      -v ${patchDir}/generator.py:${engineDir}/generator/generator.py:ro \
      -v ${patchDir}/ngram_embedding.py:${engineDir}/modules/ngram_embedding.py:ro \
      -v ${patchDir}/deepseek_v41_mod.py:${engineDir}/modules/arch_specific/deepseek_v41.py:ro \
      -v ${patchDir}/dsv4.py:${engineDir}/modules/dsv4.py:ro \
      -v ${patchDir}/dspark.py:${engineDir}/modules/arch_specific/dspark.py:ro \
      -v ${patchDir}/dsa.py:${engineDir}/cache/dsa.py:ro \
      -v ${patchDir}/model_ls.py:${engineDir}/model/model_ls.py:ro \
      -v ${patchDir}/device_copy.py:${engineDir}/util/device_copy.py:ro \
      -v ${cfg.modelsDir}:${cfg.modelsDir} \
      -v ${serveDir}:/serve:ro \
      -v ${tuneDir}:/autotune \
      ${cfg.image} \
      /tabby/main.py --config /serve/ds41-mia-${cfg.layout}.yml
  '';
in
{
  options.services.ds41-mia = {
    enable = lib.mkEnableOption "DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw (PP5/TP5 + FP4 Engram in RAM) on 5x CMP 170HX";

    layout = lib.mkOption {
      type = lib.types.enum [ "pp5" "tp5" ];
      default = "pp5";
      description = "Serving layout (pp5 = 5-stage layer split, tp5 = 5-rank tensor parallel).";
    };

    prof = lib.mkOption {
      type = lib.types.str;
      default = "0";
      description = "EXL3_PROF: log the batched-step projection vs per-job split (diagnostic).";
    };

    layersPerDevice = lib.mkOption {
      type = lib.types.str;
      default = "8,6,10,8,8";
      description = "Layer counts per GPU in PP5 mode, aligned with kv_source and index_source boundaries.";
    };

    image = lib.mkOption {
      type = lib.types.str;
      default = "localhost/ds41-mia-exl3:latest";
      description = "Container image in user dkp's rootless podman store.";
    };

    containerName = lib.mkOption {
      type = lib.types.str;
      default = "ds41-mia-pp5";
      description = "Podman container name.";
    };

    modelsDir = lib.mkOption {
      type = lib.types.str;
      default = "/home/dkp/models/1017-mia-2p9";
      description = "Host directory containing the Mia 2.9bpw EXL3 view and engram-fp4.safetensors.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 1920;
      description = "OpenAI API port (configured in ds41-mia-pp5.yml).";
    };

    autoStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Start automatically at boot (multi-user.target).";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = config.users.users ? dkp;
        message = "services.ds41-mia needs user dkp (owns the rootless podman image store).";
      }
    ];

    systemd.services.ds41-mia = {
      description = "TabbyAPI/ExLlamaV3 DeepSeek-V4.1-Flash-1017 2.9bpw (${cfg.layout}, FP4 Engram in RAM) - 5x CMP 170HX";
      wants = [ "network-online.target" ];
      after = [
        "network-online.target"
        "nvidia-container-toolkit-cdi-generator.service"
      ];
      wantedBy = lib.mkIf cfg.autoStart [ "multi-user.target" ];

      serviceConfig = {
        Delegate = true;
        Type = "exec";
        User = "dkp";
        Group = "users";
        Restart = "on-failure";
        RestartSec = 30;
        TimeoutStartSec = 1800;
        Environment = podmanEnv;

        ExecStart = "${launcher}";
        ExecStop = "${podman} stop --ignore -t 30 ${cfg.containerName}";
        ExecStopPost = "${podman} rm -f --ignore ${cfg.containerName}";
      };
    };

    networking.firewall.allowedTCPPorts = [ cfg.port ];
  };
}
