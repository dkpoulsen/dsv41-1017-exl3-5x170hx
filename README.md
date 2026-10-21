# DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw + MXFP4 Engram on 5× CMP 170HX (TP5 vs PP5 at 1M context)

Adapting [`anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m`](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m) to an all-Ampere **5× NVIDIA CMP 170HX (64 GB each, sm_80, PCIe Gen2 x4)** machine with **251 GiB host RAM** — no 96 GB RTX PRO 6000 Blackwell card, and half the host RAM of the reference box.

Three sets of changes make the full **1,048,576-token FP16 context + DSpark MTP speculative decoding + RAM-resident Engram** recipe fit comfortably and run fast on this hardware:

1. **Pack selection (`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`)**: LibertAI's 3.5 bpw pack quantizes the 3-layer DSpark MTP drafter to 6 bpw EXL3 (**20.0 GiB**), which only fits when parked on a 96 GB card. The 2.9 bpw family (`Mia-AiLab EAQ-v1 2.9bpw` and `LibertAIDAI 2.9bpw`) quantizes the MTP drafter to 4 bpw EXL3 (**6.75–6.88 GiB**) and the main trunk to **189.5 GiB**, fitting all 384 routed experts + 1M FP16 KV cache + MTP drafter across five 64 GB cards with 10–17 GiB/card to spare.
2. **Offline MXFP4 (`e2m1`) Engram conversion + RAM residency (`patches/convert_engram_fp4.py`, `patches/ngram_embedding.py`)**: The base checkpoint's two Engram n-gram lookup tables (`layers.{1,14}.engram.embed.{weight,scale}`) total **188.83 GiB** in FP8 (`F8_E4M3` + `F8_E8M0`), leaving almost no breathing room on a 251 GiB host. Converting them to packed `e2m1` nibbles (`U8 [rows, 128]`) with re-computed per-32-element `F8_E8M0` power-of-two budgets (`e + 127`) halves the tables to **97.28 GiB** (`rel-L2 = 0.1205 / 0.1209` vs FP8) so the entire Engram table stays permanently in host RAM (`EXL3_NGRAM_STREAM=0`) in both **PP5** (anonymous RAM) and **TP5** (shared `mmap` across all 5 worker processes), with **>105 GiB of free RAM** remaining.
3. **PP5 & sm_80 execution optimizations (`patches/all_optimizations.patch`, inspired by [`Morrowmake/vllm-cmp170hx`](https://github.com/Morrowmake/vllm-cmp170hx))**:
   - **Pool/indexer-aligned 5-stage layer partition (`EXL3_LAYERS_PER_DEVICE=8,6,10,8,8`)**: Greedy `autosplit` placed all 40 trunk layers on GPUs 0–3 (`[10, 10, 11, 9]`) and left **0 trunk layers on GPU 4**, while slicing right across `kv_source_layer_ids = [2, 8, 14, 20]` and `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`. Partitioning as **`[8, 6, 10, 8, 8]`** (`0..7`, `8..13`, `14..23`, `24..31`, `32..39`) activates all 5 GPUs (`[40.8, 30.8, 43.8, 41.3, 51.8]` GiB), keeps KV Pools 2, 8, and 14 strictly on their owner GPUs (zero mirrors, zero `pull()` calls), keeps **every** `index_source` group (`2..7`, `8..13`, `14..19`, `20..23`, `24..27`, `28..31`, `32..35`, `36..39`) on a single GPU (zero cross-GPU `topk` index copies), and co-locates `dspark_target_layer_ids = [37, 38, 39]` on GPU 4 with the MTP drafter (zero cross-GPU `export_states` transfers).
   - **Async P2P stage hops (`EXLLAMA_NO_P2P_COPY=0` + `non_blocking=True` in `device_copy.py` & `dsa.py`)**: The reference recipe set `EXLLAMA_NO_P2P_COPY=1` because its 5th card was a Blackwell GPU without P2P to Ampere, forcing every stage hop in `to_device()` to run `t.cpu().to(device)` (a synchronous D2H copy into pageable RAM that stalled the Python thread at every GPU boundary: `2.93 ms` per 4 hops at decode and `1,786 ms` per 4,096-token chunk at prefill). On 5× CMP 170HX with `cmpunlocker` P2P enabled, all 20 GPU pairs pass direct P2P verification (`0.61 ms` at decode, `362 ms` per 4K chunk at prefill — **4.8× faster**), and contiguous multi-page KV pool mirror pulls (`dsa.py`) collapse from `64 × 3 = 192` copies into `3` slice copies.
   - **True 5-stage pipelined prefill (`chunk_size: 4096` + `TABBY_EXL3_GENERATOR_CHUNK=65536` + GPU-resident `export_states` in `model_ls.py`)**: With `M = 16` sub-chunks of 4,096 tokens per generator chunk across `S = 5` stages (and keeping `export_states` on GPU 4 instead of calling `.cpu()` after every sub-chunk), all 5 GPUs compute concurrently during prefill (**2,573 tok/s** at 56K and **2,448 tok/s** at 258K — **2.08× faster** than baseline PP5 and **2.9× faster** than TP5).
   - **Fused `exl3_mgemm` projection fans for DeepSeek-V4.1 (`dsv4.py`, `dspark.py`)**: Upstream gated `_build_x_fan()` on `self.q_head_norm`, which is `False` on DeepSeek-V4.1, disabling both `x_fan` (`[q_a, wkv, comp.wkv, comp.wgate]`) and `q_fan` (`[q_b, idx_wq_b]`) across all 40 trunk layers and 3 MTP layers. Enabling `x_fan` and `q_fan` for `q_head_norm=False` (plus grouped `exl3_mgemm` `wo_a` output projection in `DSparkAttention.forward`) eliminates ~75 separate GEMM/Hadamard launches per step.
   - **1-kernel LUT FP4 Engram decode + background worker prefetch (`ngram_embedding.py`, `deepseek_v41_mod.py`)**: Replaced `_decode_fp4`'s per-call `torch.tensor(_E2M1_LUT, device=dev)` (which caused a synchronous H2D `cudaMemcpy` + 11 elementwise kernels per Engram layer) with per-device cached `(256, 2)` byte-pair and `(256,)` E8M0 scale LUTs (`byte_lut[packed.long()].view(U, 8, 32) * scale_lut[s.long()].unsqueeze(-1)`), and added `prefetch_ids` to `EngramLayer` so CPU n-gram hashing and RAM row staging overlap GPU compute on a worker thread.

---

## Headline Results (5× CMP 170HX, PCIe Gen2 x4, 1M FP16 Context, DSpark MTP)

| Metric | TP5 (FP8 baseline) | **TP5 (FP4 + Opt)** | PP5 (FP8 baseline) | PP5 (FP4 in RAM) | **PP5 (FP4 + Opt)** |
|---|---:|---:|---:|---:|---:|
| Engram RAM footprint | 189 GiB page cache | **97.3 GiB shared mmap** | 189 GiB page cache | 97.3 GiB anon RAM | **97.3 GiB anon RAM** |
| Free / available host RAM | ~29 GiB free | **105 GiB free (206 GiB avail)** | ~29 GiB free | 124 GiB free | **124 GiB free** |
| Model load time | ~285 s | ~285 s | 192.7 s | 192.7 s | **157.5 s** |
| VRAM per GPU (GiB) | 47.3, 44.6×4 | 47.3, 44.6×4 | 48.4, 50.6, 51.4, 48.4, **17.7** | 48.4, 50.6, 51.4, 48.4, 17.7 | **40.8, 30.8, 43.8, 41.3, 51.8** |
| **Decode: `utf8_len` (C code)** | 71.8 tok/s | **82.9–91.8 tok/s** | 48.4 tok/s | 46.4 tok/s | **64.2–65.4 tok/s (+35–48%)** |
| **Decode: `review` (600 tok)** | 55.5 tok/s | **50.5–58.0 tok/s** | 32.7 tok/s | 33.8 tok/s | **45.7 tok/s (+40%)** |
| **Decode: `agreement` (prose)** | — | **73.1–76.9 tok/s** | 44.1 tok/s | 45.6 tok/s | **51.3–54.0 tok/s (+22%)** |
| **Decode: `reason` (math)** | — | **86.4–95.6 tok/s** | 52.6 tok/s | 58.3 tok/s | **70.3 tok/s (+34%)** |
| **Cold prefill ~55K** | 883.8 tok/s (64.9 s) | 823.4 tok/s (67.8 s) | 1,235.0 tok/s (44.3 s) | 1,244.5 tok/s (44.3 s) | **2,572.7 tok/s (21.7 s, 2.08×)** |
| **Cold prefill ~255K** | 908.2 tok/s (292.6 s) | **935.8 tok/s** (275.3 s) | 1,436.3 tok/s (176.3 s) | 1,420.5 tok/s (179.6 s) | **2,447.6 tok/s (105.3 s, 1.70×)** |
| **Cold prefill 1.04M** | 832.6 tok/s (20.9 min) | — | 1,264.5 tok/s (13.8 min) | — | *~2,400 tok/s (~7.2 min est.)* |
| DSML tool calling (`get_weather`) | Pass | **Pass** | Pass | Pass | **Pass** |
| Greedy output (`agreement` & `reason`) | Reference | **Byte-identical** | **Byte-identical** | **Byte-identical** | **Byte-identical** |

### Concurrency Scaling Benchmark (1, 2, 4, 8 Concurrent Users)

Measured with `bench/bench_conc.py` on this machine (greedy, reasoning on as served, 400 max completion tokens per request, unique non-repeating prompts across requests to prevent prefix cache hits):

| Users | Layout | Tokens Generated | Wall Time | Median TTFT | Median Decode | **Aggregate Decode** | **Aggregate End-to-End** |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1 user** | **PP5** | 287 | 5.5 s | 0.62 s | 59.1 tok/s | 59.3 tok/s | 52.5 tok/s |
| | **TP5** | 305 | 4.1 s | 0.38 s | 81.6 tok/s | **81.8 tok/s** | **74.2 tok/s** |
| **2 users** | **PP5** | 687 | 16.5 s | 1.30 s | 27.5 tok/s | 44.9 tok/s | 41.5 tok/s |
| | **TP5** | 687 | 11.4 s | 0.74 s | 39.2 tok/s | **64.4 tok/s** | **60.2 tok/s** |
| **4 users** | **PP5** | 872 | 21.0 s | 2.13 s | 20.3 tok/s | 46.0 tok/s | 41.6 tok/s |
| | **TP5** | 873 | 12.6 s | 1.20 s | 32.1 tok/s | **76.2 tok/s** | **69.3 tok/s** |
| **8 users** | **PP5** | 2,114 | 40.7 s | 3.64 s | 10.9 tok/s | 54.8 tok/s | 52.0 tok/s |
| | **TP5** | 2,279 | 26.0 s | 1.99 s | 17.0 tok/s | **91.5 tok/s** | **87.5 tok/s** |

#### Why TP5 scales better on multi-user decode while PP5 dominates long prefill
- **Under TP5**: All 5 GPUs participate in every single token's matrix multiplications in parallel. Expanding batch size from 1 to 8 amortizes the HBM weight bandwidth and fills the 74 streaming multiprocessors per card, pushing aggregate decode from **81.8 tok/s → 91.5 tok/s**.
- **Under PP5**: Pipeline parallelism splits the 40 layers sequentially across the 5 GPUs. Without micro-batch interleaving across requests (the engine batches active requests into one forward pass), only 1 GPU is executing at any given instant during a decode step. Aggregate decode is bounded by the single-card throughput of the full-layer execution (~45–55 tok/s). However, during prefill, pipelining micro-chunks across all 5 stages delivers **2,573 tok/s** (2.9× faster than TP5).

---

## Hardware vs reference recipe

| Component | `anoane` reference machine | This machine (`RAM-Monster`) |
|---|---|---|
| GPUs | 4× CMP 170HX (64 GB, sm_80) + 1× RTX PRO 6000 (96 GB, sm_120) | **5× CMP 170HX (64 GB, sm_80)** |
| PCIe links | PCIe Gen2 **x16** (~6.4 GB/s per direction) | PCIe Gen2 **x4** (~1.6 GB/s per direction, risers) |
| CPU | AMD EPYC 9J14 (96C) | AMD Ryzen Threadripper PRO 3975WX (32C/64T) |
| Host RAM | 499 GiB (189 GiB FP8 Engram in page cache) | **251 GiB** (**97.3 GiB FP4 Engram in RAM**, Swap = 0) |
| Driver | NVIDIA open 610.57.04 + `cmpunlocker` (Gen2 + P2P) | NVIDIA open 615.71.09 + `cmpunlocker` (Gen2 + P2P) on NixOS 26.11 |
| Engine | `anoane/exllamav3-anemone` @ `309ef6d1` (`dsv41f-1017`, INT8 DSA indexer) | Same commit `309ef6d1` (`sm_80`) + `patches/all_optimizations.patch` |
| Server | `anoane/tabbyAPI` @ `ce250ff` (`deepseek-v41`) | Same commit `ce250ff` |

---

## 1. Survey of `DeepSeek-V4.1-Flash-1017` EXL3 packs for 5× 64 GB cards

Measured directly from the safetensors headers on Hugging Face (all derived from `deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10`):

| Pack | Trunk (excl. MTP) | MTP drafter | Fits TP5 + MTP @ 1M on 5× 64 GB? | Published fidelity / notes |
|---|---:|---:|---|---|
| `LibertAIDAI/...-3.5bpw` (reference repo) | 219.9 GiB | **20.0 GiB** (6 bpw) | **No** (draft card needs ~72+ GiB) | ppl 1.8895 code / 6.4341 prose |
| `PixelML/...-3bpw` | 200.3 GiB | 7.39 GiB (FP8 source) | Tight, but needs FP8 dense/MTP loader | **Uncalibrated** (identity Hessian), unmeasured |
| **`Mia-AiLab/...-EAQ-v1-2.9-bpw`** *(tested here)* | **189.45 GiB** | **6.75 GiB** (4 bpw) | **Yes** (47.3 / 44.6×4 GiB allocated) | **KL 0.0137 nats, top-1 96.6%** vs original 1017 |
| `LibertAIDAI/...-2.9bpw` | 189.46 GiB | **6.88 GiB** (4/6 bpw) | **Yes** (same tensor map as Mia 2.9) | ppl +1.8% code / +0.4% prose vs original 1017 |
| `jayleaton/...-2.8bpw` | 183.0 GiB | 6.75 GiB | Yes | Coding/math/tool-call weighted allocation |
| `HeNryous/...-2.75bpw-DGX-Spark` | 181.4 GiB | 3.68 GiB | Yes | KLD 0.0201 / top-1 95.96% |
| `sudoingx/...-2.7bpw` | 177.5 GiB | 5.29 GiB (K3) | Yes | Per-expert sensitivity allocation (2–5 bpw) |

---

## 2. MXFP4 (`e2m1`) Engram Conversion & In-RAM Serving

### Format (`patches/convert_engram_fp4.py`)

Each Engram table in `deepseek-ai/DeepSeek-V4.1-Flash-1017` is a `[rows, 256]` `F8_E4M3` matrix with a `[rows, 8]` `F8_E8M0` block scale (one power-of-two exponent byte per 32 values; `rows = 384,006,168` on layer 1 and `384,016,682` on layer 14).

Because the stored `e4m3` values use the full FP8 dynamic range (measured `|v|_max = 7.5`, scale bytes `117..122`), the raw `e8m0` exponent byte cannot be reused directly for `e2m1` (max magnitude `6.0`). For each 32-value block we reconstruct the true FP32 value `v = e4m3 * 2^(s - 127)` on GPU and compute a new power-of-two budget:

```
e        = ceil(log2(max(|v|) / 6.0))           # so |v| / 2^e <= 6.0
code     = sign(q) << 3 | bucketize(|q|)        # into {0, 0.5, 1, 1.5, 2, 3, 4, 6}
packed   = code[0::2] | (code[1::2] << 4)       # uint8 [rows, 128]
newscale = uint8(e + 127)                       # F8_E8M0 [rows, 8]
```

- **Conversion speed**: ~8.5 minutes total for both 384M-row tables on a single CMP 170HX (`cuda:0`, 4M rows/chunk).
- **Accuracy (`results/convert.log`)**: `rel-L2 = 0.1205` (layer 1) and `0.1209` (layer 14).
- **Size**: `188.83 GiB → 97.28 GiB` (`104,451,108,088` bytes in a single `engram-fp4.safetensors`).

---

## 3. Deep-Dive on the PP5 & sm_80 Performance Optimizations

### A. Pool- and Indexer-Aligned 5-GPU Layer Partition (`EXL3_LAYERS_PER_DEVICE=8,6,10,8,8`)
In `DeepSeek-V4.1-Flash-1017`, KV pools and DSA lightning-indexer selections are shared across layer groups:
- `kv_source_layer_ids = [2, 8, 14, 20]`
- `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`
- `dspark_target_layer_ids = [37, 38, 39]` (taps taken from the output of layers `36, 37, 38`)
- Layers `18..22` use 2-bit routed experts (`3.28 GiB`/layer vs `4.86 GiB`/layer elsewhere).

Greedy `autosplit` packed all 40 layers onto GPUs 0–3 (`10, 10, 11, 9`), leaving **0 trunk layers on GPU 4**, splitting Pool 8 (`8..13`) across GPUs 0–1, splitting Indexer 28 (`28..31`) across GPUs 2–3, and placing the DSpark tap layers (`36, 37, 38`) on GPU 3 while the MTP drafter lived on GPU 4.

Setting `EXL3_LAYERS_PER_DEVICE=8,6,10,8,8` (`patches/model_ls.py`):
- **GPU 0 (`0..7`)**: Sliding `0..1` + entire Pool/Indexer `2..7` (`40.8 GiB`)
- **GPU 1 (`8..13`)**: Entire Pool/Indexer `8..13` (`30.8 GiB`)
- **GPU 2 (`14..23`)**: Entire Pool/Indexer `14..19` + entire Indexer `20..23` (`43.8 GiB`)
- **GPU 3 (`24..31`)**: Entire Indexers `24..27` and `28..31` (`41.3 GiB`)
- **GPU 4 (`32..39` + `head` + MTP)**: Entire Indexers `32..35` and `36..39` + DSpark taps `36..38` + `head` + MTP drafter (`51.8 GiB`)

### B. Direct Async P2P Stage Hops (`EXLLAMA_NO_P2P_COPY=0`, `patches/device_copy.py`, `patches/dsa.py`)
With `cmpunlocker` P2P active across all 5 CMP 170HX cards, `canDeviceAccessPeer` is `True` for all 20 GPU pairs. Switching `EXLLAMA_NO_P2P_COPY=0` and using `non_blocking=True` on direct CUDA-to-CUDA moves eliminates the pageable CPU bounce (`t.cpu().to(device)`) and lets the Python host thread enqueue kernels across all 5 GPUs without stalling at stage boundaries. In `CacheLayer_dsa.pull()` (`patches/dsa.py`), contiguous page ranges are copied in a single slice per pool tensor instead of looping page-by-page.

### C. 5-Stage Pipelined Prefill (`TABBY_EXL3_GENERATOR_CHUNK=65536`, `patches/model_ls.py`)
`Model_LSMixin.prefill_ls_pipelined` splits a generator prefill chunk of `rows` tokens into sub-chunks of `max_chunk_size` (`4,096` tokens) and pipelines them across the devices holding trunk layers. With `TABBY_EXL3_GENERATOR_CHUNK=16384` (`4` sub-chunks) and only 4 active trunk GPUs, pipeline fill/drain bubbles consumed half of every call. Raising `TABBY_EXL3_GENERATOR_CHUNK=65536` (`16` sub-chunks of `4,096` tokens across all 5 GPUs) and keeping `export_states` on GPU 4 (instead of calling `.cpu()` at the end of every sub-chunk) doubled PP5 cold prefill speed from **1,235 → 2,573 tok/s** at 56K and **1,436 → 2,448 tok/s** at 258K.

### D. Fused `exl3_mgemm` Projection Fans (`patches/dsv4.py`, `patches/dspark.py`)
Enabled `x_fan` (`[q_a, wkv]` / `[q_a, wkv, comp.wkv, comp.wgate]`) and `q_fan` (`[q_b, idx_wq_b]`) when `q_head_norm=False` (DeepSeek-V4.1), and replaced the 8-slice Python `wo_a` loop in `DSparkAttention.forward` with `self._project_o_grouped` (`exl3_mgemm`).

### E. Optimized `exl3_moe_coop` CUDA Kernels (`patches/moe_coop/`)
Applied the 4 targeted MoE kernel optimizations from `GNmac-lab`:
1. **MMA Operand Swap (`patch_coop_mma_swap.py`)**: Transposed the operand assignments in `mma.sync.m16n8k16` so decoded weights feed the $A$ operand and activations feed the $B$ operand, cutting from two MMAs with zero padding to **one MMA per 16×16 tile** without padding (`gate/up -12.7%, down -12%`).
2. **Branch-Free Loop (`patch_coop_branchfree.py`)**: Eliminated divergent branch bookkeeping in the prefetch ring loop for 2, 3, and 4 bpw register-decode widths.
3. **512-Column Down Tiles (`patch_coop_down_wide16.py`)**: Sized down-projection tiles with 16 warps along $N$, eliminating cross-warp reductions on the short $K=512$ down stage (`102.5 → 87.2 µs`).
4. **Parallel Chunk Epilogue (`patch_coop_down_epilogue.py`)**: Parallelized completion counters and chunk accumulation.

Together, these pushed single-stream decode on `c_code` from **44.2 → 65.4 tok/s (+48%)**, `review` from **33.6 → 45.7 tok/s (+36%)**, and `reason` from **52.6 → 70.3 tok/s (+34%)** while preserving byte-exact greedy outputs.

---

## Repository Layout

- `build/` — `Containerfile`, `headers.py`, `check_engine.py`, `get_engine.sh`, `make_view.py`, `chat_template.jinja`, `tokenizer_config.json`
- `patches/` — `convert_engram_fp4.py`, `all_optimizations.patch`, `moe_coop/` (the 4 C++/CUDA MoE kernel patches + unified diffs), and the 8 patched engine modules (`generator.py`, `ngram_embedding.py`, `deepseek_v41_mod.py`, `dsv4.py`, `dspark.py`, `dsa.py`, `model_ls.py`, `device_copy.py`)
- `serving/` — `ds41-mia-tp5.yml`, `ds41-mia-pp5.yml`, `ds41-mia.nix` (NixOS systemd service), `run-mia.sh`, `run-fg.sh`, `prewarm-engram.sh`
- `bench/` — `bench_ab.py`, `bench_conc.py`, `compare.py`, `capture_ref.py`
- `results/` — `results-tp5*.json`, `results-pp5*.json`, `results-conc-*.json`, `ref-*.json`, `convert.log`
