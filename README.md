# DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw + MXFP4 Engram on 5× CMP 170HX (TP5 vs PP5 at 1M context)

Adapting [`anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m`](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m) to an all-Ampere **5× NVIDIA CMP 170HX (64 GB each, sm_80, PCIe Gen2 x4)** machine with **251 GiB host RAM** — no 96 GB RTX PRO 6000 Blackwell card, and half the host RAM of the reference box.

Three sets of changes make the full **1,048,576-token FP16 context + DSpark MTP speculative decoding + RAM-resident Engram** recipe fit comfortably and run fast on this hardware:

> **Current production default (2026-10-22):** the machine serves **GNmac-lab vLLM TP4** (`--tensor-parallel-size 4 --pipeline-parallel-size 1`, GPUs `0..3`) as `services.ds41-flash` on `http://127.0.0.1:8000/v1`. While PP5 achieves 2,268 tok/s cold prefill, single-stream interactive decode on PP5 collapses to 1.0–13.9 tok/s on real-world reasoning prompts due to PCIe Gen2 x4 pipeline serialization, whereas TP4 delivers **52.8–183.1 tok/s (10× to 60× faster)** and leaves GPU 4 completely free. See [§4](#4-alternative-serving-engines--architecture-exploration) and [§6](#6-production-deployment--serving-stack).

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
| **Decode: `utf8_len` (C code)** | 71.8 tok/s | **82.9–91.8 tok/s** | 48.4 tok/s | 46.4 tok/s | **59.8–65.4 tok/s (+35–48%)** |
| **Decode: `review` (600 tok)** | 55.5 tok/s | **50.5–58.0 tok/s** | 32.7 tok/s | 33.8 tok/s | **44.4–45.7 tok/s (+32–36%)** |
| **Decode: `agreement` (prose)** | — | **73.1–76.9 tok/s** | 44.1 tok/s | 45.6 tok/s | **55.1–55.9 tok/s (+25–27%)** |
| **Decode: `reason` (math)** | — | **86.4–95.6 tok/s** | 52.6 tok/s | 58.3 tok/s | **77.8–80.7 tok/s (+48–53%)** |
| **Cold prefill ~15K** | — | — | — | — | **1,529.3 tok/s (9.9 s)** |
| **Cold prefill ~55K** | 883.8 tok/s (64.9 s) | 823.4 tok/s (67.8 s) | 1,235.0 tok/s (44.3 s) | 1,244.5 tok/s (44.3 s) | **2,963.5 tok/s (18.9 s, 2.40×)** |
| **Cold prefill ~255K** | 908.2 tok/s (292.6 s) | **935.8 tok/s** (275.3 s) | 1,436.3 tok/s (176.3 s) | 1,420.5 tok/s (179.6 s) | **3,044.4 tok/s (84.8 s, 2.12×)** |
| **Cold prefill 1.04M** | 832.6 tok/s (20.9 min) | — | 1,264.5 tok/s (13.8 min) | — | *~3,000 tok/s (~5.8 min est.)* |
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

### F. DeepInfer-SM80 Architectural & Kernel Adaptations (`patches/row_table.py`, `patches/model_ls.py`, `patches/generator.py`)
Adapted core Ampere breakthroughs from [`HalfVulpes/Deepinfer-SM80`](https://github.com/HalfVulpes/Deepinfer-SM80):
1. **Early Async Engram Prefetch & Dual-Table Staging (`patches/row_table.py`, `patches/ngram_embedding.py`)**:
   - In baseline serving, `self.scale` lookup was never prefetched alongside `self.table`, causing synchronous CPU stalls and 100% prefetch misses at Layers 1 and 14.
   - Tied `companion_scale` into `RowTable`: both the 128-byte packed FP4 payload and the 8-byte E8M0 scale rows are gathered simultaneously on the background worker thread during prefetch and uploaded to the GPU in a single non-blocking DMA pair.
2. **IEEE-754 Bitshift Scale Decoding (`patches/ngram_embedding.py`)**:
   - Replaced transcendental `.exp2()` calls with single-instruction IEEE-754 integer bitshifts: `(scale_u8 << 23).view(torch.float32)` produces the exact power-of-two $2^{s - 127}$ on the integer pipe with zero SFU contention.
3. **DSpark MTP Dynamic Confidence-Head Draft Pruning (`EXL3_DSPARK_CONF=0.4`, `patches/generator.py`)**:
   - Evaluates `mtp.2.confidence_head.proj.weight` (`[1, 5376]`) during speculative drafting to dynamically prune low-confidence draft tokens on divergent prose, preventing target verification overhead and boosting math decode to **80.7 tok/s (+53% vs baseline)** while maintaining 100% byte-identical greedy output.
4. **YOCO Prefill Compaction at Layer 20 (`EXL3_YOCO_COMPACT=1`, `patches/model_ls.py`)**:
   - DeepSeek-V4.1's 40-layer backbone is a YOCO hybrid: Layer 20 is the final global KV producer (`kv_source_layer_ids = [2, 8, 14, 20]`). Layers 21–39 never write global KV.
   - For all intermediate prefill chunks prior to the prompt tail, intermediate tokens in Layers 21–39 are skipped, reducing pipeline compute by 40% and pushing 258K cold prefill to **2,940 tok/s**.

---

## 4. Alternative Serving Engines & Architecture Exploration

In addition to the TabbyAPI PP5 and TP5 baselines, we explored alternative serving architectures using the **GNmac-lab vLLM** stack (`localhost/ds41-cmp170hx:latest`), an Ampere-optimized fork of vLLM V1 with standalone C++/CUDA MoE kernels (`DeepSeekv41-Flash-EXL3-TC-MoE`), FlashInfer 0.7.0.post1 PCIe IPC all-reduce, and SSD-streamed Engram tables.

### Architecture Comparison (All Tested Configurations on 5× CMP 170HX)

| Architecture / Engine | GPUs Used | VRAM / GPU | Engram Residency | Single-Stream Decode (C Code) | Single-Stream Decode (Prose) | 8-User Aggregate Decode (Code / Structured) | Cold Prefill (24K–38K Tokens) | Notes / Best Use Case |
|---|---|---|---|---:|---:|---:|---:|---|
| **TabbyAPI PP5** | 5 GPUs (`0..4`) | 30.8–51.8 GiB | 97.3 GiB Host RAM (Anon) | 65.4 tok/s | 55.1 tok/s | 54.8 tok/s | **1,529–3,044 tok/s** | Best for ultra-long context prefill (>100K) |
| **TabbyAPI TP5** | 5 GPUs (`0..4`) | 44.6–47.3 GiB | 97.3 GiB Host RAM (`mmap`) | 79.9 tok/s | 76.9 tok/s | 91.5 tok/s | 871–935 tok/s | High single-stream decode across all 5 cards |
| **GNmac-lab vLLM TP2×PP2** | 4 GPUs (`0..3`) | ~58.5 GiB | NVMe SSD Streamed | 79.4 tok/s | — | 88.2 tok/s | 972.4 tok/s | Balances prefill and decode; GPU 4 free |
| **GNmac-lab vLLM TP4** | 4 GPUs (`0..3`) | ~59.2 GiB | NVMe SSD Streamed | **183.1 tok/s** | 12.3 tok/s | **177.5 / 320.6 tok/s** | 653–695 tok/s | **Fastest single-stream code & structured decode; GPU 4 100% free** |
| **GNmac-lab vLLM PP5** | **5 GPUs (`0..4`)** | **34.8–56.6 GiB** | **NVMe SSD Streamed** | **86.0 tok/s** | **50.5 tok/s** | **177.5 / 288.1 tok/s** | **2,268 tok/s (2,336 peak)** | **Best all-rounder: 3.5× faster prefill, 4× faster prose, identical 8-user aggregate** |

### Why vLLM PP5 Delivers 2,268 tok/s Prefill on PCIe Gen2 x4
- **Eliminating the 80 All-Reduces**: Under TP4, every forward step requires 80 layer-by-layer all-reduces that saturate the PCIe Gen2 x4 bus at 1.6 GB/s. Under PP5 (`--pipeline-parallel-size 5 --tensor-parallel-size 1`), all-reduce is **completely eliminated**. Tokens flow forward through pipeline stages, crossing the PCIe link only at the 4 stage boundaries per chunk.
- **Balanced Layer Partition (`VLLM_PP_LAYER_PARTITION="8,6,10,8,8"`)**:
  - GPU 0: Layers `0..7` (43.7 GiB used)
  - GPU 1: Layers `8..13` (32.3 GiB used)
  - GPU 2: Layers `14..23` (44.9 GiB used)
  - GPU 3: Layers `24..31` (42.4 GiB used)
  - GPU 4: Layers `32..39` + DSpark drafter (53.0 GiB used)
- **Concurrency Amortization**: Under 8-user concurrency, pipeline bubble overhead is fully hidden. PP5 matches TP4 on multi-user code generation (**177.5 tok/s aggregate**) and structured throughput (**288.1 tok/s aggregate**).
- **Prose Decode Superiority (50.5 vs 12.3 tok/s)**: On creative prose where speculative acceptance drops to ~2 tokens/step, TP4 suffers from latency penalties, whereas PP5 sustains a steady **50.5 tok/s** single-stream decode.

### PP5 Concurrency Scaling (measured, greedy, 400 completion tokens)

| Users | Structured JSON | Code | Prose |
|---:|---:|---:|---:|
| 1 | 97.9 tok/s | 86.0 tok/s | 50.5 tok/s |
| 2 (aggregate) | 99.9 | 70.6 | 53.7 |
| 4 (aggregate) | 174.2 | 119.2 | 71.7 |
| 8 (aggregate) | **288.1** | **177.5** | **113.2** |

### Interactive Decode Benchmark: Why PP5 Collapses on Real Prompts (1–10 tok/s vs 52–165 tok/s on TP4)

While PP5 achieved high synthetic decode numbers on repetitive counter prompts with 99% speculative acceptance (`bench/mia.py`), real-world conversational and reasoning prompts revealed an acute bottleneck under PP5. On normal chat and coding requests, DSpark MTP speculative acceptance drops to 20%–40%, exposing the raw pipeline round-trip latency across the PCIe Gen2 x4 bus:

| Test Prompt | Target Tokens | PP5 Decode (5 GPUs) | TP4 Decode (4 GPUs) | Speedup on TP4 |
|---|---:|---:|---:|---:|
| **Python Fibonacci Function** | 100 | **1.0 tok/s** (101.5 s) | **52.8 tok/s** (1.89 s) | **53× faster** |
| **Count 1 to 100 with Commas** | 100 | **2.4 tok/s** (42.5 s) | **164.5 tok/s** (0.61 s) | **68× faster** |
| **Explain TCP vs UDP** | 100 | **10.7 tok/s** (9.3 s) | **107.5 tok/s** (0.93 s) | **10× faster** |
| **Roman Empire Essay** | 300 | **7.3 tok/s** (41.0 s) | **70.3 tok/s** (4.27 s) | **10× faster** |

### Standardized Nine-Request C-Coding Benchmark (Recipe 08/09 Protocol)

Using the standardized 9-request systems C-coding benchmark protocol from [`anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m`](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m) (`bench/bench_nine_requests.py`), measured greedy against the live server on GNmac-lab vLLM TP4 **v1.1.0**:

| Request | Prompt Tokens | Time to First Token (TTFT, cold) | Prefill (cold) | Completion Tokens | Decode (1.0.0) | **Decode (1.1.0)** | Speedup vs 1.0.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`review`** | 4,464 | 5.46 s | 817 tok/s | 1,500 | 115.1 tok/s | **136.5 tok/s** | +18.6% |
| **`bug-find`** | 3,150 | 3.75 s | 840 tok/s | 600 | 122.8 tok/s | **143.0 tok/s** | +16.4% |
| **`fix-diff`** | 3,142 | 3.73 s | 841 tok/s | 350 | 139.1 tok/s | **128.4 tok/s** | −7.7% |
| **`fix-func`** | 3,148 | 3.75 s | 839 tok/s | 1,200 | 138.7 tok/s | **156.2 tok/s** | +12.6% |
| **`whole-file`** | 1,626 | 2.06 s | 790 tok/s | 2,600 | 147.0 tok/s | **176.9 tok/s** | +20.3% |
| **`agent-edit` (tools)** | 3,492 | 4.29 s | 814 tok/s | 154 | 112.2 tok/s | **134.5 tok/s** | +19.9% |
| **`agent-multi` (tools)** | 8,252 | 10.41 s | 793 tok/s | 212 | 124.5 tok/s | **138.4 tok/s** | +11.2% |
| **`new-code`** | 73 | 0.21 s | 352 tok/s | 1,500 | 133.7 tok/s | **141.9 tok/s** | +6.1% |
| **`explain`** | 4,463 | 5.74 s | 778 tok/s | 1,000 | 100.3 tok/s | **117.6 tok/s** | +17.2% |

- **Mean per-request decode**: **141.5 tok/s** cold / **145.0 tok/s** warm cache (vs 1.0.0's 125.9 tok/s — **+12.4% to +15.2% faster**; vs reference repo's 99.9 tok/s — **+41.6% faster**).
- **Total output**: **9,116 completion tokens** in 101.4 s cold (**89.9 tok/s wall throughput**) and 63.8 s warm prefix (**142.9 tok/s wall throughput**).
- **GNmac-lab 1.1.0 Engine Upgrades**:
  - MXFP8 linears on one fragment-native weight copy (`dsv41_fp8w_ext` GEMV for $M \le 8$, smem-staged kernel for $M \le 64$).
  - Decode $w_{o\_a}$ CUDA kernel (`dsv41_woa_ext`) with fused inverse GPT-J RoPE.
  - Cooperative MoE decode loop instruction reduction (7.33 $\to$ 6.39 instructions per weight).
  - Draft-only FP8 drafter head shard (`DSV41_DRAFT_HEAD_FP8=1`).
  - DSpark Markov sampling loop as a single CUDA graph with fused bias GEMV and max kernels.
  - Greedy verification using vocab-parallel argmax IDs directly (skips 2 full-vocab passes).
  - Runtime variants: `dsv41_topk_meta` (1 launch vs 5), `dsv41_h2d_lean`, `dsv41_ragged_lean`.
  - **PCIe Gen2 x4 Finding on Confidence Gating (`DSV41_GATE_TAU=0`)**: The upstream 1.1.0 default `DSV41_GATE_TAU=0.10` gates out drafts in reasoning prose when drafter confidence drops. On a machine with PCIe Gen2 x16 links, 1-token decode steps take ~17 ms. However, on physical PCIe Gen2 x4 links, a single un-speculated token step requires ~60 ms ($1 / 0.060 \approx 16.6\text{ tok/s}$). Amortizing PCIe all-reduce latency requires keeping speculative decoding active. Setting `DSV41_GATE_TAU=0` keeps all 5 drafts verified, sustaining >115 tok/s on reasoning prose and 140–177 tok/s on code.
- **INT8 Lightning Indexer**: Prefill scoring in INT8 (`VLLM_DSV4_INDEXER_DTYPE=int8`, `patches/mqa_logits_triton.py`) reduces operand transfer across the memory bus by 50% and runs on native Ampere INT8 tensor cores.
- **GPU Persistence Mode**: Initialised at boot via `systemd.services.nvidia-persistence` (`nvidia-smi -pm 1`), eliminating power-state clock latency.

#### Root Cause Analysis:
1. **Pipeline Bubble Serialization**: In PP5, during single-stream token generation, only 1 of the 5 GPUs is active at any microsecond (an 80% bubble). The activation vector must serialize sequentially across all 5 cards (`GPU 0 → 1 → 2 → 3 → 4`) and then back to GPU 0 for each speculative verification cycle.
2. **PCIe Gen2 x4 Transfer Overhead**: Over downgraded Gen2 x4 risers, inter-stage P2P handshakes add 10–20 ms per hop. A full 5-stage round trip takes ~250 ms. When speculative acceptance is low (1–2 tokens accepted per step), the effective generation speed is bounded by $1\text{ token} / 0.25\text{ s} \approx 4\text{ tok/s}$.
3. **TP4 Parallelism with FlashInfer PCIe IPC**: In TP4, all 4 GPUs compute every layer concurrently (0% pipeline bubble). FlashInfer's PCIe IPC All-Reduce writes directly into peer BAR1 allocations in microseconds. Even when 0 draft tokens are accepted, each step takes only ~15 ms, establishing a guaranteed decode floor of **>50 tok/s**.

### Deployment Decision

**TP4 is the production default.** For interactive agent workflows, single-stream turn latency dominates: 53 s vs 1 s for a 100-token response makes PP5 unusable for interactive chat, whereas TP4 responds almost instantaneously while keeping GPU 4 (64 GB VRAM) free. PP5 is preserved in `serving/ds41-flash-pp5.nix` for offline batch prefill / document ingestion.

---

## 5. Sampling Parameter Recommendations: Temperature & Top-P

For `DeepSeek-V4.1-Flash-1017` running in reasoning / thinking mode:

| Parameter | Recommended Value | Rationale |
|---|---|---|
| **`temperature`** | **`0.6`** | Recommended by DeepSeek for reasoning models (DeepSeek-R1 / V4 reasoning mode). Setting `temperature = 0.0` (greedy) causes reasoning chains to fall into repetition loops or fail to backtrack on complex multi-step reasoning. Setting `temperature > 0.8` degrades mathematical syntax and formatting stability. `0.6` provides optimal balance. |
| **`top_p`** | **`0.95`** | Nucleus sampling cutoff that prevents low-probability hallucinations in tail distributions while preserving exploratory thinking paths. |
| **`max_tokens`** | **`65536`** | Provides headroom for deep reasoning traces prior to final answers. |

*Note: For strict, non-reasoning deterministic tasks (e.g. unit test assertions, fixed schema extractions), `temperature: 0.0` (greedy) can be used.*

---

## 6. Production Deployment & Serving Stack

The production setup runs as a managed systemd service on NixOS 26.11:

### A. Systemd Service (`services.ds41-flash` in `serving/ds41-flash.nix`)
- **Engine**: GNmac-lab vLLM **TP4** container (`localhost/ds41-cmp170hx:latest`)
- **GPUs**: GPUs `0, 1, 2, 3` (`CUDA_VISIBLE_DEVICES=0,1,2,3`); GPU 4 is 100% free.
- **Flags**:
  - `--tensor-parallel-size 4 --pipeline-parallel-size 1`
  - `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`
  - `--max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096`
  - `--gpu-memory-utilization 0.95`
  - `--kv-cache-dtype fp8_ds_mla`
  - `--enable-prefix-caching --async-scheduling`
  - `--speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'`
  - `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1,2,3,4,5,6,8,12,18,24,30,36,42,48]}'`
  - `DSV41_DSPARK_V1=1 VLLM_EXL3_TC_MOE_V2=1 VLLM_EXL3_TC_MOE_PREFILL=1 VLLM_EXL3_TC_PREFILL_ROWS=4096`
- **Boot Pre-warming (`ExecStartPost = "${pkgs.python3}/bin/python3 /etc/nixos/prewarm-agent.py"`)**：
  - Automatically runs after uvicorn health check passes.
  - Warms CUDA graphs, JIT-compiled kernels, and pre-populates the paged KV prefix cache with the standard `pi` system prompt (`serving/pi-system-prompt.txt`).
  - Cuts subsequent agent turn latency from **11.4 s → 1.14 s**.
- **Measured Peak Throughput (TP4)**:
  - Structured JSON decode: **216.7 tok/s** (TTFT 0.13 s)
  - Python / C Code decode: **183.1 tok/s** (TTFT 0.13 s)
  - General chat / reasoning decode: **52.8–107.5 tok/s**
  - Concurrency: **190.0 tok/s** aggregate decode (8 general users), **320.6 tok/s** (8 structured users).
- **Network**: Binds to `127.0.0.1:8000`

> The PP5 service definition is preserved as `serving/ds41-flash-pp5.nix` for 5-GPU offline batch document ingestion, and `serving/ds41-flash-pp4.nix` provides the 4-GPU pipeline-parallel configuration (`--tensor-parallel-size 1 --pipeline-parallel-size 4`, `VLLM_PP_LAYER_PARTITION=10,11,11,8`, PIECEWISE CUDA graphs).

### B. PP4 Serving Layout (`serving/ds41-flash-pp4.nix`)
- **Engine**: GNmac-lab vLLM **PP4** container (`localhost/ds41-cmp170hx:1.1.0`)
- **Partition**: `10, 11, 11, 8` layers across GPUs `0..3`.
- **Characteristics**:
  - Eliminates all-reduce across GPUs; only inter-stage activations are transferred over PCIe links.
  - Cold prefill reaches **1,080–1,628 tok/s** (+30% to +105% faster prefill than TP4 over PCIe Gen2 x4).
  - Single-user decode throughput: **53.0–75.0 tok/s** (mean: **64.4 tok/s** on C workloads).
  - Preserves GPU 4 (64 GB) 100% free.

### C. Live Real-Time Dashboard (`vllm-tps`)
- Dashboard listening on `http://127.0.0.1:8099`.
- Polls Prometheus metrics from `http://127.0.0.1:8000/metrics` every second.
- Tracks generation tok/s, prompt compute tok/s, KV cache utilization, and active request concurrency for `deepseek-v4.1-flash`.

### C. Agent CLI Integration (`pi`)
Configured for both `root` and standard users (`/root/.pi/agent/` and `/home/dkp/.pi/agent/`):
```json
"providers": {
  "ds41-flash": {
    "name": "DeepSeek-V4.1-Flash-1017 vLLM TP4",
    "baseUrl": "http://127.0.0.1:8000/v1",
    "api": "openai-completions",
    "apiKey": "vllm",
    "compat": {
      "supportsDeveloperRole": false,
      "thinkingFormat": "chat-template"
    },
    "models": [
      {
        "id": "deepseek-v4.1-flash",
        "reasoning": true,
        "contextWindow": 262144,
        "maxTokens": 65536,
        "samplingParams": { "temperature": 0.6, "top_p": 0.95 }
      }
    ]
  }
}
```

---

## Repository Layout

- `build/` — `Containerfile`, `headers.py`, `check_engine.py`, `get_engine.sh`, `make_view.py`, `chat_template.jinja`, `tokenizer_config.json`
- `patches/` — `convert_engram_fp4.py`, `all_optimizations.patch`, `moe_coop/` (the 4 C++/CUDA MoE kernel patches + unified diffs), and the 9 patched engine modules (`row_table.py`, `generator.py`, `ngram_embedding.py`, `deepseek_v41_mod.py`, `dsv4.py`, `dspark.py`, `dsa.py`, `model_ls.py`, `device_copy.py`)
- `serving/` — `ds41-flash.nix` (NixOS vLLM **TP4** production service), `ds41-flash-pp5.nix` (NixOS vLLM PP5 service for bulk prefill), `prewarm-agent.py` (systemd boot prewarmer), `pi-system-prompt.txt` (reference agent prompt), `ds41-mia-tp5.yml`, `ds41-mia-pp5.yml`, `ds41-mia.nix` (NixOS TabbyAPI PP5 service), `run-mia.sh`, `run-fg.sh`, `prewarm-engram.sh`
- `bench/` — `bench_ab.py`, `bench_conc.py`, `compare.py`, `capture_ref.py`
- `results/` — `results-vllm-tp4.json`, `results-vllm-pp2tp2.json`, `results-vllm-pp5.json`, `results-tp5*.json`, `results-pp5*.json`, `results-conc-*.json`, `ref-*.json`, `convert.log`
