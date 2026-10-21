# DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw + MXFP4 Engram on 5× CMP 170HX (TP5 vs PP5 at 1M context)

Adapting [`anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m`](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m) to an all-Ampere **5× NVIDIA CMP 170HX (64 GB each, sm_80, PCIe Gen2 x4)** machine with **251 GiB host RAM** — no 96 GB RTX PRO 6000 Blackwell card, and half the host RAM of the reference box.

Two changes make the full **1,048,576-token FP16 context + DSpark MTP speculative decoding + RAM-resident Engram** recipe fit comfortably on this hardware:

1. **Pack selection (`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`)**: LibertAI's 3.5 bpw pack quantizes the 3-layer DSpark MTP drafter to 6 bpw EXL3 (**20.0 GiB**), which only fits when parked on a 96 GB card. The 2.9 bpw family (`Mia-AiLab EAQ-v1 2.9bpw` and `LibertAIDAI 2.9bpw`) quantizes the MTP drafter to 4 bpw EXL3 (**6.75–6.88 GiB**) and the main trunk to **189.5 GiB**, fitting all 384 routed experts + 1M FP16 KV cache + MTP drafter across five 64 GB cards with 10–17 GiB/card to spare.
2. **Offline MXFP4 (`e2m1`) Engram conversion + RAM residency (`patches/convert_engram_fp4.py`, `patches/ngram_embedding.py`)**: The base checkpoint's two Engram n-gram lookup tables (`layers.{1,14}.engram.embed.{weight,scale}`) total **188.83 GiB** in FP8 (`F8_E4M3` + `F8_E8M0`), leaving almost no breathing room on a 251 GiB host. Converting them to packed `e2m1` nibbles (`U8 [rows, 128]`) with re-computed per-32-element `F8_E8M0` power-of-two budgets (`e + 127`) halves the tables to **97.28 GiB** (`rel-L2 = 0.1205 / 0.1209` vs FP8) so the entire Engram table stays permanently in host RAM (`EXL3_NGRAM_STREAM=0`) in both **PP5** (anonymous RAM) and **TP5** (shared `mmap` across all 5 worker processes), with **>105 GiB of free RAM** remaining.

---

## Hardware vs reference recipe

| Component | `anoane` reference machine | This machine (`RAM-Monster`) |
|---|---|---|
| GPUs | 4× CMP 170HX (64 GB, sm_80) + 1× RTX PRO 6000 (96 GB, sm_120) | **5× CMP 170HX (64 GB, sm_80)** |
| PCIe links | PCIe Gen2 **x16** (~6.4 GB/s per direction) | PCIe Gen2 **x4** (~1.6 GB/s per direction, risers) |
| CPU | AMD EPYC 9J14 (96C) | AMD Ryzen Threadripper PRO 3975WX (32C/64T) |
| Host RAM | 499 GiB (189 GiB FP8 Engram in page cache) | **251 GiB** (**97.3 GiB FP4 Engram in RAM**, Swap = 0) |
| Driver | NVIDIA open 610.57.04 + `cmpunlocker` (Gen2 + P2P) | NVIDIA open 615.71.09 + `cmpunlocker` (Gen2 + P2P) on NixOS 26.11 |
| Engine | `anoane/exllamav3-anemone` @ `309ef6d1` (`dsv41f-1017`, INT8 DSA indexer) | Same commit `309ef6d1` (`sm_80`) + FP4 Engram patch (`patches/ngram_embedding.py`) |
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

## 2. TP5 vs PP5 A/B Benchmark (1M FP16 context, DSpark MTP `dynamic_draft: true`)

Measured on this machine (`bench/bench_ab.py`, greedy `temperature: 0`, `reasoning_effort: high`, `EXL3_DSA_INDEXER_DTYPE=sm_80:int8`, `EXL3_INT8_GEMV=0`, `TABBY_EXL3_GENERATOR_CHUNK=16384`, 1 stream). Cold prefill prompts inject unique block markers every 1,200 chars so TabbyAPI's block prefix cache cannot short-circuit earlier prefixes.

### Cold prefill throughput (up to 1.04M tokens)

| Prompt tokens | **TP5** TTFT | **TP5** prefill | **PP5** TTFT | **PP5** prefill | PP5 / TP5 |
|---:|---:|---:|---:|---:|---:|
| ~55–57K | 64.9 s | 883.8 tok/s | **44.3 s** | **1,235.0 tok/s** (1,244.5 FP4) | **1.40×** |
| ~253–266K | 292.6 s (272.9 s FP4) | 908.2 tok/s (**934.7** FP4) | **176.3 s** | **1,436.3 tok/s** (1,420.5 FP4) | **1.54×** |
| 537,140 | 612.6 s | 876.9 tok/s | — | — | — |
| **1,043,858** | 1,253.8 s (20.9 min) | 832.6 tok/s | **825.5 s (13.8 min)** | **1,264.5 tok/s** | **1.52×** |

### Single-stream drafted decode throughput

| Workload | **TP5** (FP8 Engram) | **TP5** (FP4 Engram in RAM) | **PP5** (FP8 Engram) | **PP5** (FP4 Engram in RAM) |
|---|---:|---:|---:|---:|
| C coding: `utf8_len` | 71.8 tok/s (TTFT 0.39 s) | **81.7 tok/s** (TTFT 0.41 s) | 48.4 tok/s (TTFT 0.79 s) | 46.4 tok/s (TTFT 0.57 s) |
| C coding: `review` (600 tok) | 55.5 tok/s (TTFT 0.41 s) | 46.7 tok/s (TTFT 0.45 s) | 32.7 tok/s (TTFT 0.69 s) | **33.8 tok/s** (TTFT 0.69 s) |
| **Mean C-coding decode** | **63.6 tok/s** | **64.2 tok/s** | **40.5 tok/s** | **40.1 tok/s** |
| 4-prompt ref suite (`agreement` / `c_code` / `review` / `reason`) | — | — | 44.1 / 44.2 / 33.6 / 52.6 | **45.6 / 46.7 / 36.2 / 58.3 tok/s** |
| DSML tool calling (`get_weather`) | Pass (`{"city": "Paris"}`) | Pass (`{"city": "Paris"}`) | Pass (`{"city": "Paris"}`) | Pass (`{"city": "Paris"}`) |
| Greedy output (`agreement` & `reason`) | Reference | **Byte-identical** | **Byte-identical** | **Byte-identical** |
| VRAM allocated per GPU (GiB) | 47.3, 44.6, 44.6, 44.6, 44.6 | 47.3, 44.6, 44.6, 44.6, 44.6 | 48.4, 50.6, 51.4, 48.4, 17.7 | 48.4, 50.6, 51.4, 48.4, 17.7 |

### Why TP5 and PP5 trade places on PCIe Gen2 x4

- **Decode (`TP5` wins by 1.57–1.60×)**: Under tensor parallel, all 5 GPUs read their 1/5th slice of the active experts simultaneously (5× aggregate HBM bandwidth) and synchronize small activations through host memory. Even on Gen2 x4 links, parallel HBM reads outweigh the all-reduce latency. Under layer split (`PP5`), only one GPU computes at a time while the other four sit idle.
- **Long prefill (`PP5` wins by 1.40–1.58×)**: During a 16K-token prefill chunk, activations are large and compute-bound; `TP5`'s host-memory all-reduce on every layer bottlenecks on the Gen2 x4 (~1.6 GB/s) links, whereas `PP5` transfers activations only 4 times across the entire 40-layer pass.

---

## 3. MXFP4 (`e2m1`) Engram Conversion & In-RAM Serving

### Format (`patches/convert_engram_fp4.py`)

Each Engram table in `deepseek-ai/DeepSeek-V4.1-Flash-1017` is a `[rows, 256]` `F8_E4M3` matrix with a `[rows, 8]` `F8_E8M0` block scale (one power-of-two exponent byte per 32 values; `rows = 384,006,168` on layer 1 and `384,016,682` on layer 14).

Because the stored `e4m3` values use the full FP8 dynamic range (measured `|v|_max = 7.5`, scale bytes `117..122`), the raw `e8m0` exponent byte cannot be reused directly for `e2m1` (max magnitude `6.0`). For each 32-value block we reconstruct the true FP32 value `v = e4m3 * 2^(s - 127)` on GPU and compute a new power-of-two budget:

```
e        = ceil(log2(max(|v|) / 6.0))           # so |v| / 2^e <= 6.0
code     = sign(q) << 3 | bucketize(|q|)        # into {0, 0.5, 1, 1.5, 2, 3, 4, 6}
packed   = code[0::2] | (code[1::2] << 4)       # uint8 [rows, 128]
newscale = uint8(e + 127)                       # F8_E8M0 [rows, 8]
```

Because `exllamav3`'s existing scale path already computes `(r.float() - 127.0).exp2()`, storing `e + 127` in `embed.scale` requires **zero changes** to the scale loader or scale math.

- **Conversion speed**: ~8.5 minutes total for both 384M-row tables on a single CMP 170HX (`cuda:0`, 4M rows/chunk).
- **Accuracy (`results/convert.log`)**:
  - Layer 1 (`384,006,168 × 256`): **`rel-L2 = 0.1205`**
  - Layer 14 (`384,016,682 × 256`): **`rel-L2 = 0.1209`**
- **Size**: `188.83 GiB → 97.28 GiB` (`104,451,108,088` bytes in a single `engram-fp4.safetensors`).

### Engine patch (`patches/ngram_embedding.py`)

1. **Auto-detected FP4 decode**: `NGramEmbedding._set_table` checks `table.row_dtype == torch.uint8` (`self.fp4 = True`) with `row_words == head_dim // 2 == 128`, and `_decode_fp4` unpacks the low/high nibbles on the target GPU before multiplying by the `F8_E8M0` block scales.
2. **True RAM residency in both PP5 and TP5 (`EXL3_NGRAM_STREAM=0`)**:
   - In **PP5**, `RowTable.open(..., stream_from_disk=False)` loads the 97.28 GiB table into anonymous process RAM (`torch.index_select`, `on_disk=False`). Even after `echo 3 > /proc/sys/vm/drop_caches`, `buff/cache` is 3 GiB, process RSS is 102.25 GiB, and 124 GiB of host RAM remains free.
   - In **TP5**, upstream `NGramEmbedding.tp_import` always forces `stream_from_disk=True` (so 5 worker processes don't allocate `5 × table_size` in anonymous RAM). Our patch adds `_from_export_mmap`, which maps `engram-fp4.safetensors` with `np.memmap(..., mode="c")` wrapped in `torch.from_numpy`. All 5 TP worker processes run `torch.index_select` (`on_disk=False`, zero `pread` syscalls) against **one shared 97.28 GiB RAM mapping**, leaving **105 GiB of free RAM** (`206 GiB available`).

---

## 4. Lessons & Gotchas Fixed Along the Way

1. **The MTP draft head loads FIRST, and its `draft_gpu_split` must include its 1M-token KV cache.**
   TabbyAPI loads `draft_model` before the main trunk. Although the Mia 2.9 MTP weights are 6.75 GiB, its 1M-token cache adds ~1.7 GiB, so `draft_gpu_split: [6.8, ...]` fails immediately with `RuntimeError: Insufficient VRAM in split for model and cache`. Setting `draft_gpu_split: [12, 0, 0, 0, 0]` (and `gpu_split: [45, 55, 55, 55, 55]` on the main trunk) loads cleanly at 47.3 GiB on GPU 0 and 44.6 GiB on GPUs 1–4.
2. **`SafeTensorsCollection` scans `*.safetensors` alphabetically, ignoring `model.safetensors.index.json`.**
   If the old `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` symlinks remain in the view folder alongside `engram-fp4.safetensors`, `exllamav3` scans `engram-fp4.safetensors` first and then overrides its tensor keys with `model-00047/00048`! Remove the `model-00047/00048` symlinks from the view folder when switching to `engram-fp4.safetensors`.
3. **Safetensors `data_offsets` are relative to `8 + header_len`, not file offset 0.**
   When `np.memmap`ing a `.safetensors` file directly, add `8 + struct.unpack('<Q', f.read(8))[0]` to `data_offsets[0]`. Forgetting that offset reads the ASCII JSON header as row 0 (which decodes as `1e40` scale bytes and `inf` values).
4. **TabbyAPI's prefix cache matches token blocks, not whole prompts.**
   A unique first line is not enough to force a cold prefill if the rest of the corpus is block-aligned; `bench/bench_ab.py` injects a unique comment marker every 1,200 characters throughout the prompt body.

---

## Repository Layout

- `build/` — `Containerfile`, `headers.py`, `check_engine.py`, `get_engine.sh`, `make_view.py`, `chat_template.jinja`, `tokenizer_config.json`
- `patches/` — `convert_engram_fp4.py` (GPU FP8→FP4 Engram converter + verifier), `ngram_embedding.py` & `ngram_embedding_fp4.patch` (exllamav3 FP4 decode + TP shared-mmap RAM support)
- `serving/` — `ds41-mia-tp5.yml`, `ds41-mia-pp5.yml`, `run-mia.sh`, `run-fg.sh`, `prewarm-engram.sh`
- `bench/` — `bench_ab.py`, `compare.py`, `capture_ref.py`
- `results/` — `results-tp5.json`, `results-pp5.json`, `results-tp5-fp4.json`, `results-pp5-fp4.json`, `ref-fp8.json`, `ref-fp4.json`, `convert.log`
