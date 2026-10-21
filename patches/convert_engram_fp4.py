#!/usr/bin/env python3
"""Convert the two Engram embed tables from fp8 (e4m3 payload + e8m0 scales) to
packed e2m1 (uint8 nibbles) + rewritten e8m0 power-of-two budgets.

Proven math (from the dsv41 vLLM engram-fp4 work, rel-L2 ~0.12 vs fp8):
    true value v = e4m3(x) * 2^(e8m0(s) - 127)          (per 32-value block)
    e      = ceil(log2(block_max / 6))                   (so q = v / 2^e fits e2m1 range <= 6)
    code   = e2m1nearest(q): sign<<3 | index into {0,.5,1,1.5,2,3,4,6}
    new s' = e + 127   (stored as e8m0; the engine's exp2(s-127) then yields 2^e directly)
    packed byte: low nibble = even column, high nibble = odd column

Output: one safetensors, tensor order [L1.weight, L1.scale, L14.weight, L14.scale].
"""
import json, os, struct, sys, time
import numpy as np
import torch

ENGRAM_DIR = "/home/dkp/models/1017-mia-2p9/engram"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/dkp/models/1017-mia-2p9/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/engram-fp4.safetensors"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)
CHUNK = 1 << 22  # 4M rows per chunk


def header_of(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def tensors(src_file, layer):
    base, h = header_of(os.path.join(ENGRAM_DIR, src_file))
    w = h[f"layers.{layer}.engram.embed.weight"]
    s = h[f"layers.{layer}.engram.embed.scale"]
    assert w["dtype"] == "F8_E4M3" and w["shape"][1] == 256, w
    assert s["dtype"] == "F8_E8M0" and s["shape"][1] == 8, s
    return base, w, s


def convert_table(src_file, layer, out_f, stats):
    base, w, s = tensors(src_file, layer)
    rows = w["shape"][0]
    src_path = os.path.join(ENGRAM_DIR, src_file)
    mm = np.memmap(src_path, dtype=np.uint8, mode="r")
    wo = base + w["data_offsets"][0]
    so = base + s["data_offsets"][0]
    t0 = time.time()
    scale_tmp = OUT + f".scale{layer}.tmp"
    with open(scale_tmp, "wb") as sf:
        for r0 in range(0, rows, CHUNK):
            r1 = min(r0 + CHUNK, rows)
            n = r1 - r0
            payload = torch.from_numpy(mm[wo + r0 * 256: wo + r1 * 256].copy()).to(DEV, non_blocking=True) \
                .view(torch.float8_e4m3fn).float()
            scale = torch.from_numpy(mm[so + r0 * 8: so + r1 * 8].copy()).to(DEV, non_blocking=True) \
                .float().view(n, 8)
            v = payload.view(n, 8, 32) * torch.exp2(scale - 127.0).unsqueeze(-1)  # true values
            amax = v.abs().amax(-1)                                               # [n,8]
            e = torch.ceil(torch.log2(amax / 6.0))
            e = torch.where(amax > 0, e, torch.zeros_like(e)).clamp(-126, 126)
            q = v / torch.exp2(e).unsqueeze(-1)
            idx = torch.bucketize(q.abs(), BOUNDS).to(torch.uint8)                # 0..7
            sign = (q < 0).to(torch.uint8)
            code = idx | (sign << 3)                                              # [n,8,32]
            code_flat = code.view(n, 256)
            packed = code_flat[:, 0::2] | (code_flat[:, 1::2] << 4)               # [n,128] uint8
            newscale = (e + 127).to(torch.uint8)                                  # [n,8] e8m0 byte
            if r0 == 0:
                with torch.no_grad():
                    mag = LUT[(code & 7).long()]
                    sgn = torch.where((code >> 3) & 1 == 1, -1.0, 1.0)
                    back = (mag * sgn * torch.exp2(e).unsqueeze(-1)).view(n, 256)
                stats[layer] = float((back - v.view(n, 256)).norm() / v.view(n, 256).norm())
            out_f.write(packed.cpu().numpy().tobytes())
            sf.write(newscale.cpu().numpy().tobytes())
            if r0 % (CHUNK * 10) == 0:
                print(f"  layers.{layer}: {r1}/{rows} rows ({time.time()-t0:.0f}s)", flush=True)
    with open(scale_tmp, "rb") as sf:
        while True:
            b = sf.read(1 << 24)
            if not b:
                break
            out_f.write(b)
    os.unlink(scale_tmp)


def verify_output(spec):
    base4, h4 = header_of(OUT)
    mm4 = np.memmap(OUT, dtype=np.uint8, mode="r")
    lut_cpu = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for src_file, layer in spec:
        base_b, w_b, s_b = tensors(src_file, layer)
        mm_b = np.memmap(os.path.join(ENGRAM_DIR, src_file), dtype=np.uint8, mode="r")
        w4 = h4[f"layers.{layer}.engram.embed.weight"]
        s4 = h4[f"layers.{layer}.engram.embed.scale"]
        for row in [0, 12345, 5_000_000, 100_000_000, 380_000_000]:
            wo_b = base_b + w_b["data_offsets"][0] + row * 256
            so_b = base_b + s_b["data_offsets"][0] + row * 8
            vb = (torch.from_numpy(mm_b[wo_b:wo_b + 256].copy()).view(torch.float8_e4m3fn).float().view(8, 32) *
                  (torch.from_numpy(mm_b[so_b:so_b + 8].copy()).float() - 127.0).exp2().unsqueeze(-1)).view(256)
            wo_4 = base4 + w4["data_offsets"][0] + row * 128
            so_4 = base4 + s4["data_offsets"][0] + row * 8
            pk = torch.from_numpy(mm4[wo_4:wo_4 + 128].copy())
            codes = torch.empty(256, dtype=torch.uint8)
            codes[0::2] = pk & 0xF
            codes[1::2] = pk >> 4
            sc4 = (torch.from_numpy(mm4[so_4:so_4 + 8].copy()).float() - 127.0).exp2()
            v4 = (lut_cpu[(codes & 7).long()] * torch.where((codes >> 3) & 1 == 1, -1.0, 1.0)).view(8, 32) * sc4.unsqueeze(-1)
            v4 = v4.view(256)
            err = float((v4 - vb).norm() / vb.norm())
            print(f"  verify layer {layer} row {row}: |vb|={vb.norm():.3f} |v4|={v4.norm():.3f} rel_err={err:.4f}", flush=True)
            assert err < 0.20, f"bad rel_err {err} at layer {layer} row {row}"


def main():
    print(f"converter device: {DEV}", flush=True)
    spec = [("model-00047-of-00048.safetensors", 1), ("model-00048-of-00048.safetensors", 14)]
    header = {}
    off = 0
    for f, layer in spec:
        _, w, s = tensors(f, layer)
        rows = w["shape"][0]
        assert s["shape"][0] == rows
        for name, dt, dim in [(f"layers.{layer}.engram.embed.weight", "U8", 128),
                              (f"layers.{layer}.engram.embed.scale", "F8_E8M0", 8)]:
            nbytes = rows * dim
            header[name] = {"dtype": dt, "shape": [rows, dim], "data_offsets": [off, off + nbytes]}
            off += nbytes
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - (len(hb) + 8) % 8) % 8)
    print(f"output {OUT}: {(off + 8 + len(hb))/2**30:.2f} GiB", flush=True)
    stats = {}
    with open(OUT + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for src, layer in spec:
            convert_table(src, layer, f, stats)
    os.rename(OUT + ".tmp", OUT)
    print("rel-L2 vs fp8 (first chunk of each table):", stats, flush=True)
    verify_output(spec)
    print("CONVERT-DONE", flush=True)


if __name__ == "__main__":
    main()
