#!/usr/bin/env python3
"""Aggregated concurrency benchmark for the ds41-mia stack on :1920 (TabbyAPI).

Usage: bench_conc.py [--conc 1,2,4,8] [--tokens 400] [--prompt-tokens 0]

For each concurrency level N it launches N streaming chat requests at once
(greedy, reasoning on as served), waits for all to finish, and reports:
  - per-request TTFT (median / max)
  - per-request decode tok/s (median)
  - aggregate decode tok/s (sum of completion tokens / wall time of the
    generation phase, i.e. from first token of the batch to last token)
  - aggregate end-to-end tok/s (sum completion tokens / total wall time)

Prompts are distinct per request and per run (unique markers) so the block
prefix cache cannot serve them.
"""
import json, os, re, statistics, sys, threading, time
import urllib.request

URL = "http://127.0.0.1:1920/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"

PROMPTS = [
    "Write a C function `int utf8_len(const char *s)` that returns the number of Unicode "
    "code points in a NUL-terminated UTF-8 string. Only the function, no main.",
    "Review this C code and list its bugs as a numbered list:\n```c\nint sum(int *a, int n) {\n"
    "  int s = 0;\n  for (int i = 0; i <= n; i++) s += a[i];\n  return s;\n}\n```",
    "Explain how a hash table achieves average O(1) lookup. Be concise.",
    "A train leaves at 15:00 and arrives at 17:42, stopping 3 times for 4 minutes each. "
    "How many minutes was it moving? Answer with the number only.",
    "Write a Python function that merges two sorted lists into one sorted list, in place "
    "where possible. Only the function.",
    "List five differences between TCP and UDP in a compact table.",
    "Implement binary search in Rust returning Option<usize>. Only the function.",
    "Summarize what a memory barrier is and why it matters on weakly ordered CPUs.",
]


def stream(body, out, idx, timeout=7200):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text = ""
    usage = {}
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        with r as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices", []):
                    delta = ch.get("delta", {})
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        if first is None:
                            first = time.time()
                        text += piece
    except Exception as e:
        out[idx] = {"error": str(e)[:300]}
        return
    t1 = time.time()
    out[idx] = {"t0": t0, "first": first, "t1": t1, "text": text, "usage": usage}


def run_level(n, tokens, prompt_tokens, run_tag):
    out = [None] * n
    threads = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        if prompt_tokens:
            prompt = f"/* conc-{run_tag}-{i} */\n" + prompt
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True}}
        t = threading.Thread(target=stream, args=(body, out, i))
        threads.append(t)
    wall0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall1 = time.time()

    ok = [r for r in out if r and "error" not in r]
    errs = [r for r in out if r and "error" in r]
    if errs:
        print(f"  !! {len(errs)} request(s) failed: {errs[0]['error']}")
    if not ok:
        return None

    ttfts = [r["first"] - r["t0"] for r in ok if r["first"]]
    decs = []
    total_tokens = 0
    for r in ok:
        ct = r["usage"].get("completion_tokens", 0)
        total_tokens += ct
        if r["first"]:
            decs.append((ct - 1) / max(r["t1"] - r["first"], 1e-9))
    gen_start = min(r["first"] for r in ok if r["first"])
    gen_end = max(r["t1"] for r in ok)
    agg_gen = total_tokens / max(gen_end - gen_start, 1e-9)
    agg_e2e = total_tokens / max(wall1 - wall0, 1e-9)
    return {
        "concurrency": n,
        "requests_ok": len(ok),
        "completion_tokens_total": total_tokens,
        "ttft_median": round(statistics.median(ttfts), 2),
        "ttft_max": round(max(ttfts), 2),
        "decode_median_tok_s": round(statistics.median(decs), 1),
        "decode_min_tok_s": round(min(decs), 1),
        "aggregate_decode_tok_s": round(agg_gen, 1),
        "aggregate_e2e_tok_s": round(agg_e2e, 1),
        "wall_s": round(wall1 - wall0, 1),
    }


def main():
    concs = [1, 2, 4, 8]
    tokens = 400
    prompt_tokens = 0
    if "--conc" in sys.argv:
        concs = [int(x) for x in sys.argv[sys.argv.index("--conc") + 1].split(",")]
    if "--tokens" in sys.argv:
        tokens = int(sys.argv[sys.argv.index("--tokens") + 1])
    if "--prompt-tokens" in sys.argv:
        prompt_tokens = int(sys.argv[sys.argv.index("--prompt-tokens") + 1])

    run_tag = str(int(time.time()))
    results = []
    print(f"model={MODEL} max_tokens={tokens} run={run_tag}")
    print(f"{'conc':>4} {'ok':>3} {'tokens':>7} {'TTFT med':>9} {'TTFT max':>9} "
          f"{'dec med':>8} {'dec min':>8} {'agg dec':>8} {'agg e2e':>8} {'wall':>7}")
    for n in concs:
        r = run_level(n, tokens, prompt_tokens, run_tag)
        if r is None:
            print(f"{n:>4}  FAILED")
            continue
        results.append(r)
        print(f"{r['concurrency']:>4} {r['requests_ok']:>3} {r['completion_tokens_total']:>7} "
              f"{r['ttft_median']:>9.2f} {r['ttft_max']:>9.2f} "
              f"{r['decode_median_tok_s']:>8.1f} {r['decode_min_tok_s']:>8.1f} "
              f"{r['aggregate_decode_tok_s']:>8.1f} {r['aggregate_e2e_tok_s']:>8.1f} "
              f"{r['wall_s']:>7.1f}")
        time.sleep(2)

    path = os.path.join(HERE, f"results-conc-{run_tag}.json")
    with open(path, "w") as f:
        json.dump({"model": MODEL, "max_tokens": tokens, "results": results}, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
