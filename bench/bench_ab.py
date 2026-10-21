#!/usr/bin/env python3
"""A/B bench for the Mia 2.9 EXL3 DS41-Flash-1017 stack on :1920 (TabbyAPI).

Usage: bench_ab.py <tp5|pp5> [--sizes 16k,64k,256k,512k[,1m]] [--quick]
Writes results-<layout>.json next to this script.

Measures, one request at a time, greedy, reasoning on (the served default):
  - smoke: one-word answer
  - tool call: DSML tool_calls emission
  - decode: two C-coding requests, 600 tokens -> TTFT + decode tok/s
  - prefill: prompts of N tokens (unique first line => cold prefix cache) -> TTFT + prefill tok/s
  - agreement: fixed prompt, 64 tokens, saved for the cross-layout comparison
"""
import json, os, re, subprocess, sys, time
import urllib.request

URL = "http://127.0.0.1:1920/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.c")
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
CHARS_PER_TOKEN = 3.15  # the recipe's ratio for C source with this tokenizer


def http(body, timeout=7200):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:400]}") from None
    with r as resp:
        first = None
        text = ""
        usage = {}
        tool_calls = []
        for raw in r:
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
                for tc in delta.get("tool_calls") or []:
                    tool_calls.append(tc)
    t1 = time.time()
    return t0, first, t1, text, usage, tool_calls


def build_corpus(target_mb=10):
    if os.path.exists(CORPUS) and os.path.getsize(CORPUS) > target_mb * 1024 * 1024 // 2:
        return
    print("building C corpus from /nix/store ...")
    out = []
    total = 0
    cap = 200 * 1024
    n = 0
    find = subprocess.run(
        ["find", "/nix/store", "-maxdepth", "5", "-type", "f",
         "(", "-name", "*.c", "-o", "-name", "*.h", ")", "-size", "+2k"],
        capture_output=True, text=True, timeout=600)
    paths = [p for p in find.stdout.splitlines() if "/patch" not in p and "iana-etc" not in p]
    for p in paths:
        if total >= target_mb * 1024 * 1024:
            break
        try:
            with open(p, "r", errors="ignore") as f:
                chunk = f.read(cap)
        except OSError:
            continue
        if len(chunk) < 2048:
            continue
        out.append(f"/* ==== {p} ==== */\n" + chunk)
        total += len(chunk)
        n += 1
    with open(CORPUS, "w") as f:
        f.write("\n".join(out))
    print(f"corpus: {n} files, {total/1e6:.1f} MB -> {CORPUS}")


def prompt_of_tokens(ntok, tag):
    """Cold-prefill prompt: unique markers every ~1200 chars (the prefix cache matches
    token blocks, so a unique first line alone is not enough) + C corpus."""
    with open(CORPUS) as f:
        body = f.read(int(ntok * CHARS_PER_TOKEN * 0.925) + 4096)
    parts = []
    step = 1200
    for i in range(0, len(body), step):
        parts.append(f"/* cold-{tag}-{i} */\n")
        parts.append(body[i:i + step])
    return ("Review the following C source. Find bugs.\n\n" + "".join(parts))


def main():
    layout = sys.argv[1]
    sizes = ["16k", "64k", "256k", "512k"]
    if "--sizes" in sys.argv:
        sizes = sys.argv[sys.argv.index("--sizes") + 1].split(",")
    def parse_size(s):
        s = s.strip().lower()
        m = re.fullmatch(r"(\d+)(k|m)?", s)
        if not m:
            raise SystemExit(f"bad size {s!r}")
        n = int(m.group(1))
        return n * 1024 if m.group(2) == "k" else n * 1024 * 1024 if m.group(2) == "m" else n
    sizes = [parse_size(s) for s in sizes]
    quick = "--quick" in sys.argv
    build_corpus()
    res = {"layout": layout, "model": MODEL, "sizes": sizes, "tests": {}}

    # 1. smoke
    t0, first, t1, text, usage, _ = http(
        {"model": MODEL, "messages": [{"role": "user", "content": "What is the capital of France? Answer with one word."}],
         "max_tokens": 64, "temperature": 0.0, "stream": True,
         "stream_options": {"include_usage": True}})
    res["tests"]["smoke"] = {"answer": text.strip()[:120], "ok": "paris" in text.lower()}
    print(f"smoke: {text.strip()[:60]!r} ok={res['tests']['smoke']['ok']}")

    # 2. tool call (DSML)
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    t0, first, t1, text, usage, tcs = http(
        {"model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}],
         "tools": tools, "max_tokens": 256, "temperature": 0.0, "stream": True,
         "stream_options": {"include_usage": True}})
    ok = bool(tcs) and any("get_weather" in json.dumps(tc) for tc in tcs)
    res["tests"]["tool_call"] = {"ok": ok, "n_tool_calls": len(tcs),
                                 "first": json.dumps(tcs[0])[:200] if tcs else text[:200]}
    print(f"tool_call: ok={ok} n={len(tcs)} {res['tests']['tool_call']['first'][:120]}")

    # 3. decode: C-coding requests
    decode = []
    reqs = [
        ("Write a C function `int utf8_len(const char *s)` that returns the number of "
         "Unicode code points in a NUL-terminated UTF-8 string. Only the function, no main.", "utf8_len"),
        ("Review this C code and list its bugs as a numbered list:\n"
         "```c\nint sum(int *a, int n) {\n  int s = 0;\n  for (int i = 0; i <= n; i++) s += a[i];\n"
         "  return s;\n}\n```", "review"),
    ]
    if quick:
        reqs = reqs[:1]
    for prompt, tag in reqs:
        t0, first, t1, text, usage, _ = http(
            {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 600, "temperature": 0.0, "stream": True,
             "stream_options": {"include_usage": True}})
        ttft = first - t0
        dec = (usage.get("completion_tokens", 0) - 1) / max(t1 - first, 1e-9)
        entry = {"tag": tag, "ttft": round(ttft, 2), "decode_tok_s": round(dec, 1),
                 "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                 "chars": len(text)}
        decode.append(entry)
        print(f"decode[{tag}]: TTFT {ttft:.2f}s  decode {dec:.1f} tok/s  "
              f"({usage.get('completion_tokens')} tokens, {len(text)} chars)")
    res["tests"]["decode"] = decode

    # 4. prefill at several sizes
    pre = []
    for ntok in sizes:
        prompt = prompt_of_tokens(ntok, f"{layout}-{ntok}-{int(time.time())}")
        try:
            t0, first, t1, text, usage, _ = http(
                {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                 "max_tokens": 8, "temperature": 0.0, "stream": True,
                 "stream_options": {"include_usage": True}})
        except Exception as e:
            print(f"prefill {ntok}: FAILED {e}")
            pre.append({"target_tokens": ntok, "error": str(e)[:300]})
            continue
        ttft = first - t0
        pt = usage.get("prompt_tokens", 0)
        entry = {"target_tokens": ntok, "prompt_tokens": pt, "ttft": round(ttft, 1),
                 "prefill_tok_s": round(pt / max(ttft, 1e-9), 1)}
        pre.append(entry)
        print(f"prefill {pt:>8} tok: TTFT {ttft:7.1f}s  prefill {entry['prefill_tok_s']:7.1f} tok/s")
    res["tests"]["prefill"] = pre

    # 5. agreement prompt (greedy, fixed) for the cross-layout comparison
    try:
        t0, first, t1, text, usage, _ = http(
            {"model": MODEL, "messages": [{"role": "user",
              "content": "In exactly two sentences, explain what a hash table is and what its average lookup cost is."}],
             "max_tokens": 64, "temperature": 0.0, "stream": True,
             "stream_options": {"include_usage": True}})
        res["tests"]["agreement"] = {"text": text}
        print(f"agreement: {text.strip()[:150]!r}")
    except Exception as e:
        res["tests"]["agreement"] = {"error": str(e)[:300]}
        print(f"agreement FAILED: {e}")

    out = os.path.join(HERE, f"results-{layout}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
