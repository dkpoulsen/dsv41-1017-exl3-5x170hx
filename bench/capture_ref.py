#!/usr/bin/env python3
"""Capture greedy reference outputs from the running server (fp8 engram) for the fp4 A/B."""
import json, time, urllib.request

URL = "http://127.0.0.1:1920/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"

PROMPTS = [
    ("agreement", "In exactly two sentences, explain what a hash table is and what its average lookup cost is.", 64),
    ("c_code", "Write a C function `int utf8_len(const char *s)` that returns the number of "
               "Unicode code points in a NUL-terminated UTF-8 string. Only the function, no main.", 400),
    ("review", "Review this C code and list its bugs as a numbered list:\n"
               "```c\nint sum(int *a, int n) {\n  int s = 0;\n  for (int i = 0; i <= n; i++) s += a[i];\n"
               "  return s;\n}\n```", 400),
    ("reason", "A train leaves at 15:00 and arrives at 17:42. It stops 3 times for 4 minutes each. "
               "How long was it moving? Answer with the number of minutes.", 200),
]

out = {}
for tag, prompt, mx in PROMPTS:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mx, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text = ""
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
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
    out[tag] = {"text": text, "completion_tokens": usage.get("completion_tokens"),
                "decode_tok_s": round((usage.get("completion_tokens", 0) - 1) / max(time.time() - first, 1e-9), 1)}
    print(f"{tag}: {usage.get('completion_tokens')} tokens, decode {out[tag]['decode_tok_s']} tok/s")
    print(f"   {text.strip()[:160]!r}")

import os
name = os.environ.get("REF_NAME", "ref-fp8.json")
with open(f"/home/dkp/serve-mia/{name}", "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote /home/dkp/serve-mia/{name}")
