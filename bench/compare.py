#!/usr/bin/env python3
"""Compare results-tp5.json and results-pp5.json: A/B table + output agreement."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(layout):
    p = os.path.join(HERE, f"results-{layout}.json")
    if not os.path.exists(p):
        print(f"missing {p}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


a = load("tp5")
b = load("pp5")

print(f"model: {a['model']}   (both layouts, greedy, reasoning on, one stream)\n")

print("== smoke / tool call ==")
print(f"  TP5: answer={a['tests']['smoke']['answer']!r} ok={a['tests']['smoke']['ok']} | "
      f"tool_call ok={a['tests']['tool_call']['ok']} n={a['tests']['tool_call']['n_tool_calls']}")
print(f"  PP5: answer={b['tests']['smoke']['answer']!r} ok={b['tests']['smoke']['ok']} | "
      f"tool_call ok={b['tests']['tool_call']['ok']} n={b['tests']['tool_call']['n_tool_calls']}")

print("\n== decode (C-coding requests, 600 tokens) ==")
print(f"  {'request':10} {'TP5 TTFT':>9} {'TP5 tok/s':>10} {'PP5 TTFT':>9} {'PP5 tok/s':>10}")
for da, db in zip(a["tests"]["decode"], b["tests"]["decode"]):
    print(f"  {da['tag']:10} {da['ttft']:9.2f} {da['decode_tok_s']:10.1f} "
          f"{db['ttft']:9.2f} {db['decode_tok_s']:10.1f}")
ma = sum(d["decode_tok_s"] for d in a["tests"]["decode"]) / len(a["tests"]["decode"])
mb = sum(d["decode_tok_s"] for d in b["tests"]["decode"]) / len(b["tests"]["decode"])
print(f"  {'mean':10} {'':9} {ma:10.1f} {'':9} {mb:10.1f}   (TP5/PP5 = {ma/mb:.2f}x)")

print("\n== prefill (cold prompts, 8 new tokens) ==")
print(f"  {'tokens':>9} {'TP5 TTFT':>9} {'TP5 tok/s':>10} {'PP5 TTFT':>9} {'PP5 tok/s':>10}")
pa = {p["prompt_tokens"]: p for p in a["tests"]["prefill"]}
pb = {p["prompt_tokens"]: p for p in b["tests"]["prefill"]}
for tok in sorted(set(pa) | set(pb)):
    x, y = pa.get(tok), pb.get(tok)
    xs = f"{x['ttft']:9.1f} {x['prefill_tok_s']:10.1f}" if x else f"{'-':>9} {'-':>10}"
    ys = f"{y['ttft']:9.1f} {y['prefill_tok_s']:10.1f}" if y else f"{'-':>9} {'-':>10}"
    print(f"  {tok:9} {xs} {ys}")

print("\n== agreement (fixed prompt, 64 greedy tokens) ==")
ta = a["tests"]["agreement"]["text"].strip()
tb = b["tests"]["agreement"]["text"].strip()
print(f"  TP5: {ta[:180]!r}")
print(f"  PP5: {tb[:180]!r}")
print(f"  identical: {ta == tb}")
if ta != tb:
    n = min(len(ta), len(tb))
    i = next((k for k in range(n) if ta[k] != tb[k]), n)
    print(f"  first difference at char {i}")
