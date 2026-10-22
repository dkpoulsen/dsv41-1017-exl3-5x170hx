#!/usr/bin/env python3
"""Prewarms ds41-flash serving engine on port 8000:
1. Smoke ping (triggers CUDA graph capture and Triton JIT compilation)
2. Agent system prompt prefix (populates KV prefix cache for near-zero TTFT)
3. 4096-token prefill chunk (warms up TC-MoE prefill workspace)
"""

import json
import os
import sys
import time
import urllib.request

BASE_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-v4.1-flash")

def post_json(path, data, timeout=60):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def wait_for_ready(max_seconds=600):
    start = time.time()
    while time.time() - start < max_seconds:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def main():
    print(f"Waiting for {BASE_URL}/health ...")
    if not wait_for_ready():
        print(f"Timeout waiting for {BASE_URL} to become ready", file=sys.stderr)
        sys.exit(1)

    print("ds41-flash engine is ready! Beginning pre-warming...")

    # 1. Smoke test: 1 token completion to warm CUDA graphs and JIT kernels
    try:
        t0 = time.perf_counter()
        res = post_json("/v1/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}
        })
        dt = time.perf_counter() - t0
        print(f"1. Smoke warmup ok ({dt:.2f}s)")
    except Exception as e:
        print(f"Smoke warmup warning: {e}", file=sys.stderr)

    # 2. Agent System Prompt Prefix Caching
    prompt_file = "/etc/nixos/pi-system-prompt.txt"
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                sys_prompt = f.read().strip()
            t0 = time.perf_counter()
            res = post_json("/v1/chat/completions", {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Ready."}
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}
            })
            dt = time.perf_counter() - t0
            cached = (res["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            total = res["usage"].get("prompt_tokens", 0)
            print(f"2. Agent prefix cache warmed ({total} prompt tokens, {cached} cached, {dt:.2f}s)")
        except Exception as e:
            print(f"Agent prefix warming warning: {e}", file=sys.stderr)

    # 3. 4K chunk prefill warmup
    try:
        dummy_text = ("Optimization of tensor-parallel all-reduce operations across PCIe Gen2 links. " * 80)
        t0 = time.perf_counter()
        res = post_json("/v1/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": dummy_text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}
        })
        dt = time.perf_counter() - t0
        tokens = res["usage"].get("prompt_tokens", 0)
        tok_s = (tokens / dt) if dt > 0 else 0
        print(f"3. 4K chunk prefill warmup ok ({tokens} tokens, {dt:.2f}s, {tok_s:.0f} tok/s)")
    except Exception as e:
        print(f"Chunk warmup warning: {e}", file=sys.stderr)

    print("Pre-warming sequence complete.")

if __name__ == "__main__":
    main()
