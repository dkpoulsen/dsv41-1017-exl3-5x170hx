#!/usr/bin/env python3
"""bench_nine_requests.py: Standardized nine-request C-coding benchmark protocol.
Mirrors anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m request categories:
  1. review       (~5,100 tok prompt, 1,500 max)
  2. bug-find     (~3,500 tok prompt,   600 max)
  3. fix-diff     (~3,400 tok prompt,   350 max)
  4. fix-func     (~3,400 tok prompt, 1,200 max)
  5. whole-file   (~1,800 tok prompt, 2,600 max)
  6. agent-edit   (~4,000 tok prompt,   154 max, with tools)
  7. agent-multi  (~9,400 tok prompt,   212 max, with tools & turns)
  8. new-code     (  ~140 tok prompt, 1,500 max)
  9. explain      (~5,100 tok prompt, 1,000 max)

Greedy, streaming, measures TTFT, completion tokens, decode tok/s.
"""

import json, os, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4.1-flash"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

CORPUS_PATH = "/home/dkp/serve-mia/corpus.c"
if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, "r", errors="ignore") as f:
        # Load ~50K characters of C code (~15K tokens)
        c_code_base = f.read(50000)
else:
    c_code_base = "/* C code sample */\n" * 2000

# Subsets for realistic prompt lengths
chunk_5k = c_code_base[:16000]
chunk_3k5 = c_code_base[:11000]
chunk_1k8 = c_code_base[:5600]
chunk_9k = c_code_base[:28500]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit lines in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"}
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    }
]

REQUESTS = [
    (
        "review",
        [
            {"role": "system", "content": "You are a senior Linux kernel and C systems auditor."},
            {"role": "user", "content": f"Review this C source for locking, memory ordering, and concurrency safety issues. Provide a detailed section-by-section breakdown:\n\n```c\n{chunk_5k}\n```"}
        ],
        None,
        1500
    ),
    (
        "bug-find",
        [
            {"role": "system", "content": "You are an expert vulnerability researcher and static analyzer."},
            {"role": "user", "content": f"Find all edge-case bugs, buffer overruns, and potential integer truncation risks in this codebase:\n\n```c\n{chunk_3k5}\n```"}
        ],
        None,
        600
    ),
    (
        "fix-diff",
        [
            {"role": "system", "content": "You are a Git patch contributor."},
            {"role": "user", "content": f"Produce a unified diff format patch fixing error checking on memory allocations in this module:\n\n```c\n{chunk_3k5}\n```"}
        ],
        None,
        350
    ),
    (
        "fix-func",
        [
            {"role": "system", "content": "You are a systems programmer specializing in refactoring."},
            {"role": "user", "content": f"Rewrite the main helper functions in this code to make them strictly reentrant and thread-safe:\n\n```c\n{chunk_3k5}\n```"}
        ],
        None,
        1200
    ),
    (
        "whole-file",
        [
            {"role": "system", "content": "You are a C library author."},
            {"role": "user", "content": f"Based on the interfaces declared below, implement the complete standalone C module with clean memory management and documentation:\n\n```c\n{chunk_1k8}\n```"}
        ],
        None,
        2600
    ),
    (
        "agent-edit",
        [
            {"role": "system", "content": "You are an automated coding agent. If a tool call is needed, call it directly using the available tool definitions."},
            {"role": "user", "content": f"Inspect this file and invoke edit_file to add a mutex guard to the global state access:\n\n```c\n{chunk_3k5}\n```"}
        ],
        TOOLS,
        154
    ),
    (
        "agent-multi",
        [
            {"role": "system", "content": "You are an automated coding agent acting on a repository."},
            {"role": "user", "content": f"Read the header file and tell me what structures are defined in:\n\n```c\n{chunk_9k}\n```"},
            {"role": "assistant", "content": "I need to check the definitions.", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"include/types.h\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "typedef struct { int fd; void *buf; size_t len; } buffer_ctx_t;"},
            {"role": "user", "content": "Now show how buffer_ctx_t is initialized in the code above and whether it is safe."}
        ],
        TOOLS,
        212
    ),
    (
        "new-code",
        [
            {"role": "system", "content": "You are an expert C11 programmer."},
            {"role": "user", "content": "Write a complete, high-performance, cache-aligned single-producer single-consumer (SPSC) lock-free ring buffer in modern C11 with atomic acquire/release fences."}
        ],
        None,
        1500
    ),
    (
        "explain",
        [
            {"role": "system", "content": "You are a computer architecture professor."},
            {"role": "user", "content": f"Explain the cache coherency implications of this codebase when executed across multiple NUMA sockets with MOESI protocol and relaxed memory ordering:\n\n```c\n{chunk_5k}\n```"}
        ],
        None,
        1000
    ),
]

def run_one(name, messages, tools, max_tokens):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    if tools:
        body["tools"] = tools

    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    first_token_time = None
    completion_tokens = 0
    prompt_tokens = 0
    tool_calls_seen = 0

    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except Exception:
                continue

            if "usage" in data and data["usage"]:
                u = data["usage"]
                prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                completion_tokens = u.get("completion_tokens", completion_tokens)

            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                tc = delta.get("tool_calls")
                if tc:
                    tool_calls_seen += len(tc)
                if first_token_time is None and (content or tc):
                    first_token_time = time.time()

    t1 = time.time()
    ttft = (first_token_time or t1) - t0
    gen_time = max(t1 - (first_token_time or t1), 1e-9)
    # decode tok/s is completion tokens after the first divided by generation time
    n_after_first = max(0, completion_tokens - 1)
    decode_tps = n_after_first / gen_time if n_after_first > 0 else 0.0
    prefill_tps = prompt_tokens / max(ttft, 1e-9)

    return {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft": ttft,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "tool_calls": tool_calls_seen,
        "total_time": t1 - t0
    }

def main():
    print(f"=== Running Nine-Request C-Coding Benchmark against {URL} ({MODEL}) ===")
    results = []
    for name, messages, tools, max_tokens in REQUESTS:
        print(f"Running request: {name:12s} ... ", end="", flush=True)
        res = run_one(name, messages, tools, max_tokens)
        results.append(res)
        tc_str = f" [tool_calls={res['tool_calls']}]" if res['tool_calls'] else ""
        print(f"prompt {res['prompt_tokens']:5d} (TTFT {res['ttft']:5.2f}s, {res['prefill_tps']:6.0f} tok/s) | out {res['completion_tokens']:4d} tokens | decode {res['decode_tps']:6.1f} tok/s{tc_str}", flush=True)

    mean_decode = sum(r["decode_tps"] for r in results) / len(results)
    total_tokens = sum(r["completion_tokens"] for r in results)
    total_wall = sum(r["total_time"] for r in results)
    print("=" * 70)
    print(f"SUMMARY: Mean per-request decode: {mean_decode:.1f} tok/s")
    print(f"Total tokens generated: {total_tokens} in {total_wall:.1f}s (wall throughput {total_tokens/total_wall:.1f} tok/s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
