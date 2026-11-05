#!/usr/bin/env python3
"""Single-stream decode throughput with a co-tenant contention gate.  Measurement only.

A shared engine makes decode tok/s look like it drifts.  It does not: the batch is simply being split
with somebody else's request.  On this rig that artefact reached 2.84x spread across byte-identical
probes and was mistaken for engine behaviour for days (docs/DISK_KV_TIER.md, section 6).

Each probe sends one identical request and brackets it with the engine's own log, then classifies the
measurement window:

  * The engine prints `thinking: N of max M tokens` and `answering: ...` once per second per stream.
  * This probe caps itself mid-thinking, so it never reaches an `answering:` phase and only ever
    reports its own `--max-tokens` as M.
  * Therefore an `answering:` line, or any `of max M` with an M we did not send, is somebody else's
    request decoding inside our window -- the sample is CONTENDED and must not be pooled with the
    clean ones.

Reported separately, never averaged together.  Pool only the ALONE rows; if there are none, the box
is too busy to measure and no number from it means anything.

    python decode_contention_probe.py --runs 14
    python decode_contention_probe.py --runs 20 --url http://host:8080/v1/chat/completions --log /path/server.log

Exits non-zero if no clean sample was obtained, so it can gate a benchmark in CI or a shell script.
"""
import argparse, json, statistics, subprocess, sys, time, urllib.request

PROMPT = "Explain in detail how a write-back CPU cache decides which line to evict, and why."


def log_lines(path, first, last):
    if not path or last <= first:
        return []
    out = subprocess.run(["sed", "-n", f"{first + 1},{last}p", path], capture_output=True, text=True)
    return out.stdout.splitlines()


def log_len(path):
    if not path:
        return 0
    try:
        return int(subprocess.run(["wc", "-l", path], capture_output=True, text=True).stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def probe(url, model, max_tokens, timeout):
    """One streaming request.  Returns decode tok/s measured between the first and last token."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                delta = (json.loads(payload).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                continue
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
                last = time.time()
                if first is None:
                    first = last
    if first is None or n < 2 or last <= first:
        return None
    # n-1 inter-token gaps, so the prefill is excluded from the measured window.
    return {"ttft": first - t0, "tokens": n, "decode": (n - 1) / (last - first)}


def foreign(lines, max_tokens):
    mine = f"of max {max_tokens} tokens"
    return [l for l in lines if "answering:" in l or ("of max" in l and mine not in l)]


def report(name, samples):
    if not samples:
        print(f"{name}: none")
        return
    lo, hi = min(samples), max(samples)
    line = (f"{name}: n={len(samples)}  min {lo:.1f}  median {statistics.median(samples):.1f}  "
            f"max {hi:.1f}  spread {hi / lo:.2f}x")
    if len(samples) > 2:
        line += f"  cv {statistics.stdev(samples) / statistics.mean(samples) * 100:.1f}%"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--log", default="/opt/strata/server.log",
                    help="engine log used to detect co-tenants; pass empty to disable the gate")
    ap.add_argument("--model", default="local")
    ap.add_argument("--runs", type=int, default=14)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    alone, shared = [], []
    for i in range(a.runs):
        before = log_len(a.log)
        r = probe(a.url, a.model, a.max_tokens, a.timeout)
        after = log_len(a.log)
        if r is None:
            print(f"  run {i + 1:2d}: no tokens returned")
            continue
        intruders = foreign(log_lines(a.log, before, after), a.max_tokens)
        (shared if intruders else alone).append(r["decode"])
        tag = f"CONTENDED ({len(intruders)} foreign line(s))" if intruders else "alone"
        print(f"  run {i + 1:2d}: decode {r['decode']:6.1f} tok/s  ttft {r['ttft']:5.2f}s  {tag}")

    print()
    report("ALONE ", alone)
    report("SHARED", shared)
    if not alone:
        print("\nNo clean sample: the engine was busy for every probe.  No tok/s number here is usable.")
        return 1
    if shared:
        print("\nThe SHARED rows are an artefact of sharing the batch, not engine drift.  Discard them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
