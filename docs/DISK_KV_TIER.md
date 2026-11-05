# Disk KV — session files on a layer split, and an automatic evict-to-disk tier

Reusing a long prompt is the difference between a 0.2 s reply and a 31 s one. Strata already keeps
recent conversations in host RAM (`--conversation-cache-*`), but that cache is small and bounded: once
it evicts a conversation, the K/V is **gone** and the next visit re-reads the whole prompt.

This fork adds disk as a tier below that cache, so an evicted conversation comes back **without
prefill** — including after an engine restart, days later.

Two separate things live here. Most people want the second.

| | explicit session files | automatic disk tier |
|---|---|---|
| trigger | client sends `SESSION save` / `restore` | automatic, on cache eviction |
| works with `--layer-split` | yes (this fork's change) | yes |
| works with `--batch` (parallel) | **no** — still refused | **yes** |
| survives engine restart | yes | yes (sidecar index) |
| needs client support | yes | **no** |

---

## 1. Explicit session files on a layer split

Upstream refused session files whenever stages were in use: *"session files do not support this
multi-GPU shape"*. A layer split is the one multi-GPU shape where it is expressible, so the gate is
narrowed to the shapes that really cannot (`stages.empty() && multi_gpu`).

**Design: one v1 session file per stage** — stage 0 at `path`, later stages at `path.stage1 … .stageN`.

No file-format change was needed, and that is not a coincidence:
`conversation_checkpoints_split()` moves `stage_parts` *out* of every checkpoint (stage 0's are
cleared, each stage's are freshly constructed), so every image written is free of `stage_parts` and
passes `has_stage_parts()`. `kVersion` stays `1` and `conversation_file.cpp` is untouched.

**The trap, if you port this:** `serve/server.py` enforces `if done > total or done < last_done: raise`
on SESSION progress — and that *kills the engine*. A file per stage restarts `done` at 0, so both the
write and read sides need a cumulative offset (`moving(prog_base + done, prog_base + total)`). With
the offset, **stock `server.py` needs no change**.

Still refused, by design: `--peer-device`, other multi-GPU shapes, `--batch > 0` (each parallel slot
is its own session — a concurrency problem, not a format one), and `--prompt-cache <= 0`.

> Because `--batch > 0` is still refused, the explicit API does **not** work on a config with
> `parallel: 2` or higher. On such a config, use the automatic tier below — it bypasses these gates
> because it does not go through the SESSION command handler.

---

## 2. Automatic disk tier (recommended)

A conversation the RAM cache is about to drop is written to disk instead, and read back when a later
request matches it.

- **Spill** — `ConversationCache::set_on_evict()` fires inside `evict_oldest()`, so both eviction
  paths (the `make_room` loop and the physical-RAM gate) funnel through one hook.
- **Read back** — before the prefill decision, the tier is searched for the longest prefix of the
  incoming prompt: the entry's live tail *or* any of its indexed checkpoints. A checkpoint match
  mounts only that depth (`conversation_snapshot_restore_prefix`), exactly as a RAM borrow does.
- **Supersede** — the same rule the RAM cache applies in `put()` via `drop_superseded()`.
- **Index persistence** — each spill writes a small `.idx` sidecar, scanned at startup.

### Enable

| variable | default | meaning |
|---|---|---|
| `STRATA_DISK_TIER` | *(unset — feature off)* | directory for spilled conversations |
| `STRATA_DISK_TIER_GIB` | `200` | cap; the least-recently-used entry is dropped above it |

```jsonc
// strata-<model>.json
"parallel": 4,
"env": {
  "STRATA_DISK_TIER": "/opt/strata/sessions/auto",
  "STRATA_DISK_TIER_GIB": "250"
}
```

On disk per conversation: `conv-<serial>.bin` (stage 0), `conv-<serial>.bin.stage1 … .stageN`,
`conv-<serial>.bin.idx`. Put the directory on NVMe — a hit's floor is its read bandwidth.

---

## 3. What a hit can match — and why checkpoints are scarce

This is the part that decides your hit rate. Read it before tuning anything.

An entry is matched when a **stored sequence is a prefix of the incoming prompt**. There are two
kinds of stored sequence, and they fail in opposite situations:

| stored sequence | matches when | fails when |
|---|---|---|
| the entry's **live tail** (`prompt + everything the model generated`) | the new request **extends** it | the new request is **shorter** than it |
| an indexed **checkpoint** (a prefix the engine snapshotted mid-prefill) | the request reaches that depth | no checkpoint sits at a depth the request reaches |

Now the two request shapes a client actually sends:

- **Re-ask** — the same prompt again, alone (subagents, retries, a fan-out that rebuilds the same
  context). It is **shorter** than the stored tail, so **only a checkpoint can match it.**
- **Continuation** — `prompt + prior reply + a new question`. This *extends* the stored tail, so the
  tail path matches it… **provided the client replays what the model actually generated.**

> **Reasoning models:** the stored tail contains the generated tokens *including reasoning*. A client
> that replays only the visible answer produces a sequence that diverges from the stored tail
> immediately after the prompt, so its continuation **cannot** tail-match either — it falls back to
> needing a checkpoint, exactly like a re-ask. If you front the engine with something that replays
> `reasoning_content`, continuations tail-match; if you do not, assume every shape needs a checkpoint.

### The trap: checkpoints are a by-product of prefill

**The engine creates checkpoints while reading a prompt. This tier exists to stop reading prompts.**
So the better it works, the fewer checkpoints later spills contain — and checkpoints are the only
thing a re-ask can match. Measured on a 15-minute run: `ckpt` hits froze at 11 while `tail` hits
climbed to 100, and after a restart **all 50 surviving entries had zero checkpoints**, so
cross-restart resume failed **0/3** (report E).

Two rules follow, and both are implemented here:

1. **Never supersede a checkpointed entry with a checkpoint-less one.** A newer image is not a better
   one if it dropped the only thing a re-ask can match.
2. **Index the whole checkpoint chain**, not the deepest few — a request can arrive at any depth,
   and the useful checkpoint is the one at the *prompt boundary*, which is the shallowest, not the
   deepest.

## 4. Parameters — what to set, and why

| knob | recommended | why |
|---|---|---|
| `STRATA_DISK_TIER_GIB` | `(distinct conversations) × (image size)` | Image size is **~940 MiB per 41k-token history** on this rig. It is not linear in tokens — fixed state/checkpoint overhead dominates short conversations: **~30 KB/token at 15k, ~23 KB/token at 41k, ~19 KB/token at 62k**. 250 GiB ≈ **265** 41k-token histories. |
| `parallel` | **= stage count** (4 on a 4-way split) | `auto_groups()` picks the most groups ≤ stages that *divide* batch. 4 on 4 stages → **4 groups × 1 slot** (balanced); 3 → 3 groups (a stage idles); 2 → 2 groups (two idle). This is a structural argument from the scheduler, not a measured throughput win — see the caveat in §6. |
| `--prompt-cache` | leave at the default (`6`) or higher density | Sets checkpoint spacing, which decides whether a checkpoint exists **at the prompt boundary**. This is what the tier matches (§3). `--prompt-cache 0` disables checkpoints *and* the explicit session API. |
| `--conversation-cache-slots` | `8` | The RAM tier is still faster than disk (~0.33 s vs ~2.9–3.9 s to mount). Spill is the fallback, not the goal — give RAM what it can hold. |
| `--kv` | `int8` | Measured **no** throughput benefit from smaller KV: `k8v4` and `q4_0` landed within **±1.3%** of `int8`. And `q4_0` costs quality that *grows* with context (+8% perplexity at 1k, **+12% at 8k**). Smaller KV does shrink the images, so it is a capacity knob, not a speed one. |
| tier directory | NVMe | A hit transfers the whole image (~940 MiB here). Observed ~2.9–3.9 s, i.e. a few hundred MB/s is the floor on TTFT. |
| *(internal)* indexed checkpoints per entry | 16 | Each one costs its full prefix in the sidecar and in the in-RAM index (~164 KB per 41k-token prefix). 16 bounds a full 250 GiB tier to a few hundred MB of index. Raise it only if re-asks arrive at many distinct depths. |

---

## 5. Test reports

**Rig for every number below:** VM100 — 4 × RTX 3080 20 GB, 4-stage layer split
(layers 0–11 / 12–24 / 25–36 / 37–47), Qwen3.8-Flash-Next 125B MoE UD-IQ4_XS,
`--max-context 262144`, `--kv int8`, `--conversation-cache-slots 8`,
`--conversation-cache-mib 12288`, engine 0.1.51.

### A. Explicit session files, across a full engine restart
62,189-token prompt, `parallel: 1`.

| | |
|---|---|
| save | 4 files, 1,189,696,928 B (1.11 GiB) in **2.92 s** |
| restore (after restart) | 1.11 GiB in **3.32 s** |
| cold prefill | ttft **31.16 s**, prompt_read 62,189 |
| after restore | ttft **0.79 s**, prompt_read **5**, reused **100%** |

`stage3` is ~59 MB larger than its siblings — it carries the MTP draft layer, which confirms draft
routing lands on the last stage.

### B. Eviction survival, with a control arm
14,935-token conversation: save, then 9 other conversations fill the 8-slot cache, then re-ask.

| arm | reuse | prompt_read | ttft |
|---|---|---|---|
| **control** (no restore) | **0%** | 14,935 | 8.48 s |
| restore from disk (445 MB in 1.17 s) | **100%** | **5** | **0.22 s** |

The control arm is the point: without it, a 100% figure proves nothing, because the conversation
might simply never have left RAM.

### C. Automatic tier — the checkpoint path
10 conversations × ~41k tokens vs an 8-slot cache, `parallel: 2`, serialised, 18 min.

| | |
|---|---|
| turns | **212** |
| tier hits | **212 / 212**, all checkpoint matches |
| reuse | **100%** on every turn |
| failures / engine restarts | **0 / none** |
| ttft on a hit | **2.9 – 3.8 s** (vs **16 – 23 s** cold) |

### D. Automatic tier — supersede + 4-way concurrency
`parallel: 4`, 4 concurrent clients, 15.2 min, `max_tokens 150`.

| | |
|---|---|
| turns | **304** (304 ok, 0 failed, 304 metrics-matched) |
| tier | spills 146, hits **146**, fail **0** |
| hit mode | ckpt 12 / tail 134 |
| reuse > 90% | **304 / 304 (100%)** |
| ttft p50 / p90 / max | **1.88 / 8.34 / 12.86 s** |
| aggregate decode | 50.1 tok/s |
| store | **flat at 10,245 MB** |

Two things to read here. First, **the store stays flat** — that is the supersede rule working. Without
it the same 10 conversations grew to **118 entries / 111 GiB in 10 minutes** and hit a 200 GiB cap at
t+18m, at which point GC starts dropping live entries and quietly erodes the hit rate.

Second, the hit mode shifts to mostly `tail`. Under concurrency a conversation can be evicted
*mid-prefill*, before its reply exists — and then its live tail genuinely *is* the prompt, so the tail
path matches. Both paths are exercised; report C is the checkpoint path's test.

### E. Confirmation on the deployed production config — long replies
The promoted production engine (`parallel: 4`, tier on), 4 concurrent clients, 15.6 min,
`max_tokens 2000`. Decode-weighted rather than prompt-weighted.

| | |
|---|---|
| turns | **113 ok, 0 failed** (113 metrics-matched) |
| reuse > 90% | **113 / 113 (100%)** |
| aggregate decode | **112.7 tok/s** (4 concurrent) |
| per-stream decode | p50 **33.8**, min 2.4, max 68.0 tok/s |
| ttft p50 / p90 / max | **3.02 / 4.58 / 9.11 s** |
| tier | spills 157, hits 111, fail 0, superseded 107 |
| hit mode | ckpt 11 / tail 100 |

The per-stream spread is the batch sharing the GPU: a stream that ends up decoding alone runs near
68 tok/s, four overlapping ones fall to ~14. Lower `parallel` for latency, raise it for aggregate.

### F. The two failures this feature hit — and what they cost
Both were found only by testing resume explicitly. Worth repeating if you port this.

**F1 — every entry lost its checkpoints.** After 15 minutes the tier held 50 entries, *all* with
`ck=0`, and cross-restart resume failed **0/3** (each re-prefilled ~41k tokens at ~15.8 s). Cause:
checkpoints are made by prefill, the tier removes prefill, and the supersede rule was happily
replacing a checkpointed entry with a checkpoint-less one. Fixes: never supersede a checkpointed
entry with one carrying none, and index the whole chain rather than the deepest few (§3).

**F2 — a restarted engine never looked at its own index.** The lookup was gated on the session
identity, which only a *spill* sets. A fresh engine that recovered its index from disk therefore
skipped the lookup entirely: `recovered 2 conversations … dropped 0` followed by
**`tier reads after restart: 0`**, resume **0/2**. Fix: gate on having candidates
(`!dt.empty()`), and let the read path establish the identity itself.

The lesson for both: *a tier that recovers state at startup must be tested from a cold process.*
Every in-process number (reports C and D, 100% reuse on 516 turns) stayed green through both bugs.

### G. Cross-restart resume — verified
Two conversations present **only on disk**, engine restarted so RAM holds nothing, each prompt
re-asked. Compare the same requests before the F2 fix.

| | before F2 fix | after |
|---|---|---|
| conv 00 | reuse 0%, read 40950, ttft **15.17 s** | reuse **100%**, read **5**, ttft **2.34 s** |
| conv 01 | reuse 0%, read 41216, ttft **15.95 s** | reuse **100%**, read **5**, ttft **2.76 s** |
| tier reads | **0** | **2**, both `mode=ckpt` |

```
disk tier: recovered 2 conversations (1884 MiB) from /opt/strata/sessions/auto, dropped 0
disk tier: read 40945 tokens (940 MiB) back - no prefill (mode=ckpt)
disk tier: read 41211 tokens (944 MiB) back - no prefill (mode=ckpt)
```

5 tokens prefilled instead of 40,950. This is the capability the tier exists for: compute a context
today, resume it after the engine has been down, pay no prefill.

### Known-good configuration

Everything above was measured with:

```jsonc
"exe": "<patched build>",
"parallel": 4,                       // = stage count
"layer_split": "auto",
"env": { "STRATA_DISK_TIER": "/opt/strata/sessions/auto", "STRATA_DISK_TIER_GIB": "250" }
```
plus `--max-context 262144 --kv int8 --conversation-cache-slots 8 --conversation-cache-mib 12288`.

## 6. What is measured, and what is not

Be careful with the throughput figures. Stated plainly:

- **Reuse, hit rate, store size, ttft, and cross-restart resume are measured** and reproducible, with
  a control arm where one was needed (report B).
- **`parallel: 4` vs `parallel: 2` was never compared at matched concurrency.** Report C is
  serialised at `parallel: 2`; report D is 4-way concurrent at `parallel: 4`. They differ in more than
  one variable, so **D ÷ C is not a speedup**. The `parallel = stage count` recommendation rests on
  the `auto_groups()` argument in §4 plus "validated working", not on a measured win.
- This rig shows **run-to-run decode drift of up to 2.8×** on byte-identical configs (49.5 → 79.1 →
  137.2 tok/s across three identical runs). Cause unresolved; cold page cache, engine swapping,
  sidecar RAM and GPU throttling were each ruled out. **Treat any single tok/s number from this rig,
  including the ones above, as indicative only.**
- Reports C–E are prompt-dominated by design (41k-token prompts against 150–2000 output tokens) to
  stress the tier. They **understate** what the config can do on decode.

---

## 7. Tuning — symptom to action

| symptom | likely cause | action |
|---|---|---|
| hits near zero on **re-asks** | no checkpoint at the prompt boundary (§3) | raise checkpoint density (`--prompt-cache`); confirm the build preserves checkpointed entries (§3) |
| hits near zero on **continuations** with a reasoning model | the client replays only the visible answer, so its history diverges from the stored tail (§3) | replay `reasoning_content`, or accept that continuations need a checkpoint too |
| `ckpt` hits frozen while `tail` hits climb | later spills carry no checkpoints because the tier removed their prefill (§3) | expected; make sure checkpointed entries are not being superseded away |
| hit rate ~48% ceiling | live-tail matching only | you are on a build without checkpoint indexing |
| store grows without bound | no supersede rule | you are on a build without the `drop_superseded` parity fix (§2) |
| store at the cap, hit rate falling | genuinely more conversations than the cap holds | raise `STRATA_DISK_TIER_GIB`, or shrink images with `--kv k8v4` (capacity, not speed — §4) |
| ttft 3 s+ on every hit | image transfer is the floor | faster disk; or keep more in RAM (`--conversation-cache-slots`, `--conversation-cache-mib`) |
| per-stream decode feels slow under load | 4 slots sharing the GPU | expected — per-stream fell to ~14 tok/s with 4 concurrent vs ~60 alone. Lower `parallel` for latency, raise it for aggregate throughput |
| `recovered 0 … dropped N` at startup | images written by a build without sidecars, or corrupt `.idx` | expected once after upgrading; the orphans are reclaimed, not leaked |
| explicit `SESSION` refused | `--batch > 0`, `--peer-device`, `--prompt-cache 0` | use the automatic tier instead (§2) |

---

## 8. Limitations

- The explicit SESSION API is still refused with `--batch > 0`, `--peer-device`, other multi-GPU
  shapes, and `--prompt-cache <= 0`. The automatic tier is not.
- The tier's index keeps the **4 deepest** checkpoints per entry, so the supersede rule is
  *conservative*: it can miss a supersede the full chain would catch, but never drops an entry the
  rule would keep. Erring toward a duplicate is the safe direction for a cache.
- `.idx` sidecars are **host byte order** and carry no version negotiation beyond their magic. This is
  a local cache: a sidecar that does not parse costs its image and nothing more.
- A split save materialises each stage in host RAM (the streaming `conversation_snapshot_sources`
  path needs a `QsaState&` and cannot express "no draft").
- The tier is per engine process. Two engines pointed at one directory are not coordinated.
- Checkpoint availability, not disk space, is what usually limits re-ask hits — see §3. A conversation
  whose turns are always served from cache stops producing new checkpoints.
- A continuation from a **reasoning** model only tail-matches if the client replays
  `reasoning_content`; otherwise it needs a checkpoint like a re-ask (§3).

## 9. Rollback

Production pins the engine in `strata-<model>.json`. Promotion backs the file up first:

```bash
cp /opt/strata/Strata/strata-unsloth-ud-iq4_xs.json.pre-disktier.<stamp> \
   /opt/strata/Strata/strata-unsloth-ud-iq4_xs.json
sudo systemctl restart strata.service
```

Removing `STRATA_DISK_TIER` alone disables the tier while keeping the patched binary; the spilled
files are then inert and can be deleted.

## 10. Where the code is

All of it is in `src/program/generate.cpp`, plus the one hook in
`include/strata/core/conversation_cache.hpp`. Search for `#diskkv`.
