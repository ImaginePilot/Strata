# Measured tuning on 4× RTX 3080‑20G (pipeline parallel, 125B MoE @ IQ3_XXS)

Host: 2× Xeon Platinum 8269CY (104 threads), 376 GiB. Engine: 4 stages, layer split `13,26,39`,
`--kv int8 --kv-resident 65536`, `--ple-io ram`, `--prefill-pipe 0`, conversation cache 12288 MiB.

**Benchmark protocol.** Every A/B below stopped the co-tenant routers first — ungated warm rounds on
this box drift **1.24×**, larger than most effects worth measuring. Engine restarts wait for the
`ready:` **count to rise** (the log is appended across restarts, so matching on presence returns
instantly and benchmarks a still-loading engine). Each arm discards a warmup round.

---

## Decode throughput vs context (gated, 4 lanes, 3 reps, prefill excluded)

| context | per-lane | fleet decode | TTFT |
|---|---|---|---|
| 20k | 74.2 tok/s | 269.2 tok/s | 1.75 s |
| 40k | 68.4 | 237.0 | 2.35 s |
| 80k | 59.0 | 187.7 | 3.23 s |
| 106k | 54.5 | 166.6 | 8.79 s |

Within-arm spread on per-lane decode ≤1%. **Decode falls only −27% across a 5.3× context range.**
Marginal gain per halving: 106k→80k +12.7%, **80k→40k +26.3%**, 40k→20k +13.6% — the knee is ~40k.

## The cliff is the conversation-cache budget, not context length

Same context, same config, only the resident set differs:

| 80k | within budget | over budget |
|---|---|---|
| per-lane | 59.0 | 46.1 (−22%) |
| fleet decode | 187.7 | **98.8 (−47%)** |
| TTFT | 3.23 s | 9.94 s |

**−47% exceeds the entire 20k→106k context effect (−38%).** Size the working set, not the context:
parked snapshot fits `291 MiB + 19.4 KB/token`, so 4 lanes need 4.1 GiB at 40k but 9.2 GiB at 106k
against a 12 GiB budget — already marginal before cold conversations arrive.

## MTP speculative decoding is gated by CONCURRENCY, not context

| concurrency | n | drafts/token | tok/s |
|---|---|---|---|
| 1–2 | 11 | 0.861 | 129.5 |
| 2–3 | 21 | 0.764 | 125.3 |
| **3–5** | 14 | **0.000** | 85.4 |

Context showed **no** monotonic relationship (the 40–70k bucket drafted *more* than 20–40k). Verified
within each client independently, so it is not a client artifact. Decode tracks draft firing: ~0.85
drafts/token → 125–165 tok/s, 0.00 → 28–85.

Practical consequence: **do not add concurrent streams you do not need.** Bounding a background
workload to one concurrent request raised draft firing from 53% to 68% and median decode +22%.

This reproduces, on a different engine and model, the conclusion already recorded for 27B/vLLM: MTP
is a low-concurrency optimisation.

## `--batch-groups` semantics (why `--batch-mtp` cannot run pipelined)

`auto_groups()` picks the largest divisor of `--batch` that is ≤ stages. At `--batch 4` on a 4-stage
split that is 4 groups of 1 slot — full pipeline depth, every stage busy. `--batch-groups 1` is one
window of 4 slots walking the stages in lockstep, leaving three cards idle at each instant; measured
**−31%**, matching the source's own "25–30%".

`--batch-mtp` needs **two rows per slot** (one committed token, one proposal); the pipelined path
builds **one row per slot** and commits differently (`commit_slot_prefixes` vs `commit_slots`).
Removing the guard would not enable drafting — it would allocate drafters (~0.9 GB/slot) that produce
nothing. Upstream's own measurement bounds the prize: at four concurrent streams, serial+MTP lost
25–30% against pipelined-without-MTP, which with our −31% for serial implies an MTP multiplier of
only **M ≈ 1.01–1.09** under batch.

## Slot count at a compacted context (gated, 50.2k/lane, 3 reps + discarded warmup)

| | batch 4 | batch 8 |
|---|---|---|
| aggregate | 171.2 / 176.1 / 187.4 (**still rising**) | 210.2 / 212.3 / 211.9 (**±0.5%**) |
| per-lane | 42.8–46.9 | 26.5 |
| latency p100 | 8.5–9.3 s | 15.1 s |
| expert cache | 6656 / 9.41 GiB per stage | **unchanged** |
| VRAM | — | +6.0 GiB (~1.5 GiB/slot at this context) |

**+13% aggregate**, quoted against batch 4's *best* rep because batch 4 never converged. The extra
slots did not come out of the expert arena. ~1.5 GiB/slot here, **not** the 0.39 GiB/slot measured at
short context — slot cost is regime-dependent.

---

## Corrections to earlier claims in this fork's history

- **"3 of 4 cards idle by construction"** — false for prompt reads; all four cards are busy on a
  single request at `--prefill-pipe 0`. (Still true for `--batch-groups 1` decode windows, which is a
  different mechanism.)
- **sm% and watts both mislead.** `--prefill-pipe 4` showed *higher* utilisation at 7.6× *lower*
  throughput — that util was kernel residency while stalling. Tune on `prompt_ms`, not utilisation.
- **Aggregate decode tok/s is the wrong health metric** on a prefill-bound workload; quote prefill
  tok/s and reuse %.
- **"SUM-decode" is not a throughput.** Summing each lane's own average over its own window
  overstates by ~1.6×, because a lane that freezes during another lane's prompt read shows a low
  average rather than idle time. Use total tokens ÷ wall.
