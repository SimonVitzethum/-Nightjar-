# What the engine actually costs, and how that was found

Every number here was measured on the machine named beside it. Where a guess was recorded and
later contradicted, both are kept — the contradictions are the useful part.

## Decode, Ornith-1.5-35B-A3B (Q4_K_M), RTX 5070 Laptop 8 GB

| Stage | tok/s | ms/tok | GPU busy | idle |
|---|---|---|---|---|
| Start | 25.6 | 39.4 | 41.2% | 23.8 ms |
| Selection sampler | 44.5 | 22.8 | 89.9% | 2.3 ms |
| Expert misses on the copy stream | 46.8 | 21.5 | 93.0% | 1.5 ms |
| gemv decomposition | 55.3 | 19.0 | 88.5% | 2.2 ms |
| rmsnorm, coalesced pointer uploads, hoist | 56.5 | 17.7 | 92% | 1.5 ms |

**2.2x**, and the shape of the problem changed underneath it: at 41% occupancy the bottleneck
was the host, at 92% it is memory bandwidth.

## The three instruments

- **`QWEN_CUPTI=1`** — exact GPU timeline via CUPTI, the library nsys records with. Reports
  the union of op intervals (busy), the sum of durations (which exceeds busy exactly when
  streams overlap, so it measures overlap directly), and the idle gap before each op
  *attributed to the op that ended it*. That last column is what turns a table into a
  diagnosis.
- **`QWEN_HOSTPROF=1`** — the host side, invisible to CUPTI. `forward 17.68 ms | sync 14.13,
  topk 0.10, issue 0.37`.
- **`tests/qwen35_gemv_bench`** — decode kernels at real shapes on synthetic weights, checked
  against the CPU dequant. No model load, so kernel work iterates in seconds.

## Four claims that measurement killed

1. **"The GPU is launch-bound."** 59% idle, but 94% of it was one bubble per token, before the
   embedding upload, where only the CPU runs: the sampler, sorting a 248320-entry vocabulary to
   keep twenty candidates. 18.4 ms/token, larger than every CUDA kernel combined.
2. **"Q4_K is slower than Q6_K per byte."** Same time for the same shape. The GB/s gap was
   Q6_K storing more bytes for the same arithmetic. The real win was the shape dispatch:
   one block per row collapses when the row is short.
3. **"The card does 600 GB/s."** It does 283 (5070 Laptop) / 890 (5080). The big kernels were
   already at 81–87% of achievable, so a budget written against 600 was off by two.
4. **"Hot-expert pinning beats LRU."** 81.8% hit for LRU against 76.8% for LFU. At the
   capacity-limited plateau there is little for a policy to decide.

## Speculative decoding

Acceptance is good and throughput is not: 2.96 of 4 drafted tokens accepted per round,
50.1 tok/s against 50.6 plain. A verify over S positions costs S *sequential* gated-delta-net
state updates — the recurrence does not batch. Only the expert weights amortise, and a
five-token window still touches 3.14x the experts of one token.

This is architectural. On an attention-only model the same machinery should pay; here it does
not, which is why it ships off by default and switchable per request.

## Prefill, and a correction

Prefill runs at **33-38 tok/s**, flat in prompt length: 689 tokens at 33.2, 2714 at 38.1,
10514 at 35.4. The prefix cache works as arithmetic says it should — a turn that genuinely
extends the previous one reuses it, measured `330 prompt (310 cached) in 0.32s`.

The correction is worth recording because the wrong number came from this repository. Earlier
commentary in the source quotes 8-12 tok/s prefill (`19504 prompt (8192 cached) in 1497.43s`),
and that figure was repeated here without being re-measured. It predates the work in this
document, and the current engine is three to four times faster than it. Two other numbers were
taken the same way and were also wrong: a 600 GB/s card that delivers 283, and a markdown
parser blamed for a hang it did not cause.

A cached prefix only helps when the request EXTENDS the previous one exactly. Changing the
middle of a prompt discards it, and correctly so: the recurrent half of this architecture
cannot be rewound, only replayed from a checkpoint. Checkpoints are written every 512 tokens,
so a prompt shorter than that has none to fall back on.

What still costs is the first request of a conversation, which has nothing to extend, and the
first request after a model switch, which pays residency (`ram residency 5.30s`) and a cold
expert cache on top -- measured at 1.66 tok/s decode while warming, against 45 warm.

## Known open: nondeterminism in the MoE path

Two runs of the same binary at `--temp 0` with the same prompt diverge, and one degenerated
into garbage. It predates this work. Excluded so far: the sampler, OpenMP, atomics (there are
none), the GDN and attention kernels (bit-stable across runs), VRAM allocation, and the expert
cache (still diverges at 0% hit rate). The dense model is deterministic, so it is MoE-specific.
