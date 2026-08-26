# Nightjar

An inference engine in C and CUDA for hybrid Mixture-of-Experts models, written for the case
where the model does not fit: 18.6 GB of weights, an 8 GB card, one laptop.

Nightjar runs Qwen3.5-architecture MoE models -- 40 layers, 30 gated-delta-net and 10
grouped-query attention, 256 experts with 8 active per token. Almost all of that is dormant at
any instant, so the engine treats residency as the central problem: a hot working set lives in
VRAM under an LRU, misses stream over PCIe on a dedicated copy stream while the resident
experts are already computing, and the CPU takes the overflow when the transfer budget is
spent. Weights are read in place from a memory-mapped GGUF, with no dependencies beyond CUDA
and libc.

It decodes a 35B-A3B model at **55 tok/s** on an RTX 5070 Laptop at 92% GPU occupancy, up from
25 when the work started -- and every step came from measurement rather than intuition. The
engine ships the instruments it was tuned with:

- `QWEN_CUPTI=1` gives an exact GPU timeline, including the idle gap before every operation,
  attributed to the op that ended it. That is what found the largest item in the decode
  budget: not a kernel, but the sampler, sorting a 248320-entry vocabulary once per token to
  keep twenty candidates.
- `QWEN_HOSTPROF=1` covers the host side, which CUPTI cannot see. It is why there are no CUDA
  Graphs here -- the host already spends 14 of every 18 ms blocked waiting for the GPU, so
  capturing its launches would return a resource that is idle.
- `tests/qwen35_gemv_bench` runs the decode kernels at their real shapes on synthetic weights,
  checked against the CPU dequant. It measures what the card delivers rather than what the
  datasheet claims -- here, a factor of two apart.

Speculative decoding is included -- a batched MoE verify path and an n-gram drafter,
switchable per request. Off by default: it measures break-even here, because acceptance is
good but the recurrent state update does not amortise across a draft window.

Third-party components and their licences are listed in `THIRD-PARTY-NOTICES.md`.
