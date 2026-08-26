# Nightjar

An inference engine in C and CUDA for running large Mixture-of-Experts models on hardware that
cannot hold them: 18.6 GB of weights, an 8 GB card, one laptop.

Nightjar is the hybrid-MoE engine of a multi-model runtime. One server, one web UI, a registry
of GGUFs, one resident at a time -- switching is a re-exec, so each architecture gets its own
engine chosen at startup rather than a universal loader that fits none of them well. Today that
covers the Qwen3.5 family, dense and MoE: 40 layers, 30 gated-delta-net and 10 grouped-query
attention, 256 experts with 8 active per token. GLM-5.2 and Kimi K3 are the next engines behind
the same registry.

Almost all of an MoE is dormant at any instant, so residency is the central problem. A hot
working set lives in VRAM under an LRU, misses stream over PCIe on a dedicated copy stream
while the resident experts are already computing, and the CPU takes the overflow when the
transfer budget is spent. Weights are read in place from a memory-mapped GGUF, with no
dependencies beyond CUDA and libc.

It decodes a 35B-A3B model at **55 tok/s** on an RTX 5070 Laptop at 92% GPU occupancy, up from
25 when the work started -- and every step came from measurement. The engine ships the
instruments it was tuned with:

- `QWEN_CUPTI=1` -- an exact GPU timeline, with the idle gap before each operation attributed
  to the op that ended it. It found the largest item in the decode budget: not a kernel, but
  the sampler, sorting a 248320-entry vocabulary per token to keep twenty candidates.
- `QWEN_HOSTPROF=1` -- the host side, which CUPTI cannot see. It is why there are no CUDA
  Graphs here: the host already spends 14 of every 18 ms blocked on the GPU.
- `tests/qwen35_gemv_bench` -- decode kernels at their real shapes on synthetic weights, against
  the CPU dequant. It measures what the card delivers, not the datasheet: a factor of two apart.

Speculative decoding is included, off by default: it measures break-even on a recurrent
architecture. Third-party components and licences: `THIRD-PARTY-NOTICES.md`.
