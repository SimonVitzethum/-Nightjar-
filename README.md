# Nightjar

**Large Mixture-of-Experts models on hardware that cannot hold them.** 18.6 GB of weights, an
8 GB card, one laptop — 55 tok/s.

Pure C and CUDA. No Python at runtime, no BLAS, no inference framework: a GGUF reader, k-quant
kernels, CUDA kernels, a three-tier KV cache, an HTTP server, an agent tool layer and a web UI.

---

## What it runs

One server, one UI, a registry of models, one resident at a time. Switching is a re-exec, so
each architecture gets its own engine chosen at startup — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Model | Architecture | Engine | Status |
|---|---|---|---|
| Qwen 3.8 27B | `qwen35` — dense hybrid GDN/attention | Nightjar | runs |
| Ornith 1.5 35B-A3B | `qwen35moe` — 256 experts, 8 active | Nightjar | runs, 55 tok/s |
| GLM 5.2 744B | `glm5` | Colibri | dispatch in place, engine to port |
| Kimi K3 2.8T | `kimi` | Colibri | dispatch in place, engine to port |

## Why it is shaped this way

A sparse model's problem is residency, not arithmetic. A token touches 8 of 256 experts per
layer, so almost everything is dormant at any instant. Nightjar keeps a hot working set in VRAM
under an LRU, streams misses over PCIe on a dedicated copy stream **while the resident experts
are already computing**, and lets the CPU take the overflow when the transfer budget is spent.
Weights are read in place from a memory-mapped GGUF.

Nightjars are among the few birds that enter torpor: nearly everything dormant, only what is
needed awake.

## Getting started

```sh
make -C c qwen35_run qwen35_server      # CUDA; add HAVE_CUDA=0 for CPU-only
make -C c serve                         # web UI + OpenAI-compatible API on :8080
```

Models live under `$NIGHTJAR_MODELS`, else `/models`, else `$HOME/Models`. Nothing in the tree
hardcodes a path.

## Measurement

Decode went from 25.6 to 56.5 tok/s, and every step came from measuring rather than guessing —
including four confident theories that measurement killed. The instruments ship with the
engine: `QWEN_CUPTI=1` (GPU timeline with idle attributed to the op that ended it),
`QWEN_HOSTPROF=1` (the host side CUPTI cannot see), and `tests/qwen35_gemv_bench` (kernels at
real shapes, no model load). The numbers and the four dead theories:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the runtime, the engine interface, the memory hierarchy
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — what it costs and how that was found
- [docs/SERVER.md](docs/SERVER.md) — the API, the tool layer, the UI
- [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) — third-party components and their licences

## Licence

See [LICENSE](LICENSE). Third-party components are listed separately in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md); their weights are not redistributed here.
