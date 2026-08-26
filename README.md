# Nightjar

**Large Mixture-of-Experts models on hardware that cannot hold them.** 18.6 GB of weights, an
8 GB card, one laptop — 55 tok/s.

> **This is a fork of [Colibrì](https://github.com/JustVugg/colibri)** by JustVugg, which
> provides the GLM 5.2, Kimi K3, DeepSeek V4 and OLMoE engines and the runtime they sit in.
> Nightjar is the hybrid GDN/attention engine added on top, for the Qwen3.5 family. Upstream's
> documentation is kept verbatim at [docs/README.colibri.md](docs/README.colibri.md) —
> everything it describes still applies.

Pure C and CUDA. No Python at runtime, no BLAS, no inference framework: a GGUF reader, k-quant
kernels, CUDA kernels, a three-tier KV cache, an HTTP server, an agent tool layer and a web UI.

---

## What runs

One server, one UI, a registry of models, one resident at a time. Switching is a re-exec, so
each architecture gets its own engine chosen at startup — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Model | Architecture | Engine | |
|---|---|---|---|
| Qwen 3.8 27B | `qwen35` — dense hybrid GDN/attention | Nightjar | 
| Ornith 1.5 35B-A3B | `qwen35moe` — 256 experts, 8 active | Nightjar | 55 tok/s |
| GLM 5.2 744B | `glm5` | Colibrì | |
| Kimi K3 2.8T | `kimi` | Colibrì | |
| DeepSeek V4 Flash 284B | `dsv4` | Colibrì | |

`nj_arch.h` reads `general.architecture` out of a GGUF header without loading the file, so an
unsupported model is reported by name rather than as a failure.

## Why Nightjar is shaped this way

A sparse model's problem is residency, not arithmetic. A token touches 8 of 256 experts per
layer, so almost everything is dormant at any instant. Nightjar keeps a hot working set in VRAM
under an LRU, streams misses over PCIe on a dedicated copy stream **while the resident experts
are already computing**, and lets the CPU take the overflow when the transfer budget is spent.

Nightjars are among the few birds that enter torpor: nearly everything dormant, only what is
needed awake.

## Building

```sh
make nightjar          # the Nightjar engine and its tests
make nightjar-serve    # web UI + OpenAI-compatible API on :8080
make all               # the Colibrì engines, unchanged
```

Both live in `c/`. Nightjar's rules are in `c/Makefile.nightjar` and its targets are namespaced,
so nothing upstream builds is affected. Models live under `$NIGHTJAR_MODELS`, else `/models`,
else `$HOME/Models` — no path is hardcoded in the tree.

## Measurement

Decode went from 25.6 to 56.5 tok/s, and every step came from measuring rather than guessing —
including four confident theories that measurement killed. The instruments ship with the
engine: `QWEN_CUPTI=1` (GPU timeline with idle attributed to the op that ended it),
`QWEN_HOSTPROF=1` (the host side CUPTI cannot see) and `tests/qwen35_gemv_bench` (kernels at
real shapes, no model load). Numbers and dead theories:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the runtime, the engine interface, the memory hierarchy
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — what it costs and how that was found
- [docs/SERVER.md](docs/SERVER.md) — the API, the tool layer, the UI
- [docs/README.colibri.md](docs/README.colibri.md) — upstream Colibrì's documentation
- [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) — third-party components and their licences

## Licence

See [LICENSE](LICENSE), inherited from Colibrì. Third-party components are listed separately in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md); their weights are not redistributed here.
