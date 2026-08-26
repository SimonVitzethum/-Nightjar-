# How the runtime is put together

One server, one web UI, a registry of models, **one resident at a time**. That last constraint
is not a limitation to work around — it is what keeps the design small.

## Why there is no plugin system

Models do not fit in RAM together, so only one is ever loaded. Switching is a re-exec:

    /v1/model/select  ->  reply sent first  ->  execv("/proc/self/exe", argv)

with the listening socket inherited across the exec, so open connections survive and the
client sees a reconnect rather than a dead port. The consequence is the important part:
**two engines never share an address space.** A new architecture therefore needs a dispatch at
*startup*, not runtime polymorphism, not dynamic loading, not a common object model threaded
through every call site.

    nj_arch_of(path)      read general.architecture from the GGUF header, without loading
       |
       +-- qwen35, qwen35moe   -> the Nightjar engine (this tree)
       +-- glm5, kimi, dsv4    -> the Colibri engines
       +-- unknown             -> refuse with the architecture name, not a crash

## The engine interface

Narrow on purpose — an engine is not asked to be generic, only to answer five questions:

| | |
|---|---|
| `load(path, ctx)` | mmap the GGUF, build device state, report VRAM and residency |
| `prefill(toks, n)` | run the prompt, leave the KV and recurrent state at position n |
| `decode(tok, pos, logits)` | one token |
| `hidden(out)` | the residual stream, for a draft head |
| `free()` | release |

`kv_tier.h` (the three-tier KV cache) and `harness.h` (tools, policy, output budget) are
already model-agnostic and sit above this line. They do not need touching for a new engine.

## The MoE memory hierarchy

For a sparse model the central problem is residency, not arithmetic. A token touches 8 of 256
experts per layer, so almost everything is dormant:

    VRAM cache (LRU)  --hit-->  compute in place
         |miss
         v
    host RAM (pinned, mmapped GGUF)  --DMA on the copy stream-->  VRAM slot
         |budget spent
         v
    CPU computes the expert in place, result added on the GPU

Hits are computed **first**, deliberately: the misses are still crossing PCIe while the
resident experts run, which is the only window the transfer can hide in. See
`q35cu_moe_layer` and `docs/PERFORMANCE.md`.

## Where models live

`$NIGHTJAR_MODELS`, else `/models`, else `$HOME/Models` — resolved identically by `models.h`,
the Makefile and `serve.sh`, so there is one place to change and no path in a source file.
