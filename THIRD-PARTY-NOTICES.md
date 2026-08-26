# Third-party components

## MTP draft head — `shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`

Ornith-1.5-35B-A3B ships `mtp.*` tensors, but they are randomly initialised rather than
trained: every projection has std ≈ 0.0200 and the norm weights sit near 0.02 instead of ~1.
Used as a speculative draft they accept ~13% of tokens, which is chance. See the upstream
report at <https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/discussions/10>. Our GGUF build
does not contain them at all (`mtp_included: false` in the conversion manifest), so the engine
has no usable built-in draft head.

This engine can optionally load a **trained replacement** head:

| | |
|---|---|
| Source | <https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY> |
| File | `model-mtp.safetensors` — 844.6M parameters, 19 fused BF16 tensors, 1.689 GB |
| Author | Shisa.AI |
| License | Apache License 2.0 (see `licenses/LICENSE-shisa-mtp-head.txt`) |
| Initialised from | `Qwen/Qwen3.6-35B-A3B` MTP head — Apache 2.0, <https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE> |
| Method credit | protoLabsAI, <https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP> — the result that full-vocabulary KL beats hard CE for speculative acceptance |

The head is a derivative of Qwen's MTP component, re-aligned to Ornith-1.5's hidden states by
full-vocabulary KL distillation. Reported acceptance: **69.27% on code, 3.078 accepted tokens
per step**; 60.51% on a mixed suite.

**The weights are NOT redistributed in this repository.** `make mtp-head` fetches them from the
source above at the user's request. Apache 2.0 requires that the licence text and attribution
travel with any redistribution, which is what this file and `licenses/` exist for.

### Caveat this engine must measure, not assume

Our target is the **abliterated** conversion. The head was distilled against the hidden states
of the *base* Ornith-1.5, and abliteration projects a refusal direction out of the weights,
which moves the residual stream the head was aligned to. The published 69.27% is therefore an
upper bound here, not a prediction. Acceptance is measured per workload before the head is
trusted — see `QWEN_SPEC_STAT`.
