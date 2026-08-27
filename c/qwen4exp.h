/* qwen4exp.h — Qwen4 preview ("Qwen3.8 Flash Next", 512x56B), config and binding.
 *
 * WHAT IS THE SAME, AND WHY THAT MATTERS
 *
 * The expensive half of this architecture is the half Nightjar already runs. The trunk is the
 * familiar hybrid — full_attention_interval 4 over 48 blocks, so 12 attention layers and 36
 * recurrent ones — and the FFN is a sparse MoE with a shared expert. Every bandwidth-heavy
 * tensor maps onto a kernel that exists:
 *
 *   attn_q/k/v/output, attn_q_norm/k_norm, attn_gate     -> the GQA path
 *   ssm_conv1d, ssm_a, ssm_dt, ssm_norm, ssm_out, alpha  -> the gated delta net
 *   ffn_{gate,up,down}_exps, ffn_gate_inp, *_shexp       -> the bundled expert kernels
 *
 * WHAT IS NEW IS STRUCTURAL, NOT ARITHMETIC
 *
 * Four mechanisms have no counterpart here. All four are small in bytes and large in control
 * flow, which is the opposite of the work this engine has been tuned for. The descriptions
 * below are taken from the llama.cpp port (ggml-org/llama.cpp#27742), not inferred from the
 * tensor names -- guessing at a formulation from its parameter shapes has a poor record.
 *
 *   hc_*   HYPER-CONNECTIONS. The residual stream is hc_count = 4 times as wide:
 *          n_embd_out_impl = hc_count * n_embd, laid out [n_embd, hc, n_tokens]. Each mixer is
 *          a low-rank down / silu / up sigmoid gate (rank 320) and then collapses the four
 *          streams by a PLAIN MEAN. There is no output_norm in this model at all -- the final
 *          mixer's hc_norm is the last norm before the LM head, which is a thing to get right
 *          rather than discover.
 *
 *   ple_*  PER-LAYER EMBEDDINGS. per_layer_token_embd is a hash table, 51.2 BILLION elements
 *          (97.7 GiB unquantised; it is IQ4_NL in the Q4_K_XL build, which is why that type
 *          had to exist first). Each token is hashed together with its ngram_size-1 = 2
 *          predecessors, host-side, with 64-bit multipliers around 2.4e13 and an xor -- they
 *          do not fit in int32, which is worth knowing before writing the hash. The rows are
 *          gathered and combined as a sum of shifted, per-channel-scaled copies, then run
 *          through ple_conv1d.
 *
 *          Note what this means for streaming: it is a row gather, a few rows per token per
 *          layer out of an enormous table. Random access, but tiny and bounded -- which is
 *          the friendly case for a disk tier, not the hostile one.
 *
 *   indexer.*  SPARSE ATTENTION (QSA). The indexer scores one mean-pooled key per block of
 *          compress_ratio tokens and keeps a budget of indexer.top_k = 2048. The opening is
 *          exact rather than approximate: below indexer_top_k + compress_ratio - 1 cached
 *          tokens EVERY block fits in the budget, so the result is not merely close to dense
 *          attention, it is bit-identical to it. The whole model can therefore be built and
 *          verified at short context against the dense path already in this engine, and the
 *          selection added afterwards.
 *
 *   rope   Interleaved mRoPE, partial rotary 64 of 256.
 *
 * So the honest shape of the port: the kernels are reusable, the schedule is not.
 *
 * This header does the part that can be settled against the file rather than argued about —
 * read the config, bind the tensors, and say precisely what is present and what is missing.
 */
#ifndef QWEN4EXP_H
#define QWEN4EXP_H

#include <stdio.h>
#include <string.h>
#include "gguf.h"

typedef struct {
    int n_layer, d_model, n_head, n_head_kv, d_key, d_value;
    int n_expert, n_expert_used, d_ff_exp, d_ff_shexp;
    int full_attn_interval;          /* every Nth block is full attention */
    int idx_heads, idx_key_len, idx_top_k;
    int hc_count, hc_low_rank;       /* hyper-connections */
    int ple_per_layer, ple_ngram, ple_heads_per_ngram, ple_conv;
    int ssm_conv, ssm_groups, ssm_inner, ssm_state, ssm_dt_rank;
    int n_ctx, rope_dims, rope_sections;
    float rope_base, eps;
    int vocab;
} Q4XCfg;

static int q4x_cfg(const gguf_model *m, Q4XCfg *c){
    const char *arch = gguf_str(m, "general.architecture", "");
    if(strcmp(arch, "qwen4exp") != 0) return 0;
    memset(c, 0, sizeof *c);
    #define U(k) gguf_u64(m, "qwen4exp." k, 0)
    c->n_layer      = (int)U("block_count");
    c->d_model      = (int)U("embedding_length");
    c->n_head       = (int)U("attention.head_count");
    c->n_head_kv    = (int)U("attention.head_count_kv");
    c->d_key        = (int)U("attention.key_length");
    c->d_value      = (int)U("attention.value_length");
    c->n_expert     = (int)U("expert_count");
    c->n_expert_used= (int)U("expert_used_count");
    c->d_ff_exp     = (int)U("expert_feed_forward_length");
    c->d_ff_shexp   = (int)U("expert_shared_feed_forward_length");
    c->full_attn_interval = (int)U("full_attention_interval");
    c->idx_heads    = (int)U("attention.indexer.head_count");
    c->idx_key_len  = (int)U("attention.indexer.key_length");
    c->idx_top_k    = (int)U("attention.indexer.top_k");
    c->hc_count     = (int)U("hyper_connection.count");
    c->hc_low_rank  = (int)U("hyper_connection.low_rank");
    c->ple_per_layer= (int)U("embedding_length_per_layer_input");
    c->ple_ngram    = (int)U("ple.ngram_size");
    c->ple_heads_per_ngram = (int)U("ple.heads_per_ngram");
    c->ple_conv     = (int)U("ple.conv_kernel");
    c->ssm_conv     = (int)U("ssm.conv_kernel");
    c->ssm_groups   = (int)U("ssm.group_count");
    c->ssm_inner    = (int)U("ssm.inner_size");
    c->ssm_state    = (int)U("ssm.state_size");
    c->ssm_dt_rank  = (int)U("ssm.time_step_rank");
    c->n_ctx        = (int)U("context_length");
    c->rope_dims    = (int)U("rope.dimension_count");
    #undef U
    c->rope_base = (float)gguf_f64(m, "qwen4exp.rope.freq_base", 10000.0);
    c->eps       = (float)gguf_f64(m, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-6);
    return c->n_layer > 0 && c->d_model > 0 && c->n_expert > 0;
}

/* What this engine can and cannot run yet, said against the file rather than from memory. */
static void q4x_report(const gguf_model *m, const Q4XCfg *c){
    fprintf(stderr, "  qwen4exp: %d blocks (%d attention, %d recurrent), d_model %d\n",
            c->n_layer, c->full_attn_interval ? c->n_layer/c->full_attn_interval : 0,
            c->full_attn_interval ? c->n_layer - c->n_layer/c->full_attn_interval : c->n_layer,
            c->d_model);
    fprintf(stderr, "            %d experts, %d active, d_ff %d (+shared %d)\n",
            c->n_expert, c->n_expert_used, c->d_ff_exp, c->d_ff_shexp);
    fprintf(stderr, "            context %d, rope base %.0f\n", c->n_ctx, c->rope_base);
    fprintf(stderr, "  not yet implemented:\n");
    fprintf(stderr, "    hyper-connections   %d streams, low rank %d  (replaces every residual add)\n",
            c->hc_count, c->hc_low_rank);
    fprintf(stderr, "    sparse attention    %d indexer heads, top-%d of %d positions\n",
            c->idx_heads, c->idx_top_k, c->n_ctx);
    fprintf(stderr, "                        -- below %d positions it selects everything, so the\n"
                    "                           dense path is correct there and can be built first\n",
            c->idx_top_k);
    fprintf(stderr, "    per-layer embeddings %d dims/layer, %d-gram, conv %d\n",
            c->ple_per_layer, c->ple_ngram, c->ple_conv);
    (void)m;
}

#endif
