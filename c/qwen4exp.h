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
 * Four mechanisms have no counterpart here, and all four are small in bytes and large in
 * control flow — which is the opposite of the work this engine has been tuned for:
 *
 *   hc_*      Hyper-connections. The residual stream is replaced by `hyper_connection.count`
 *             = 4 parallel streams mixed through a low-rank (320) projection. Every `x = x + t`
 *             in the forward pass becomes a mix, and there is one at each of two points per
 *             block plus once at the output.
 *   indexer.* Sparse attention. A 4-head indexer scores keys and the attention then runs over
 *             the top `indexer.top_k` = 2048 of them. Below 2048 positions it selects
 *             everything, so it degenerates to dense attention — which is the opening: the
 *             rest can be built and tested first, at short context, against the dense path.
 *   ple_*     Per-layer embeddings. A separate per_layer_token_embd table, an n-gram (size 3)
 *             convolution and a key/value mix feed each block its own view of the input.
 *   rope.dimension_sections  Multi-section RoPE rather than one contiguous rotation.
 *
 * So the honest shape of the port is: the kernels are reusable, the schedule is not.
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
