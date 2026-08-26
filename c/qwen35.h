#ifndef QWEN_QWEN35_H
#define QWEN_QWEN35_H
/* qwen35.h — Qwen3.5 (arch "qwen35") geometry, tensor binding and placement plan.
 *
 * WHY THIS ARCH NEEDED A NEW FRONT-END, AND WHY THE OLD ENGINE STILL APPLIES
 *
 * colibri was built for GLM-5.2/DeepSeek: MLA attention + 256 routed experts streamed off
 * NVMe. Qwen3.5-27B has NEITHER. It is a HYBRID:
 *
 *     64 trunk layers, of which
 *       48 are Gated DeltaNet ("linear attention") — a RECURRENT layer whose state is
 *          O(1) in sequence length: 48 v-heads x 128x128 fp32 = 3 MB per layer, forever.
 *       16 are full GQA attention (every 4th: (i+1) % 4 == 0) — these are the ONLY layers
 *          that own a KV cache.
 *     + 1 MTP/nextn block (blk.64), skipped unless speculative decoding is on.
 *
 * That 48/16 split is the whole reason a 250k context is affordable on 8 GB of VRAM: a
 * dense 64-layer model would want 260 KB/token of KV, this wants 34 KB/token at q8_0, so
 * 250k tokens cost 8.1 GiB instead of 62.
 *
 * WHAT MAPS ONTO WHAT (measured from the real file, see qwen35_test):
 *
 *     ffn (64 layers)     9.650 GiB   <- the "expert" mass. RAM-resident, computed on the
 *                                        CPU, exactly the moe_cpu.h principle: it is
 *                                        ALREADY in RAM, so pushing it over PCIe to be
 *                                        computed on the GPU is pure overhead.
 *     gdn/ssm (48)        3.224 GiB   \
 *     attn (16)           0.890 GiB    >  5.085 GiB VRAM-resident on the 5070. Small,
 *     output              0.971 GiB   /   hot every token, and the GDN recurrence is a
 *                                        latency chain the CPU is bad at.
 *     token_embd          0.666 GiB   <- RAM (a gather, one row per token)
 *     mtp                 0.245 GiB   <- skipped
 *
 * The GDN layers are NOT a rounding error (3.2 GiB, 63% of the VRAM budget) — but they are
 * read once per token either way, and keeping them resident is what makes the 48 linear
 * layers cost ~0 PCIe. Only the KV stream then crosses the bus, which is the one thing that
 * GROWS with context and therefore the one thing worth spending bus bandwidth on.
 *
 * THE THREE THINGS THAT ARE NOT GUESSABLE FROM THE TENSOR SHAPES:
 *
 *   attn_q is [n_embd, n_head * head_dim * 2] — the second half is a per-head GATE, applied
 *   as sigmoid(gate) * attn_out BEFORE wo. Reading it as a 2x-wide Q gives fluent garbage.
 *
 *   ssm_a already holds -exp(A_log), not A_log. The decay is exp(softplus(alpha + dt_bias)
 *   * ssm_a), which is in (0,1] only because ssm_a is negative.
 *
 *   q/k inside a GDN layer are L2-normalized (x / sqrt(sum x^2 + eps)), NOT RMS-normalized.
 *   The two differ by sqrt(d) and the model will not notice loudly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#include "gguf.h"
#include "kquant.h"

#define Q35_MAX_LAYER 128

enum { Q35_LAYER_GDN = 0, Q35_LAYER_ATTN = 1, Q35_LAYER_MTP = 2 };

typedef struct {
    /* trunk */
    int   n_layer;         /* 64  — trunk layers actually executed */
    int   n_layer_all;     /* 65  — including the MTP block */
    int   n_layer_mtp;     /* 1 */
    int   d_model;         /* 5120 */
    int   d_ff;            /* 17408 */
    int   vocab;           /* 248320 */
    float eps;             /* 1e-6 */

    /* full attention (16 layers) */
    int   n_head;          /* 24 */
    int   n_head_kv;       /* 4 */
    int   d_head;          /* 256 (key_length == value_length) */
    int   n_rot;           /* 64 — PARTIAL rope: only the first 64 of 256 dims */
    int   rope_sect[4];    /* [11,11,10,0] mrope sections; text-only => all pos equal */
    float rope_base;       /* 1e7 */
    int   full_attn_iv;    /* 4 */

    /* gated delta net (48 layers) */
    int   d_conv;          /* 4  */
    int   d_inner;         /* 6144 */
    int   d_state;         /* 128 — head_k_dim == head_v_dim */
    int   n_k_heads;       /* 16 (ssm.group_count) */
    int   n_v_heads;       /* 48 (ssm.time_step_rank) */
    int   d_head_v;        /* d_inner / n_v_heads = 128 */
    int   conv_dim;        /* 2*d_state*n_k_heads + d_inner = 10240 */

    /* sampling defaults shipped in the file */
    float top_p, temp; int top_k;

    /* MoE (qwen35moe / Ornith). is_moe=0 => dense qwen35 and every field below is 0.
     * The routed FFN replaces the dense one: n_expert experts of width d_ff_exp, n_expert_used
     * of them per token, plus one always-on shared expert of width d_ff_shexp. */
    int   is_moe;
    int   n_expert;        /* 256 */
    int   n_expert_used;   /* 8   */
    int   d_ff_exp;        /* expert_feed_forward_length = 512 */
    int   d_ff_shexp;      /* expert_shared_feed_forward_length = 512 */

    int   kind[Q35_MAX_LAYER];   /* Q35_LAYER_* per block index */
} Q35Cfg;

typedef struct {
    /* shared by both layer kinds */
    const float *attn_norm;          /* [d_model] */
    const float *post_attn_norm;     /* [d_model] */
    const gguf_tensor *ffn_gate, *ffn_up, *ffn_down;   /* dense FFN (is_moe==0) */

    /* MoE FFN (is_moe==1). The experts are stacked 3-D tensors [., ., n_expert]; a per-expert
     * 2-D slice is contiguous, so kq_gemv_fast reads it straight out of the mapping. */
    const gguf_tensor *ffn_gate_inp;                                  /* [d_model, n_expert] F32 router */
    const gguf_tensor *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;   /* [d_model,d_ff_exp,E] / down [d_ff_exp,d_model,E] */
    const gguf_tensor *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp;/* the shared expert, dense */
    const float       *ffn_gate_inp_shexp;                            /* [d_model] F32 — sigmoid gate on the shared expert */

    /* full-attention layers */
    const gguf_tensor *wq, *wk, *wv, *wo;      /* wq is [d_model, n_head*d_head*2] */
    const float *q_norm, *k_norm;              /* [d_head] */

    /* gdn layers */
    const gguf_tensor *wqkv;         /* [d_model, conv_dim] */
    const gguf_tensor *wgate;        /* [d_model, d_inner] — the z gate */
    const gguf_tensor *w_alpha, *w_beta;   /* [d_model, n_v_heads] */
    const gguf_tensor *w_out;        /* [d_inner, d_model] */
    const float *conv1d;             /* [d_conv, conv_dim] f32 */
    const float *ssm_a;              /* [n_v_heads] — already -exp(A_log) */
    const float *dt_bias;            /* [n_v_heads] */
    const float *ssm_norm;           /* [d_head_v] */

    /* MTP / nextn block (blk.64 only). A DRAFT HEAD TRAINED WITH THE MODEL: it predicts
     * token t+2 from the main model's hidden state at t+1 plus that token's embedding, for
     * one layer's worth of work instead of 64. Speculative decoding with it needs no second
     * model and no extra weights — they are already in the file. */
    const gguf_tensor *eh_proj;      /* [2*d_model, d_model] */
    const float *enorm, *hnorm, *head_norm;

    int kind;
} Q35Layer;

typedef struct {
    gguf_model  m;
    Q35Cfg      c;
    Q35Layer    L[Q35_MAX_LAYER];
    const gguf_tensor *tok_embd, *output;
    const float *output_norm;
    int         opened;

    /* Relocated weights: tensor index -> a private anonymous copy, or NULL for "still in the
     * mmap". See q35_reside_anon. */
    uint8_t   **tw;
    uint8_t    *anon;
    int64_t     anon_bytes;
} Q35Model;

/* Where a tensor's bytes actually live right now. */
static inline const void *q35_wdata(const Q35Model *M, const gguf_tensor *T){
    if(M->tw){
        const int idx = (int)(T - M->m.t);
        if(idx >= 0 && idx < M->m.n_tensors && M->tw[idx]) return M->tw[idx];
    }
    return gguf_data(&M->m, T);
}

/* ---------------- binding ---------------- */

static const float *q35_f32(const gguf_model *m, const char *fmt, int il){
    char nm[128];
    if(il >= 0) snprintf(nm, sizeof nm, fmt, il); else snprintf(nm, sizeof nm, "%s", fmt);
    const gguf_tensor *t = gguf_find(m, nm);
    if(!t || t->type != KQ_F32) return NULL;
    return (const float*)gguf_data(m, t);
}
static const gguf_tensor *q35_t(const gguf_model *m, const char *fmt, int il){
    char nm[128];
    if(il >= 0) snprintf(nm, sizeof nm, fmt, il); else snprintf(nm, sizeof nm, "%s", fmt);
    return gguf_find(m, nm);
}

/* A tensor is bound only if it exists AND has the shape the geometry implies. Every silent
 * failure on this project started with a tensor that was present but the wrong shape. */
static int q35_want(const gguf_tensor *t, const char *nm, int64_t ne0, int64_t ne1){
    if(!t){ fprintf(stderr, "qwen35: missing tensor %s\n", nm); return 0; }
    if(t->ne[0] != ne0 || (ne1 >= 0 && t->ne[1] != ne1)){
        fprintf(stderr, "qwen35: %s has ne=[%lld,%lld], expected [%lld,%lld]\n",
                nm, (long long)t->ne[0], (long long)t->ne[1], (long long)ne0, (long long)ne1);
        return 0;
    }
    return 1;
}

/* Like q35_want, but for a stacked expert tensor: ne=[ne0, ne1, n_expert]. */
static int q35_want3(const gguf_tensor *t, const char *nm, int64_t ne0, int64_t ne1, int64_t ne2){
    if(!t){ fprintf(stderr, "qwen35: missing tensor %s\n", nm); return 0; }
    if(t->n_dims != 3 || t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != ne2){
        fprintf(stderr, "qwen35: %s has ne=[%lld,%lld,%lld], expected [%lld,%lld,%lld]\n", nm,
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                (long long)ne0, (long long)ne1, (long long)ne2);
        return 0;
    }
    return 1;
}

static int q35_cfg(Q35Model *M){
    const gguf_model *m = &M->m;
    Q35Cfg *c = &M->c;
    const char *arch = gguf_str(m, "general.architecture", "");
    /* One engine, two Qwen3.5 architectures: the dense trunk ("qwen35") and the MoE
     * ("qwen35moe", e.g. Ornith). They share every attention/GDN tensor and differ only in the
     * FFN, so the whole config below is identical bar the metadata PREFIX and the four MoE keys.
     * llama.cpp prefixes each key with the architecture name, so the prefix is the arch. */
    c->is_moe = !strcmp(arch, "qwen35moe");
    if(strcmp(arch, "qwen35") != 0 && !c->is_moe){
        fprintf(stderr, "qwen35: architecture is \"%s\", not qwen35 or qwen35moe\n", arch);
        return 0;
    }
    const char *P = c->is_moe ? "qwen35moe" : "qwen35";
    char kb[96];
    #define K(suf) (snprintf(kb, sizeof kb, "%s.%s", P, suf), kb)
    c->n_layer_all = (int)gguf_u64(m, K("block_count"), 0);
    c->n_layer_mtp = (int)gguf_u64(m, K("nextn_predict_layers"), 0);
    c->n_layer     = c->n_layer_all - c->n_layer_mtp;
    c->d_model     = (int)gguf_u64(m, K("embedding_length"), 0);
    c->d_ff        = (int)gguf_u64(m, K("feed_forward_length"), 0);
    c->eps         = (float)gguf_f64(m, K("attention.layer_norm_rms_epsilon"), 1e-6);

    c->n_head      = (int)gguf_u64(m, K("attention.head_count"), 0);
    c->n_head_kv   = (int)gguf_u64(m, K("attention.head_count_kv"), 0);
    c->d_head      = (int)gguf_u64(m, K("attention.key_length"), 0);
    c->n_rot       = (int)gguf_u64(m, K("rope.dimension_count"), 0);
    c->rope_base   = (float)gguf_f64(m, K("rope.freq_base"), 10000.0);
    c->full_attn_iv= (int)gguf_u64(m, K("full_attention_interval"), 4);

    c->d_conv      = (int)gguf_u64(m, K("ssm.conv_kernel"), 0);
    c->d_inner     = (int)gguf_u64(m, K("ssm.inner_size"), 0);
    c->d_state     = (int)gguf_u64(m, K("ssm.state_size"), 0);
    c->n_k_heads   = (int)gguf_u64(m, K("ssm.group_count"), 0);
    c->n_v_heads   = (int)gguf_u64(m, K("ssm.time_step_rank"), 0);

    if(c->is_moe){
        c->n_expert      = (int)gguf_u64(m, K("expert_count"), 0);
        c->n_expert_used = (int)gguf_u64(m, K("expert_used_count"), 0);
        c->d_ff_exp      = (int)gguf_u64(m, K("expert_feed_forward_length"), 0);
        c->d_ff_shexp    = (int)gguf_u64(m, K("expert_shared_feed_forward_length"), 0);
        if(c->n_expert <= 0 || c->n_expert_used <= 0 || c->d_ff_exp <= 0){
            fprintf(stderr, "qwen35moe: bad expert config E=%d used=%d ff=%d\n",
                    c->n_expert, c->n_expert_used, c->d_ff_exp); return 0;
        }
    }
    #undef K

    c->top_k = (int)gguf_u64(m, "general.sampling.top_k", 20);
    c->top_p = (float)gguf_f64(m, "general.sampling.top_p", 0.95);
    c->temp  = (float)gguf_f64(m, "general.sampling.temp", 1.0);

    const int32_t *sect = NULL;
    uint64_t ns = gguf_i32s(m, c->is_moe ? "qwen35moe.rope.dimension_sections"
                                          : "qwen35.rope.dimension_sections", &sect);
    for(int i = 0; i < 4; i++) c->rope_sect[i] = (i < (int)ns && sect) ? sect[i] : 0;

    const gguf_tensor *te = gguf_find(m, "token_embd.weight");
    c->vocab = te ? (int)te->ne[1] : 0;

    if(c->n_v_heads <= 0 || c->d_inner % c->n_v_heads != 0){
        fprintf(stderr, "qwen35: d_inner %d not divisible by n_v_heads %d\n",
                c->d_inner, c->n_v_heads); return 0;
    }
    c->d_head_v = c->d_inner / c->n_v_heads;
    c->conv_dim = 2 * c->d_state * c->n_k_heads + c->d_inner;

    if(c->n_layer_all > Q35_MAX_LAYER){
        fprintf(stderr, "qwen35: %d layers exceeds Q35_MAX_LAYER\n", c->n_layer_all); return 0;
    }
    /* llama.cpp: is_recurrent(i) = i < n_layer && (i+1) % full_attention_interval != 0.
     * The MTP block sits past n_layer and is a full-attention block. */
    for(int i = 0; i < c->n_layer_all; i++)
        c->kind[i] = (i >= c->n_layer) ? Q35_LAYER_MTP
                   : (((i + 1) % c->full_attn_iv) != 0 ? Q35_LAYER_GDN : Q35_LAYER_ATTN);

    /* the rope sections must cover exactly n_rot/2 pairs, or mrope silently rotates the
     * wrong dims and long context degrades in a way that looks like a sampling problem */
    int sum = c->rope_sect[0] + c->rope_sect[1] + c->rope_sect[2] + c->rope_sect[3];
    if(sum != 0 && sum * 2 != c->n_rot){
        fprintf(stderr, "qwen35: rope sections sum %d, expected n_rot/2 = %d\n", sum, c->n_rot/2);
        return 0;
    }
    return 1;
}

static int q35_bind(Q35Model *M){
    const gguf_model *m = &M->m;
    const Q35Cfg *c = &M->c;
    int ok = 1;

    M->tok_embd    = q35_t(m, "token_embd.weight", -1);
    M->output      = q35_t(m, "output.weight", -1);
    M->output_norm = q35_f32(m, "output_norm.weight", -1);
    ok &= q35_want(M->tok_embd, "token_embd.weight", c->d_model, c->vocab);
    ok &= q35_want(M->output,   "output.weight",     c->d_model, c->vocab);
    if(!M->output_norm){ fprintf(stderr, "qwen35: missing output_norm.weight\n"); ok = 0; }

    const int q_gate_dim= c->n_head * c->d_head * 2;     /* 12288 */
    const int kv_dim    = c->n_head_kv * c->d_head;      /* 1024  */
    char nm[128];

    for(int il = 0; il < c->n_layer_all; il++){
        Q35Layer *L = &M->L[il];
        memset(L, 0, sizeof *L);
        L->kind = c->kind[il];

        L->attn_norm      = q35_f32(m, "blk.%d.attn_norm.weight", il);
        L->post_attn_norm = q35_f32(m, "blk.%d.post_attention_norm.weight", il);
        if(!L->attn_norm || !L->post_attn_norm){
            fprintf(stderr, "qwen35: layer %d missing a norm\n", il); ok = 0;
        }
        if(c->is_moe){
            const int E = c->n_expert, F = c->d_ff_exp, Fs = c->d_ff_shexp;
            L->ffn_gate_inp   = q35_t(m, "blk.%d.ffn_gate_inp.weight",   il);
            L->ffn_gate_exps  = q35_t(m, "blk.%d.ffn_gate_exps.weight",  il);
            L->ffn_up_exps    = q35_t(m, "blk.%d.ffn_up_exps.weight",    il);
            L->ffn_down_exps  = q35_t(m, "blk.%d.ffn_down_exps.weight",  il);
            L->ffn_gate_shexp = q35_t(m, "blk.%d.ffn_gate_shexp.weight", il);
            L->ffn_up_shexp   = q35_t(m, "blk.%d.ffn_up_shexp.weight",   il);
            L->ffn_down_shexp = q35_t(m, "blk.%d.ffn_down_shexp.weight", il);
            L->ffn_gate_inp_shexp = q35_f32(m, "blk.%d.ffn_gate_inp_shexp.weight", il);
            snprintf(nm,sizeof nm,"blk.%d.ffn_gate_inp",  il); ok &= q35_want (L->ffn_gate_inp,  nm, c->d_model, E);
            snprintf(nm,sizeof nm,"blk.%d.ffn_gate_exps", il); ok &= q35_want3(L->ffn_gate_exps, nm, c->d_model, F, E);
            snprintf(nm,sizeof nm,"blk.%d.ffn_up_exps",   il); ok &= q35_want3(L->ffn_up_exps,   nm, c->d_model, F, E);
            snprintf(nm,sizeof nm,"blk.%d.ffn_down_exps", il); ok &= q35_want3(L->ffn_down_exps, nm, F, c->d_model, E);
            snprintf(nm,sizeof nm,"blk.%d.ffn_gate_shexp",il); ok &= q35_want (L->ffn_gate_shexp,nm, c->d_model, Fs);
            snprintf(nm,sizeof nm,"blk.%d.ffn_up_shexp",  il); ok &= q35_want (L->ffn_up_shexp,  nm, c->d_model, Fs);
            snprintf(nm,sizeof nm,"blk.%d.ffn_down_shexp",il); ok &= q35_want (L->ffn_down_shexp,nm, Fs, c->d_model);
            if(!L->ffn_gate_inp_shexp){ fprintf(stderr,"qwen35moe: layer %d missing ffn_gate_inp_shexp\n", il); ok = 0; }
        } else {
        L->ffn_gate = q35_t(m, "blk.%d.ffn_gate.weight", il);
        L->ffn_up   = q35_t(m, "blk.%d.ffn_up.weight",   il);
        L->ffn_down = q35_t(m, "blk.%d.ffn_down.weight", il);
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate", il); ok &= q35_want(L->ffn_gate, nm, c->d_model, c->d_ff);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up",   il); ok &= q35_want(L->ffn_up,   nm, c->d_model, c->d_ff);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down", il); ok &= q35_want(L->ffn_down, nm, c->d_ff, c->d_model);
        }

        if(L->kind == Q35_LAYER_GDN){
            L->wqkv    = q35_t(m, "blk.%d.attn_qkv.weight",   il);
            L->wgate   = q35_t(m, "blk.%d.attn_gate.weight",  il);
            L->w_alpha = q35_t(m, "blk.%d.ssm_alpha.weight",  il);
            L->w_beta  = q35_t(m, "blk.%d.ssm_beta.weight",   il);
            L->w_out   = q35_t(m, "blk.%d.ssm_out.weight",    il);
            L->conv1d  = q35_f32(m, "blk.%d.ssm_conv1d.weight", il);
            L->ssm_a   = q35_f32(m, "blk.%d.ssm_a", il);
            L->dt_bias = q35_f32(m, "blk.%d.ssm_dt.bias", il);
            L->ssm_norm= q35_f32(m, "blk.%d.ssm_norm.weight", il);
            snprintf(nm,sizeof nm,"blk.%d.attn_qkv", il);  ok &= q35_want(L->wqkv,   nm, c->d_model, c->conv_dim);
            snprintf(nm,sizeof nm,"blk.%d.attn_gate",il);  ok &= q35_want(L->wgate,  nm, c->d_model, c->d_inner);
            snprintf(nm,sizeof nm,"blk.%d.ssm_alpha",il);  ok &= q35_want(L->w_alpha,nm, c->d_model, c->n_v_heads);
            snprintf(nm,sizeof nm,"blk.%d.ssm_beta", il);  ok &= q35_want(L->w_beta, nm, c->d_model, c->n_v_heads);
            snprintf(nm,sizeof nm,"blk.%d.ssm_out",  il);  ok &= q35_want(L->w_out,  nm, c->d_inner, c->d_model);
            if(!L->conv1d || !L->ssm_a || !L->dt_bias || !L->ssm_norm){
                fprintf(stderr, "qwen35: layer %d missing an ssm f32 tensor\n", il); ok = 0;
            }
            const gguf_tensor *cv = q35_t(m, "blk.%d.ssm_conv1d.weight", il);
            snprintf(nm,sizeof nm,"blk.%d.ssm_conv1d", il); ok &= q35_want(cv, nm, c->d_conv, c->conv_dim);
        } else {
            L->wq = q35_t(m, "blk.%d.attn_q.weight", il);
            L->wk = q35_t(m, "blk.%d.attn_k.weight", il);
            L->wv = q35_t(m, "blk.%d.attn_v.weight", il);
            L->wo = q35_t(m, "blk.%d.attn_output.weight", il);
            if(L->kind == Q35_LAYER_MTP){
                L->eh_proj   = q35_t(m, "blk.%d.nextn.eh_proj.weight", il);
                L->enorm     = q35_f32(m, "blk.%d.nextn.enorm.weight", il);
                L->hnorm     = q35_f32(m, "blk.%d.nextn.hnorm.weight", il);
                L->head_norm = q35_f32(m, "blk.%d.nextn.shared_head_norm.weight", il);
                snprintf(nm,sizeof nm,"blk.%d.nextn.eh_proj", il);
                ok &= q35_want(L->eh_proj, nm, 2LL*c->d_model, c->d_model);
                if(!L->enorm || !L->hnorm){
                    fprintf(stderr, "qwen35: MTP block %d missing enorm/hnorm\n", il); ok = 0;
                }
            }
            L->q_norm = q35_f32(m, "blk.%d.attn_q_norm.weight", il);
            L->k_norm = q35_f32(m, "blk.%d.attn_k_norm.weight", il);
            snprintf(nm,sizeof nm,"blk.%d.attn_q",     il); ok &= q35_want(L->wq, nm, c->d_model, q_gate_dim);
            snprintf(nm,sizeof nm,"blk.%d.attn_k",     il); ok &= q35_want(L->wk, nm, c->d_model, kv_dim);
            snprintf(nm,sizeof nm,"blk.%d.attn_v",     il); ok &= q35_want(L->wv, nm, c->d_model, kv_dim);
            snprintf(nm,sizeof nm,"blk.%d.attn_output",il); ok &= q35_want(L->wo, nm, c->n_head*c->d_head, c->d_model);
            if(!L->q_norm || !L->k_norm){
                fprintf(stderr, "qwen35: layer %d missing q/k norm\n", il); ok = 0;
            }
        }
    }
    return ok;
}

static int q35_open(Q35Model *M, const char *path){
    memset(M, 0, sizeof *M);
    if(!gguf_open(&M->m, path)) return 0;
    M->opened = 1;
    if(!q35_cfg(M))  return 0;
    if(!q35_bind(M)) return 0;
    return 1;
}

static void q35_close(Q35Model *M){ if(M->opened) gguf_close(&M->m); M->opened = 0; }

/* ---------------- residency ----------------
 * THE MODEL MUST NOT RUN FROM DISK. It is mmapped, which makes it page-cache backed, which
 * makes it EVICTABLE — and an evicted FFN is not a slow FFN, it is a different machine. This
 * was measured, not theorized: after an unrelated 8 GiB of IO pushed the model out of the
 * page cache, the FFN gemv fell from 33.7 GB/s to 2.67 GB/s and the token went from 0.5 s to
 * 5.4 s. Same code, same weights, 10x slower, and nothing in the process reports why.
 *
 * So residency is made explicit rather than hoped for:
 *   1. MADV_WILLNEED so the kernel reads it back sequentially instead of by page fault,
 *   2. a touch of every page so "will need" becomes "is here",
 *   3. mlock where the rlimit allows, so nothing can take it away again,
 *   4. mincore afterwards, so the answer is measured and not assumed.
 *
 * Step 4 is the one that matters. Without it this is three syscalls that might have done
 * nothing, which is exactly the failure it is meant to prevent. */
typedef struct { int64_t want, resident, locked, huge; double secs; } Q35Resident;

/* Duplicated from qwen35_plan.h on purpose: residency must be able to refuse an allocation
 * without dragging the planner in, and the planner needs the same two numbers. */
static int64_t q35_meminfo_kb(const char *key){
    FILE *f = fopen("/proc/meminfo", "r");
    if(!f) return 0;
    char k[64]; long v; char u[16]; int64_t out = 0;
    while(fscanf(f, "%63s %ld %15s", k, &v, u) >= 2) if(!strcmp(k, key)){ out = (int64_t)v*1024; break; }
    fclose(f);
    return out;
}
static int64_t q35_ram_avail_bytes(void){ return q35_meminfo_kb("MemAvailable:"); }

/* ---------------- the memory floor ----------------
 * The context is NOT capped at a number. It grows for as long as the machine can carry it and
 * spills to disk after that, so the only real limit is the drive. What must never happen is
 * the engine allocating its way into an OOM kill: anonymous pages cannot be reclaimed, so the
 * kernel's only recourse is to shoot something, and on a desktop that something is usually
 * not this process.
 *
 * Two rules, both hard:
 *   1. Stop GROWING RAM below QWEN_MEM_FLOOR_GB (default 2). Everything past that point
 *      goes to the disk tier, which is why the disk tier exists.
 *   2. If an allocation actually FAILS, stop. Do not retry, do not fall back to a smaller
 *      request, do not free something and try again — by then the machine is already in
 *      trouble and another attempt just makes this process the one that tips it over.
 *      Exit, and say what was being asked for.
 */
static int64_t q35_mem_floor(void){
    const char *e = getenv("QWEN_MEM_FLOOR_GB");
    double gb = e ? atof(e) : 2.0;
    if(gb < 0.25) gb = 0.25;
    return (int64_t)(gb*(1024.0*1024.0*1024.0));
}

/* Is there room to grow by `bytes` and still leave the floor intact? */
static int q35_mem_room(int64_t bytes){
    const int64_t avail = q35_ram_avail_bytes();
    if(avail <= 0) return 0;                 /* cannot tell => assume not */
    return (avail - bytes) >= q35_mem_floor();
}

/* Terminal. Called only when an allocation has already failed.
 *
 * Formats into a stack buffer and write(2)s to fd 2 directly. NOT fprintf: _exit(2) skips
 * stdio flushing, so a buffered diagnostic is lost exactly in the case it exists for. Same
 * reasoning as the rest of this policy — the failure path must not depend on machinery that
 * may itself be starved. */
static void q35_oom(const char *what, int64_t bytes){
    const int64_t avail = q35_ram_avail_bytes();
    char buf[768];
    const int n = snprintf(buf, sizeof buf,
        "\n[qwen35] OUT OF MEMORY - stopping instead of trying again.\n"
        "  wanted    : %.3f GiB for %s\n"
        "  available : %.3f GiB   (floor %.2f GiB)\n"
        "  The context grows until memory runs out by design; this is that point. Nothing\n"
        "  further is allocated. Free memory, lower QWEN_KV_RAM_GB, or point\n"
        "  QWEN_KV_SPILL at a drive with room, and restart.\n",
        bytes/(1024.0*1024*1024), what, avail/(1024.0*1024*1024),
        q35_mem_floor()/(1024.0*1024.0*1024.0));
    if(n > 0){
        ssize_t off = 0;
        while(off < n){
            const ssize_t w = write(2, buf + off, (size_t)(n - off));
            if(w <= 0) break;
            off += w;
        }
    }
    _exit(2);            /* _exit, not exit: no atexit handler should allocate on the way out */
}

/* malloc that either succeeds or ends the process. Used for anything on the growth path. */
/* A LENGTH IS NEVER NEGATIVE, AND size_t DOES NOT KNOW THAT.
 *
 * Casting a negative int to size_t yields ~1.8e19, and memcpy, malloc and cudaMemcpy will all
 * accept it. The engine uses -1 as "empty" in several places — an unassigned checkpoint, a
 * layer's slot index when it is not that kind of layer — so the sentinel and the length share
 * a type and the conversion is silent. One instance of this segfaulted inside libc with no
 * message, ~512 tokens into a prefill.
 *
 * Anywhere a position or a count becomes a size, it goes through here. The cost is a
 * predictable branch; what it buys is that the sentinel can never become 18 exabytes. */
static size_t q35_len(int64_t n, const char *what){
    if(n < 0){
        char b[192];
        const int k = snprintf(b, sizeof b,
            "qwen35: %s is %lld — a negative length would become 1.8e19 as size_t\n",
            what ? what : "length", (long long)n);
        if(k > 0) { ssize_t w = write(2, b, (size_t)k); (void)w; }
        abort();
    }
    return (size_t)n;
}

static void *q35_xalloc(size_t n, const char *what){
    void *p = malloc(n);
    if(!p) q35_oom(what, (int64_t)n);
    return p;
}
static void *q35_xrealloc(void *old, size_t n, const char *what){
    void *p = realloc(old, n);
    if(!p) q35_oom(what, (int64_t)n);
    return p;
}
static int64_t q35_reserve_bytes(void){
    const char *e = getenv("QWEN_RESERVE_GB");
    double gb = e ? atof(e) : 6.0;
    if(gb < 1) gb = 1;
    return (int64_t)(gb*(1024.0*1024.0*1024.0));
}

static void q35_reside(Q35Model *M, Q35Resident *R, int do_lock){
    memset(R, 0, sizeof *R);
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    const long ps = sysconf(_SC_PAGESIZE);

    for(int sh = 0; sh < M->m.n_shards; sh++){
        gguf_shard *S = &M->m.shard[sh];
        if(!S->map || !S->size) continue;
        R->want += (int64_t)S->size;

        madvise(S->map, S->size, MADV_WILLNEED);
        /* touch: one byte per page, into a sink the compiler cannot fold away */
        volatile uint8_t sink = 0;
        for(size_t off = 0; off < S->size; off += (size_t)ps) sink ^= S->map[off];
        (void)sink;

        if(do_lock && mlock(S->map, S->size) == 0) R->locked += (int64_t)S->size;

        const size_t npg = (S->size + ps - 1)/ps;
        unsigned char *vec = (unsigned char*)malloc(npg);
        if(vec){
            if(mincore(S->map, S->size, vec) == 0){
                int64_t res = 0;
                for(size_t i = 0; i < npg; i++) if(vec[i] & 1) res++;
                R->resident += res*(int64_t)ps;
            }
            free(vec);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    R->secs = (t1.tv_sec-t0.tv_sec) + 1e-9*(t1.tv_nsec-t0.tv_nsec);
}

/* Old scripts and old shells still say COLIBRI_*. Accept them: an environment variable that
 * silently stopped being read is the worst kind of rename, because nothing fails — the engine
 * just quietly runs with a different KV spill path or a different memory floor than the person
 * setting it believes. Renaming is free; breaking someone's shell profile is not. */
extern char **environ;
static void q35_env_compat(void){
    for(char **e = environ; *e; e++){
        if(strncmp(*e, "COLIBRI_", 8)) continue;
        const char *eq = strchr(*e, '=');
        if(!eq) continue;
        char name[128];
        const size_t n = (size_t)(eq - *e) - 8;
        if(n + 6 >= sizeof name) continue;
        memcpy(name, "QWEN_", 5);
        memcpy(name + 5, *e + 8, n);
        name[5 + n] = 0;
        if(!getenv(name)) setenv(name, eq + 1, 1);
    }
}

/* THE CPU-SIDE WEIGHTS, IN MEMORY THE KERNEL CANNOT TAKE BACK.
 *
 * MADV_WILLNEED + touch only asks the page cache nicely, and the page cache says no when the
 * machine is busy: measured here, 75% of the model stayed resident and the missing 25% was
 * re-read from NVMe on every single token, which cost more than the resident 75% did to
 * compute. mlock would fix it, but RLIMIT_MEMLOCK is 8 MB on this box and the hard limit is
 * the same, so it is not available without root.
 *
 * Anonymous memory is. It is charged to the process, not to the cache, so the kernel will
 * not drop it to make room for file pages — only swap can take it, and that is visible.
 *
 * Only what the CPU reads gets copied: the FFN and the embeddings. The GDN, attention and
 * output tensors are bound for VRAM, so a host copy of them would be 5 GiB of RAM spent to
 * hold something the CPU never touches. */
/* what = Q35_RES_FFN  : only the FFN + embeddings (the GPU takes gdn/attn/output)
 * what = Q35_RES_ALL   : every trunk tensor (CPU-only mode: nothing may come from disk) */
enum { Q35_RES_FFN = 0, Q35_RES_ALL = 1 };

/* Tensors already living somewhere else. The GPU path uploads part of the FFN to VRAM
 * BEFORE residency runs, and copying those same layers into anonymous RAM as well buys
 * nothing and costs 1.5 GiB of the KV budget — which at 250k is the whole game. Set by
 * q35cu_mark_resident_elsewhere(); NULL means "nothing is". */
static const uint8_t *g_q35_elsewhere = NULL;
static inline int q35_is_elsewhere(const Q35Model *M, const gguf_tensor *t){
    if(!g_q35_elsewhere || !t) return 0;
    const int i = (int)(t - M->m.t);
    return i >= 0 && i < M->m.n_tensors && g_q35_elsewhere[i];
}

static int q35_reside_anon(Q35Model *M, Q35Resident *R, int what){
    memset(R, 0, sizeof *R);
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    const Q35Cfg *c = &M->c;

    const gguf_tensor **want = (const gguf_tensor**)malloc(sizeof(void*)*(size_t)(12*c->n_layer_all + 4));
    int nw = 0;
    for(int il = 0; il < c->n_layer; il++){          /* trunk only: the MTP block is skipped */
        const Q35Layer *L = &M->L[il];
        if(c->is_moe){
            /* The MoE FFN is the experts: keeping them in RAM is the whole point of RES_FFN
             * for Ornith. The router and shared expert are small but on the same hot path. */
            const gguf_tensor *g[8] = { L->ffn_gate_inp, L->ffn_gate_exps, L->ffn_up_exps,
                                        L->ffn_down_exps, L->ffn_gate_shexp, L->ffn_up_shexp,
                                        L->ffn_down_shexp, NULL };
            for(int k = 0; k < 8; k++) if(g[k] && !q35_is_elsewhere(M, g[k])) want[nw++] = g[k];
        } else {
        if(L->ffn_gate && !q35_is_elsewhere(M, L->ffn_gate)) want[nw++] = L->ffn_gate;
        if(L->ffn_up   && !q35_is_elsewhere(M, L->ffn_up))   want[nw++] = L->ffn_up;
        if(L->ffn_down && !q35_is_elsewhere(M, L->ffn_down)) want[nw++] = L->ffn_down;
        }
        if(what == Q35_RES_ALL){
            const gguf_tensor *g[9] = { L->wq, L->wk, L->wv, L->wo,
                                        L->wqkv, L->wgate, L->w_alpha, L->w_beta, L->w_out };
            for(int k = 0; k < 9; k++) if(g[k]) want[nw++] = g[k];
        }
    }
    if(what == Q35_RES_ALL && M->output) want[nw++] = M->output;
    /* The MTP block too, when it exists. It sits at index n_layer — PAST the trunk loop —
     * so a `for(il < n_layer)` residency pass silently leaves it in the mmap that this
     * function is about to MADV_DONTNEED. The draft head then faults its 0.2 GiB off NVMe on
     * every single draft: measured 0.299 s for a block that should cost 0.05, i.e. 55% of a
     * full 15.6 GiB pass to run 1.4% of the weights. Exactly the trap q35_wdata exists for,
     * one loop bound further out. */
    for(int il = c->n_layer; il < c->n_layer_all; il++){
        const Q35Layer *L = &M->L[il];
        const gguf_tensor *g[8] = { L->ffn_gate, L->ffn_up, L->ffn_down,
                                    L->wq, L->wk, L->wv, L->wo, L->eh_proj };
        for(int k = 0; k < 8; k++) if(g[k]) want[nw++] = g[k];
    }
    if(M->tok_embd) want[nw++] = M->tok_embd;

    int64_t need = 0;
    for(int i = 0; i < nw; i++) need += (want[i]->bytes + 63) & ~63LL;
    R->want = need;

    /* Refuse rather than OOM. Anonymous pages cannot be reclaimed the way file pages can, so
     * an over-optimistic copy does not degrade — it takes the desktop with it. */
    {
        const int64_t avail = q35_ram_avail_bytes();
        const int64_t floor_ = q35_reserve_bytes();
        if(avail - need < floor_){
            fprintf(stderr, "qwen35: refusing to make %.2f GiB resident: only %.2f GiB is "
                    "available and the floor is %.2f GiB.\n"
                    "        Free memory, or lower QWEN_RESERVE_GB, or use Q35_RES_FFN and "
                    "put gdn/attn/output on the GPU.\n",
                    need/(1024.0*1024*1024), avail/(1024.0*1024*1024), floor_/(1024.0*1024*1024));
            free(want);
            return 0;
        }
    }

    /* NOT MAP_POPULATE. Ordering matters here and it is easy to get backwards: MAP_POPULATE
     * faults the whole range in immediately as 4 KiB pages, and a later MADV_HUGEPAGE can
     * then only ask khugepaged to collapse them in the background — measured, that granted
     * 0 of 15.6 GiB. Mapping lazily, marking the range, and letting the copy below fault it
     * in gets 2 MiB pages from the start. */
    /* MEASURED, AND IT DOES NOT HELP — off unless asked for.
     *
     * 15.6 GiB of 4 KiB pages is 4.0M page-table entries against a ~2048-entry L2 STLB, which
     * looks like it should be ruinous. It is not, because this access pattern is a pure
     * SEQUENTIAL STREAM: the page-walk caches hold the upper levels, only the last level
     * misses, and that miss overlaps with streaming loads the prefetcher already has in
     * flight. TLB pressure costs you on RANDOM access, which this is not.
     *
     * A/B on the real model, 16 tokens: 4 KiB -> 0.43 s/token, FFN 36.39 GB/s; 2 MiB pages
     * (40% granted, rest lost to fragmentation) -> 0.43 s/token, FFN 36.25 GB/s. Identical
     * inside noise, and the hugepage run cost 29 s MORE to load because defrag=madvise does
     * direct compaction. Available, measured, not worth it: QWEN_HUGEPAGE=1 to retry on a
     * machine with less fragmented memory. */
    const int want_huge = getenv("QWEN_HUGEPAGE") != NULL;
    M->anon = (uint8_t*)mmap(NULL, (size_t)need, PROT_READ|PROT_WRITE,
                             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(M->anon == MAP_FAILED){
        M->anon = NULL;
        fprintf(stderr, "qwen35: cannot reserve %.2f GiB of anonymous memory for the CPU "
                "weights\n", need/(1024.0*1024*1024));
        free(want);
        return 0;
    }
    /* HUGEPAGES. 15.4 GiB of 4 KiB pages is 4.0 MILLION page-table entries, against an L2
     * STLB of ~2048 on Raptor Lake — a reach of 8 MiB. A streaming pass over the weights
     * therefore takes a TLB miss almost every page, and each miss is a 4-level page walk
     * that is itself memory traffic competing with the weights it is trying to fetch.
     *
     * 2 MiB pages cut that to 7,880 entries and give the same STLB a reach of 4 GiB. THP is
     * set to `madvise` on this box, so it does nothing unless asked — this is the ask. It is
     * also the userspace answer to "would direct physical addressing be faster": the win is
     * FEWER TRANSLATIONS, and no kernel module is required to get it.
     *
     * Best-effort: if the kernel cannot find contiguous 2 MiB blocks it silently falls back
     * to 4 KiB, which is why the report prints what was actually granted rather than what
     * was asked for. */
#ifdef MADV_HUGEPAGE
    if(want_huge) madvise(M->anon, (size_t)need, MADV_HUGEPAGE);
#endif
    M->anon_bytes = need;
    M->tw = (uint8_t**)calloc((size_t)M->m.n_tensors, sizeof(uint8_t*));
    if(!M->tw){ free(want); return 0; }

    /* PARALLEL, AND READ WITH pread RATHER THAN COPIED OUT OF THE MMAP.
     *
     * Two separate costs, both measured, and the second one only became visible after the
     * first was fixed:
     *
     *  1. The destination is a fresh anonymous mapping, so every 4 KiB written is a minor
     *     fault before it is a store. 19 MiB chunks, this box: 1 thread 2.01 GB/s (2040
     *     ns/page), 4 threads 7.07, 16 threads 13.98. Faults are taken per-CPU, so they
     *     parallelize — which is why this is 7x and not the 4.3x the copy alone would give.
     *
     *  2. The SOURCE was the mmapped file, and a memcpy out of it faults the pages in one at
     *     a time behind 128 KiB of readahead. One sequential stream gives an NVMe a queue
     *     depth of one. Measured with the source cold: 9.65 GiB took 9.48 s = 1.02 GB/s,
     *     against 6.27 GB/s that the same disk gives for 19 MiB pread requests.
     *
     * So: pread, in chunks, in parallel. The chunk is 19 MiB because that is what this
     * project already measured the disk to like, and chunking rather than parallelizing over
     * TENSORS matters because an FFN matrix is 85 MiB and a norm vector is 20 KiB — a split by
     * tensor hands one thread five big ones and another five small ones.
     *
     * The offsets must be laid out BEFORE the parallel region: `off` was a running total, and
     * a running total is exactly what a parallel loop cannot have. */
    int64_t *offs = (int64_t*)malloc(sizeof(int64_t)*(size_t)(nw > 0 ? nw : 1));
    if(!offs){ free(want); return 0; }
    {
        int64_t off = 0;
        for(int i = 0; i < nw; i++){ offs[i] = off; off += (want[i]->bytes + 63) & ~63LL; }
    }

    typedef struct { uint8_t *dst; int fd; int64_t foff, len; const void *src; } ResChunk;
    const int64_t CHUNK = 19LL<<20;
    int nc = 0;
    for(int i = 0; i < nw; i++) nc += (int)((want[i]->bytes + CHUNK - 1)/CHUNK);
    ResChunk *ch = (ResChunk*)malloc(sizeof(ResChunk)*(size_t)(nc > 0 ? nc : 1));
    if(!ch){ free(offs); free(want); return 0; }
    nc = 0;
    for(int i = 0; i < nw; i++){
        const gguf_tensor *T = want[i];
        const gguf_shard  *sh = &M->m.shard[T->shard];
        const void *src = gguf_data(&M->m, T);
        for(int64_t o = 0; o < T->bytes; o += CHUNK){
            const int64_t len = (T->bytes - o < CHUNK) ? T->bytes - o : CHUNK;
            ch[nc].dst  = M->anon + offs[i] + o;
            ch[nc].fd   = sh->fd;
            ch[nc].foff = (int64_t)T->off + o;
            ch[nc].len  = len;
            ch[nc].src  = src ? (const uint8_t*)src + o : NULL;   /* fallback if fd is gone */
            nc++;
        }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < nc; i++){
        if(ch[i].fd >= 0){
            int64_t done = 0;
            while(done < ch[i].len){
                const ssize_t r = pread(ch[i].fd, ch[i].dst + done,
                                        (size_t)(ch[i].len - done), (off_t)(ch[i].foff + done));
                if(r <= 0) break;
                done += r;
            }
            if(done == ch[i].len){
                /* same as the weight upload: it is in anonymous RAM now, so a second copy in
                 * the page cache only competes with the half of startup that has not run yet */
                posix_fadvise(ch[i].fd, (off_t)ch[i].foff, (off_t)ch[i].len, POSIX_FADV_DONTNEED);
                continue;
            }
        }
        if(ch[i].src) memcpy(ch[i].dst, ch[i].src, (size_t)ch[i].len);
    }
    free(ch);
    for(int i = 0; i < nw; i++){
        const int idx = (int)(want[i] - M->m.t);
        if(idx >= 0 && idx < M->m.n_tensors) M->tw[idx] = M->anon + offs[i];
    }
    free(offs);
    free(want);

    /* Measure, do not assume: mincore on the anonymous range says what is really there. */
    const long ps = sysconf(_SC_PAGESIZE);
    const size_t npg = ((size_t)need + ps - 1)/ps;
    unsigned char *vec = (unsigned char*)malloc(npg);
    if(vec){
        if(mincore(M->anon, (size_t)need, vec) == 0){
            int64_t res = 0;
            for(size_t i = 0; i < npg; i++) if(vec[i] & 1) res++;
            R->resident = res*(int64_t)ps;
        }
        free(vec);
    }
    if(mlock(M->anon, (size_t)need) == 0) R->locked = need;

    /* How much actually came back as hugepages? smaps_rollup is the only honest source —
     * madvise returning 0 means "noted", not "granted". */
    {
        /* line-oriented, NOT fscanf("%s %ld %s"): the first line of smaps_rollup is a mapping
         * header ("21fd...-7286... ---p 0 00:00 0 [rollup]"), so a scanf expecting a number in
         * field 2 fails there and the loop exits before reaching any key. That reported 0%
         * hugepages while the kernel was in fact granting 100% — a measurement bug that looked
         * exactly like a kernel refusing the request. */
        FILE *f = fopen("/proc/self/smaps_rollup", "r");
        if(f){
            char line[256];
            while(fgets(line, sizeof line, f))
                if(!strncmp(line, "AnonHugePages:", 14)){
                    long v = 0; sscanf(line+14, "%ld", &v); R->huge = (int64_t)v*1024; break;
                }
            fclose(f);
        }
    }

    /* The mmap has done its job; tell the kernel it may reclaim those file pages instead of
     * keeping a second copy of everything we just duplicated. */
    for(int sh = 0; sh < M->m.n_shards; sh++)
        if(M->m.shard[sh].map) madvise(M->m.shard[sh].map, M->m.shard[sh].size, MADV_DONTNEED);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    R->secs = (t1.tv_sec-t0.tv_sec) + 1e-9*(t1.tv_nsec-t0.tv_nsec);
    return 1;
}

static void q35_reside_report(const Q35Resident *R, FILE *o){
    const double gb = 1024.0*1024.0*1024.0;
    fprintf(o, "  residency: %.2f of %.2f GiB in RAM (%.1f%%), %.2f GiB mlocked, "
               "%.2f GiB in 2 MiB pages, %.1f s\n",
            R->resident/gb, R->want/gb, R->want ? 100.0*R->resident/R->want : 0.0,
            R->locked/gb, R->huge/gb, R->secs);
    if(R->want && R->huge < R->want/2)
        fprintf(o, "  note: only %.0f%% of the weights got 2 MiB pages. 4 KiB pages mean ~%.1fM\n"
                   "        page-table entries against a ~2048-entry STLB — a TLB miss per page\n"
                   "        on every streaming pass. Check /sys/kernel/mm/transparent_hugepage.\n",
                R->want ? 100.0*R->huge/R->want : 0.0, R->want/4096.0/1e6);
    if(R->want && R->resident < R->want*98/100)
        fprintf(o, "  WARNING: %.2f GiB of the model is NOT resident — those bytes will be\n"
                   "           re-read from disk on every token. Free memory or raise\n"
                   "           RLIMIT_MEMLOCK (ulimit -l) so mlock can hold it.\n",
                (R->want - R->resident)/gb);
    if(R->locked == 0 && R->want)
        fprintf(o, "  note: nothing mlocked (ulimit -l too low) — residency is best-effort and\n"
                   "        memory pressure elsewhere can still evict the weights.\n");
}

/* ---------------- byte accounting ---------------- */

static int64_t q35_tbytes(const gguf_tensor *t){ return t ? t->bytes : 0; }

typedef struct {
    int64_t ffn, gdn, attn, out, embd, mtp, norms, total;
} Q35Bytes;

static void q35_bytes(const Q35Model *M, Q35Bytes *B){
    memset(B, 0, sizeof *B);
    const Q35Cfg *c = &M->c;
    for(int il = 0; il < c->n_layer_all; il++){
        const Q35Layer *L = &M->L[il];
        int64_t ffn = q35_tbytes(L->ffn_gate) + q35_tbytes(L->ffn_up) + q35_tbytes(L->ffn_down);
        int64_t att = q35_tbytes(L->wq) + q35_tbytes(L->wk) + q35_tbytes(L->wv) + q35_tbytes(L->wo);
        int64_t gdn = q35_tbytes(L->wqkv) + q35_tbytes(L->wgate) + q35_tbytes(L->w_alpha)
                    + q35_tbytes(L->w_beta) + q35_tbytes(L->w_out)
                    + (int64_t)c->d_conv * c->conv_dim * 4 + (int64_t)c->n_v_heads * 8
                    + (int64_t)c->d_head_v * 4;
        if(L->kind == Q35_LAYER_MTP){ B->mtp += ffn + att; continue; }
        B->ffn  += ffn;
        B->attn += att;
        if(L->kind == Q35_LAYER_GDN) B->gdn += gdn;
        B->norms += (int64_t)c->d_model * 4 * 2;
    }
    B->out  = q35_tbytes(M->output);
    B->embd = q35_tbytes(M->tok_embd);
    B->total = B->ffn + B->gdn + B->attn + B->out + B->embd + B->mtp + B->norms;
}

/* ---------------- KV geometry ----------------
 * Deliberately NOT including kvstore.h: that header is MLA-shaped (one shared 576-value
 * latent per token) and drags mla.h with it. Qwen3.5 has ordinary GQA K and V, so the
 * format enum is redefined here with the same wire layout ggml uses, which keeps the
 * numbers comparable to llama.cpp's -ctk/-ctv. */
enum { Q35_KV_F32 = 0, Q35_KV_F16 = 1, Q35_KV_Q8_0 = 2, Q35_KV_Q4_0 = 3 };

static const char *q35_kv_fmt_name(int f){
    switch(f){ case Q35_KV_F32:return "fp32"; case Q35_KV_F16:return "fp16";
               case Q35_KV_Q8_0:return "q8_0"; case Q35_KV_Q4_0:return "q4_0"; default:return "?"; }
}
static int64_t q35_kv_entry_bytes(int fmt, int64_t n){
    switch(fmt){
        case Q35_KV_F32:  return n * 4;
        case Q35_KV_F16:  return n * 2;
        case Q35_KV_Q8_0: return (n / 32) * (2 + 32);
        case Q35_KV_Q4_0: return (n / 32) * (2 + 16);
        default: return 0;
    }
}


static int q35_n_attn_layers(const Q35Cfg *c){
    int n = 0;
    for(int i = 0; i < c->n_layer; i++) if(c->kind[i] == Q35_LAYER_ATTN) n++;
    return n;
}

/* Bytes of K+V for ONE token across ALL full-attention layers. This is the only thing in the
 * model that grows with context — the 48 GDN layers carry a fixed-size state instead. */
static int64_t q35_kv_bytes_per_token(const Q35Cfg *c, int fmt){
    const int n = c->n_head_kv * c->d_head;            /* 1024 values, K and V each */
    return (int64_t)q35_n_attn_layers(c) * 2 * q35_kv_entry_bytes(fmt, n);
}

/* Fixed recurrent state: the GDN layers' [d_state x d_head_v] per v-head, plus the conv
 * ring. Independent of context length — this is the whole point of the hybrid. */
static int64_t q35_gdn_state_bytes(const Q35Cfg *c){
    int ng = 0;
    for(int i = 0; i < c->n_layer; i++) if(c->kind[i] == Q35_LAYER_GDN) ng++;
    int64_t s = (int64_t)c->d_state * c->d_head_v * c->n_v_heads * 4;   /* recurrent */
    int64_t v = (int64_t)(c->d_conv - 1) * c->conv_dim * 4;             /* conv ring */
    return (int64_t)ng * (s + v);
}

#endif
