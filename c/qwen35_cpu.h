#ifndef QWEN_QWEN35_CPU_H
#define QWEN_QWEN35_CPU_H
/* qwen35_cpu.h — the Qwen3.5 forward pass, in plain C, on the CPU.
 *
 * This is the REFERENCE. It is not the fast path and does not try to be: its job is to be
 * the thing every optimized path is diffed against. On this project every silent bug came
 * from an optimization that had nothing to be compared to, so the order is fixed — correct
 * first, measured against llama.cpp, and only then split across GPU and CPU.
 *
 * THE FOUR PLACES THIS ARCHITECTURE HIDES A TRAP
 *
 * 1. attn_q is [d_model, n_head*d_head*2] and the layout is INTERLEAVED PER HEAD:
 *        [ q_h0 (256) | gate_h0 (256) | q_h1 (256) | gate_h1 (256) | ... ]
 *    not [all q | all gates]. Reading it as the latter gives a model that still produces
 *    grammatical English, which is why it has to be asserted rather than eyeballed.
 *
 * 2. RoPE is PARTIAL and NeoX-paired: only dims [0,64) of each 256-wide head rotate, pairing
 *    i with i+32, and dims [64,256) pass through untouched. It is declared as mrope with
 *    sections [11,11,10,0], but for text-only input all three position channels carry the
 *    same value, so mrope collapses exactly onto NeoX. (Feed it images and that stops being
 *    true — see q35_rope_mrope.)
 *
 * 3. Inside a GDN layer q and k are L2-normalized, not RMS-normalized. The two differ by a
 *    factor of sqrt(128) which the delta rule partly absorbs, so the damage is gradual.
 *
 * 4. ssm_a holds -exp(A_log). The decay is exp(softplus(alpha+dt_bias) * ssm_a) and lands in
 *    (0,1] only because that value is negative. qwen35_test asserts the sign.
 *
 * THE ONE THING THAT IS GENUINELY AMBIGUOUS, AND IS THEREFORE A SWITCH
 *
 * There are 48 v-heads and 16 k-heads, so each k-head serves three v-heads. HF's reference
 * uses repeat_interleave -> v-head h reads k-head h/3. llama.cpp's non-fused graph reaches
 * the same shape with ggml_repeat, which TILES -> v-head h reads k-head h%16. Those are
 * different models. q35_set_gdn_head_map picks one; qwen35_fwd_test settles it against the
 * real oracle rather than against an argument.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "qwen35.h"
#include "kquant.h"
#include "kquant_simd.h"
#include "kv_tier.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* v-head -> k-head mapping. 0 = interleave (HF), 1 = tile.
 *
 * SETTLED AGAINST THE ORACLE, NOT AGAINST THE ARGUMENT. HF's modeling_qwen3_5.py uses
 * repeat_interleave, so v-head h reads k-head h/3. ggml does NOT: the fused kernel indexes
 * iq1 = iv1 % neq1 (ops.cpp, ggml_compute_forward_gated_delta_net_one_chunk) and the
 * non-fused graph reaches the same thing through ggml_repeat, which tiles. Those are
 * different models, so it had to be measured: on layer 0 of this file, tiling reproduces
 * llama.cpp's linear_attn_out to 0.1% and interleaving is off by 21%.
 *
 * The conversion script is what reconciles it — the GGUF's head order is whatever makes the
 * converter's own engine right — so for a llama.cpp-converted file, TILE is correct. */
static int g_q35_gdn_tile = 1;
static void q35_set_gdn_head_map(int tile){ g_q35_gdn_tile = tile; }
static inline int q35_kh(int hv, int n_v, int n_k){
    return g_q35_gdn_tile ? (hv % n_k) : (hv / (n_v / n_k));
}

/* ---------------- small numerics ---------------- */

static void q35_rmsnorm(float *out, const float *x, const float *w, int n, float eps){
    double s = 0;
    for(int i = 0; i < n; i++) s += (double)x[i]*x[i];
    float r = 1.0f / sqrtf((float)(s/n) + eps);
    for(int i = 0; i < n; i++) out[i] = x[i]*r*w[i];
}

/* ggml_l2_norm: x / sqrt(sum(x^2) + eps) — the sum is NOT divided by n. That missing 1/n is
 * the whole difference from rmsnorm and it is worth sqrt(128) here. */
static void q35_l2norm(float *v, int n, float eps){
    double s = 0;
    for(int i = 0; i < n; i++) s += (double)v[i]*v[i];
    float r = 1.0f / fmaxf(sqrtf((float)s), eps);      /* ggml: max(), not sum+eps */
    for(int i = 0; i < n; i++) v[i] *= r;
}

static inline float q35_sigmoid(float x){ return 1.0f/(1.0f + expf(-x)); }
static inline float q35_silu(float x){ return x/(1.0f + expf(-x)); }
/* softplus, guarded: log1p(exp(x)) overflows for large x and log1p(exp(-inf)) is fine */
static inline float q35_softplus(float x){ return x > 20.0f ? x : log1pf(expf(x)); }

static void q35_softmax(float *x, int n){
    float m = -INFINITY;
    for(int i = 0; i < n; i++) if(x[i] > m) m = x[i];
    float s = 0;
    for(int i = 0; i < n; i++){ x[i] = expf(x[i]-m); s += x[i]; }
    float r = 1.0f/s;
    for(int i = 0; i < n; i++) x[i] *= r;
}

/* NeoX-paired partial RoPE over the first n_rot dims of a d_head-wide head.
 * theta_scale = base^(-2/n_rot); pair (i, i + n_rot/2) rotates by pos*theta_scale^i. */
static void q35_rope(float *v, int pos, int n_rot, float base){
    const int h = n_rot/2;
    const float ts = powf(base, -2.0f/(float)n_rot);
    float th = (float)pos;
    for(int i = 0; i < h; i++){
        float c = cosf(th), s = sinf(th);
        float x0 = v[i], x1 = v[i+h];
        v[i]   = x0*c - x1*s;
        v[i+h] = x0*s + x1*c;
        th *= ts;
    }
}

static int g_q35_dbg = 0;
/* Per-stage wall time, so "how fast can this go" is answered from where the time actually
 * is, not from where it is assumed to be. */
static double g_t_ffn = 0, g_t_attn = 0, g_t_gdn = 0, g_t_out = 0, g_t_embd = 0;
static double q35_clk(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
static void q35_dbg(const char *tag, const float *v, int n){
    if(!g_q35_dbg) return;
    double s = 0; for(int i=0;i<n;i++) s += (double)v[i]*v[i];
    fprintf(stderr, "    %-24s norm=%12.5f ", tag, sqrt(s));
    for(int i=0;i<6 && i<n;i++) fprintf(stderr, "%11.5f", v[i]);
    fprintf(stderr, "\n");
}

/* ---------------- runtime state ---------------- */

typedef struct {
    const Q35Model *M;
    int   n_ctx;

    /* KV lives in the tiered store: quantized, RAM/NVMe, prefetched a chunk ahead. */
    KvTier kv;
    int    n_attn;              /* full-attention layers */
    int   *attn_slot;           /* layer -> slot, or -1 */

    /* GDN recurrent state: [gdn_slot][n_v_heads][d_state*d_head_v] */
    float *S;
    float *conv;                /* [gdn_slot][(d_conv-1)*conv_dim] */
    int   *gdn_slot;
    int    n_gdn;

    /* scratch, one token wide */
    float *x, *xn, *tmp, *qg, *k, *v, *att, *o, *ffn_g, *ffn_u, *qkv, *z, *ab;
    float *moe_rlog, *moe_od;   /* MoE only: router logits [n_expert], one experts down output [d_model] */
    int    pos;
} Q35State;

static void *q35_calloc(size_t n){ void *p = calloc(1, n); if(!p){ fprintf(stderr,"qwen35: OOM %zu\n", n); exit(1);} return p; }

static int q35_state_init_ex(Q35State *R, const Q35Model *M, int n_ctx,
                             int kv_fmt, int64_t kv_ram_budget, const char *spill_dir,
                             int chunk_tokens){
    memset(R, 0, sizeof *R);
    const Q35Cfg *c = &M->c;
    R->M = M; R->n_ctx = n_ctx;

    R->attn_slot = (int*)q35_calloc(sizeof(int)*c->n_layer_all);
    R->gdn_slot  = (int*)q35_calloc(sizeof(int)*c->n_layer_all);
    for(int i = 0; i < c->n_layer_all; i++){ R->attn_slot[i] = -1; R->gdn_slot[i] = -1; }
    for(int i = 0; i < c->n_layer; i++){
        if(c->kind[i] == Q35_LAYER_ATTN) R->attn_slot[i] = R->n_attn++;
        else                             R->gdn_slot[i]  = R->n_gdn++;
    }
    /* The MTP block is a full-attention block and needs a KV slot of its own when
     * speculative decoding is on. Reserving it unconditionally costs one slot's worth of
     * arena; NOT reserving it means the draft head silently writes into trunk layer 15's
     * cache, which corrupts the real model rather than just the draft. */
    const int kv_layers = R->n_attn + (c->n_layer_mtp ? 1 : 0);

    const int kvd = c->n_head_kv * c->d_head;
    if(!kvt_open_n(&R->kv, c, kv_layers, n_ctx, kv_fmt, kv_ram_budget, spill_dir, chunk_tokens, 8)){
        fprintf(stderr, "qwen35: kv_tier open failed\n"); return 0;
    }

    R->S    = (float*)q35_calloc((size_t)R->n_gdn * c->n_v_heads * c->d_state * c->d_head_v * sizeof(float));
    R->conv = (float*)q35_calloc((size_t)R->n_gdn * (c->d_conv-1) * c->conv_dim * sizeof(float));

    R->x     = (float*)q35_calloc(sizeof(float)*c->d_model);
    R->xn    = (float*)q35_calloc(sizeof(float)*c->d_model);
    R->tmp   = (float*)q35_calloc(sizeof(float)*c->d_model);
    R->qg    = (float*)q35_calloc(sizeof(float)*2*c->n_head*c->d_head);
    R->k     = (float*)q35_calloc(sizeof(float)*kvd);
    R->v     = (float*)q35_calloc(sizeof(float)*kvd);
    R->att   = (float*)q35_calloc(sizeof(float)*(size_t)n_ctx*c->n_head);
    R->o     = (float*)q35_calloc(sizeof(float)*(c->n_head*c->d_head > c->d_inner ? c->n_head*c->d_head : c->d_inner));
    /* The FFN scratch must hold the widest hidden vector the model uses. Dense: d_ff. MoE:
     * the routed expert (d_ff_exp) and the shared expert (d_ff_shexp), whichever is wider —
     * d_ff itself is 0 for a MoE file. */
    int ffn_scratch = c->d_ff;
    if(c->is_moe){
        if(c->d_ff_exp   > ffn_scratch) ffn_scratch = c->d_ff_exp;
        if(c->d_ff_shexp > ffn_scratch) ffn_scratch = c->d_ff_shexp;
    }
    R->ffn_g = (float*)q35_calloc(sizeof(float)*ffn_scratch);
    R->ffn_u = (float*)q35_calloc(sizeof(float)*ffn_scratch);
    if(c->is_moe){
        R->moe_rlog = (float*)q35_calloc(sizeof(float)*c->n_expert);
        R->moe_od   = (float*)q35_calloc(sizeof(float)*c->d_model);
    }
    R->qkv   = (float*)q35_calloc(sizeof(float)*c->conv_dim);
    R->z     = (float*)q35_calloc(sizeof(float)*c->d_inner);
    R->ab    = (float*)q35_calloc(sizeof(float)*2*c->n_v_heads);
    return 1;
}

/* the default: everything in RAM, q8_0. The planner picks the real budget. */
static int q35_state_init(Q35State *R, const Q35Model *M, int n_ctx){
    const int64_t all = q35_kv_bytes_per_token(&M->c, Q35_KV_Q8_0) * (int64_t)n_ctx;
    return q35_state_init_ex(R, M, n_ctx, Q35_KV_Q8_0, all + (1<<20), NULL, 4096);
}

static void q35_state_free(Q35State *R){
    kvt_close(&R->kv);
    free(R->attn_slot); free(R->gdn_slot); free(R->S); free(R->conv);
    free(R->x); free(R->xn); free(R->tmp); free(R->qg); free(R->k); free(R->v);
    free(R->att); free(R->o); free(R->ffn_g); free(R->ffn_u); free(R->qkv); free(R->z); free(R->ab);
    free(R->moe_rlog); free(R->moe_od);
}

/* reset the sequence: recurrent state to zero, KV logically empty */
static void q35_state_reset(Q35State *R){
    const Q35Cfg *c = &R->M->c;
    memset(R->S, 0, (size_t)R->n_gdn * c->n_v_heads * c->d_state * c->d_head_v * sizeof(float));
    memset(R->conv, 0, (size_t)R->n_gdn * (c->d_conv-1) * c->conv_dim * sizeof(float));
    R->pos = 0;
}

/* ---------------- gemv over a quantized [O, I] tensor ---------------- */

/* The whole decode is gemv: one token, so S=1 always. kq_gemv_fast routes Q4_K and Q6_K —
 * which is all of this model's FFN, 9.65 GiB per token — through the AVX2 kernels and
 * everything else through the scalar reference. Measured: 2.89 -> 32.2 GB/s. */
/* EVERY weight read goes through q35_wdata. Not "most" — q35_reside_anon calls
 * MADV_DONTNEED on the mmap once it has copied the weights out, so a gemv that still points
 * at the mapping does not read a slow copy, it reads a DISCARDED page and faults it back off
 * the NVMe every single token. That is how the GDN ended up at 1.2 GB/s while the FFN next
 * to it ran at 35: same model, same run, one of them still pointing at the mmap. */
static void q35_mv2(float *y, const float *x, const Q35Model *M, const gguf_tensor *T){
    const void *W = q35_wdata(M, T);
    kq_gemv_fast(y, x, W, T->type, (int)T->ne[0], (int)T->ne[1]);
}

/* ---------------- FFN: swiglu, gate/up parallel ---------------- */

static void q35_ffn_dense(Q35State *R, const Q35Layer *L, const float *xn, float *out){
    const Q35Cfg *c = &R->M->c;
    const Q35Model *M = R->M;
    q35_mv2(R->ffn_g, xn, M, L->ffn_gate);
    q35_mv2(R->ffn_u, xn, M, L->ffn_up);
    for(int i = 0; i < c->d_ff; i++) R->ffn_g[i] = q35_silu(R->ffn_g[i]) * R->ffn_u[i];
    q35_mv2(out, R->ffn_g, M, L->ffn_down);
}

/* One dense SwiGLU FFN of hidden width F, straight out of quantized weights: same shape as the
 * dense path but with the width and pointers passed in, so both the routed experts and the
 * shared expert share it. acc is written (not accumulated). */
/* One dense SwiGLU FFN of hidden width F straight out of quantized weights, state-free: the
 * caller passes the two hidden scratch buffers (gbuf,ubuf, each >= F) and the accumulator.
 * Shared by the routed experts and the shared expert, decode and prefill alike. */
static void q35_swiglu_row(const Q35Model *M, const float *xn, int D, int F,
                           const gguf_tensor *Tg, const gguf_tensor *Tu, const gguf_tensor *Td,
                           size_t goff, size_t uoff, size_t doff,
                           float *gbuf, float *ubuf, float *acc){
    const uint8_t *bg = (const uint8_t*)q35_wdata(M, Tg) + goff;
    const uint8_t *bu = (const uint8_t*)q35_wdata(M, Tu) + uoff;
    const uint8_t *bd = (const uint8_t*)q35_wdata(M, Td) + doff;
    kq_gemv_fast(gbuf, xn, bg, Tg->type, D, F);
    kq_gemv_fast(ubuf, xn, bu, Tu->type, D, F);
    for(int i = 0; i < F; i++) gbuf[i] = q35_silu(gbuf[i]) * ubuf[i];
    kq_gemv_fast(acc, gbuf, bd, Td->type, F, D);
}

/* Routing only: softmax over all E experts, take the top-K, renormalize their weights to sum
 * 1 (norm_topk_prob). Fills idx[K], wt[K]; returns the shared expert's scalar sigmoid gate.
 * The router weight is F32 and small, so this is a plain dot loop. Shared by the CPU FFN and
 * the GPU hetero FFN — they route identically and only differ in where the experts run. */
static float q35_moe_route(const Q35Model *M, const Q35Layer *L, const float *xn,
                           float *rlog, int *idx, float *wt){
    const Q35Cfg *c = &M->c;
    const int D = c->d_model, E = c->n_expert, K = c->n_expert_used;
    const float *Wr = (const float*)q35_wdata(M, L->ffn_gate_inp);
    float mx = -1e30f;
    for(int e = 0; e < E; e++){
        const float *w = Wr + (size_t)e*D;
        float a = 0.f; for(int i = 0; i < D; i++) a += w[i]*xn[i];
        rlog[e] = a; if(a > mx) mx = a;
    }
    float sum = 0.f;
    for(int e = 0; e < E; e++){ rlog[e] = expf(rlog[e]-mx); sum += rlog[e]; }
    const float inv = sum > 0.f ? 1.f/sum : 0.f;
    for(int e = 0; e < E; e++) rlog[e] *= inv;
    float wsum = 0.f;
    for(int j = 0; j < K; j++){
        int best = -1; float bv = -1.f;
        for(int e = 0; e < E; e++){
            if(rlog[e] > bv){ int taken = 0;
                for(int t = 0; t < j; t++) if(idx[t] == e){ taken = 1; break; }
                if(!taken){ bv = rlog[e]; best = e; } }
        }
        idx[j] = best; wt[j] = bv; wsum += bv;
    }
    const float rn = wsum > 0.f ? 1.f/wsum : 0.f;
    for(int j = 0; j < K; j++) wt[j] *= rn;
    const float *ws = L->ffn_gate_inp_shexp;
    float g = 0.f; for(int i = 0; i < D; i++) g += ws[i]*xn[i];
    return 1.f/(1.f+expf(-g));
}

/* ---------------- FFN: routed MoE (qwen35moe / Ornith), ONE token, all on the CPU ----------
 * State-free (every scratch buffer is the caller's) so decode and the prefill batch share it.
 * rlog >= n_expert, gbuf/ubuf >= max(d_ff_exp, d_ff_shexp), od >= d_model. */
static void q35_moe_row(const Q35Model *M, const Q35Layer *L, const float *xn, float *out,
                        float *rlog, float *gbuf, float *ubuf, float *od){
    const Q35Cfg *c = &M->c;
    const int D = c->d_model, F = c->d_ff_exp, K = c->n_expert_used;
    int idx[64]; float wt[64];
    if(K > 64) return;
    const float sg_gate = q35_moe_route(M, L, xn, rlog, idx, wt);

    for(int i = 0; i < D; i++) out[i] = 0.f;
    const int64_t sg = (int64_t)F * kq_row_bytes(L->ffn_gate_exps->type, D);
    const int64_t su = (int64_t)F * kq_row_bytes(L->ffn_up_exps->type,   D);
    const int64_t sd = (int64_t)D * kq_row_bytes(L->ffn_down_exps->type, F);
    for(int j = 0; j < K; j++){
        const int e = idx[j];
        q35_swiglu_row(M, xn, D, F, L->ffn_gate_exps, L->ffn_up_exps, L->ffn_down_exps,
                       (size_t)e*sg, (size_t)e*su, (size_t)e*sd, gbuf, ubuf, od);
        for(int i = 0; i < D; i++) out[i] += wt[j] * od[i];
    }
    q35_swiglu_row(M, xn, D, c->d_ff_shexp, L->ffn_gate_shexp, L->ffn_up_shexp, L->ffn_down_shexp,
                   0, 0, 0, gbuf, ubuf, od);
    for(int i = 0; i < D; i++) out[i] += sg_gate * od[i];
}

static void q35_ffn_moe(Q35State *R, const Q35Layer *L, const float *xn, float *out){
    q35_moe_row(R->M, L, xn, out, R->moe_rlog, R->ffn_g, R->ffn_u, R->moe_od);
}

static void q35_ffn(Q35State *R, const Q35Layer *L, const float *xn, float *out){
    if(R->M->c.is_moe) q35_ffn_moe(R, L, xn, out);
    else               q35_ffn_dense(R, L, xn, out);
}

/* ---------------- full attention (one token, decode) ---------------- */

static void q35_attn(Q35State *R, const Q35Layer *L, int il, const float *xn, int pos, float *out){
    const Q35Cfg *c = &R->M->c;
    const int dh = c->d_head, nh = c->n_head, nkv = c->n_head_kv;
    const int grp = nh / nkv;

    q35_mv2(R->qg, xn, R->M, L->wq);      /* [nh][2*dh], q and gate interleaved per head */
    q35_mv2(R->k,  xn, R->M, L->wk);
    q35_mv2(R->v,  xn, R->M, L->wv);

    /* per-head QK-norm, then partial rope */
    for(int h = 0; h < nh; h++){
        float *q = R->qg + (size_t)h*2*dh;
        q35_rmsnorm(q, q, L->q_norm, dh, c->eps);
        q35_rope(q, pos, c->n_rot, c->rope_base);
    }
    for(int h = 0; h < nkv; h++){
        float *k = R->k + (size_t)h*dh;
        q35_rmsnorm(k, k, L->k_norm, dh, c->eps);
        q35_rope(k, pos, c->n_rot, c->rope_base);
    }

    /* append to the tiered store */
    const int slot = R->attn_slot[il];
    kvt_put(&R->kv, slot, pos, R->k, R->v);

    const float scale = 1.0f/sqrtf((float)dh);
    const int T = pos + 1;
    KvTier *KV = &R->kv;
    const int64_t hb  = q35_kv_entry_bytes(KV->fmt, dh);      /* bytes of ONE head's K (or V) */
    const int n_ch = (T + KV->chunk - 1) / KV->chunk;

    memset(R->o, 0, sizeof(float)*(size_t)nh*dh);

    /* Chunk-at-a-time so the disk tier can be read AHEAD of the compute. Chunk c+1 is asked
     * for before chunk c is touched, which is the whole reason the drive stays busy. The
     * softmax is done in one pass at the end over the scores collected here — running max
     * and rescale would work too, but two passes over kilobytes is not the cost that
     * matters when the other side of the loop is moving gigabytes. */
    for(int ci = 0; ci < n_ch; ci++){
        kvt_prefetch_window(KV, slot, ci, 1, n_ch);
        const uint8_t *blk = kvt_chunk(KV, slot, ci);
        if(!blk) continue;
        const int t0 = ci*KV->chunk;
        const int t1 = (t0 + KV->chunk < T) ? t0 + KV->chunk : T;
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++){
            const float *q = R->qg + (size_t)h*2*dh;
            float *sc = R->att + (size_t)h*R->n_ctx;
            const int hk = h / grp;
            for(int t = t0; t < t1; t++){
                /* the K region: this pass touches nothing but K, back to back */
                const uint8_t *kt = blk + kvt_koff(KV, t - t0) + hk*hb;
                sc[t] = kvt_dot(KV, q, kt, dh)*scale;
            }
        }
    }

    /* one softmax over the whole scanned history, per head */
    #pragma omp parallel for schedule(static)
    for(int h = 0; h < nh; h++) q35_softmax(R->att + (size_t)h*R->n_ctx, T);

    for(int ci = 0; ci < n_ch; ci++){
        kvt_prefetch_window(KV, slot, ci, 1, n_ch);
        const uint8_t *blk = kvt_chunk(KV, slot, ci);
        if(!blk) continue;
        const int t0 = ci*KV->chunk;
        const int t1 = (t0 + KV->chunk < T) ? t0 + KV->chunk : T;
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++){
            const float *sc = R->att + (size_t)h*R->n_ctx;
            const int hk = h / grp;
            float *oh = R->o + (size_t)h*dh;
            for(int t = t0; t < t1; t++){
                /* the V region, likewise contiguous */
                const uint8_t *vt = blk + kvt_voff(KV, t - t0) + hk*hb;
                kvt_axpy(KV, oh, sc[t], vt, dh);
            }
        }
    }

    /* the per-head output gate, sigmoid, from the second half of this head's q slice */
    for(int h = 0; h < nh; h++){
        const float *g = R->qg + (size_t)h*2*dh + dh;
        float *oh = R->o + (size_t)h*dh;
        for(int i = 0; i < dh; i++) oh[i] *= q35_sigmoid(g[i]);
    }

    q35_mv2(out, R->o, R->M, L->wo);
}

/* ---------------- gated delta net (one token, decode) ---------------- */

static void q35_gdn(Q35State *R, const Q35Layer *L, int il, const float *xn, float *out){
    const Q35Cfg *c = &R->M->c;
    const int dk = c->d_state, dv = c->d_head_v;
    const int nk = c->n_k_heads, nv = c->n_v_heads;
    const int cd = c->conv_dim, kdim = dk*nk;
    const int slot = R->gdn_slot[il];

    if(il==0) q35_dbg("attn_norm-0", xn, c->d_model);
    q35_mv2(R->qkv, xn, R->M, L->wqkv);          /* [conv_dim] = q | k | v */
    if(il==0) q35_dbg("qkv_mixed-0", R->qkv, cd);
    q35_mv2(R->z,   xn, R->M, L->wgate);         /* [d_inner] */
    q35_mv2(R->ab,          xn, R->M, L->w_beta);   /* [nv] */
    q35_mv2(R->ab + nv,     xn, R->M, L->w_alpha);  /* [nv] */

    /* per-head decay and delta strength.
     * g = exp(softplus(alpha + dt_bias) * ssm_a), ssm_a < 0 => g in (0,1] */
    float *beta = R->ab, *gdec = R->ab + nv;
    if(il==0){ q35_dbg("beta_raw-0", beta, nv); q35_dbg("alpha-0", gdec, nv); }
    for(int h = 0; h < nv; h++){
        beta[h] = q35_sigmoid(beta[h]);
        gdec[h] = expf(q35_softplus(gdec[h] + L->dt_bias[h]) * L->ssm_a[h]);
    }

    if(il==0){ q35_dbg("beta_sigmoid-0", beta, nv); q35_dbg("gdecay-0", gdec, nv); }

    /* causal depthwise conv, kernel d_conv, over all conv_dim channels, then silu.
     * The state holds the previous d_conv-1 inputs per channel, oldest first. */
    float *cs = R->conv + (size_t)slot*(c->d_conv-1)*cd;
    const int K1 = c->d_conv - 1;
    for(int ch = 0; ch < cd; ch++){
        const float xin = R->qkv[ch];                     /* PRE-conv: this is what the ring keeps */
        float acc = 0;
        /* ssm_conv1d.weight is [d_conv, conv_dim] with ne[0]=d_conv FASTEST: the kernel taps
         * of one channel are contiguous, channel stride is d_conv. Indexing it the other way
         * round is silent — it still reads in-bounds floats. */
        const float *kk = L->conv1d + (size_t)ch*c->d_conv;
        for(int j = 0; j < K1; j++)
            acc += kk[j] * cs[(size_t)j*cd + ch];
        acc += kk[K1] * xin;
        for(int j = 0; j < K1-1; j++) cs[(size_t)j*cd + ch] = cs[(size_t)(j+1)*cd + ch];
        cs[(size_t)(K1-1)*cd + ch] = xin;
        R->qkv[ch] = q35_silu(acc);
    }

    if(il==0) q35_dbg("conv_out_silu-0", R->qkv, cd);
    float *q = R->qkv, *k = R->qkv + kdim, *vv = R->qkv + 2*kdim;
    for(int h = 0; h < nk; h++){
        q35_l2norm(q + (size_t)h*dk, dk, c->eps);
        q35_l2norm(k + (size_t)h*dk, dk, c->eps);
    }

    const float qscale = 1.0f/sqrtf((float)dk);
    float *Sall = R->S + (size_t)slot*nv*dk*dv;

    #pragma omp parallel for schedule(static)
    for(int h = 0; h < nv; h++){
        const int hk = q35_kh(h, nv, nk);
        float *S = Sall + (size_t)h*dk*dv;               /* S[i*dv + j], i over k-dim, j over v-dim */
        const float *kh = k + (size_t)hk*dk;
        const float *qh = q + (size_t)hk*dk;
        const float *vh = vv + (size_t)h*dv;
        const float g = gdec[h], b = beta[h];

        /* S *= g ; kv = S^T k ; d = b*(v - kv) ; S += k (x) d ; o = S^T (q/sqrt(dk)) */
        float kvv[512];                                   /* dv <= 512 by construction */
        for(int j = 0; j < dv; j++) kvv[j] = 0;
        for(int i = 0; i < dk; i++){
            float *Si = S + (size_t)i*dv;
            const float ki = kh[i];
            for(int j = 0; j < dv; j++){ Si[j] *= g; kvv[j] += Si[j]*ki; }
        }
        float dd[512];
        for(int j = 0; j < dv; j++) dd[j] = b*(vh[j] - kvv[j]);

        float *oh = R->o + (size_t)h*dv;
        for(int j = 0; j < dv; j++) oh[j] = 0;
        for(int i = 0; i < dk; i++){
            float *Si = S + (size_t)i*dv;
            const float ki = kh[i], qi = qh[i]*qscale;
            for(int j = 0; j < dv; j++){ Si[j] += ki*dd[j]; oh[j] += Si[j]*qi; }
        }
    }

    if(il==0) q35_dbg("attn_output-0", R->o, c->d_inner);
    /* gated per-head RMS norm: rmsnorm(o_h, ssm_norm) * silu(z_h) */
    for(int h = 0; h < nv; h++){
        float *oh = R->o + (size_t)h*dv;
        q35_rmsnorm(oh, oh, L->ssm_norm, dv, c->eps);
        const float *zh = R->z + (size_t)h*dv;
        for(int j = 0; j < dv; j++) oh[j] *= q35_silu(zh[j]);
    }

    if(il==0) q35_dbg("final_output-0", R->o, c->d_inner);
    q35_mv2(out, R->o, R->M, L->w_out);
    if(il==0) q35_dbg("linear_attn_out-0", out, c->d_model);
}

/* ---------------- one token through the whole trunk ---------------- */

static void q35_embed(Q35State *R, int tok, float *x){
    const Q35Model *M = R->M;
    const gguf_tensor *T = M->tok_embd;
    const int64_t rb = kq_row_bytes(T->type, T->ne[0]);
    const uint8_t *W = (const uint8_t*)q35_wdata(M, T);
    kq_dequant_row(T->type, W + rb*(int64_t)tok, x, T->ne[0]);
}

static double q35_nrm(const float *v, int n){
    double s = 0; for(int i=0;i<n;i++) s += (double)v[i]*v[i]; return sqrt(s);
}

/* logits may be NULL to prefill without paying for the 248320-wide output projection. */
static void q35_forward(Q35State *R, int tok, int pos, float *logits){
    const Q35Model *M = R->M;
    const Q35Cfg *c = &M->c;
    float *x = R->x, *xn = R->xn, *t = R->tmp;

    { const double t0 = q35_clk(); q35_embed(R, tok, x); g_t_embd += q35_clk()-t0; }
    if(g_q35_dbg) fprintf(stderr, "  embed norm %.4f\n", q35_nrm(x, c->d_model));

    for(int il = 0; il < c->n_layer; il++){
        const Q35Layer *L = &M->L[il];

        q35_rmsnorm(xn, x, L->attn_norm, c->d_model, c->eps);
        { const double t0 = q35_clk();
          if(L->kind == Q35_LAYER_ATTN){ q35_attn(R, L, il, xn, pos, t); g_t_attn += q35_clk()-t0; }
          else                         { q35_gdn (R, L, il, xn, t);      g_t_gdn  += q35_clk()-t0; } }
        double n_mix = g_q35_dbg ? q35_nrm(t, c->d_model) : 0;
        for(int i = 0; i < c->d_model; i++) x[i] += t[i];
        double n_res = g_q35_dbg ? q35_nrm(x, c->d_model) : 0;

        q35_rmsnorm(xn, x, L->post_attn_norm, c->d_model, c->eps);
        { const double t0 = q35_clk(); q35_ffn(R, L, xn, t); g_t_ffn += q35_clk()-t0; }
        double n_ffn = g_q35_dbg ? q35_nrm(t, c->d_model) : 0;
        for(int i = 0; i < c->d_model; i++) x[i] += t[i];
        if(g_q35_dbg)
            fprintf(stderr, "  L%-2d %-4s  mix %9.3f  res %9.3f  ffn %9.3f  out %9.3f\n",
                    il, L->kind == Q35_LAYER_ATTN ? "attn" : "gdn", n_mix, n_res, n_ffn,
                    q35_nrm(x, c->d_model));
    }

    if(logits){
        const double t0 = q35_clk();
        q35_rmsnorm(xn, x, M->output_norm, c->d_model, c->eps);
        q35_mv2(logits, xn, M, M->output);
        g_t_out += q35_clk()-t0;
    }
    R->pos = pos + 1;
}

#endif
