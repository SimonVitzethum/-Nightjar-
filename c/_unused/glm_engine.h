#ifndef COLIBRI_GLM_ENGINE_H
#define COLIBRI_GLM_ENGINE_H
/* glm_engine.h — the whole thing, assembled: GLM-5.2 forward pass on a streamed GGUF.
 *
 * Everything below is built from pieces that were each validated against an oracle before
 * being wired in, because every one of them fails SILENTLY when it is wrong:
 *
 *   kquant.h   9 quant types, bit-identical to ggml
 *   mla.h      absorbed MLA, matched to llama.cpp on real layer-0 tensors
 *   gguf.h     tensor directory, expert slicing verified against the real 254 GB file
 *   kvstore.h  KV quantization (q8_0: 0.79% score error; q4_0 was 10% and rejected)
 *
 * What this file adds is the glue, and the glue has its own traps. The three that are NOT
 * guessable from the tensor shapes, and that this file gets from the GGUF metadata:
 *
 *   rope.freq_base = 8e6      not the usual 10000
 *   expert_gating_func = 2    SIGMOID, not the softmax most MoEs use (and not the
 *                             sqrt-softplus DeepSeek-V4 uses — same family, different gate)
 *   expert_weights_scale=2.5  applied AFTER renormalizing the top-8 weights
 *
 * and the one that is a genuine subtlety of DeepSeek-style routing:
 *
 *   exp_probs_b steers WHICH experts are picked, but the mixing weights are read from the
 *   UNBIASED probabilities. Using the biased ones keeps the output fluent and quietly wrong.
 *
 * Layers 0..2 are dense FFN (leading_dense_block_count); 3..78 are MoE. Layer 78 is the
 * MTP/nextn head and is skipped unless speculative decoding is on.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf.h"
#include "kquant.h"
#include "mla.h"

typedef struct {
    int d_model, n_layer, n_dense, n_head, vocab;
    int q_lora, kv_lora, qk_nope, qk_rope, v_head;
    int n_exp, topk, n_shared, d_ff, d_ff_exp;
    float eps, rope_base, w_scale;
    int   w_norm, gating;
} GCfg;

typedef struct {
    /* attention */
    const void *q_a, *q_b, *kv_a, *k_b, *v_b, *o;
    int t_q_a, t_q_b, t_kv_a, t_k_b, t_v_b, t_o;
    const float *attn_norm, *q_a_norm, *kv_a_norm, *ffn_norm;
    /* ffn */
    const void *gate, *up, *down;                 /* dense layers */
    int t_gate, t_up, t_down;
    const void *sh_gate, *sh_up, *sh_down;        /* shared expert */
    int t_sh_gate, t_sh_up, t_sh_down;
    const float *router;                          /* [n_exp, d_model] f32 */
    const float *router_bias;                     /* exp_probs_b */
    const gguf_tensor *e_gate, *e_up, *e_down;    /* the streamed experts */
    int is_moe;
} GLayer;

/* Optional: run a layer's routed experts somewhere else (the GPU). Given the layer index,
 * the normed input, host pointers to the k experts' quantized weights and their mixing
 * weights, it writes the weighted sum of their FFN outputs into `acc` [d_model].
 * NULL means the CPU does it — 150x slower, and slow enough that the NVMe idles 77% of the
 * time waiting for it. */
typedef void (*glm_moe_hook)(void *ud, int layer, float *acc, const float *xn,
                             const void **eg, const void **eu, const void **ed,
                             const float *w, int k);

/* Same idea for the attention. On the CPU its six projections decode k-quants scalar and
 * run at 0.85 GB/s out of RAM that does 62 — 156 ms per layer, 9x the disk read they were
 * meant to hide behind. The hook lets the GPU do them instead. */
/* The output projection is 535 MB (Q4_K, 6144 x 154880) and runs ONCE per token — but on
 * the CPU's scalar k-quant path that is 0.63 s, which at 7 s/token is 9% of everything. */
typedef void (*glm_out_hook)(void *ud, float *logits, const float *xn);

typedef void (*glm_attn_hook)(void *ud, int layer, float *out, const float *x, int pos,
                              const MLACfg *mc, const MLALayer *ml,
                              MLACache *kv, MLAScratch *s, float rope_base);

typedef struct {
    gguf_model  m;
    GCfg        c;
    GLayer     *L;
    const gguf_tensor *emb;
    const void *out_w;   int t_out;
    const float *out_norm;
    MLACache   *kv;      /* one per layer */
    MLAScratch  s;
    glm_moe_hook  moe;    /* NULL = CPU */
    void         *moe_ud;
    glm_attn_hook attn;   /* NULL = CPU */
    glm_out_hook  out;    /* NULL = CPU */
    void         *out_ud;
    /* The shared expert is dimensionally identical to a routed one — same [6144x2048] gate/up
     * and [2048x6144] down. So a GPU MoE hook can just take it as one more expert with weight
     * 1.0 instead of leaving it on the CPU, where its 27.6 MB cost 32.5 ms EVERY layer:
     * 2.44 s/token, a third of the whole budget. When the hook does that, we must not also
     * run it here. */
    int           moe_shared;
    void         *attn_ud;
} GModel;

/* ---- the FFN, shared by the dense layers, the shared expert, and every routed expert ---- */
static void glm_ffn(float *out, const float *x, int d_in, int d_hid,
                    const void *wg, int tg, const void *wu, int tu,
                    const void *wd, int td, float *g, float *u){
    kq_gemm(g, x, wg, tg, 1, d_in, d_hid);
    kq_gemm(u, x, wu, tu, 1, d_in, d_hid);
    for(int i=0;i<d_hid;i++)
        g[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];      /* SwiGLU; glm-dsa has no clamp */
    kq_gemm(out, g, wd, td, 1, d_hid, d_in);
}

/* ---- the router ----
 * gating = 2 is SIGMOID (DeepSeek-V3 family). The bias picks; the unbiased probs weigh. */
#include <time.h>
static double glm_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* what is left on the CPU, and what it costs */
double GLM_T_dense = 0, GLM_T_route = 0, GLM_T_rest = 0;

/* ROUTING STABILITY — the premise of any predictive prefetch.
 *
 * The disk is a barrier: layer L+1's experts cannot be read until L+1's router has run, and
 * that needs L's output. So the drive idles through every compute phase. The only way to fill
 * it is to GUESS which experts the next token will want — and the natural guess is "the same
 * ones this layer wanted last token".
 *
 * That guess is worth building on only if routing is actually stable across tokens. If a
 * layer picks 8 fresh experts out of 256 every time, a predictive prefetch just floods the
 * drive with reads nobody will use, and makes things worse. So: measure the overlap first,
 * build second. GLM_ROUTE_LOG=1 turns this on. */
static int   GLM_route_log = 0;
static int  *GLM_prev_idx  = NULL;    /* [n_layer][topk] from the previous token */
long   GLM_route_hits = 0, GLM_route_total = 0;
long   GLM_route_hist[9] = {0,0,0,0,0,0,0,0,0};   /* how many of topk repeated */
static float GLM_last_score[64];
double GLM_margin_sum = 0; long GLM_margin_n = 0;
long   GLM_margin_hist[6] = {0,0,0,0,0,0};       /* margin between the 8th and 9th expert */

static void glm_route_log_init(int n_layer, int topk){
    GLM_route_log = getenv("GLM_ROUTE_LOG") != NULL;
    if(!GLM_route_log) return;
    GLM_prev_idx = (int*)malloc((size_t)n_layer*topk*sizeof(int));
    for(int i=0;i<n_layer*topk;i++) GLM_prev_idx[i] = -1;
}

static void glm_route(const GCfg *c, const float *x, const GLayer *L,
                      int *idx, float *w, float *probs){
    /* 256 experts x 6144 = 1.57M MACs, and this was taking 4.7 ms — about 334 MFLOPS out of a
     * machine that does tens of GFLOPS. It was a scalar, single-threaded loop, on a body whose
     * 256 iterations are completely independent of each other. Just parallelize it.
     *
     * The accumulator STAYS double. Dropping to float would vectorize better, but the router's
     * output is a discrete choice: 8 experts out of 256, by a margin that is often tiny. A
     * rounding difference that flips one pick does not make the model slightly worse, it makes
     * it a DIFFERENT model — and it would do so silently. Speed here is not worth that. */
    #pragma omp parallel for schedule(static)
    for(int e=0;e<c->n_exp;e++){
        double s = 0;
        const float *r = L->router + (size_t)e*c->d_model;
        for(int i=0;i<c->d_model;i++) s += (double)r[i]*x[i];
        probs[e] = 1.0f/(1.0f + expf(-(float)s));
    }
    for(int k=0;k<c->topk;k++){
        int best = -1; float bv = -INFINITY;
        for(int e=0;e<c->n_exp;e++){
            int taken = 0;
            for(int j=0;j<k;j++) if(idx[j]==e){ taken = 1; break; }
            if(taken) continue;
            float s = probs[e] + (L->router_bias ? L->router_bias[e] : 0.0f);
            if(s > bv){ bv = s; best = e; }
        }
        idx[k] = best;
        w[k]   = probs[best];                    /* UNBIASED — this is the trap */
        GLM_last_score[k] = bv;
    }

    /* HOW CLOSE IS THE CUT?
     *
     * The GPU attention agrees with the CPU one to 2.6e-06 — pure fp32 summation order, not a
     * bug — and yet the two produce different WORDS. The only mechanism that can amplify 1e-6
     * into a different token is this loop: it picks 8 experts out of 256 by score, and if the
     * 8th and the 9th are separated by less than the numerical noise, the choice is a coin
     * flip. A different expert is not a small perturbation. It is a different function.
     *
     * So measure the margin. If it is routinely near zero, then "the correct output" is not a
     * well-defined thing for this model, and no amount of engine correctness will fix that. */
    if(GLM_route_log){
        float ninth = -INFINITY;
        for(int e=0;e<c->n_exp;e++){
            int taken = 0;
            for(int j=0;j<c->topk;j++) if(idx[j]==e){ taken=1; break; }
            if(taken) continue;
            float s = probs[e] + (L->router_bias ? L->router_bias[e] : 0.0f);
            if(s > ninth) ninth = s;
        }
        double margin = (double)GLM_last_score[c->topk-1] - ninth;
        GLM_margin_sum += margin; GLM_margin_n++;
        if(margin < 1e-6) GLM_margin_hist[0]++;
        else if(margin < 1e-5) GLM_margin_hist[1]++;
        else if(margin < 1e-4) GLM_margin_hist[2]++;
        else if(margin < 1e-3) GLM_margin_hist[3]++;
        else if(margin < 1e-2) GLM_margin_hist[4]++;
        else GLM_margin_hist[5]++;
    }
    if(c->w_norm){
        float sum = 0;
        for(int k=0;k<c->topk;k++) sum += w[k];
        if(sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for(int k=0;k<c->topk;k++) w[k] /= sum;
    }
    for(int k=0;k<c->topk;k++) w[k] *= c->w_scale;
}

/* Fetch ALL topk experts of a layer at once.
 *
 * Not a convenience — a necessity. Fetching them one at a time means one outstanding read
 * at a time, and an NVMe at queue depth 1 is latency-bound, not bandwidth-bound: measured,
 * 2.36 GB/s at QD1 against 5.39 GB/s at QD8. Worse, a sequential fetch interleaves each
 * read with the matmul that consumes it, so the drive idles through the compute and the
 * observed rate collapses to under 1 GB/s.
 *
 * The router hands us all 8 expert ids at once, so there is no reason to ask for them one
 * at a time. The implementation behind this callback issues them concurrently. */
typedef int (*glm_fetch)(void *ud, int layer, const int *eid, int k,
                         const void **g, const void **u, const void **d);

static void glm_forward(GModel *M, int token, int pos, float *logits,
                        glm_fetch fetch, void *ud){
    const GCfg *c = &M->c;
    const int D = c->d_model;

    float *x   = (float*)malloc((size_t)D*sizeof(float));
    float *xn  = (float*)malloc((size_t)D*sizeof(float));
    float *att = (float*)malloc((size_t)D*sizeof(float));
    float *ffn = (float*)malloc((size_t)D*sizeof(float));
    float *acc = (float*)malloc((size_t)D*sizeof(float));
    int   nh   = c->d_ff > c->d_ff_exp ? c->d_ff : c->d_ff_exp;
    float *g   = (float*)malloc((size_t)nh*sizeof(float));
    float *u   = (float*)malloc((size_t)nh*sizeof(float));
    float *probs = (float*)malloc((size_t)c->n_exp*sizeof(float));
    int   *idx   = (int*)  malloc((size_t)c->topk*sizeof(int));
    float *w     = (float*)malloc((size_t)c->topk*sizeof(float));

    /* embedding */
    const int64_t erb = kq_row_bytes(M->emb->type, M->emb->ne[0]);
    kq_dequant_row(M->emb->type,
                   (const uint8_t*)gguf_data(&M->m, M->emb) + (size_t)token*erb, x, D);

    MLACfg mc = { c->n_head, c->qk_nope, c->qk_rope, c->v_head, c->kv_lora,
                  c->q_lora, D, c->eps,
                  1.0f/sqrtf((float)(c->qk_nope + c->qk_rope)) };

    for(int l = 0; l < c->n_layer; l++){
        const GLayer *L = &M->L[l];

        /* ---- attention ---- */
        MLALayer mm = {
            .pq_a = L->q_a, .pq_b = L->q_b, .pkv_a = L->kv_a,
            .pk_b = L->k_b, .pv_b = L->v_b, .po = L->o,
            .tq_a = L->t_q_a, .tq_b = L->t_q_b, .tkv_a = L->t_kv_a,
            .tk_b = L->t_k_b, .tv_b = L->t_v_b, .to = L->t_o,
            .attn_norm = L->attn_norm, .q_a_norm = L->q_a_norm, .kv_a_norm = L->kv_a_norm
        };
        if(M->attn) M->attn(M->attn_ud, l, att, x, pos, &mc, &mm, &M->kv[l], &M->s, c->rope_base);
        else        mla_step(att, x, pos, &mc, &mm, &M->kv[l], &M->s, c->rope_base);

        if(getenv("GLM_CHECK")) for(int i=0;i<D;i++) if(!isfinite(att[i])){
            fprintf(stderr,"[nan] ATTENTION output of layer %d (elem %d)\n", l, i); exit(1); }
        for(int i=0;i<D;i++) x[i] += att[i];

        /* ---- FFN / MoE ---- */
        if(getenv("GLM_CHECK")){
            double mx=0; int bad=-1;
            for(int i=0;i<D;i++){ if(!isfinite(x[i])) bad=i; if(fabs(x[i])>mx) mx=fabs(x[i]); }
            if(bad>=0){ fprintf(stderr,"[nan] x before ffn_norm, layer %d elem %d\n", l, bad); exit(1); }
            for(int i=0;i<D;i++) if(!isfinite(L->ffn_norm[i])){
                fprintf(stderr,"[nan] ffn_norm weight of layer %d is not finite\n", l); exit(1); }
            if(l>=6 && l<=9) fprintf(stderr,"[dbg] layer %d: max|x| before ffn_norm = %.3e\n", l, mx);
        }
        mla_rms(xn, x, L->ffn_norm, D, c->eps);

        if(!L->is_moe){
            const double td0 = glm_now();
            glm_ffn(ffn, xn, D, c->d_ff, L->gate, L->t_gate, L->up, L->t_up,
                    L->down, L->t_down, g, u);
            GLM_T_dense += glm_now() - td0;
            for(int i=0;i<D;i++) x[i] += ffn[i];
        } else {
            if(getenv("GLM_CHECK")) for(int i=0;i<D;i++) if(!isfinite(xn[i])){
                fprintf(stderr,"[nan] router input (xn) of layer %d is not finite\n", l);
                exit(1); }
            const double tr0 = glm_now();
            glm_route(c, xn, L, idx, w, probs);
            GLM_T_route += glm_now() - tr0;

            if(GLM_route_log && GLM_prev_idx){
                int *pv = GLM_prev_idx + (size_t)l*c->topk;
                if(pv[0] >= 0){          /* not the first token for this layer */
                    int same = 0;
                    for(int a=0;a<c->topk;a++)
                        for(int b=0;b<c->topk;b++)
                            if(idx[a] == pv[b]){ same++; break; }
                    GLM_route_hits += same;
                    GLM_route_total += c->topk;
                    GLM_route_hist[same > 8 ? 8 : same]++;
                }
                for(int a=0;a<c->topk;a++) pv[a] = idx[a];
            }
            if(getenv("GLM_CHECK")) for(int kk=0;kk<c->topk;kk++) if(idx[kk] < 0){
                fprintf(stderr,"[bad] router of layer %d picked expert -1\n", l); exit(1); }

            /* every routed expert of this layer, in flight at once */
            const void *eg[64], *eu[64], *ed[64];
            if(!fetch(ud, l, idx, c->topk, eg, eu, ed)){
                fprintf(stderr, "expert fetch failed at layer %d\n", l); exit(1); }

            if(M->moe){
                M->moe(M->moe_ud, l, acc, xn, eg, eu, ed, w, c->topk);
            } else {
                for(int i=0;i<D;i++) acc[i] = 0.f;
                for(int k=0;k<c->topk;k++){
                    glm_ffn(ffn, xn, D, c->d_ff_exp,
                            eg[k], L->e_gate->type, eu[k], L->e_up->type, ed[k], L->e_down->type, g, u);
                    for(int i=0;i<D;i++) acc[i] += w[k]*ffn[i];
                }
            }
            if(M->moe_shared){
                for(int i=0;i<D;i++) x[i] += acc[i];   /* the hook already folded it in */
            } else {
                /* the shared expert runs for EVERY token, no routing, no weight */
                glm_ffn(ffn, xn, D, c->d_ff_exp, L->sh_gate, L->t_sh_gate,
                        L->sh_up, L->t_sh_up, L->sh_down, L->t_sh_down, g, u);
                if(getenv("GLM_CHECK")) for(int i=0;i<D;i++) if(!isfinite(ffn[i])){
                    fprintf(stderr,"[nan] SHARED EXPERT of layer %d\n", l); exit(1); }
                for(int i=0;i<D;i++) x[i] += acc[i] + ffn[i];
            }
        }
    }

    mla_rms(xn, x, M->out_norm, D, c->eps);
    if(M->out) M->out(M->out_ud, logits, xn);
    else       kq_gemm(logits, xn, M->out_w, M->t_out, 1, D, c->vocab);

    free(x); free(xn); free(att); free(ffn); free(acc);
    free(g); free(u); free(probs); free(idx); free(w);
}

#endif /* COLIBRI_GLM_ENGINE_H */
