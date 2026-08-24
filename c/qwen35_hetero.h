#ifndef COLIBRI_QWEN35_HETERO_H
#define COLIBRI_QWEN35_HETERO_H
/* qwen35_hetero.h — the per-layer scheduler: which half of the model runs where.
 *
 * THE SPLIT, AND THE ONE NUMBER THAT DECIDES IT
 *
 * Per generated token the model reads 15.6 GiB of weights. The FFN is 9.65 of them and it
 * stays in RAM: it does not fit in 8 GiB of VRAM, and streaming it over this machine's x8
 * link at ~25 GB/s would be slower than reading it in place at 36. Everything else — gdn
 * 3.222, attn 0.889, output 0.971 — does fit, and once resident it is read at VRAM speed
 * and costs the DRAM controller nothing.
 *
 * That is the entire mechanism. Not overlap: the layer chain is strictly sequential inside a
 * token (attn -> +res -> ffn -> +res -> attn), so there is nothing to overlap. The saving is
 * bytes that stop being read on the slow side. Measured here: GPU 5.08 GiB at ~254 GB/s is
 * ~20 ms against the CPU's ~288 ms, so serializing the two costs ~7% and buys 37%.
 *
 * WHAT THE LEFTOVER VRAM DOES
 *
 * After the compute tensors and the 146 MiB of recurrent state there is ~1.8 GiB left. Two
 * candidates want it and they are worth EXACTLY the same: a KV window and a slice of FFN
 * layers both move ~1.8 GiB from the 36 GB/s side to the 254 GB/s side. So the tie is broken
 * on shape, not on speed — KV grows with context and FFN does not, so KV is served first up
 * to what the context actually needs and the FFN takes the remainder.
 *
 * SPLIT CONTEXT
 *
 * When the context outgrows the window, tokens [0, win) are attended on the GPU and
 * [win, T) on the CPU where those bytes already are, and the two partial results are merged
 * by the standard flash-decoding combine. Two vectors per layer cross PCIe instead of the
 * cache. It is exact up to rounding, which is what keeps the CPU comparison a hard gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "qwen35_cpu.h"
#include "qwen35_batch.h"
#include "qwen35_cuda.h"

typedef struct {
    void  *wqkv, *wgate, *w_alpha, *w_beta, *w_out;    /* gdn, quantized, device */
    float *conv1d, *ssm_a, *dt_bias, *ssm_norm;
    void  *wq, *wk, *wv, *wo;                          /* attn, quantized, device */
    float *q_norm, *k_norm;
    float *attn_norm, *post_attn_norm;
    void  *ffn_gate, *ffn_up, *ffn_down;               /* NULL => this layer's FFN is CPU-side */
    int    ffn_gpu;
} Q35CuLayer;

/* Per-stage timing costs a stream sync per stage: 192 a token instead of the 54 the schedule
 * actually needs, and each one is ~15 us of the CPU spinning instead of running the FFN. It
 * is how you find out where the time goes, so it stays -- behind a switch. */
static int g_q35_stage_timing = 0;

typedef struct {
    const Q35Model *M;
    Q35CuLayer  L[Q35_MAX_LAYER];
    void       *output;
    float      *output_norm;

    /* work, device */
    float *x, *xn, *t, *qkv, *z, *ab, *o, *qg, *k, *v;
    float *lse, *o2, *lse2, *scratch, *logits, *ffn_g, *ffn_u;
    float *S, *conv;

    /* the device KV window: tokens [0, win) of every attention layer */
    uint8_t *Kc, *Vc;
    int      win, kv_fmt, kv_layers;
    int64_t  kv_half;          /* bytes of one token's K (or V) in one layer */

    /* pinned staging for the FFN hand-off */
    float *hxn, *ht, *hqg, *ho, *hlse;

    int64_t vram_weights, vram_state, vram_kv, vram_ffn, vram_work;
    int     n_ffn_gpu;
    int     ok;

    /* ---- the streamed FFN ----
     * For layers whose weights did not fit in VRAM, the GPU still does a share of the work
     * by pulling its slice over PCIe WHILE the CPU computes the rest in place. The two read
     * the same DRAM, so the win is not free -- but the CPU reaches 38 GB/s and the DMA
     * engine adds 27.7 against a 64.8 GB/s ceiling, and the sides run at the same time.
     * Measured split point is therefore near 42% to the GPU; `stream_f` tracks it live,
     * because the balance moves with context length and with how busy the copy engine is. */
    uint8_t *stage;              /* device landing zone for one layer's slice */
    int64_t  stage_bytes;
    float   *sg, *su, *st;       /* device: gate/up partials and the output */
    float   *hg, *hu, *ht2;      /* pinned host: the CPU's partials */
    double   stream_f;           /* fraction of rows given to the GPU, 0 = off */
    struct { int64_t gb, db; int r, s_unused; } slice[2];
    int      stage_half;
    double   t_cpu_side, t_gpu_side;
    int64_t  stream_bytes;
    int      stream_on, host_pinned;
} Q35Cu;

/* ---------------- upload ---------------- */

static void *q35cu_up_t(Q35Cu *G, const Q35Model *M, const gguf_tensor *T, int64_t *acc){
    if(!T) return NULL;
    if(!q35cu_type_ok(T->type)){
        fprintf(stderr, "qwen35_cuda: %s has type %s, which has no device kernel\n",
                T->name, kq_name(T->type));
        G->ok = 0; return NULL;
    }
    void *d = q35cu_alloc((size_t)T->bytes);
    if(!d){
        fprintf(stderr, "qwen35_cuda: out of VRAM uploading %s (%.1f MiB): %s\n",
                T->name, T->bytes/1048576.0, q35cu_error());
        G->ok = 0; return NULL;
    }
    if(!q35cu_h2d(d, q35_wdata(M, T), (size_t)T->bytes)){
        fprintf(stderr, "qwen35_cuda: upload of %s failed: %s\n", T->name, q35cu_error());
        G->ok = 0; return NULL;
    }
    *acc += T->bytes;
    return d;
}

static float *q35cu_up_f(Q35Cu *G, const float *src, int n, int64_t *acc){
    if(!src) return NULL;
    float *d = (float*)q35cu_alloc((size_t)n*4);
    if(!d){ G->ok = 0; return NULL; }
    if(!q35cu_h2d(d, src, (size_t)n*4)){ G->ok = 0; return NULL; }
    *acc += (int64_t)n*4;
    return d;
}

static float *q35cu_work(Q35Cu *G, int64_t n){
    float *d = (float*)q35cu_alloc((size_t)n*4);
    if(!d){ fprintf(stderr, "qwen35_cuda: work buffer (%lld floats): %s\n",
                    (long long)n, q35cu_error()); G->ok = 0; return NULL; }
    G->vram_work += n*4;
    return d;
}

/* How many tokens of KV window fit in `budget` bytes. */
static int q35cu_win_tokens(const Q35Cu *G, int64_t budget){
    const int64_t per = (int64_t)G->kv_layers * G->kv_half * 2;   /* K and V */
    if(per <= 0) return 0;
    int64_t n = budget/per;
    if(n < 0) n = 0;
    return (int)(n > (int64_t)1<<30 ? (int64_t)1<<30 : n);
}

/* ---------------- init ----------------
 *
 * Order matters: the compute tensors are mandatory and go first, so a capacity failure
 * happens here and loudly rather than as a mystery slowdown three layers into a token. */
static int q35cu_model_init(Q35Cu *G, const Q35Model *M, int n_ctx, int kv_fmt){
    memset(G, 0, sizeof *G);
    G->M = M; G->ok = 1; G->kv_fmt = kv_fmt;
    const Q35Cfg *c = &M->c;

    if(!q35cu_init(0)){
        fprintf(stderr, "qwen35_cuda: %s\n", q35cu_error());
        return 0;
    }
    size_t vfree = 0, vtot = 0;
    q35cu_mem(&vfree, &vtot);
    if(q35cu_log_on()) fprintf(stderr, "  [kernel] device        %s, %.2f GiB free\n",
                               q35cu_name(), vfree/1073741824.0);

    int64_t w = 0;
    for(int il = 0; il < c->n_layer && G->ok; il++){
        const Q35Layer *S = &M->L[il];
        Q35CuLayer *D = &G->L[il];
        D->attn_norm      = q35cu_up_f(G, S->attn_norm,      c->d_model, &w);
        D->post_attn_norm = q35cu_up_f(G, S->post_attn_norm, c->d_model, &w);
        if(S->kind == Q35_LAYER_ATTN){
            D->wq = q35cu_up_t(G, M, S->wq, &w);
            D->wk = q35cu_up_t(G, M, S->wk, &w);
            D->wv = q35cu_up_t(G, M, S->wv, &w);
            D->wo = q35cu_up_t(G, M, S->wo, &w);
            D->q_norm = q35cu_up_f(G, S->q_norm, c->d_head, &w);
            D->k_norm = q35cu_up_f(G, S->k_norm, c->d_head, &w);
        } else {
            D->wqkv    = q35cu_up_t(G, M, S->wqkv,    &w);
            D->wgate   = q35cu_up_t(G, M, S->wgate,   &w);
            D->w_alpha = q35cu_up_t(G, M, S->w_alpha, &w);
            D->w_beta  = q35cu_up_t(G, M, S->w_beta,  &w);
            D->w_out   = q35cu_up_t(G, M, S->w_out,   &w);
            D->conv1d   = q35cu_up_f(G, S->conv1d,  c->d_conv*c->conv_dim, &w);
            D->ssm_a    = q35cu_up_f(G, S->ssm_a,   c->n_v_heads, &w);
            D->dt_bias  = q35cu_up_f(G, S->dt_bias, c->n_v_heads, &w);
            D->ssm_norm = q35cu_up_f(G, S->ssm_norm, c->d_head_v, &w);
        }
    }
    G->output      = q35cu_up_t(G, M, M->output, &w);
    G->output_norm = q35cu_up_f(G, M->output_norm, c->d_model, &w);
    G->vram_weights = w;
    if(!G->ok) return 0;

    /* recurrent state: 48 layers x 48 heads x 64 KiB, constant in context length */
    int n_gdn = 0, n_attn = 0;
    for(int il = 0; il < c->n_layer; il++)
        (c->kind[il] == Q35_LAYER_ATTN) ? n_attn++ : n_gdn++;
    const int64_t s_n = (int64_t)n_gdn*c->n_v_heads*c->d_state*c->d_head_v;
    const int64_t c_n = (int64_t)n_gdn*(c->d_conv-1)*c->conv_dim;
    G->S    = (float*)q35cu_alloc((size_t)s_n*4);
    G->conv = (float*)q35cu_alloc((size_t)c_n*4);
    if(!G->S || !G->conv){ fprintf(stderr, "qwen35_cuda: gdn state: %s\n", q35cu_error()); return 0; }
    q35cu_zero(G->S, (size_t)s_n*4);
    q35cu_zero(G->conv, (size_t)c_n*4);
    G->vram_state = (s_n + c_n)*4;

    /* work buffers */
    const int nh = c->n_head, dh = c->d_head;
    G->x   = q35cu_work(G, c->d_model);
    G->xn  = q35cu_work(G, c->d_model);
    G->t   = q35cu_work(G, c->d_model);
    G->qkv = q35cu_work(G, c->conv_dim);
    G->z   = q35cu_work(G, c->d_inner);
    G->ab  = q35cu_work(G, 2*c->n_v_heads);
    G->o   = q35cu_work(G, nh*dh > c->d_inner ? nh*dh : c->d_inner);
    G->qg  = q35cu_work(G, 2*nh*dh);
    G->k   = q35cu_work(G, c->n_head_kv*dh);
    G->v   = q35cu_work(G, c->n_head_kv*dh);
    G->lse = q35cu_work(G, nh);
    G->o2  = q35cu_work(G, nh*dh);
    G->lse2= q35cu_work(G, nh);
    G->scratch = q35cu_work(G, (int64_t)q35cu_attn_scratch(nh, dh, n_ctx));
    G->logits  = q35cu_work(G, c->vocab);
    if(!G->ok) return 0;

    /* the KV window, then whatever is left goes to FFN layers */
    G->kv_layers = n_attn + (c->n_layer_mtp ? 1 : 0);
    G->kv_half   = q35_kv_entry_bytes(kv_fmt, c->n_head_kv*c->d_head);
    q35cu_mem(&vfree, &vtot);
    int64_t spare = (int64_t)vfree - (256<<20);        /* leave headroom for the allocator */
    if(spare < 0) spare = 0;

    /* The window is a CACHE, not a context limit: anything past it is attended on the CPU
     * out of the host tier and merged back with the flash combine (see §7.4). Capping it
     * explicitly is the knob that trades KV window against resident FFN layers — they are
     * worth the same per byte, so which one wins depends on how long the context actually
     * gets. */
    const int64_t kv_need = (int64_t)n_ctx * G->kv_layers * G->kv_half * 2;
    int64_t kv_budget = kv_need < spare ? kv_need : spare;
    G->win = q35cu_win_tokens(G, kv_budget);
    if(G->win > n_ctx) G->win = n_ctx;
    /* DEFAULT CAP, and the arithmetic behind it.
     *
     * A resident FFN layer costs 165 MiB and saves ~2.6 ms of every token, forever. The same
     * 165 MiB as KV window holds ~4978 tokens, and those only save anything once the context
     * has actually grown past the window — everything beyond it is attended on the CPU at
     * 36 GB/s, which is 2.6 ms per 2688 tokens. Solve the two against each other and a
     * 32768-token window does not overtake an 8192-token one until the context reaches
     * ~35k; at 8192 the breakeven is ~24k. So the window is capped low by default and the
     * VRAM goes to FFN layers, which pay from the first token. Raise it with
     * COLIBRI_CUDA_KV_TOK when the sessions really are that long. */
    { const char *we = getenv("COLIBRI_CUDA_KV_TOK");
      int cap = we ? atoi(we) : 8192;
      if(cap >= 0 && cap < G->win) G->win = cap; }
    if(G->win > 0){
        const size_t nb = (size_t)G->kv_layers*(size_t)G->win*(size_t)G->kv_half;
        G->Kc = (uint8_t*)q35cu_alloc(nb);
        G->Vc = (uint8_t*)q35cu_alloc(nb);
        if(!G->Kc || !G->Vc){ G->win = 0; q35cu_free(G->Kc); q35cu_free(G->Vc); G->Kc = G->Vc = NULL; }
        else G->vram_kv = 2*(int64_t)nb;
    }

    /* The streaming staging area, sized for the largest single-layer slice we would ever
     * ask for. Capped at half the remaining VRAM: a landing zone big enough to starve the
     * resident FFN of a layer is a net loss. */
    {
        const char *se = getenv("COLIBRI_CUDA_STREAM");
        G->stream_f = se ? atof(se) : 0.60;
        if(G->stream_f > 0.9) G->stream_f = 0.9;
        if(G->stream_f > 0){
            q35cu_mem(&vfree, &vtot);
            int64_t cap = (int64_t)vfree - (160<<20);
            int64_t need = 0;
            for(int il = 0; il < c->n_layer; il++){
                const Q35Layer *S = &M->L[il];
                const int64_t g = q35_tbytes(S->ffn_gate), u = q35_tbytes(S->ffn_up),
                              d = q35_tbytes(S->ffn_down);
                const int64_t t = 2*(int64_t)(G->stream_f*1.05*(double)(g + u + d));
                if(t > need) need = t;
            }
            if(need > cap) need = cap;
            /* Each half must be 256-byte aligned: k_gemv_q4k_r pulls d/dmin and the twelve
             * scale bytes as 32-bit loads, and an odd half-offset makes every one of them a
             * misaligned access. CUDA reports that as "misaligned address" from the NEXT
             * stream sync, several layers away from the cause, and the logits come back all
             * zero — which reads like a scheduling bug and is an alignment bug. */
            need &= ~(int64_t)511;
            if(need > (16<<20)){
                G->stage = (uint8_t*)q35cu_alloc((size_t)need);
                G->sg  = (float*)q35cu_alloc((size_t)c->d_ff*4);
                G->su  = (float*)q35cu_alloc((size_t)c->d_ff*4);
                G->st  = (float*)q35cu_alloc((size_t)c->d_model*4);
                G->hg  = (float*)q35cu_host_alloc((size_t)c->d_ff*4);
                G->hu  = (float*)q35cu_host_alloc((size_t)c->d_ff*4);
                G->ht2 = (float*)q35cu_host_alloc((size_t)c->d_model*4);
                if(G->stage && G->sg && G->su && G->st && G->hg && G->hu && G->ht2){
                    G->stage_bytes = need; G->stream_on = 1;
                } else {
                    q35cu_free(G->stage); G->stage = NULL;
                }
            }
        }
    }
    /* FFN layers into the remainder. Whole layers, not row slices: a split gate/up/down needs
     * the down projection cut along its CONTRACTED dim, which is a strided slice of every row
     * and buys nothing here, because the two sides cannot overlap anyway. */
    q35cu_mem(&vfree, &vtot);
    spare = (int64_t)vfree - (320<<20);
    const char *fe = getenv("COLIBRI_CUDA_FFN_GB");
    if(fe){ const double gb = atof(fe); const int64_t cap = (int64_t)(gb*1073741824.0);
            if(cap < spare) spare = cap; }
    if(spare > 0){
        int64_t need_ffn = 0;
        G->ffn_g = q35cu_work(G, c->d_ff);
        G->ffn_u = q35cu_work(G, c->d_ff);
        spare -= 2*(int64_t)c->d_ff*4;
        for(int il = 0; il < c->n_layer; il++){
            const Q35Layer *S = &M->L[il];
            const int64_t need = q35_tbytes(S->ffn_gate) + q35_tbytes(S->ffn_up) + q35_tbytes(S->ffn_down);
            if(need > spare) break;
            int64_t got = 0;
            Q35CuLayer *D = &G->L[il];
            D->ffn_gate = q35cu_up_t(G, M, S->ffn_gate, &got);
            D->ffn_up   = q35cu_up_t(G, M, S->ffn_up,   &got);
            D->ffn_down = q35cu_up_t(G, M, S->ffn_down, &got);
            if(!G->ok || !D->ffn_gate || !D->ffn_up || !D->ffn_down){
                q35cu_free(D->ffn_gate); q35cu_free(D->ffn_up); q35cu_free(D->ffn_down);
                D->ffn_gate = D->ffn_up = D->ffn_down = NULL;
                G->ok = 1;                        /* not fatal: the CPU keeps this layer */
                break;
            }
            D->ffn_gpu = 1; G->n_ffn_gpu++; spare -= got; need_ffn += got;
        }
        G->vram_ffn = need_ffn;
    }

    /* pinned staging: the FFN hand-off and, at split context, the attention partials */
    G->hxn  = (float*)q35cu_host_alloc((size_t)c->d_model*4);
    G->ht   = (float*)q35cu_host_alloc((size_t)c->d_model*4);
    G->hqg  = (float*)q35cu_host_alloc((size_t)2*nh*dh*4);
    G->ho   = (float*)q35cu_host_alloc((size_t)nh*dh*4);
    G->hlse = (float*)q35cu_host_alloc((size_t)nh*4);
    if(!G->hxn || !G->ht || !G->hqg || !G->ho || !G->hlse){
        fprintf(stderr, "qwen35_cuda: pinned host staging failed\n"); return 0;
    }
    if(!q35cu_sync()){ fprintf(stderr, "qwen35_cuda: %s\n", q35cu_error()); return 0; }
    return G->ok;
}

/* The weights must be PINNED or the DMA runs at 6.8 GB/s instead of 27.7 and the whole
 * split is a loss. q35_reside_anon puts them all in one anonymous mapping, so this is one
 * call -- and it must happen after residency, not before. */
static void q35cu_pin_weights(Q35Cu *G, Q35Model *M){
    if(!G->stream_on || !M->anon || M->anon_bytes <= 0) return;
    if(q35cu_host_register(M->anon, (size_t)M->anon_bytes)) G->host_pinned = 1;
    else {
        fprintf(stderr, "qwen35_cuda: cannot pin weights (%s) — streaming disabled,"
                        " pageable DMA would be slower than the CPU\n", q35cu_error());
        G->stream_on = 0;
    }
}

static void q35cu_model_free(Q35Cu *G){
    const Q35Cfg *c = &G->M->c;
    for(int il = 0; il < c->n_layer; il++){
        Q35CuLayer *D = &G->L[il];
        void *p[] = { D->wqkv,D->wgate,D->w_alpha,D->w_beta,D->w_out,D->conv1d,D->ssm_a,
                      D->dt_bias,D->ssm_norm,D->wq,D->wk,D->wv,D->wo,D->q_norm,D->k_norm,
                      D->attn_norm,D->post_attn_norm,D->ffn_gate,D->ffn_up,D->ffn_down };
        for(unsigned i = 0; i < sizeof p/sizeof *p; i++) q35cu_free(p[i]);
    }
    void *p[] = { G->output,G->output_norm,G->x,G->xn,G->t,G->qkv,G->z,G->ab,G->o,G->qg,
                  G->k,G->v,G->lse,G->o2,G->lse2,G->scratch,G->logits,G->ffn_g,G->ffn_u,
                  G->S,G->conv,G->Kc,G->Vc,G->stage,G->sg,G->su,G->st };
    for(unsigned i = 0; i < sizeof p/sizeof *p; i++) q35cu_free(p[i]);
    q35cu_host_free(G->hxn); q35cu_host_free(G->ht);
    q35cu_host_free(G->hqg); q35cu_host_free(G->ho); q35cu_host_free(G->hlse);
    q35cu_host_free(G->hg); q35cu_host_free(G->hu); q35cu_host_free(G->ht2);
    memset(G, 0, sizeof *G);
}

/* The device window ends on a host-tier chunk boundary. Otherwise one chunk is half on the
 * GPU and half in RAM, and the CPU's half-pass has to skip inside a chunk it just faulted in
 * whole — correct, but it reads bytes nobody uses. */
static void q35cu_align_win(Q35Cu *G, int chunk){
    if(chunk > 0 && G->win > chunk) G->win -= G->win % chunk;
}

/* Tell q35_reside_anon which FFN tensors are already in VRAM, so it does not pull a second
 * copy into RAM. Call between q35cu_model_init and q35_reside_anon. */
static void q35cu_mark_resident_elsewhere(Q35Cu *G){
    const Q35Model *M = G->M;
    static uint8_t *flag = NULL;
    free(flag);
    flag = (uint8_t*)calloc((size_t)M->m.n_tensors, 1);
    if(!flag) return;
    for(int il = 0; il < M->c.n_layer; il++){
        if(!G->L[il].ffn_gpu) continue;
        const gguf_tensor *g[3] = { M->L[il].ffn_gate, M->L[il].ffn_up, M->L[il].ffn_down };
        for(int k = 0; k < 3; k++){
            if(!g[k]) continue;
            const int i = (int)(g[k] - M->m.t);
            if(i >= 0 && i < M->m.n_tensors) flag[i] = 1;
        }
    }
    g_q35_elsewhere = flag;
}

static void q35cu_state_reset(Q35Cu *G){
    const Q35Cfg *c = &G->M->c;
    int n_gdn = 0;
    for(int il = 0; il < c->n_layer; il++) if(c->kind[il] != Q35_LAYER_ATTN) n_gdn++;
    q35cu_zero(G->S,    (size_t)n_gdn*c->n_v_heads*c->d_state*c->d_head_v*4);
    q35cu_zero(G->conv, (size_t)n_gdn*(c->d_conv-1)*c->conv_dim*4);
    q35cu_sync();
}

static void q35cu_report(const Q35Cu *G, FILE *o){
    size_t f = 0, t = 0; q35cu_mem(&f, &t);
    fprintf(o, "\n  device: %s\n", q35cu_name());
    fprintf(o, "    weights %6.3f GiB   state %6.3f   kv window %6.3f (%d tok)   ffn %6.3f (%d layers)   work %6.3f\n",
            G->vram_weights/1073741824.0, G->vram_state/1073741824.0,
            G->vram_kv/1073741824.0, G->win, G->vram_ffn/1073741824.0, G->n_ffn_gpu,
            G->vram_work/1073741824.0);
    fprintf(o, "    vram %.2f used / %.2f GiB, %.2f free\n",
            (t-f)/1073741824.0, t/1073741824.0, f/1073741824.0);
    if(G->stream_on)
        fprintf(o, "    streamed ffn: %.0f%% of each non-resident layer to the GPU over PCIe,"
                   " %.0f MiB staging%s\n", 100*G->stream_f, G->stage_bytes/1048576.0,
                G->host_pinned ? ", weights pinned" : ", NOT PINNED (slow)");
}

/* ---------------- the CPU's half of a split context ----------------
 *
 * Computes attention over tokens [t_lo, t_hi) straight out of the host tier — where those
 * bytes already are — and returns the partial in flash-decoding form: the normalized output
 * and the log-sum-exp of its scores. The GPU's half arrives the same way and the two combine
 * exactly. This is what makes the KV window a CACHE rather than a context limit. */
static void q35_attn_host_partial(Q35State *R, const float *qg, int slot,
                                  int t_lo, int t_hi, float *o, float *lse){
    const Q35Cfg *c = &R->M->c;
    const int dh = c->d_head, nh = c->n_head, nkv = c->n_head_kv, grp = nh/nkv;
    const float scale = 1.0f/sqrtf((float)dh);
    KvTier *KV = &R->kv;
    const int64_t hb = q35_kv_entry_bytes(KV->fmt, dh);
    const int c0 = t_lo/KV->chunk, c1 = (t_hi + KV->chunk - 1)/KV->chunk;

    for(int h = 0; h < nh; h++) lse[h] = -INFINITY;
    memset(o, 0, sizeof(float)*(size_t)nh*dh);
    if(t_hi <= t_lo) return;

    for(int ci = c0; ci < c1; ci++){
        kvt_prefetch(KV, slot, ci+1);
        const uint8_t *blk = kvt_chunk(KV, slot, ci);
        if(!blk) continue;
        int a = ci*KV->chunk, b = a + KV->chunk;
        if(a < t_lo) a = t_lo;
        if(b > t_hi) b = t_hi;
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++){
            const float *q = qg + (size_t)h*2*dh;
            float *sc = R->att + (size_t)h*R->n_ctx;
            const int hk = h/grp;
            for(int t = a; t < b; t++){
                const uint8_t *kt = blk + kvt_koff(KV, t - ci*KV->chunk) + hk*hb;
                sc[t] = kvt_dot(KV, q, kt, dh)*scale;
            }
        }
    }
    /* one pass for the max, then an unnormalized weighted sum; the caller's merge divides */
    #pragma omp parallel for schedule(static)
    for(int h = 0; h < nh; h++){
        const float *sc = R->att + (size_t)h*R->n_ctx;
        float m = -INFINITY;
        for(int t = t_lo; t < t_hi; t++) if(sc[t] > m) m = sc[t];
        lse[h] = m;
    }
    for(int ci = c0; ci < c1; ci++){
        kvt_prefetch(KV, slot, ci+1);
        const uint8_t *blk = kvt_chunk(KV, slot, ci);
        if(!blk) continue;
        int a = ci*KV->chunk, b = a + KV->chunk;
        if(a < t_lo) a = t_lo;
        if(b > t_hi) b = t_hi;
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++){
            const float *sc = R->att + (size_t)h*R->n_ctx;
            const int hk = h/grp;
            float *oh = o + (size_t)h*dh;
            const float m = lse[h];
            for(int t = a; t < b; t++){
                const float p = expf(sc[t] - m);
                const uint8_t *vt = blk + kvt_voff(KV, t - ci*KV->chunk) + hk*hb;
                kvt_axpy(KV, oh, p, vt, dh);
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for(int h = 0; h < nh; h++){
        const float *sc = R->att + (size_t)h*R->n_ctx;
        const float m = lse[h];
        double s = 0;
        for(int t = t_lo; t < t_hi; t++) s += exp((double)(sc[t] - m));
        float *oh = o + (size_t)h*dh;
        const float inv = s > 0 ? (float)(1.0/s) : 0.f;
        for(int i = 0; i < dh; i++) oh[i] *= inv;
        lse[h] = m + (float)log(s > 0 ? s : 1.0);
    }
}

/* ---------------- layers ---------------- */

/* ---------------- the split FFN: CPU in place, GPU over PCIe, at the same time ----------
 *
 * The layer is cut so that NEITHER SIDE NEEDS THE OTHER'S RESULT until the very end.
 *
 * gate and up are [d_model, d_ff] with the output index slowest, so rows [0,r) are a
 * contiguous byte range -- the GPU takes those, the CPU takes [r, d_ff).
 *
 * down is [d_ff, d_model] and the obvious cut, by output row, does NOT work: every output
 * row needs the WHOLE swiglu vector, so each side would have to wait for the other's half
 * and the pipeline collapses into two barriers. Cutting it along the CONTRACTION instead --
 * the GPU sums i in [0,r), the CPU sums i in [r,d_ff) -- means each side uses exactly the
 * part of the swiglu vector it computed itself. The two partial sums are added at the end.
 * That costs one strided DMA (a column band of every row, cudaMemcpy2DAsync) and buys the
 * only true concurrency in the whole engine.
 *
 * r is a multiple of 256 because that is a k-quant super-block: cutting inside one would
 * mean decoding a block from the middle, which the format does not allow. */
typedef struct { int64_t gb, db; int r, s_unused; } Q35Slice;

static Q35Slice q35cu_slice(const Q35Cu *G, const Q35Layer *L, double f){
    const Q35Cfg *c = &G->M->c;
    Q35Slice S; memset(&S, 0, sizeof S);
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, c->d_model);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, c->d_ff);
    int r = (int)(f*(double)c->d_ff);
    r -= r % KQ_QK_K;                                 /* whole super-blocks only */
    for(;;){
        if(r < KQ_QK_K){ r = 0; break; }
        S.gb = (int64_t)r*grb;
        S.db = (int64_t)(r/KQ_QK_K)*kq_typesize(L->ffn_down->type)*(int64_t)c->d_model;
        if(2*S.gb + S.db <= G->stage_bytes/2) break;   /* double buffered */
        r -= KQ_QK_K;
    }
    S.r = r;
    if(r == 0){ S.gb = S.db = 0; }
    return S;
}

/* Issue one layer's slice onto the copy stream. Called a layer AHEAD of where it is used,
 * so the 2.6 ms of DMA sits underneath the previous layer's compute instead of in front of
 * this one's. */
static void q35cu_stream_issue(Q35Cu *G, int il, int half){
    uint8_t *dst = G->stage + (int64_t)half*(G->stage_bytes/2);
    Q35Slice *S = (Q35Slice*)&G->slice[half];
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35Layer *L = &M->L[il];
    *S = q35cu_slice(G, L, G->stream_f);
    if(!S->r){ q35cu_copy_mark(half); return; }
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, c->d_model);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, c->d_ff);
    const int64_t dw  = (int64_t)(S->r/KQ_QK_K)*kq_typesize(L->ffn_down->type);
    q35cu_h2d_async(dst,            q35_wdata(M, L->ffn_gate), (size_t)S->gb);
    q35cu_h2d_async(dst + S->gb,    q35_wdata(M, L->ffn_up),   (size_t)S->gb);
    q35cu_h2d_2d_async(dst + 2*S->gb, (size_t)dw,
                       q35_wdata(M, L->ffn_down), (size_t)drb,
                       (size_t)dw, (size_t)c->d_model);
    G->stream_bytes += 2*S->gb + S->db;
    q35cu_copy_mark(half);
}

/* The next layer whose FFN is not already resident in VRAM, or -1. */
static int q35cu_next_stream_layer(const Q35Cu *G, int from){
    for(int il = from; il < G->M->c.n_layer; il++) if(!G->L[il].ffn_gpu) return il;
    return -1;
}

static void q35cu_ffn_cpu_half(Q35Cu *G, const Q35Layer *L, const float *xn, float *out, int r);

/* One layer, both sides at once. Returns 0 if there was no slice and the caller should run
 * the plain CPU FFN. */
static int q35cu_ffn_split(Q35Cu *G, int il, const float *xn_dev, float *out_dev, int half){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35Layer *L = &M->L[il];
    const int r = G->slice[half].r;
    if(r <= 0) return 0;
    const int64_t gb = G->slice[half].gb;
    uint8_t *base = G->stage + (int64_t)half*(G->stage_bytes/2);
    const int Dm = c->d_model;

    q35cu_sync();
    q35cu_d2h(G->hxn, xn_dev, sizeof(float)*Dm);

    /* queue the GPU half behind its DMA -- this does not block the host */
    const double tg = q35_clk();
    q35cu_stream_join(half);
    q35cu_gemv(G->sg, xn_dev, base,            L->ffn_gate->type, Dm, r);
    q35cu_gemv(G->su, xn_dev, base + gb,       L->ffn_up->type,   Dm, r);
    q35cu_swiglu(G->sg, G->su, r);
    q35cu_gemv(out_dev, G->sg, base + 2*gb,    L->ffn_down->type, r, Dm);

    /* the CPU half runs NOW, against the GPU and against the next layer's DMA */
    const double tc = q35_clk();
    q35cu_ffn_cpu_half(G, L, G->hxn, G->ht2, r);
    G->t_cpu_side += q35_clk() - tc;

    q35cu_h2d(G->st, G->ht2, sizeof(float)*Dm);
    q35cu_add(out_dev, G->st, Dm);
    G->t_gpu_side += q35_clk() - tg;
    return 1;
}

/* The CPU's half: rows [r, d_ff) of gate and up, then the tail of the contraction in down. */
static void q35cu_ffn_cpu_half(Q35Cu *G, const Q35Layer *L, const float *xn, float *out, int r){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const int Dff = c->d_ff, Dm = c->d_model;
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, Dm);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, Dff);
    const int nrow = Dff - r;
    kq_gemv_fast(G->hg + r, xn, (const uint8_t*)q35_wdata(M, L->ffn_gate) + (int64_t)r*grb,
                 L->ffn_gate->type, Dm, nrow);
    kq_gemv_fast(G->hu + r, xn, (const uint8_t*)q35_wdata(M, L->ffn_up) + (int64_t)r*grb,
                 L->ffn_up->type, Dm, nrow);
    for(int i = r; i < Dff; i++) G->hg[i] = q35_silu(G->hg[i])*G->hu[i];
    const uint8_t *Wd = (const uint8_t*)q35_wdata(M, L->ffn_down);
    const int64_t skip = (int64_t)(r/KQ_QK_K)*kq_typesize(L->ffn_down->type);
    /* kq_dot_fast, NOT kq_dot: the latter is the scalar reference at 0.71 GB/s per core
     * against AVX2's 2.66. Using it here made the CPU half of a 76% share cost 0.383 s where
     * the WHOLE FFN costs 0.246 — a 4x regression that looks exactly like "the split does
     * not work" and is not. */
    #pragma omp parallel for schedule(static)
    for(int o = 0; o < Dm; o++)
        out[o] = kq_dot_fast(L->ffn_down->type, Wd + (int64_t)o*drb + skip, G->hg + r, nrow);
}



static void q35cu_gdn_layer_x(Q35Cu *G, int il, int slot, const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *S = &M->L[il];
    const int nv = c->n_v_heads, nk = c->n_k_heads, dk = c->d_state, dv = c->d_head_v;
    const int kdim = dk*nk;

    q35cu_gemv(G->qkv, xn, D->wqkv,    S->wqkv->type,    c->d_model, c->conv_dim); q35cu_mark("wqkv");
    q35cu_gemv(G->z,   xn, D->wgate,   S->wgate->type,   c->d_model, c->d_inner); q35cu_mark("wgate");
    q35cu_gemv(G->ab,        xn, D->w_beta,  S->w_beta->type,  c->d_model, nv);
    q35cu_gemv(G->ab + nv,   xn, D->w_alpha, S->w_alpha->type, c->d_model, nv); q35cu_mark("w_alpha+beta");
    q35cu_gdn_gates(G->ab, G->ab + nv, D->dt_bias, D->ssm_a, nv); q35cu_mark("gates");

    q35cu_gdn_conv(G->qkv, G->conv + (size_t)slot*(c->d_conv-1)*c->conv_dim,
                   D->conv1d, c->conv_dim, c->d_conv); q35cu_mark("conv1d");
    q35cu_gdn_l2norm(G->qkv, 2*nk, dk, c->eps);          /* q and k, NOT rmsnorm */ q35cu_mark("l2norm");
    q35cu_gdn_delta(G->S + (size_t)slot*nv*dk*dv, G->o,
                    G->qkv, G->qkv + kdim, G->qkv + 2*kdim,
                    G->ab, G->ab + nv, nv, nk, dk, dv, g_q35_gdn_tile); q35cu_mark("delta");
    q35cu_gdn_norm_gate(G->o, G->z, D->ssm_norm, nv, dv, c->eps); q35cu_mark("norm_gate");
    q35cu_gemv(out, G->o, D->w_out, S->w_out->type, c->d_inner, c->d_model); q35cu_mark("w_out");
}

static void q35cu_attn_layer_x(Q35Cu *G, Q35State *R, int il, int slot, int pos,
                               const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *S = &M->L[il];
    const int nh = c->n_head, nkv = c->n_head_kv, dh = c->d_head;
    const int n_kv = nkv*dh, T = pos + 1;
    const float scale = 1.0f/sqrtf((float)dh);

    q35cu_gemv(G->qg, xn, D->wq, S->wq->type, c->d_model, 2*nh*dh); q35cu_mark("wq");
    q35cu_gemv(G->k,  xn, D->wk, S->wk->type, c->d_model, n_kv);
    q35cu_gemv(G->v,  xn, D->wv, S->wv->type, c->d_model, n_kv); q35cu_mark("wk+wv");
    q35cu_attn_qk(G->qg, G->k, D->q_norm, D->k_norm, nh, nkv, dh,
                  c->n_rot, c->rope_base, pos, c->eps); q35cu_mark("qknorm+rope");

    const int64_t lane = (int64_t)slot*G->win*G->kv_half;
    if(pos < G->win){
        q35cu_kv_store(G->Kc, G->Vc, G->k, G->v,
                       lane + (int64_t)pos*G->kv_half, lane + (int64_t)pos*G->kv_half,
                       n_kv, G->kv_fmt);
    } else {
        /* past the window: this token's KV belongs to the host tier */
        q35cu_sync();
        q35cu_d2h(R->k, G->k, sizeof(float)*n_kv);
        q35cu_d2h(R->v, G->v, sizeof(float)*n_kv);
        kvt_put(&R->kv, slot, pos, R->k, R->v);
    }

    const int T_gpu = T < G->win ? T : G->win;
    if(T_gpu > 0)
        q35cu_attn_flash(G->o, G->lse, G->qg, G->Kc + lane, G->Vc + lane,
                         nh, nkv, dh, T_gpu, G->kv_half, G->kv_half,
                         scale, G->kv_fmt, G->scratch); q35cu_mark("flash");

    if(T > G->win){
        q35cu_sync();
        q35cu_d2h(G->hqg, G->qg, sizeof(float)*2*nh*dh);
        q35_attn_host_partial(R, G->hqg, slot, G->win, T, G->ho, G->hlse);
        q35cu_h2d(G->o2,   G->ho,   sizeof(float)*nh*dh);
        q35cu_h2d(G->lse2, G->hlse, sizeof(float)*nh);
        if(T_gpu > 0) q35cu_lse_merge(G->o, G->lse, G->o2, G->lse2, nh, dh);
        else          q35cu_h2d(G->o, G->ho, sizeof(float)*nh*dh);
    }

    q35cu_attn_gate(G->o, G->qg, nh, dh); q35cu_mark("gate");
    q35cu_gemv(out, G->o, D->wo, S->wo->type, (int)S->wo->ne[0], c->d_model); q35cu_mark("wo");
}

static void q35cu_ffn_layer_x(Q35Cu *G, Q35State *R, int il, const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *S = &M->L[il];
    if(D->ffn_gpu){
        q35cu_gemv(G->ffn_g, xn, D->ffn_gate, S->ffn_gate->type, c->d_model, c->d_ff);
        q35cu_gemv(G->ffn_u, xn, D->ffn_up,   S->ffn_up->type,   c->d_model, c->d_ff);
        q35cu_swiglu(G->ffn_g, G->ffn_u, c->d_ff);
        q35cu_gemv(out, G->ffn_g, D->ffn_down, S->ffn_down->type, c->d_ff, c->d_model);
    } else if(G->stream_on && q35cu_ffn_split(G, il, xn, out, G->stage_half)){
        /* handled: CPU and GPU each did a share, concurrently */
    } else {
        /* 20 KiB out, 20 KiB back, against 150 MiB of weights read in place. The hand-off is
         * not the cost; the alternative — moving those weights to VRAM over a x8 link — is. */
        q35cu_sync();
        q35cu_d2h(G->hxn, xn, sizeof(float)*c->d_model);
        q35_ffn(R, S, G->hxn, G->ht);
        q35cu_h2d(out, G->ht, sizeof(float)*c->d_model);
    }
}

static void q35cu_gdn_layer(Q35Cu *G, int il, int slot){
    q35cu_gdn_layer_x(G, il, slot, G->xn, G->t);
}
static void q35cu_attn_layer(Q35Cu *G, Q35State *R, int il, int slot, int pos){
    q35cu_attn_layer_x(G, R, il, slot, pos, G->xn, G->t);
}
static void q35cu_ffn_layer(Q35Cu *G, Q35State *R, int il){
    q35cu_ffn_layer_x(G, R, il, G->xn, G->t);
}

/* ---------------- one token ---------------- */

static void q35_forward_cu(Q35Cu *G, Q35State *R, int tok, int pos, float *logits){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;

    {   const double t0 = g_q35_stage_timing ? q35_clk() : 0;
        q35_embed(R, tok, R->x);                       /* one row of a 0.67 GiB table */
        q35cu_h2d(G->x, R->x, sizeof(float)*c->d_model);
        if(g_q35_stage_timing) g_t_embd += q35_clk()-t0; }

    if(G->stream_on){
        const int f0 = q35cu_next_stream_layer(G, 0);
        G->stage_half = 0;
        if(f0 >= 0) q35cu_stream_issue(G, f0, 0);
    }
    for(int il = 0; il < c->n_layer; il++){
        const Q35CuLayer *D = &G->L[il];
        /* Prefetch the NEXT streamed layer into the other half before touching this one, so
         * its 2.6 ms of DMA sits underneath this layer's compute instead of in front of the
         * next one's. */
        if(G->stream_on && !D->ffn_gpu){
            const int nl = q35cu_next_stream_layer(G, il+1);
            if(nl >= 0) q35cu_stream_issue(G, nl, 1 - G->stage_half);
        }
        q35cu_rmsnorm(G->xn, G->x, D->attn_norm, c->d_model, c->eps);
        { const double t0 = g_q35_stage_timing ? q35_clk() : 0;
          if(c->kind[il] == Q35_LAYER_ATTN) q35cu_attn_layer(G, R, il, R->attn_slot[il], pos);
          else                              q35cu_gdn_layer(G, il, R->gdn_slot[il]);
          if(g_q35_stage_timing){ q35cu_sync();
              *(c->kind[il] == Q35_LAYER_ATTN ? &g_t_attn : &g_t_gdn) += q35_clk()-t0; } }
        q35cu_add(G->x, G->t, c->d_model);

        q35cu_rmsnorm(G->xn, G->x, D->post_attn_norm, c->d_model, c->eps);
        { const double t0 = g_q35_stage_timing ? q35_clk() : 0;
          q35cu_ffn_layer(G, R, il);
          if(G->stream_on && !D->ffn_gpu) G->stage_half = 1 - G->stage_half;
          if(g_q35_stage_timing){ q35cu_sync(); g_t_ffn += q35_clk()-t0; } }
        q35cu_add(G->x, G->t, c->d_model);
    }

    if(logits){
        const double t0 = g_q35_stage_timing ? q35_clk() : 0;
        q35cu_rmsnorm(G->xn, G->x, G->output_norm, c->d_model, c->eps);
        q35cu_gemv(G->logits, G->xn, G->output, M->output->type, c->d_model, c->vocab);
        q35cu_sync();
        q35cu_d2h(logits, G->logits, sizeof(float)*c->vocab);
        if(g_q35_stage_timing) g_t_out += q35_clk()-t0;
    }
    if(!q35cu_sync()) fprintf(stderr, "qwen35_cuda: %s\n", q35cu_error());
    /* Mirror the residual stream back to the host. 20 KiB against 15.6 GiB of weights, and
     * without it q35_hidden() feeds the MTP draft head the LAST TOKEN'S EMBEDDING instead of
     * the model's hidden state — which does not crash, does not warn, and quietly drops
     * draft acceptance from ~85% to ~8%, i.e. turns speculation from a 1.5x win into an
     * 18% loss. */
    q35cu_d2h(R->x, G->x, sizeof(float)*c->d_model);
    R->pos = pos + 1;
}

#endif
