#ifndef COLIBRI_MLA_H
#define COLIBRI_MLA_H
/* mla.h — GLM-5.2 / DeepSeek-style Multi-head Latent Attention, absorbed form.
 *
 * WHY THE ABSORBED FORM, AND WHY THIS IS NOT JUST A LAYOUT DETAIL
 *
 * MLA compresses the KV cache: instead of storing K and V per head, it stores one latent
 * c[kv_lora=512] per token (plus a small RoPE part). Colibri today RECONSTRUCTS K and V
 * from that latent on every step, using one fused kv_b [H*(nope+v), kv_lora].
 *
 * The GGUF ships something different, and better:
 *
 *      attn_k_b  [192, 512, 64]   = per head, maps q_nope(192) -> latent space(512)
 *      attn_v_b  [512, 256, 64]   = per head, maps latent(512) -> v(256)
 *
 * Note the direction of k_b: 192 -> 512, which is the TRANSPOSE of the reconstruction
 * matrix (512 -> 192). That is deliberate. The attention score for head h and cached
 * token t is
 *
 *      score = q_nope . k_nope(t)  =  q_nope . (W_k c_t)  =  (W_k^T q_nope) . c_t
 *                \_____________/                              \____________/
 *                 reconstruct K for EVERY cached token         project Q ONCE
 *
 * Associativity is trivial; the cost is not. The left form rebuilds a 192-vector for every
 * one of the T cached tokens, every step: O(T * 192 * 512) per head. The right form pushes
 * the query into latent space once, O(192 * 512), and then scores directly against the
 * stored latents: O(T * 512). At long context the first term is what kills you, and it is
 * exactly the term colibri currently pays.
 *
 * Same for V: the attention output stays in latent space and is projected out ONCE per
 * step through v_b, instead of materializing V for all T tokens.
 *
 * So adopting the GGUF layout is not a concession to a file format — it is the algorithm
 * we want. The KV cache holds only [kv_lora + qk_rope] = 576 floats per token per layer,
 * for all 64 heads. That is why a 1M context is even thinkable on this box.
 *
 * Every quantity below is contracted straight off the quantized bytes (kquant.h); no
 * weight matrix is ever materialized.
 */
#include <math.h>
#include <string.h>
#include "kquant.h"

typedef struct {
    int n_head;       /* 64  */
    int qk_nope;      /* 192 : query dims that see no RoPE */
    int qk_rope;      /* 64  : query dims that do */
    int v_head;       /* 256 */
    int kv_lora;      /* 512 : the latent width — the whole point */
    int q_lora;       /* 2048 */
    int d_model;      /* 6144 */
    float eps;
    float attn_scale; /* 1/sqrt(qk_nope + qk_rope) */
} MLACfg;

/* Derive the shape from the tensors themselves and cross-check against the GGUF metadata.
 * A silent mismatch here (e.g. reading k_b as [512,192] instead of [192,512]) does not
 * crash — it produces fluent, wrong text. So we assert instead of assuming. */
static int mla_cfg_from_tensors(MLACfg *c,
                                const gguf_tensor *k_b, const gguf_tensor *v_b,
                                const gguf_tensor *q_a, const gguf_tensor *q_b,
                                const gguf_tensor *kv_a,
                                int n_head_meta, int kv_lora_meta, int rope_dim,
                                int d_model, float eps){
    if(!k_b || !v_b || !q_a || !q_b || !kv_a) return 0;
    if(k_b->n_dims != 3 || v_b->n_dims != 3) return 0;

    c->n_head  = (int)k_b->ne[2];
    c->qk_nope = (int)k_b->ne[0];
    c->kv_lora = (int)k_b->ne[1];
    c->v_head  = (int)v_b->ne[1];
    c->qk_rope = rope_dim;
    c->q_lora  = (int)q_a->ne[1];
    c->d_model = d_model;
    c->eps     = eps;
    c->attn_scale = 1.0f / sqrtf((float)(c->qk_nope + c->qk_rope));

    /* the shapes must agree with each other and with the header, or we are misreading */
    if(c->n_head  != n_head_meta)                       return 0;
    if(c->kv_lora != kv_lora_meta)                      return 0;
    if(v_b->ne[0] != c->kv_lora)                        return 0;   /* v_b contracts the latent */
    if(v_b->ne[2] != c->n_head)                         return 0;
    if(q_b->ne[1] != (int64_t)c->n_head*(c->qk_nope + c->qk_rope)) return 0;
    if(kv_a->ne[1] != c->kv_lora + c->qk_rope)          return 0;   /* latent + rope key */
    if(q_a->ne[0] != c->d_model)                        return 0;
    return 1;
}

/* The compressed KV cache: ONE latent plus one RoPE key per token per layer, shared by
 * all 64 heads. 576 floats instead of 64*(192+256) = 28672. That factor of 50 is the
 * entire reason a long context fits. */
typedef struct {
    float *c;       /* [max_t, kv_lora]  normalized latent            */
    float *kr;      /* [max_t, qk_rope]  the RoPE'd key, shared by all heads */
    int    n;       /* tokens currently cached */
    int    max_t;
} MLACache;

static int mla_cache_init(MLACache *kv, const MLACfg *c, int max_t){
    kv->c  = (float*)calloc((size_t)max_t * c->kv_lora, sizeof(float));
    kv->kr = (float*)calloc((size_t)max_t * c->qk_rope, sizeof(float));
    kv->n = 0; kv->max_t = max_t;
    return kv->c && kv->kr;
}
static int64_t mla_cache_bytes(const MLACfg *c, int max_t){
    return (int64_t)max_t * (c->kv_lora + c->qk_rope) * (int64_t)sizeof(float);
}

/* ggml's CPU matmuls do NOT contract fp32 activations against the weights: they quantize x
 * to Q8_K (256-wide blocks, one absmax scale, int8 values) first, because an int8 dot is
 * much faster than an fp32 one. That is a real numerical difference, worth ~1% per element
 * — and it is why colibri's exact fp32 contraction does not reproduce llama.cpp bit for bit
 * even when the tensor layout is right.
 *
 * Reproducing it is how mla_test proves the difference is ggml's rounding and not our bug:
 * with this applied, all of layer 0 matches llama.cpp to four decimals. The engine itself
 * does NOT call this — fp32 is both simpler and more accurate, and the GPU path is not
 * bound by the multiply anyway. */
static void mla_quant_act_blk(float *x, int n, int blk){
    for(int b = 0; b < n; b += blk){
        int e = (b + blk < n) ? b + blk : n;
        float amax = 0;
        for(int i = b; i < e; i++){ float a = fabsf(x[i]); if(a > amax) amax = a; }
        if(amax < 1e-12f) continue;
        const float s = amax / 127.0f, inv = 1.0f / s;
        for(int i = b; i < e; i++) x[i] = s * (float)((int)lrintf(x[i] * inv));
    }
}
/* The block size is not a free choice: it follows the WEIGHT's type, because ggml pairs
 * each weight format with the activation format its vec_dot expects. Q8_0 weights are
 * contracted against Q8_0 activations (32-wide blocks); the k- and i-quants use Q8_K
 * (256-wide). Using 256 everywhere leaves a visible residue on exactly the Q8_0 tensors. */
static void mla_quant_act(float *x, int n, int wtype){
    mla_quant_act_blk(x, n, (wtype == KQ_Q8_0) ? 32 : 256);
}

/* RMS norm with a learned weight (ggml_rms_norm + mul). */
static void mla_rms(float *y, const float *x, const float *w, int n, float eps){
    double ss = 0;
    for(int i=0;i<n;i++) ss += (double)x[i]*x[i];
    float s = 1.0f/sqrtf((float)(ss/n) + eps);
    for(int i=0;i<n;i++) y[i] = x[i]*s*w[i];
}

/* RoPE, NORM style: ADJACENT pairs (0,1), (2,3), ... — not the NeoX halves (i, i+n/2).
 *
 * llama.cpp maps LLM_ARCH_GLM_DSA to LLAMA_ROPE_TYPE_NORM. This is worth spelling out
 * because getting it wrong is invisible in the obvious test: at pos = 0 the rotation angle
 * is zero and RoPE is the IDENTITY, so a layer-0 logit comparison against llama.cpp passes
 * perfectly with either pairing. Everything from position 1 on is then quietly wrong, and
 * what you see is a model that opens a sentence correctly and then loses the thread —
 * which is exactly what it did until this was fixed.
 *
 * Any oracle for a positional encoding has to run at pos > 0 or it proves nothing. */
static void mla_rope(float *v, int n, int pos, float theta_base){
    for(int i = 0; i < n/2; i++){
        float inv = powf(theta_base, -2.0f*(float)i/(float)n);
        float ang = (float)pos * inv;
        float cs = cosf(ang), sn = sinf(ang);
        float a = v[2*i], b = v[2*i + 1];
        v[2*i    ] = a*cs - b*sn;
        v[2*i + 1] = a*sn + b*cs;
    }
}

/* Per-layer attention weights, straight out of the GGUF. Note that every one of these has
 * ne0 = contraction dim and ne1 = output dim, which is EXACTLY the layout kq_gemm already
 * expects — so nothing is transposed or repacked anywhere, and the quantized bytes are
 * contracted in place. */
typedef struct {
    const void *pq_a, *pq_b, *pkv_a, *pk_b, *pv_b, *po;   /* mapped weight blobs */
    int         tq_a,  tq_b,  tkv_a,  tk_b,  tv_b,  to;   /* their ggml types */
    const float *attn_norm, *q_a_norm, *kv_a_norm;
} MLALayer;

/* Scratch for one token. Sized once, reused for every layer. */
typedef struct {
    float *xn;        /* d_model  */
    float *qa;        /* q_lora   */
    float *q;         /* n_head * (qk_nope+qk_rope) */
    float *kva;       /* kv_lora + qk_rope */
    float *qlat;      /* kv_lora  : the query, pushed into latent space */
    float *scores;    /* max_t    */
    float *olat;      /* kv_lora  : attention output, still in latent space */
    float *heads;     /* n_head * v_head */
    float *qlat_all;  /* n_head * kv_lora : all heads' absorbed queries, one batched launch */
    float *olat_all;  /* n_head * kv_lora : all heads' latent outputs, likewise */
} MLAScratch;

static int mla_scratch_init(MLAScratch *s, const MLACfg *c, int max_t){
    s->xn     = (float*)malloc((size_t)c->d_model*sizeof(float));
    s->qa     = (float*)malloc((size_t)c->q_lora *sizeof(float));
    s->q      = (float*)malloc((size_t)c->n_head*(c->qk_nope+c->qk_rope)*sizeof(float));
    s->kva    = (float*)malloc((size_t)(c->kv_lora+c->qk_rope)*sizeof(float));
    s->qlat   = (float*)malloc((size_t)c->kv_lora*sizeof(float));
    s->scores = (float*)malloc((size_t)max_t*sizeof(float));
    s->olat   = (float*)malloc((size_t)c->kv_lora*sizeof(float));
    s->heads  = (float*)malloc((size_t)c->n_head*c->v_head*sizeof(float));
    s->qlat_all = (float*)malloc((size_t)c->n_head*c->kv_lora*sizeof(float));
    s->olat_all = (float*)malloc((size_t)c->n_head*c->kv_lora*sizeof(float));
    return s->xn && s->qa && s->q && s->kva && s->qlat && s->scores && s->olat && s->heads
        && s->qlat_all && s->olat_all;
}

/* One decode step of MLA attention, absorbed form. Appends this token to the cache and
 * writes the layer's attention output (pre-residual) into `out` [d_model].
 *
 * The DSA lightning indexer is deliberately absent: glm-dsa sets index_topk=2048, and the
 * indexer selects ALL keys whenever the sequence is shorter than that. So for seq < 2048
 * it is mathematically a no-op, and adding it would only add a way to be wrong. It must
 * come back before long context works.
 */
static void mla_step(float *out, const float *x, int pos,
                     const MLACfg *c, const MLALayer *L, MLACache *kv, MLAScratch *s,
                     float rope_theta){
    const int nope = c->qk_nope, rope = c->qk_rope, qk = nope + rope;
    const int H = c->n_head, DL = c->kv_lora, DV = c->v_head;

    mla_rms(s->xn, x, L->attn_norm, c->d_model, c->eps);

    /* Q: down to q_lora, norm, back up to all heads */
    kq_gemm(s->qa, s->xn, L->pq_a, L->tq_a, 1, c->d_model, c->q_lora);
    mla_rms(s->qa, s->qa, L->q_a_norm, c->q_lora, c->eps);
    kq_gemm(s->q, s->qa, L->pq_b, L->tq_b, 1, c->q_lora, H*qk);

    /* KV: one projection gives the latent AND the shared RoPE key */
    kq_gemm(s->kva, s->xn, L->pkv_a, L->tkv_a, 1, c->d_model, DL + rope);

    float *c_t  = kv->c  + (size_t)pos*DL;
    float *kr_t = kv->kr + (size_t)pos*rope;
    mla_rms(c_t, s->kva, L->kv_a_norm, DL, c->eps);      /* the latent IS the cache entry */
    memcpy(kr_t, s->kva + DL, (size_t)rope*sizeof(float));
    mla_rope(kr_t, rope, pos, rope_theta);               /* one RoPE key for all 64 heads */
    if(pos + 1 > kv->n) kv->n = pos + 1;
    const int T = kv->n;

    const int64_t kb_row = kq_row_bytes(L->tk_b, (int64_t)nope*DL);   /* per head */
    const int64_t vb_row = kq_row_bytes(L->tv_b, (int64_t)DL*DV);

    for(int h = 0; h < H; h++){
        float *qh = s->q + (size_t)h*qk;
        mla_rope(qh + nope, rope, pos, rope_theta);

        /* THE ABSORPTION. Push q_nope into latent space once: q_lat = W_k^T q_nope.
         * k_b is [nope -> DL] per head, so this is a single kq_gemm with I=nope, O=DL.
         * Afterwards the score against a cached token is a plain dot in latent space —
         * no K is ever reconstructed, for any of the T cached tokens. */
        const uint8_t *kb = (const uint8_t*)L->pk_b + (size_t)h*kb_row;
        kq_gemm(s->qlat, qh, kb, L->tk_b, 1, nope, DL);

        float mx = -INFINITY;
        for(int t = 0; t < T; t++){
            const float *ct  = kv->c  + (size_t)t*DL;
            const float *krt = kv->kr + (size_t)t*rope;
            float sc = 0.f;
            for(int i = 0; i < DL;   i++) sc += s->qlat[i]  * ct[i];    /* the nope part */
            for(int i = 0; i < rope; i++) sc += qh[nope+i]  * krt[i];   /* the rope part */
            sc *= c->attn_scale;
            s->scores[t] = sc;
            if(sc > mx) mx = sc;
        }
        float sum = 0.f;
        for(int t = 0; t < T; t++){ s->scores[t] = expf(s->scores[t]-mx); sum += s->scores[t]; }
        const float inv = 1.0f/sum;

        /* Weighted sum stays in LATENT space: 512 wide, not 64*256. */
        for(int i = 0; i < DL; i++) s->olat[i] = 0.f;
        for(int t = 0; t < T; t++){
            const float w = s->scores[t]*inv;
            const float *ct = kv->c + (size_t)t*DL;
            for(int i = 0; i < DL; i++) s->olat[i] += w*ct[i];
        }
        /* ...and only NOW is it projected out to V, once per step instead of once per
         * cached token. */
        const uint8_t *vb = (const uint8_t*)L->pv_b + (size_t)h*vb_row;
        kq_gemm(s->heads + (size_t)h*DV, s->olat, vb, L->tv_b, 1, DL, DV);
    }

    kq_gemm(out, s->heads, L->po, L->to, 1, H*DV, c->d_model);
}

#endif /* COLIBRI_MLA_H */
