#ifndef QWEN_QWEN35_BATCH_H
#define QWEN_QWEN35_BATCH_H
/* qwen35_batch.h — prefill S tokens at a time.
 *
 * WHY THIS IS THE GATE FOR EVERYTHING ELSE
 *
 * Decode reads all 15.6 GiB of weights to produce ONE token, and is therefore bandwidth
 * bound: 0.52 s/token against 57.5 GB/s of RAM, and no kernel can beat that. Prefill does not
 * have to pay that. A batch of S tokens uses each weight S times, so the weight read is
 * amortized S-fold and what is left is arithmetic — a wall several times further out.
 *
 * Without this, prefill costs exactly what decode costs, and an 8k-token RAG context takes
 * 70 minutes to ingest. With it the same 8k is a few minutes on CPU and seconds on a GPU.
 * Speculative decoding needs it too: verifying a drafted token means running the main model
 * on 2+ positions at once, which is a batch of 2.
 *
 * WHAT BATCHES AND WHAT CANNOT
 *
 * Every projection is a gemm and batches perfectly: qkv, gate, alpha/beta, o, ffn, output.
 * Those are ~97% of the FLOPs and all of the bandwidth.
 *
 * Two things are SEQUENTIAL BY CONSTRUCTION and are kept that way:
 *
 *   - The Gated DeltaNet recurrence. Its state after token t depends on the state after
 *     t-1; there is no reordering of that. (A chunked parallel form exists — llama.cpp's
 *     build_delta_net_chunking — but it is a different set of numerics, and the sequential
 *     form is what the decode path already validates against.) The projections around it
 *     still batch, which is where the cost actually is.
 *   - The causal conv1d ring, for the same reason.
 *
 * Attention is causal, so token s attends to positions 0..pos_base+s. Keys and values are
 * appended in order and each query scans its own prefix. The scan is O(S*T) and is NOT the
 * expensive part at prefill time — the projections are — so it stays simple and shares the
 * decode path's tiered-KV reader.
 *
 * VALIDATION: qwen35_batch_test requires that prefilling S tokens in one call produces the
 * same logits as S separate q35_forward calls. Anything else means a batch is leaking state
 * across positions, which reads as a subtly worse model rather than as a crash.
 */
#include "qwen35_cpu.h"

typedef struct {
    const Q35Model *M;
    Q35State *R;                 /* KV store, GDN state and geometry are shared with decode */
    int S_max;

    float *x, *xn, *t;           /* [S][d_model] */
    float *qg;                   /* [S][2*n_head*d_head] */
    float *kk, *vv;              /* [S][n_head_kv*d_head] */
    float *o;                    /* [S][max(n_head*d_head, d_inner)] */
    float *ffn_g, *ffn_u;        /* [S][d_ff] */
    float *qkv;                  /* [S][conv_dim] */
    float *z;                    /* [S][d_inner] */
    float *ab;                   /* [S][2*n_v_heads] */

    /* Speculation support. Verifying a 1-token draft runs a batch of 2; on rejection the
     * state must be rewound to "after token 0", NOT to "before the batch" — token 0 was the
     * real token and its work is valid. The layer loop is the outer one, so "after token 0"
     * is not a single point in time: each layer reaches it separately. So each GDN layer
     * copies its own state out the moment it finishes token `snap_after`. */
    int    snap_after;           /* -1 = off */
    float *snap_S, *snap_conv;
} Q35Batch;

static int q35_batch_init(Q35Batch *B, Q35State *R, int S_max){
    memset(B, 0, sizeof *B);
    const Q35Model *M = R->M;
    const Q35Cfg *c = &M->c;
    B->M = M; B->R = R; B->S_max = S_max;
    const int kvd = c->n_head_kv * c->d_head;
    const int od  = (c->n_head*c->d_head > c->d_inner) ? c->n_head*c->d_head : c->d_inner;

    #define BALLOC(p, n) do { B->p = (float*)calloc((size_t)(n)*S_max, sizeof(float)); \
                              if(!B->p) return 0; } while(0)
    BALLOC(x, c->d_model);   BALLOC(xn, c->d_model);  BALLOC(t, c->d_model);
    BALLOC(qg, 2*c->n_head*c->d_head);
    BALLOC(kk, kvd);         BALLOC(vv, kvd);         BALLOC(o, od);
    BALLOC(ffn_g, c->d_ff);  BALLOC(ffn_u, c->d_ff);
    BALLOC(qkv, c->conv_dim);BALLOC(z, c->d_inner);   BALLOC(ab, 2*c->n_v_heads);
    #undef BALLOC
    B->snap_after = -1;
    B->snap_S    = (float*)malloc((size_t)R->n_gdn*c->n_v_heads*c->d_state*c->d_head_v*sizeof(float));
    B->snap_conv = (float*)malloc((size_t)R->n_gdn*(c->d_conv-1)*c->conv_dim*sizeof(float));
    return B->snap_S && B->snap_conv;
}

static void q35_batch_free(Q35Batch *B){
    free(B->x); free(B->xn); free(B->t); free(B->qg); free(B->kk); free(B->vv);
    free(B->o); free(B->ffn_g); free(B->ffn_u); free(B->qkv); free(B->z); free(B->ab);
    free(B->snap_S); free(B->snap_conv);
    memset(B, 0, sizeof *B);
}

/* batched gemm through the relocated (resident) weights */
static void q35_mm(float *y, const float *x, const Q35Model *M, const gguf_tensor *T, int S){
    kq_gemm_batched(y, x, q35_wdata(M, T), T->type, S, (int)T->ne[0], (int)T->ne[1]);
}

static void q35_rmsnorm_rows(float *out, const float *x, const float *w, int n, int S, float eps){
    #pragma omp parallel for schedule(static) if(S > 4)
    for(int s = 0; s < S; s++) q35_rmsnorm(out + (size_t)s*n, x + (size_t)s*n, w, n, eps);
}

/* Per-component timers. The batch-2 cost measured 2.7x a single pass, and anything ABOVE
 * 2.0x is not a failure to amortize — two independent single passes cost exactly 2.0x by
 * construction, so 2.7x is overhead ADDED by the batch dimension. That cannot come from
 * inside a correctly-structured gemm, so the bisection has to be by COMPONENT. */
static double g_b_ffn = 0, g_b_attn = 0, g_b_gdn = 0, g_b_out = 0, g_b_norm = 0;
static void q35_batch_timers_reset(void){ g_b_ffn = g_b_attn = g_b_gdn = g_b_out = g_b_norm = 0; }

/* ---------------- batched FFN ---------------- */

static void q35_ffn_b(Q35Batch *B, const Q35Layer *L, const float *xn, float *out, int S){
    const Q35Cfg *c = &B->M->c;
    if(c->is_moe){
        /* Prefill for a MoE model: each token routes to its own experts, so this is a
         * per-token loop over the same q35_moe_row the decode path uses. Batching the experts
         * across tokens buys little when every token picks a different eight, and correctness
         * comes first — the tokens are independent, so OpenMP splits them cleanly. */
        const int D = c->d_model, E = c->n_expert;
        const int Fw = c->d_ff_exp > c->d_ff_shexp ? c->d_ff_exp : c->d_ff_shexp;
        #pragma omp parallel
        {
            float *rlog = (float*)malloc(sizeof(float)*E);
            float *gbuf = (float*)malloc(sizeof(float)*Fw);
            float *ubuf = (float*)malloc(sizeof(float)*Fw);
            float *od   = (float*)malloc(sizeof(float)*D);
            #pragma omp for schedule(static)
            for(int s = 0; s < S; s++)
                q35_moe_row(B->M, L, xn + (size_t)s*D, out + (size_t)s*D, rlog, gbuf, ubuf, od);
            free(rlog); free(gbuf); free(ubuf); free(od);
        }
        return;
    }
    q35_mm(B->ffn_g, xn, B->M, L->ffn_gate, S);
    q35_mm(B->ffn_u, xn, B->M, L->ffn_up,   S);
    const int64_t n = (int64_t)S*c->d_ff;
    for(int64_t i = 0; i < n; i++) B->ffn_g[i] = q35_silu(B->ffn_g[i]) * B->ffn_u[i];
    q35_mm(out, B->ffn_g, B->M, L->ffn_down, S);
}

/* ---------------- batched full attention ----------------
 * Projections batch; the causal scan is per-token because each query sees a different prefix. */

static void q35_attn_b(Q35Batch *B, const Q35Layer *L, int il, const float *xn,
                       int pos_base, float *out, int S){
    Q35State *R = B->R;
    const Q35Cfg *c = &B->M->c;
    const int dh = c->d_head, nh = c->n_head, nkv = c->n_head_kv;
    const int grp = nh / nkv, kvd = nkv * dh;
    const int slot = R->attn_slot[il];

    q35_mm(B->qg, xn, B->M, L->wq, S);
    q35_mm(B->kk, xn, B->M, L->wk, S);
    q35_mm(B->vv, xn, B->M, L->wv, S);

    /* per-head QK-norm and partial rope, then append in position order */
    for(int s = 0; s < S; s++){
        const int pos = pos_base + s;
        float *qgs = B->qg + (size_t)s*2*nh*dh;
        for(int h = 0; h < nh; h++){
            float *q = qgs + (size_t)h*2*dh;
            q35_rmsnorm(q, q, L->q_norm, dh, c->eps);
            q35_rope(q, pos, c->n_rot, c->rope_base);
        }
        float *ks = B->kk + (size_t)s*kvd;
        for(int h = 0; h < nkv; h++){
            float *k = ks + (size_t)h*dh;
            q35_rmsnorm(k, k, L->k_norm, dh, c->eps);
            q35_rope(k, pos, c->n_rot, c->rope_base);
        }
        kvt_put(&R->kv, slot, pos, ks, B->vv + (size_t)s*kvd);
    }

    /* Each query scans its own causal prefix. Chunked so the tiered store can read ahead;
     * the prefetch window is issued before the chunk is touched, exactly as in decode. */
    const float scale = 1.0f/sqrtf((float)dh);
    KvTier *KV = &R->kv;
    const int64_t hb  = q35_kv_entry_bytes(KV->fmt, dh);
    memset(B->o, 0, sizeof(float)*(size_t)S*nh*dh);

    for(int s = 0; s < S; s++){
        const int T = pos_base + s + 1;
        const int n_ch = (T + KV->chunk - 1) / KV->chunk;
        const float *qgs = B->qg + (size_t)s*2*nh*dh;
        float *os = B->o + (size_t)s*nh*dh;

        for(int ci = 0; ci < n_ch; ci++){
            kvt_prefetch_window(KV, slot, ci, 6, n_ch);
            const uint8_t *blk = kvt_chunk(KV, slot, ci);
            if(!blk) continue;
            const int t0 = ci*KV->chunk;
            const int t1 = (t0 + KV->chunk < T) ? t0 + KV->chunk : T;
            #pragma omp parallel for schedule(static)
            for(int h = 0; h < nh; h++){
                const float *q = qgs + (size_t)h*2*dh;
                float *sc = R->att + (size_t)h*R->n_ctx;
                const int hk = h / grp;
                for(int t = t0; t < t1; t++)
                    sc[t] = kvt_dot(KV, q, blk + kvt_koff(KV, t-t0) + hk*hb, dh)*scale;
            }
        }
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++) q35_softmax(R->att + (size_t)h*R->n_ctx, T);

        for(int ci = 0; ci < n_ch; ci++){
            kvt_prefetch_window(KV, slot, ci, 6, n_ch);
            const uint8_t *blk = kvt_chunk(KV, slot, ci);
            if(!blk) continue;
            const int t0 = ci*KV->chunk;
            const int t1 = (t0 + KV->chunk < T) ? t0 + KV->chunk : T;
            #pragma omp parallel for schedule(static)
            for(int h = 0; h < nh; h++){
                const float *sc = R->att + (size_t)h*R->n_ctx;
                const int hk = h / grp;
                float *oh = os + (size_t)h*dh;
                for(int t = t0; t < t1; t++)
                    kvt_axpy(KV, oh, sc[t], blk + kvt_voff(KV, t-t0) + hk*hb, dh);
            }
        }
        for(int h = 0; h < nh; h++){
            const float *g = qgs + (size_t)h*2*dh + dh;
            float *oh = os + (size_t)h*dh;
            for(int i = 0; i < dh; i++) oh[i] *= q35_sigmoid(g[i]);
        }
    }
    q35_mm(out, B->o, B->M, L->wo, S);
}

/* ---------------- batched gated delta net ----------------
 * Projections batch; conv ring and delta rule step token by token because both are recurrent. */

static void q35_gdn_b(Q35Batch *B, const Q35Layer *L, int il, const float *xn, float *out, int S){
    Q35State *R = B->R;
    const Q35Cfg *c = &B->M->c;
    const int dk = c->d_state, dv = c->d_head_v;
    const int nk = c->n_k_heads, nv = c->n_v_heads;
    const int cd = c->conv_dim, kdim = dk*nk, K1 = c->d_conv - 1;
    const int slot = R->gdn_slot[il];

    q35_mm(B->qkv, xn, B->M, L->wqkv,  S);
    q35_mm(B->z,   xn, B->M, L->wgate, S);
    /* beta and alpha are [d_model, n_v_heads]: two tiny gemms, kept separate so the layout
     * stays [S][nv] each rather than interleaved */
    float *beta_all  = B->ab;
    float *alpha_all = B->ab + (size_t)S*nv;
    q35_mm(beta_all,  xn, B->M, L->w_beta,  S);
    q35_mm(alpha_all, xn, B->M, L->w_alpha, S);

    /* gdn_slot is -1 for an attention layer; the dispatch guarantees we are not one, and
     * this makes that guarantee checked rather than assumed. */
    const size_t sl = q35_len(slot, "gdn slot");
    float *cs   = R->conv + sl*K1*cd;
    float *Sall = R->S + sl*nv*dk*dv;
    const float qscale = 1.0f/sqrtf((float)dk);

    for(int s = 0; s < S; s++){
        float *qkv = B->qkv + (size_t)s*cd;
        float *beta = beta_all + (size_t)s*nv, *gdec = alpha_all + (size_t)s*nv;
        for(int h = 0; h < nv; h++){
            beta[h] = q35_sigmoid(beta[h]);
            gdec[h] = expf(q35_softplus(gdec[h] + L->dt_bias[h]) * L->ssm_a[h]);
        }
        for(int ch = 0; ch < cd; ch++){
            const float xin = qkv[ch];
            const float *kk = L->conv1d + (size_t)ch*c->d_conv;
            float acc = 0;
            for(int j = 0; j < K1; j++) acc += kk[j] * cs[(size_t)j*cd + ch];
            acc += kk[K1] * xin;
            for(int j = 0; j < K1-1; j++) cs[(size_t)j*cd + ch] = cs[(size_t)(j+1)*cd + ch];
            cs[(size_t)(K1-1)*cd + ch] = xin;
            qkv[ch] = q35_silu(acc);
        }
        float *q = qkv, *k = qkv + kdim, *vvv = qkv + 2*kdim;
        for(int h = 0; h < nk; h++){
            q35_l2norm(q + (size_t)h*dk, dk, c->eps);
            q35_l2norm(k + (size_t)h*dk, dk, c->eps);
        }
        float *os = B->o + (size_t)s*c->d_inner;

        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nv; h++){
            const int hk = q35_kh(h, nv, nk);
            float *St = Sall + (size_t)h*dk*dv;
            const float *kh = k + (size_t)hk*dk, *qh = q + (size_t)hk*dk;
            const float *vh = vvv + (size_t)h*dv;
            const float g = gdec[h], b = beta[h];
            float kvv[512], dd[512];
            for(int j = 0; j < dv; j++) kvv[j] = 0;
            for(int i = 0; i < dk; i++){
                float *Si = St + (size_t)i*dv;
                const float ki = kh[i];
                for(int j = 0; j < dv; j++){ Si[j] *= g; kvv[j] += Si[j]*ki; }
            }
            for(int j = 0; j < dv; j++) dd[j] = b*(vh[j] - kvv[j]);
            float *oh = os + (size_t)h*dv;
            for(int j = 0; j < dv; j++) oh[j] = 0;
            for(int i = 0; i < dk; i++){
                float *Si = St + (size_t)i*dv;
                const float ki = kh[i], qi = qh[i]*qscale;
                for(int j = 0; j < dv; j++){ Si[j] += ki*dd[j]; oh[j] += Si[j]*qi; }
            }
        }
        const float *zs = B->z + (size_t)s*c->d_inner;
        for(int h = 0; h < nv; h++){
            float *oh = os + (size_t)h*dv;
            q35_rmsnorm(oh, oh, L->ssm_norm, dv, c->eps);
            const float *zh = zs + (size_t)h*dv;
            for(int j = 0; j < dv; j++) oh[j] *= q35_silu(zh[j]);
        }

        /* this layer just finished token `snap_after`: freeze its recurrent state and conv
         * ring, so a rejected draft can be rewound to exactly here */
        if(B->snap_after == s){
            const size_t sb = (size_t)nv*dk*dv, cb = (size_t)K1*cd;
            memcpy(B->snap_S    + (size_t)slot*sb, Sall, sb*sizeof(float));
            memcpy(B->snap_conv + (size_t)slot*cb, cs,   cb*sizeof(float));
        }
    }
    q35_mm(out, B->o, B->M, L->w_out, S);
}

/* ---------------- the batched trunk ----------------
 * logits, if non-NULL, receives ONLY the last token's row: prefill does not need the other
 * S-1, and the output projection is 248320 wide — computing it for a whole batch would cost
 * more than the rest of the batch put together. */
/* all_logits: 0 = last token only (prefill), 1 = every token (speculation needs the
 * distribution at position p to check the draft AND the one at p+1 to continue from). */
static void q35_forward_batch_ex(Q35Batch *B, const int *toks, int S, int pos_base,
                                 float *logits, int all_logits){
    const Q35Model *M = B->M;
    const Q35Cfg *c = &M->c;
    const int D = c->d_model;

    for(int s = 0; s < S; s++) q35_embed(B->R, toks[s], B->x + (size_t)s*D);

    for(int il = 0; il < c->n_layer; il++){
        const Q35Layer *L = &M->L[il];

        { double t0 = q35_clk();
          q35_rmsnorm_rows(B->xn, B->x, L->attn_norm, D, S, c->eps);
          g_b_norm += q35_clk()-t0; }
        { double t0 = q35_clk();
          if(L->kind == Q35_LAYER_ATTN){ q35_attn_b(B, L, il, B->xn, pos_base, B->t, S);
                                         g_b_attn += q35_clk()-t0; }
          else                         { q35_gdn_b (B, L, il, B->xn, B->t, S);
                                         g_b_gdn  += q35_clk()-t0; } }
        for(int64_t i = 0; i < (int64_t)S*D; i++) B->x[i] += B->t[i];

        { double t0 = q35_clk();
          q35_rmsnorm_rows(B->xn, B->x, L->post_attn_norm, D, S, c->eps);
          g_b_norm += q35_clk()-t0; }
        { double t0 = q35_clk(); q35_ffn_b(B, L, B->xn, B->t, S); g_b_ffn += q35_clk()-t0; }
        for(int64_t i = 0; i < (int64_t)S*D; i++) B->x[i] += B->t[i];
    }

    if(logits){
        const int s0 = all_logits ? 0 : S-1;
        for(int s = s0; s < S; s++){
            q35_rmsnorm(B->xn, B->x + (size_t)s*D, M->output_norm, D, c->eps);
            q35_mv2(logits + (size_t)(s - s0)*c->vocab, B->xn, M, M->output);
        }
    }
    /* q35_hidden reads R->x; keep the batch's last row there so the draft head can use it */
    memcpy(B->R->x, B->x + (size_t)(S-1)*D, sizeof(float)*(size_t)D);
    B->R->pos = pos_base + S;
}

static void q35_forward_batch(Q35Batch *B, const int *toks, int S, int pos_base, float *logits){
    q35_forward_batch_ex(B, toks, S, pos_base, logits, 0);
}

/* Prefill a whole prompt in chunks of S_max. Returns the position after the last token. */
static int q35_prefill(Q35Batch *B, const int *toks, int n, int pos_base, float *logits){
    int pos = pos_base;
    for(int i = 0; i < n; i += B->S_max){
        const int S = (i + B->S_max <= n) ? B->S_max : (n - i);
        const int last = (i + S >= n);
        q35_forward_batch(B, toks + i, S, pos, last ? logits : NULL);
        pos += S;
    }
    return pos;
}

#endif
