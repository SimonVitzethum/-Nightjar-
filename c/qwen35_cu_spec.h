#ifndef COLIBRI_QWEN35_CU_SPEC_H
#define COLIBRI_QWEN35_CU_SPEC_H
/* qwen35_cu_spec.h — batched verification and speculative decode on the hetero path.
 *
 * WHY THIS IS THE ONLY REMAINING LEVER
 *
 * After the CUDA split, 90% of a token is the CPU reading FFN weights, and it does that at
 * 36 GB/s while the machine's own ceiling measures 64.8 — so the FFN kernel is decode-bound,
 * not bandwidth-bound, and there is no bandwidth left to find. What IS left is reading those
 * weights FEWER TIMES: a batch of S tokens reads them once. Measured on this CPU, S=8 gets
 * 3.75x out of one pass. Speculation converts that into decode speed, because the draft head
 * shipped inside the model agrees with the trunk 95.8% of the time.
 *
 * THE SHAPE OF ONE ROUND
 *
 *   draft  = MTP(hidden, tok)                one block instead of 64, plus the output
 *                                            projection ON THE GPU (0.97 GiB of Q6_K; doing
 *                                            it on the CPU would cost more than the draft)
 *   verify = trunk over [tok, draft] as a BATCH — one pass over the weights, two positions
 *   accept -> two tokens confirmed;  reject -> one, and the recurrent state rewinds
 *
 * A rejected round still advances one token, so speculation never loses ground.
 *
 * THE ROLLBACK IS THE AWKWARD PART, AND IT IS THE GDN'S FAULT
 *
 * Attention KV is append-only: a rejected position is simply overwritten and needs no undo.
 * The gated delta net is not — its state was multiplied by a decay and rank-1 updated, and
 * that is not invertible. So the state has to be copied. The rollback target is "after token
 * 0", not "before the batch", because token 0 was the real token and its update is valid;
 * rewinding further would be correct but would cost an extra pass to redo it. The layer loop
 * is outer, so "after token 0" is not one instant — each GDN layer reaches it separately and
 * snapshots its own slot the moment it does.
 */
#define KQ_CU_BATCH_MAX 8
#include "qwen35_hetero.h"
#include "qwen35_mtp.h"

typedef struct {
    Q35Cu    *G;
    Q35State *R;
    int       S_max;

    /* device */
    float *xB, *xnB, *tB, *gB, *uB, *logitsB;
    float *snapS, *snapConv;
    int64_t snapS_bytes, snapConv_bytes;

    /* pinned host staging for the FFN hand-off */
    float *hxnB, *htB;
    float *hgB, *huB;            /* [S][d_ff] for the CPU half of a split layer */
    float *stB;                  /* [S][d_model] device landing zone for the CPU's partial */

    /* the CPU's batched FFN scratch (kq_gemm_batched lives behind it) */
    Q35Batch B;
    int      snap_after;          /* -1 = off */
    int64_t  vram;
} Q35CuBatch;

static int q35cu_batch_init(Q35CuBatch *Z, Q35Cu *G, Q35State *R, int S_max){
    memset(Z, 0, sizeof *Z);
    const Q35Cfg *c = &G->M->c;
    Z->G = G; Z->R = R; Z->S_max = S_max; Z->snap_after = -1;
    if(!q35_batch_init(&Z->B, R, S_max)) return 0;

    const size_t D = (size_t)c->d_model*S_max;
    Z->xB  = (float*)q35cu_alloc(D*4);
    Z->xnB = (float*)q35cu_alloc(D*4);
    Z->tB  = (float*)q35cu_alloc(D*4);
    Z->logitsB = (float*)q35cu_alloc((size_t)c->vocab*S_max*4);
    Z->vram += (int64_t)(3*D + (size_t)c->vocab*S_max)*4;
    if(G->n_ffn_gpu){
        Z->gB = (float*)q35cu_alloc((size_t)c->d_ff*S_max*4);
        Z->uB = (float*)q35cu_alloc((size_t)c->d_ff*S_max*4);
        Z->vram += 2*(int64_t)c->d_ff*S_max*4;
    }
    Z->snapS_bytes    = (int64_t)R->n_gdn*c->n_v_heads*c->d_state*c->d_head_v*4;
    Z->snapConv_bytes = (int64_t)R->n_gdn*(c->d_conv-1)*c->conv_dim*4;
    Z->snapS    = (float*)q35cu_alloc((size_t)Z->snapS_bytes);
    Z->snapConv = (float*)q35cu_alloc((size_t)Z->snapConv_bytes);
    Z->vram += Z->snapS_bytes + Z->snapConv_bytes;

    Z->hxnB = (float*)q35cu_host_alloc(D*4);
    Z->htB  = (float*)q35cu_host_alloc(D*4);
    Z->stB  = (float*)q35cu_alloc(D*4);
    Z->vram += (int64_t)D*4;
    Z->hgB  = (float*)q35cu_host_alloc((size_t)c->d_ff*S_max*4);
    Z->huB  = (float*)q35cu_host_alloc((size_t)c->d_ff*S_max*4);
    if(!Z->xB || !Z->xnB || !Z->tB || !Z->logitsB || !Z->snapS || !Z->snapConv
       || !Z->hxnB || !Z->htB || !Z->hgB || !Z->huB || !Z->stB
       || (G->n_ffn_gpu && (!Z->gB || !Z->uB))){
        fprintf(stderr, "qwen35_cu_spec: out of memory (%s)\n", q35cu_error());
        return 0;
    }
    return q35cu_sync();
}

static void q35cu_batch_free(Q35CuBatch *Z){
    void *p[] = { Z->xB, Z->xnB, Z->tB, Z->gB, Z->uB, Z->logitsB, Z->snapS, Z->snapConv, Z->stB };
    for(unsigned i = 0; i < sizeof p/sizeof *p; i++) q35cu_free(p[i]);
    q35cu_host_free(Z->hxnB); q35cu_host_free(Z->htB);
    q35cu_host_free(Z->hgB);  q35cu_host_free(Z->huB);
    q35_batch_free(&Z->B);
    memset(Z, 0, sizeof *Z);
}

/* ---------------- the batched FFN ----------------
 * The whole reason the batch exists. On CPU layers kq_gemm_batched dequantizes each weight
 * once and reuses it for every token in the batch; on GPU layers the same thing happens
 * inside k_gemv_q4k. */
/* The split FFN, batched.
 *
 * This is where speculation actually pays. The PCIe traffic of a batch is the SAME as of a
 * single token — the weights cross the bus once and serve every position — so the GPU half
 * gets S tokens for the price of one, and only the CPU half pays a marginal cost. That turns
 * the batch from "3.75x on the CPU" into something much better on the streamed layers.
 *
 * The down projection is again cut along the contraction, so neither side needs the other's
 * swiglu vector. kq_dot_fast_x2 exists for exactly the S=2 case that a 1-token draft makes:
 * one pass over the weight row, two accumulators. */
static int q35cu_ffn_split_b(Q35CuBatch *Z, int il, const float *xnB, float *tB, int S, int half){
    Q35Cu *G = Z->G;
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35Layer *L = &M->L[il];
    const int r = G->slice[half].r;
    if(r <= 0 || S > KQ_CU_BATCH_MAX) return 0;
    const int64_t gb = G->slice[half].gb;
    uint8_t *base = G->stage + (int64_t)half*(G->stage_bytes/2);
    const int Dm = c->d_model, Dff = c->d_ff, nrow = Dff - r;
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, Dm);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, Dff);
    const int64_t skip = (int64_t)(r/KQ_QK_K)*kq_typesize(L->ffn_down->type);

    q35cu_sync();
    q35cu_d2h(Z->hxnB, xnB, (size_t)S*Dm*4);

    q35cu_stream_join(half);
    q35cu_gemm(Z->gB, xnB, base,         L->ffn_gate->type, Dm, r, S);
    q35cu_gemm(Z->uB, xnB, base + gb,    L->ffn_up->type,   Dm, r, S);
    q35cu_swiglu(Z->gB, Z->uB, S*r);
    q35cu_gemm(tB, Z->gB, base + 2*gb,   L->ffn_down->type, r, Dm, S);
    q35cu_compute_mark(half);     /* this half is free once these have run */

    /* CPU half, concurrent with all of that */
    const double tc = q35_clk();
    kq_gemm_batched(Z->hgB, Z->hxnB,
                    (const uint8_t*)q35_wdata(M, L->ffn_gate) + (int64_t)r*grb,
                    L->ffn_gate->type, S, Dm, nrow);
    kq_gemm_batched(Z->huB, Z->hxnB,
                    (const uint8_t*)q35_wdata(M, L->ffn_up) + (int64_t)r*grb,
                    L->ffn_up->type, S, Dm, nrow);
    for(int64_t i = 0; i < (int64_t)S*nrow; i++)
        Z->hgB[i] = q35_silu(Z->hgB[i])*Z->huB[i];
    const uint8_t *Wd = (const uint8_t*)q35_wdata(M, L->ffn_down);
    #pragma omp parallel for schedule(static)
    for(int o = 0; o < Dm; o++){
        const uint8_t *w = Wd + (int64_t)o*drb + skip;
        int s = 0;
        for(; s + 1 < S; s += 2){
            float ra, rb;
            kq_dot_fast_x2(L->ffn_down->type, w, Z->hgB + (int64_t)s*nrow,
                           Z->hgB + (int64_t)(s+1)*nrow, nrow, &ra, &rb);
            Z->htB[(int64_t)s*Dm + o] = ra;
            Z->htB[(int64_t)(s+1)*Dm + o] = rb;
        }
        for(; s < S; s++)
            Z->htB[(int64_t)s*Dm + o] =
                kq_dot_fast(L->ffn_down->type, w, Z->hgB + (int64_t)s*nrow, nrow);
    }
    G->t_cpu_side += q35_clk() - tc;

    /* NOT into xnB. xnB is this layer's INPUT and the gate/up gemms above are still reading
     * it on the compute stream while the host runs the CPU half — a synchronous h2d into it
     * is a race, and it lands differently depending on how long the CPU half takes. Measured:
     * exact at S=2 and wrong by 1e-1 at S=4 and S=8, which reads like "batching is broken"
     * and is really "the input buffer was overwritten mid-flight". The single-token path
     * always used a dedicated buffer; this one has one now too. */
    q35cu_h2d(Z->stB, Z->htB, (size_t)S*Dm*4);
    q35cu_add(tB, Z->stB, S*Dm);
    return 1;
}

static void q35cu_ffn_b(Q35CuBatch *Z, int il, const float *xnB, float *tB, int S){
    Q35Cu *G = Z->G;
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *L = &M->L[il];
    if(D->ffn_gpu){
        q35cu_gemm(Z->gB, xnB, D->ffn_gate, L->ffn_gate->type, c->d_model, c->d_ff, S);
        q35cu_gemm(Z->uB, xnB, D->ffn_up,   L->ffn_up->type,   c->d_model, c->d_ff, S);
        q35cu_swiglu(Z->gB, Z->uB, S*c->d_ff);
        q35cu_gemm(tB, Z->gB, D->ffn_down, L->ffn_down->type, c->d_ff, c->d_model, S);
    } else if(G->stream_on && q35cu_ffn_split_b(Z, il, xnB, tB, S, G->stage_half)){
        /* handled: both sides at once, weights over PCIe once for the whole batch */
    } else {
        q35cu_sync();
        q35cu_d2h(Z->hxnB, xnB, (size_t)S*c->d_model*4);
        q35_ffn_b(&Z->B, L, Z->hxnB, Z->htB, S);
        q35cu_h2d(tB, Z->htB, (size_t)S*c->d_model*4);
    }
}

/* ---------------- one batch through the trunk ----------------
 * The mix stage stays per-token: the gated delta net is recurrent and attention is causal,
 * so token s genuinely has to see the state token s-1 left. That costs nothing that matters,
 * because the mix reads 4.1 GiB from VRAM at 254 GB/s while the FFN reads 9.65 from RAM at
 * 36 — the batch is there for the slow side. */
static void q35_forward_cu_batch_ex(Q35CuBatch *Z, const int *toks, int S, int pos_base,
                                    float *logits, int all_logits, float *x_all){
    Q35Cu *G = Z->G; Q35State *R = Z->R;
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const int Dm = c->d_model;

    for(int s = 0; s < S; s++){
        q35_embed(R, toks[s], R->x);
        q35cu_h2d(Z->xB + (size_t)s*Dm, R->x, sizeof(float)*Dm);
    }

    /* A batch uses each weight S times, so the CPU side is compute-bound where decode was
     * bandwidth-bound, and the GPU should take a much larger share. */
    G->stream_batch = (S > 1);
    if(G->stream_on){
        const int f0 = q35cu_next_stream_layer(G, 0);
        G->stage_half = 0;
        if(f0 >= 0) q35cu_stream_issue(G, f0, 0);
    }
    for(int il = 0; il < c->n_layer; il++){
        const Q35CuLayer *D = &G->L[il];
        if(G->stream_on && !D->ffn_gpu){
            const int nl = q35cu_next_stream_layer(G, il+1);
            if(nl >= 0) q35cu_stream_issue(G, nl, 1 - G->stage_half);
        }
        q35cu_rmsnorm_rows(Z->xnB, Z->xB, D->attn_norm, Dm, S, c->eps);
        const int is_attn = (c->kind[il] == Q35_LAYER_ATTN);
        const int slot = is_attn ? R->attn_slot[il] : R->gdn_slot[il];
        for(int s = 0; s < S; s++){
            const float *xn = Z->xnB + (size_t)s*Dm;
            float *out = Z->tB + (size_t)s*Dm;
            if(is_attn) q35cu_attn_layer_x(G, R, il, slot, pos_base + s, xn, out);
            else        q35cu_gdn_layer_x (G, il, slot, xn, out);
            /* freeze this layer's slot the moment it has finished token `snap_after` */
            if(!is_attn && Z->snap_after == s){
                const size_t sn = (size_t)c->n_v_heads*c->d_state*c->d_head_v;
                const size_t cn = (size_t)(c->d_conv-1)*c->conv_dim;
                q35cu_d2d(Z->snapS + (size_t)slot*sn, G->S + (size_t)slot*sn, sn*4);
                q35cu_d2d(Z->snapConv + (size_t)slot*cn, G->conv + (size_t)slot*cn, cn*4);
            }
        }
        q35cu_add(Z->xB, Z->tB, S*Dm);
        q35cu_rmsnorm_rows(Z->xnB, Z->xB, D->post_attn_norm, Dm, S, c->eps);
        q35cu_ffn_b(Z, il, Z->xnB, Z->tB, S);
        if(G->stream_on && !D->ffn_gpu) G->stage_half = 1 - G->stage_half;
        q35cu_add(Z->xB, Z->tB, S*Dm);
    }

    if(logits){
        const int n = all_logits ? S : 1;
        const float *src = all_logits ? Z->xB : Z->xB + (size_t)(S-1)*Dm;
        q35cu_rmsnorm_rows(Z->xnB, src, G->output_norm, Dm, n, c->eps);
        q35cu_gemm(Z->logitsB, Z->xnB, G->output, M->output->type, Dm, c->vocab, n);
        q35cu_sync();
        q35cu_d2h(logits, Z->logitsB, sizeof(float)*(size_t)n*c->vocab);
    }
    /* the residual stream of the LAST token, which is what the draft head reads next */
    q35cu_sync();
    q35cu_d2h(R->x, Z->xB + (size_t)(S-1)*Dm, sizeof(float)*Dm);
    if(x_all) q35cu_d2h(x_all, Z->xB, sizeof(float)*(size_t)S*Dm);
    R->pos = pos_base + S;
}

static void q35_forward_cu_batch(Q35CuBatch *Z, const int *toks, int S, int pos_base,
                                 float *logits, int all_logits){
    q35_forward_cu_batch_ex(Z, toks, S, pos_base, logits, all_logits, NULL);
}

static void q35cu_spec_restore(Q35CuBatch *Z){
    q35cu_d2d(Z->G->S,    Z->snapS,    (size_t)Z->snapS_bytes);
    q35cu_d2d(Z->G->conv, Z->snapConv, (size_t)Z->snapConv_bytes);
    q35cu_sync();
}

/* ---------------- the loop ----------------
 * Returns tokens produced. `tok` at `pos` is already real; R holds the state for < pos and
 * R->x the residual at pos-1. */
static int q35_spec_generate_cu(Q35CuBatch *Z, Q35Mtp *P, int tok, int pos,
                                int *out, int n_out, int eos, Q35SpecStats *st,
                                float temp, float top_p, int top_k,
                                int (*pick)(float*, int, float, float, int, void*), void *pctx){
    Q35Cu *G = Z->G; Q35State *R = Z->R;
    const Q35Cfg *c = &G->M->c;
    const int V = c->vocab, Dm = c->d_model;

    float *h  = (float*)malloc(sizeof(float)*Dm);
    float *dr = (float*)malloc(sizeof(float)*V);
    float *lg = (float*)malloc(sizeof(float)*2*(size_t)V);
    if(!h || !dr || !lg){ free(h); free(dr); free(lg); return 0; }
    int n = 0;

    while(n < n_out){
        double t0 = q35_clk();
        /* draft: the MTP block on the CPU, its output projection on the GPU */
        q35_hidden(R, h);
        q35_mtp_hidden(P, h, tok, pos);
        q35cu_h2d(G->xn, P->xn, sizeof(float)*Dm);
        q35cu_gemv(G->logits, G->xn, G->output, G->M->output->type, Dm, V);
        q35cu_sync();
        q35cu_d2h(dr, G->logits, sizeof(float)*V);
        const int draft = q35_argmax(dr, V);
        st->t_draft += q35_clk() - t0;

        const int pair[2] = { tok, draft };
        Z->snap_after = 0;
        t0 = q35_clk();
        q35_forward_cu_batch(Z, pair, 2, pos, lg, 1);
        st->t_verify += q35_clk() - t0;
        Z->snap_after = -1;

        const int truth = pick ? pick(lg, V, temp, top_p, top_k, pctx) : q35_argmax(lg, V);
        st->rounds++;

        /* Speculation is not free: a verify pass over two positions costs ~1.22 single
         * passes, so a round that accepts nothing LOSES 22%. Acceptance is content
         * dependent — measured 85% continuing a factual sentence and 38% writing a summary
         * at temperature — so the break-even has to be watched rather than assumed. Below
         * it, fall back to plain decode for the rest of this call. */
        if(st->rounds >= 12 && (double)st->accepts/st->rounds < 0.25){
            out[n++] = tok;
            if(n < n_out && truth != eos){
                Z->snap_after = -1;
                int cur = truth, p2 = pos + 1;
                if(truth != draft) q35cu_spec_restore(Z);
                while(n < n_out){
                    out[n++] = cur;
                    if(cur == eos) break;
                    q35_forward_cu(G, R, cur, p2++, lg);
                    cur = pick ? pick(lg, V, temp, top_p, top_k, pctx) : q35_argmax(lg, V);
                }
            }
            break;
        }

        out[n++] = tok;
        if(n >= n_out || tok == eos) break;

        if(truth == draft){
            st->accepts++;
            out[n++] = draft;
            if(n >= n_out || draft == eos) break;
            tok = pick ? pick(lg + V, V, temp, top_p, top_k, pctx) : q35_argmax(lg + V, V);
            pos += 2;
        } else {
            t0 = q35_clk();
            q35cu_spec_restore(Z);
            st->t_rollback += q35_clk() - t0;
            tok = truth;
            pos += 1;
        }
    }
    st->tokens += n;
    free(h); free(dr); free(lg);
    return n;
}

#endif
