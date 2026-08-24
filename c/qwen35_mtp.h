#ifndef COLIBRI_QWEN35_MTP_H
#define COLIBRI_QWEN35_MTP_H
/* qwen35_mtp.h — the draft head that ships with the model, and speculative decoding on it.
 *
 * WHY THIS IS THE BEST SPEEDUP-PER-EFFORT AVAILABLE
 *
 * Decode is bandwidth bound: 15.6 GiB of weights read to produce ONE token, 0.52 s of it,
 * against 57.5 GB/s of RAM. Kernels cannot beat that and neither can a GPU that the model
 * does not fit into. The only way past a bandwidth wall is to get MORE THAN ONE TOKEN out of
 * a single pass over the weights.
 *
 * That is exactly what speculative decoding does, and this model ships the draft head for it
 * already trained: `blk.64` with `nextn.eh_proj / enorm / hnorm / shared_head_norm` and
 * `qwen35.nextn_predict_layers = 1`. No second model, no extra download, no distillation —
 * 0.218 GiB against the main model's 15.6, so a draft costs ~1.4% of a real pass.
 *
 * HOW THE HEAD WORKS
 *
 * After the main model emits token t+1 it also has h = rmsnorm(x_final, output_norm) — the
 * same hidden state the LM head consumed. The draft head takes that h TOGETHER WITH the
 * embedding of the token just emitted, and predicts t+2:
 *
 *     cur = eh_proj . concat( rmsnorm(embed(t+1), enorm), rmsnorm(h, hnorm) )
 *     cur = cur + attention(rmsnorm(cur, attn_norm))        <- blk.64's own weights and KV
 *     cur = cur + ffn(rmsnorm(cur, post_attn_norm))
 *     logits = output . rmsnorm(cur, shared_head_norm)
 *
 * It is one transformer layer conditioned on the full model's state, which is why its
 * agreement rate is far above what an independent small model would manage.
 *
 * THE ROLLBACK PROBLEM, WHICH IS SPECIFIC TO THIS ARCHITECTURE
 *
 * Verifying a draft means running the main model on [t+1, t+2] as a batch of 2. If the draft
 * is wrong, position t+2 must be undone. For a pure attention model that is trivial — move
 * the KV write cursor back. Here 48 of 64 layers are Gated DeltaNet, whose recurrent state
 * was ADVANCED by both tokens and cannot be reversed: S_{t+2} = f(S_{t+1}) is not invertible.
 *
 * So the state is snapshotted before the verification batch and restored on rejection.
 * That is 146 MiB of GDN state + 6 MiB of conv ring — ~2.6 ms at 57.5 GB/s, against a
 * ~520 ms token. 0.5% overhead to make the whole scheme possible. The conv ring must be
 * snapshotted too, and forgetting it is silent: the model keeps talking, with three
 * positions of convolution history that belong to a rejected branch.
 */
#include "qwen35_batch.h"
#include <time.h>

typedef struct {
    const Q35Model *M;
    Q35State *R;
    const Q35Layer *L;           /* blk.64 */
    int mtp_slot;                /* its own attention KV slot */

    float *cat;                  /* [2*d_model] concat(e_norm, h_norm) */
    float *cur, *xn, *tmp;       /* [d_model] */

    /* rollback snapshot */
    float *snap_S, *snap_conv;
    int64_t snap_S_bytes, snap_conv_bytes;

    /* accounting */
    int64_t drafted, accepted;
    double  t_draft, t_verify;
} Q35Mtp;

static int q35_mtp_init(Q35Mtp *P, Q35State *R){
    memset(P, 0, sizeof *P);
    const Q35Model *M = R->M;
    const Q35Cfg *c = &M->c;
    if(c->n_layer_mtp < 1) return 0;
    P->M = M; P->R = R;
    P->L = &M->L[c->n_layer];                 /* the block just past the trunk */
    if(!P->L->eh_proj || !P->L->enorm || !P->L->hnorm) return 0;

    /* The MTP block owns an attention layer, so it needs a KV slot of its own. The tiered
     * store was sized for the trunk's 16; this reuses the LAST slot, which is safe only
     * because the trunk's slot 15 and the MTP block are never live at the same position...
     * they are. So: require the store to have been opened with one extra layer. */
    P->mtp_slot = R->n_attn;                  /* see q35_mtp_slots() */

    P->cat  = (float*)calloc(2*(size_t)c->d_model, sizeof(float));
    P->cur  = (float*)calloc((size_t)c->d_model, sizeof(float));
    P->xn   = (float*)calloc((size_t)c->d_model, sizeof(float));
    P->tmp  = (float*)calloc((size_t)c->d_model, sizeof(float));

    P->snap_S_bytes    = (int64_t)R->n_gdn * c->n_v_heads * c->d_state * c->d_head_v * sizeof(float);
    P->snap_conv_bytes = (int64_t)R->n_gdn * (c->d_conv-1) * c->conv_dim * sizeof(float);
    P->snap_S    = (float*)malloc((size_t)P->snap_S_bytes);
    P->snap_conv = (float*)malloc((size_t)P->snap_conv_bytes);
    return P->cat && P->cur && P->xn && P->tmp && P->snap_S && P->snap_conv;
}

static void q35_mtp_free(Q35Mtp *P){
    free(P->cat); free(P->cur); free(P->xn); free(P->tmp);
    free(P->snap_S); free(P->snap_conv);
    memset(P, 0, sizeof *P);
}

/* The KV store must be opened with room for the MTP block's attention layer too. */
static int q35_mtp_slots(const Q35Cfg *c){ return q35_n_attn_layers(c) + (c->n_layer_mtp ? 1 : 0); }

/* Snapshot / restore the parts of the state that a rejected draft would corrupt. The KV
 * cache needs neither: it is append-only and a rejected position is simply overwritten. */
static void q35_mtp_snapshot(Q35Mtp *P){
    memcpy(P->snap_S,    P->R->S,    (size_t)P->snap_S_bytes);
    memcpy(P->snap_conv, P->R->conv, (size_t)P->snap_conv_bytes);
}
static void q35_mtp_restore(Q35Mtp *P){
    memcpy(P->R->S,    P->snap_S,    (size_t)P->snap_S_bytes);
    memcpy(P->R->conv, P->snap_conv, (size_t)P->snap_conv_bytes);
}

/* One draft. h is the main model's post-output_norm hidden state for position `pos`, tok is
 * the token emitted AT `pos`; logits come back for position pos+1. */
/* KV ONLY, for prefill.
 *
 * The draft head owns an attention block, and that block attends over the whole sequence —
 * but nothing runs it during prefill, so its KV slot is empty for every prompt position.
 * Empty entries are not neutral: they are zeros, they score 0, and after the softmax a few
 * hundred of them drown whatever real history exists. Measured on a 382-token prompt that
 * takes draft acceptance from ~63% to 8%, which makes speculation a net LOSS while looking
 * like "the draft head is just bad on this content".
 *
 * Filling it needs eh_proj, the norm and the k/v projections — 36 MiB a token against the
 * block's full 240 — because the attention OUTPUT at a prompt position is never used, only
 * the keys and values it would have written. */
static void q35_mtp_prefill_kv(Q35Mtp *P, const float *h, int tok, int pos){
    const Q35Model *M = P->M;
    const Q35Cfg *c = &M->c;
    const Q35Layer *L = P->L;
    const int D = c->d_model, dh = c->d_head, nkv = c->n_head_kv;
    Q35State *R = P->R;

    q35_embed(R, tok, P->tmp);
    q35_rmsnorm(P->cat,     P->tmp, L->enorm, D, c->eps);
    q35_rmsnorm(P->cat + D, h,      L->hnorm, D, c->eps);
    q35_mv2(P->cur, P->cat, M, L->eh_proj);
    q35_rmsnorm(P->xn, P->cur, L->attn_norm, D, c->eps);
    q35_mv2(R->k, P->xn, M, L->wk);
    q35_mv2(R->v, P->xn, M, L->wv);
    for(int hh = 0; hh < nkv; hh++){
        float *k = R->k + (size_t)hh*dh;
        q35_rmsnorm(k, k, L->k_norm, dh, c->eps);
        q35_rope(k, pos, c->n_rot, c->rope_base);
    }
    kvt_put(&R->kv, P->mtp_slot, pos, R->k, R->v);
}

/* Everything except the output projection. Split out because on the GPU path that
 * projection is 0.97 GiB of Q6_K that lives in VRAM, and running it on the CPU instead
 * costs 30 ms — which alone would take the per-draft cost `e` from 0.04 to 0.14 and eat
 * most of what speculation is supposed to win. */
static void q35_mtp_hidden(Q35Mtp *P, const float *h, int tok, int pos){
    const Q35Model *M = P->M;
    const Q35Cfg *c = &M->c;
    const Q35Layer *L = P->L;
    const int D = c->d_model;
    Q35State *R = P->R;

    /* concat(rmsnorm(embed(tok), enorm), rmsnorm(h, hnorm)) — embedding first, matching
     * ggml_concat(e_norm, h_norm, dim=0) */
    q35_embed(R, tok, P->tmp);
    q35_rmsnorm(P->cat,     P->tmp, L->enorm, D, c->eps);
    q35_rmsnorm(P->cat + D, h,      L->hnorm, D, c->eps);
    q35_mv2(P->cur, P->cat, M, L->eh_proj);

    /* one full-attention block on blk.64's own weights and KV slot */
    memcpy(P->tmp, P->cur, sizeof(float)*(size_t)D);          /* inpSA */
    q35_rmsnorm(P->xn, P->cur, L->attn_norm, D, c->eps);

    const int saved = R->attn_slot[c->n_layer];
    R->attn_slot[c->n_layer] = P->mtp_slot;
    q35_attn(R, L, c->n_layer, P->xn, pos, P->cur);
    R->attn_slot[c->n_layer] = saved;

    for(int i = 0; i < D; i++) P->cur[i] += P->tmp[i];
    memcpy(P->tmp, P->cur, sizeof(float)*(size_t)D);          /* ffn residual */
    q35_rmsnorm(P->xn, P->cur, L->post_attn_norm, D, c->eps);
    q35_ffn(R, L, P->xn, P->cur);
    for(int i = 0; i < D; i++) P->cur[i] += P->tmp[i];

    /* shared_head_norm if present, else the model's output_norm (llama.cpp's fallback) */
    q35_rmsnorm(P->xn, P->cur, L->head_norm ? L->head_norm : M->output_norm, D, c->eps);
}

static void q35_mtp_draft(Q35Mtp *P, const float *h, int tok, int pos, float *logits){
    q35_mtp_hidden(P, h, tok, pos);
    q35_mv2(logits, P->xn, P->M, P->M->output);
}

/* The hidden state the draft head needs. q35_forward computes it internally and throws it
 * away; this recomputes the two cheap steps from the residual stream it leaves behind. */
static void q35_hidden(const Q35State *R, float *h){
    const Q35Model *M = R->M;
    q35_rmsnorm(h, R->x, M->output_norm, M->c.d_model, M->c.eps);
}

static int q35_argmax(const float *v, int n){
    int b = 0;
    for(int i = 1; i < n; i++) if(v[i] > v[b]) b = i;
    return b;
}

/* ---------------- the speculative decode loop ----------------
 *
 * One round, with 1-token lookahead:
 *
 *   1. draft   = MTP(h, tok_p)                      cheap: 8.3% of a main pass
 *   2. verify  = main model on [tok_p, draft] as a BATCH OF 2, one pass over the weights
 *   3. logits at position p give the TRUE token at p+1.
 *        accept  -> two tokens confirmed, continue from the logits at p+1
 *        reject  -> one token confirmed; rewind the recurrent state to "after token 0"
 *
 * The batch is what makes this pay: it reads all 15.6 GiB once and produces two candidate
 * positions. A rejected round still advances one token, so speculation never loses ground —
 * it only fails to gain.
 *
 * The rewind is the part this architecture makes awkward, and it is why B->snap_after exists:
 * token 0 of the batch was the real token and its state update is VALID, so the rollback
 * target is "after token 0", not "before the batch". Rewinding too far would be correct but
 * would cost a whole extra pass to redo token 0. */
typedef struct {
    int64_t rounds, tokens, accepts;
    double  t_draft, t_verify, t_rollback;
} Q35SpecStats;

/* Generates up to n_out tokens starting from `tok` at `pos`. Returns how many were produced.
 * R must already hold the state for positions < pos, with R->x the residual at pos-1. */
static int q35_spec_generate(Q35Mtp *P, Q35Batch *B, int tok, int pos,
                             int *out, int n_out, int eos, Q35SpecStats *st){
    const Q35Model *M = P->M;
    const int V = M->c.vocab;
    Q35State *R = P->R;
    const Q35Cfg *c = &M->c;

    float *h  = (float*)malloc(sizeof(float)*c->d_model);
    float *dr = (float*)malloc(sizeof(float)*V);
    float *lg = (float*)malloc(sizeof(float)*2*(size_t)V);     /* both batch positions */
    if(!h || !dr || !lg){ free(h); free(dr); free(lg); return 0; }

    const size_t sb = (size_t)R->n_gdn*c->n_v_heads*c->d_state*c->d_head_v;
    const size_t cb = (size_t)R->n_gdn*(c->d_conv-1)*c->conv_dim;
    int n = 0;

    while(n < n_out){
        double t0 = q35_clk();
        q35_hidden(R, h);
        q35_mtp_draft(P, h, tok, pos, dr);
        const int draft = q35_argmax(dr, V);
        st->t_draft += q35_clk() - t0;

        /* verify: one pass over the weights, two positions */
        const int pair[2] = { tok, draft };
        B->snap_after = 0;                       /* freeze state after token 0, per layer */
        t0 = q35_clk();
        q35_forward_batch_ex(B, pair, 2, pos, lg, 1);
        st->t_verify += q35_clk() - t0;
        B->snap_after = -1;

        const int truth = q35_argmax(lg, V);     /* the real token at pos+1 */
        st->rounds++;

        out[n++] = tok;                          /* tok at `pos` was always real */
        if(n >= n_out || tok == eos) break;

        if(truth == draft){
            st->accepts++;
            out[n++] = draft;
            if(n >= n_out || draft == eos) break;
            tok = q35_argmax(lg + V, V);         /* continue from position pos+1's logits */
            pos += 2;
        } else {
            /* the drafted position never happened: rewind to just after token 0. Its KV entry
             * at pos+1 needs no undo — it is append-only and the next round overwrites it. */
            t0 = q35_clk();
            memcpy(R->S,    B->snap_S,    sb*sizeof(float));
            memcpy(R->conv, B->snap_conv, cb*sizeof(float));
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
