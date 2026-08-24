/* engine_test.c — the assembled GLM-5.2 forward pass, against llama.cpp's logits.
 *
 * Every piece under this was validated on its own, but a forward pass is not the sum of its
 * parts: the residual stream, the RoPE base, the router's gate, the weight scale and the
 * shared expert all have to be right TOGETHER, and every one of them fails quietly. The
 * only honest test is the end of the chain — the logits themselves.
 *
 * Reference: llama-eval-callback on the same GGUF, token 77451 ("Hallo"), pos 0:
 *
 *      result_norm    sum -61.851807   [ 0.5331,  1.1695, -0.1678, ...  1.0483,  1.0136,  0.9699]
 *      result_output  sum -1118735.5   [-3.1656, -2.3425, -2.3782, ... -11.1635,-11.2077,-11.1894]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../glm_engine.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* Expert fetch: a plain pread into a scratch slab. The tiered cache is deliberately NOT in
 * the way here — this test is about whether the MATH is right, not how fast it is. */
typedef struct { GModel *M; int fd[GGUF_MAX_SHARD]; uint8_t *slab; int64_t off[3]; } Fetcher;

static int fetch_expert(void *ud, int layer, int eid,
                        const void **g, const void **u, const void **d){
    Fetcher *F = (Fetcher*)ud;
    const GLayer *L = &F->M->L[layer];
    const gguf_tensor *T[3] = { L->e_gate, L->e_up, L->e_down };
    const void **out[3] = { g, u, d };
    for(int q=0;q<3;q++){
        int fd; uint64_t off; int64_t nb, rows, cols;
        if(!gguf_expert_slice(&F->M->m, T[q], eid, &fd, &off, &nb, &rows, &cols)) return 0;
        if(pread(F->fd[T[q]->shard], F->slab + F->off[q], nb, off) != nb) return 0;
        *out[q] = F->slab + F->off[q];
    }
    return 1;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    GModel M;
    memset(&M, 0, sizeof(M));
    if(!gguf_open(&M.m, argv[1])){ printf("open failed\n"); return 1; }

    const char *arch = gguf_str(&M.m, "general.architecture", "");
    char k[160];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&M.m,k,0))
    #define KVF(s) (snprintf(k,sizeof(k),"%s." s,arch), (float)gguf_f64(&M.m,k,0))
    GCfg *c = &M.c;
    c->d_model  = KVI("embedding_length");
    c->n_layer  = KVI("block_count");
    c->n_dense  = KVI("leading_dense_block_count");
    c->n_head   = KVI("attention.head_count");
    c->vocab    = KVI("vocab_size");
    c->q_lora   = KVI("attention.q_lora_rank");
    c->kv_lora  = KVI("attention.kv_lora_rank");
    c->qk_rope  = KVI("rope.dimension_count");
    c->n_exp    = KVI("expert_count");
    c->topk     = KVI("expert_used_count");
    c->n_shared = KVI("expert_shared_count");
    c->d_ff     = KVI("feed_forward_length");
    c->d_ff_exp = KVI("expert_feed_forward_length");
    c->gating   = KVI("expert_gating_func");
    c->w_norm   = KVI("expert_weights_norm");
    c->eps      = KVF("attention.layer_norm_rms_epsilon");
    c->rope_base= KVF("rope.freq_base");
    c->w_scale  = KVF("expert_weights_scale");
    #undef KVI
    #undef KVF

    /* The MTP head is the last block; it predicts the NEXT-next token and is not part of
     * the ordinary forward pass. llama.cpp lists all of its tensors as "unused" for exactly
     * this reason, and running it would corrupt the residual stream. */
    const int n_run = c->n_layer - 1;

    printf("glm-dsa: %d blocks (%d dense, 1 MTP head skipped), %d experts top-%d + %d shared\n",
           c->n_layer, c->n_dense, c->n_exp, c->topk, c->n_shared);
    printf("  rope_base %.0f  gating %d (sigmoid)  weight_scale %.2f  norm %d\n\n",
           c->rope_base, c->gating, c->w_scale, c->w_norm);

    M.L = calloc(c->n_layer, sizeof(GLayer));
    #define T(fmt, ...) (snprintf(k,sizeof(k),fmt,__VA_ARGS__), gguf_find(&M.m,k))
    #define NEED(p,n) do{ if(!(p)){ printf("missing %s\n", n); return 1; } }while(0)

    M.emb = gguf_find(&M.m, "token_embd.weight");
    const gguf_tensor *ow = gguf_find(&M.m, "output.weight");
    const gguf_tensor *on = gguf_find(&M.m, "output_norm.weight");
    NEED(M.emb,"token_embd"); NEED(ow,"output"); NEED(on,"output_norm");
    M.out_w = gguf_data(&M.m, ow); M.t_out = ow->type;
    M.out_norm = (const float*)gguf_data(&M.m, on);

    for(int i=0;i<n_run;i++){
        GLayer *L = &M.L[i];
        const gguf_tensor *t;
        #define AT(field, tf, name) do{ t = T(name, i); NEED(t, name); \
            L->field = gguf_data(&M.m, t); L->tf = t->type; }while(0)
        AT(q_a,  t_q_a,  "blk.%d.attn_q_a.weight");
        AT(q_b,  t_q_b,  "blk.%d.attn_q_b.weight");
        AT(kv_a, t_kv_a, "blk.%d.attn_kv_a_mqa.weight");
        AT(k_b,  t_k_b,  "blk.%d.attn_k_b.weight");
        AT(v_b,  t_v_b,  "blk.%d.attn_v_b.weight");
        AT(o,    t_o,    "blk.%d.attn_output.weight");
        #undef AT
        #define NORM(field, name) do{ t = T(name, i); NEED(t, name); \
            L->field = (const float*)gguf_data(&M.m, t); }while(0)
        NORM(attn_norm, "blk.%d.attn_norm.weight");
        NORM(q_a_norm,  "blk.%d.attn_q_a_norm.weight");
        NORM(kv_a_norm, "blk.%d.attn_kv_a_norm.weight");
        NORM(ffn_norm,  "blk.%d.ffn_norm.weight");
        #undef NORM

        if(i < c->n_dense){
            L->is_moe = 0;
            t = T("blk.%d.ffn_gate.weight", i); NEED(t,"ffn_gate"); L->gate = gguf_data(&M.m,t); L->t_gate = t->type;
            t = T("blk.%d.ffn_up.weight",   i); NEED(t,"ffn_up");   L->up   = gguf_data(&M.m,t); L->t_up   = t->type;
            t = T("blk.%d.ffn_down.weight", i); NEED(t,"ffn_down"); L->down = gguf_data(&M.m,t); L->t_down = t->type;
        } else {
            L->is_moe = 1;
            L->e_gate = T("blk.%d.ffn_gate_exps.weight", i); NEED(L->e_gate,"gate_exps");
            L->e_up   = T("blk.%d.ffn_up_exps.weight",   i); NEED(L->e_up,  "up_exps");
            L->e_down = T("blk.%d.ffn_down_exps.weight", i); NEED(L->e_down,"down_exps");
            t = T("blk.%d.ffn_gate_shexp.weight", i); NEED(t,"gate_shexp"); L->sh_gate = gguf_data(&M.m,t); L->t_sh_gate = t->type;
            t = T("blk.%d.ffn_up_shexp.weight",   i); NEED(t,"up_shexp");   L->sh_up   = gguf_data(&M.m,t); L->t_sh_up   = t->type;
            t = T("blk.%d.ffn_down_shexp.weight", i); NEED(t,"down_shexp"); L->sh_down = gguf_data(&M.m,t); L->t_sh_down = t->type;
            t = T("blk.%d.ffn_gate_inp.weight", i); NEED(t,"ffn_gate_inp");
            L->router = (const float*)gguf_data(&M.m,t);
            t = T("blk.%d.exp_probs_b.bias", i);
            L->router_bias = t ? (const float*)gguf_data(&M.m,t) : NULL;
        }
    }
    #undef T
    #undef NEED

    /* shapes, cross-checked against the header (a misread k_b would pass silently) */
    const gguf_tensor *kb = gguf_find(&M.m, "blk.0.attn_k_b.weight");
    const gguf_tensor *vb = gguf_find(&M.m, "blk.0.attn_v_b.weight");
    c->qk_nope = (int)kb->ne[0];
    c->v_head  = (int)vb->ne[1];
    if(kb->ne[1] != c->kv_lora || vb->ne[0] != c->kv_lora){
        printf("FAIL: k_b/v_b do not contract the latent — we are misreading them\n"); return 1; }

    MLACfg mc = { c->n_head, c->qk_nope, c->qk_rope, c->v_head, c->kv_lora, c->q_lora,
                  c->d_model, c->eps, 1.0f/sqrtf((float)(c->qk_nope + c->qk_rope)) };
    M.kv = calloc(c->n_layer, sizeof(MLACache));
    for(int i=0;i<n_run;i++) mla_cache_init(&M.kv[i], &mc, 8);
    mla_scratch_init(&M.s, &mc, 8);

    /* expert fetch */
    Fetcher F; memset(&F,0,sizeof(F)); F.M = &M;
    for(int s=0;s<M.m.n_shards;s++) F.fd[s] = open(M.m.shard[s].path, O_RDONLY);
    int64_t big[3] = {0,0,0};
    for(int i=c->n_dense;i<n_run;i++){
        const gguf_tensor *T3[3] = { M.L[i].e_gate, M.L[i].e_up, M.L[i].e_down };
        for(int q=0;q<3;q++){
            int64_t b = kq_row_bytes(T3[q]->type, T3[q]->ne[0]*T3[q]->ne[1]);
            if(b > big[q]) big[q] = b;
        }
    }
    F.off[0]=0; F.off[1]=big[0]; F.off[2]=big[0]+big[1];
    F.slab = malloc(big[0]+big[1]+big[2]);

    /* ---- the forward pass ---- */
    float *logits = malloc((size_t)c->vocab*sizeof(float));
    printf("forward: token 77451 (\"Hallo\"), pos 0, %d layers, %d experts streamed...\n",
           n_run, (n_run - c->n_dense)*c->topk);
    double t0 = now();
    GCfg saved = *c; saved.n_layer = n_run;
    GCfg *cp = &M.c; int keep = cp->n_layer; cp->n_layer = n_run;
    glm_forward(&M, 77451, 0, logits, fetch_expert, &F);
    cp->n_layer = keep;
    double dt = now() - t0;
    printf("  %.1f s\n\n", dt);

    /* ---- against llama.cpp ---- */
    double sum = 0;
    for(int i=0;i<c->vocab;i++) sum += logits[i];
    const double ref_sum = -1118735.5;
    const double ref[6]  = { -3.1656, -2.3425, -2.3782, -11.1635, -11.2077, -11.1894 };
    const int    ix[6]   = { 0, 1, 2, c->vocab-3, c->vocab-2, c->vocab-1 };

    printf("logits vs llama.cpp\n");
    printf("  %-10s", "ours");
    for(int i=0;i<6;i++) printf(" %9.4f", logits[ix[i]]);
    printf("\n  %-10s", "llama.cpp");
    for(int i=0;i<6;i++) printf(" %9.4f", ref[i]);
    printf("\n\n");

    double worst = 0;
    for(int i=0;i<6;i++){
        double r = fabs(logits[ix[i]] - ref[i]) / (fabs(ref[i]) + 1.0);
        if(r > worst) worst = r;
    }
    printf("  sum        %14.1f   llama.cpp %14.1f   rel %.2e\n",
           sum, ref_sum, fabs(sum-ref_sum)/fabs(ref_sum));
    printf("  worst element deviation: %.2e\n", worst);

    /* argmax is what actually decides the next token */
    int am = 0;
    for(int i=1;i<c->vocab;i++) if(logits[i] > logits[am]) am = i;
    printf("  argmax token: %d  (logit %.4f)\n", am, logits[am]);

    int ok = worst < 0.05 && fabs(sum-ref_sum)/fabs(ref_sum) < 0.05;
    printf("\n%s\n", ok ? "the assembled forward pass matches llama.cpp"
                        : "MISMATCH — something in the assembly is wrong");
    return ok ? 0 : 1;
}
