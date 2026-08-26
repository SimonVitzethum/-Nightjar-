/* qwen35_test.c — does the binding agree with the real file, and does the plan fit?
 *
 * Everything downstream (the CPU forward pass, the KV tiering, the GPU split) is built on
 * the geometry this test checks. A wrong n_rot or a q-projection read as 2x-wide Q instead
 * of Q|gate produces FLUENT GARBAGE, which is the single most expensive failure mode on
 * this project: it looks like it works.
 *
 * So this asserts the derived geometry against the tensor shapes independently, rather than
 * trusting either one.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../models.h"
#include "../qwen35_plan.h"

static int fails = 0;
static void ck(int cond, const char *what, long long got, long long want){
    if(cond) return;
    printf("  FAIL  %-42s got %lld, want %lld\n", what, got, want);
    fails++;
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : nj_model_path("Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf");
    int n_ctx = argc > 2 ? atoi(argv[2]) : 250000;

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open/bind FAILED\n"); return 1; }
    const Q35Cfg *c = &M.c;

    printf("qwen35_test: %s\n", path);

    /* --- geometry that the tensor shapes must corroborate --- */
    ck(c->d_head_v == c->d_state, "head_v_dim == d_state", c->d_head_v, c->d_state);
    ck(c->conv_dim == 2*c->d_state*c->n_k_heads + c->d_inner, "conv_dim", c->conv_dim,
       2*c->d_state*c->n_k_heads + c->d_inner);
    ck(c->n_v_heads % c->n_k_heads == 0, "n_v_heads divisible by n_k_heads",
       c->n_v_heads % c->n_k_heads, 0);
    ck(c->n_head % c->n_head_kv == 0, "n_head divisible by n_head_kv",
       c->n_head % c->n_head_kv, 0);
    ck(c->n_rot <= c->d_head, "n_rot <= d_head", c->n_rot, c->d_head);

    /* the q projection carries Q *and* the output gate: exactly 2x n_head*d_head */
    int n_attn = 0, n_gdn = 0, n_mtp = 0;
    for(int i = 0; i < c->n_layer_all; i++){
        if(c->kind[i] == Q35_LAYER_ATTN) n_attn++;
        else if(c->kind[i] == Q35_LAYER_GDN) n_gdn++;
        else n_mtp++;
    }
    ck(n_gdn + n_attn == c->n_layer, "gdn + attn == n_layer", n_gdn + n_attn, c->n_layer);
    ck(n_mtp == c->n_layer_mtp, "mtp count", n_mtp, c->n_layer_mtp);

    for(int i = 0; i < c->n_layer_all; i++){
        const Q35Layer *L = &M.L[i];
        if(L->kind == Q35_LAYER_GDN){
            ck(L->wqkv && L->wqkv->ne[1] == c->conv_dim, "gdn wqkv ne[1]",
               L->wqkv ? L->wqkv->ne[1] : -1, c->conv_dim);
            ck(L->wq == NULL, "gdn layer has no attn_q", L->wq ? 1 : 0, 0);
        } else {
            ck(L->wq && L->wq->ne[1] == 2LL*c->n_head*c->d_head, "attn wq ne[1] == 2*n_head*d_head",
               L->wq ? L->wq->ne[1] : -1, 2LL*c->n_head*c->d_head);
            ck(L->wqkv == NULL, "attn layer has no attn_qkv", L->wqkv ? 1 : 0, 0);
            ck(L->wo && L->wo->ne[0] == (int64_t)c->n_head*c->d_head, "attn wo ne[0]",
               L->wo ? L->wo->ne[0] : -1, (long long)c->n_head*c->d_head);
        }
    }

    /* ssm_a must be NEGATIVE — it holds -exp(A_log), not A_log. If a future conversion
     * changes that, exp(gate) leaves (0,1] and the recurrence explodes instead of decaying.
     * That is a silent, slow divergence, so it is checked here rather than discovered. */
    for(int i = 0; i < c->n_layer_all; i++){
        const Q35Layer *L = &M.L[i];
        if(L->kind != Q35_LAYER_GDN || !L->ssm_a) continue;
        for(int h = 0; h < c->n_v_heads; h++){
            if(!(L->ssm_a[h] < 0.0f)){
                printf("  FAIL  layer %d ssm_a[%d] = %g, expected < 0 (holds -exp(A_log))\n",
                       i, h, L->ssm_a[h]);
                fails++; i = c->n_layer_all; break;
            }
        }
    }

    /* --- placement plan --- */
    int64_t vram = 8151LL << 20;              /* the 5070's 8151 MiB, per nvidia-smi */
    const char *vs = getenv("QWEN_VRAM_MB");
    if(vs) vram = (int64_t)atoll(vs) << 20;
    /* leave the driver's own context out of the budget */
    vram -= 400LL << 20;

    Q35Plan P;
    q35_plan(&P, &M, n_ctx, Q35_KV_Q8_0, vram, q35_ram_avail());
    q35_plan_print(&P, &M, stdout);

    Q35Bytes B; q35_bytes(&M, &B);
    ck(B.total > 0, "total bytes > 0", B.total, 1);

    printf("%s  (%d checks failed)\n", fails ? "FAILED" : "PASS", fails);
    q35_close(&M);
    return fails ? 1 : 0;
}
