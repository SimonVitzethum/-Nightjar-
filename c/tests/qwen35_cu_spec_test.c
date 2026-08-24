/* qwen35_cu_spec_test.c — the batch must reproduce decode, and speculation must reproduce greedy.
 *
 * Two gates, and the second is the one that matters:
 *
 *  1. A batch of S tokens must land on the same logits as S sequential single-token passes.
 *     If it does not, the batch dimension is leaking between positions.
 *  2. GREEDY SPECULATION MUST EMIT EXACTLY THE SAME TOKEN SEQUENCE AS GREEDY DECODE.
 *     Speculative decoding is only an optimization if it is output-identical; a draft that
 *     is accepted when it should not be gives fluent text that is not the model's. Since the
 *     acceptance test here is argmax equality, any divergence is a bug in the rollback — and
 *     the rollback is the part the gated delta net makes hard, because its state update is
 *     not invertible.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../qwen35_cu_spec.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
static int g_fail = 0;
static double score(const float *a, const float *b, int n){
    double d = 0, r = 0;
    for(int i = 0; i < n; i++){ const double e = (double)a[i]-b[i]; d += e*e; r += (double)b[i]*b[i]; }
    return r == 0 ? (d == 0 ? 0 : INFINITY) : sqrt(d/r);
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : "/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf";
    const int n_gen = argc > 2 ? atoi(argv[2]) : 24;
    const int n_ctx = 4096;

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    const Q35Cfg *c = &M.c;

    Q35State R;
    const int64_t kvb = q35_kv_bytes_per_token(c, Q35_KV_Q8_0)*(int64_t)n_ctx + (32<<20);
    if(!q35_state_init_ex(&R, &M, n_ctx, Q35_KV_Q8_0, kvb, getenv("QWEN_KV_SPILL"), 4096)){
        printf("kv init failed\n"); return 1; }

    Q35Cu G;
    if(!q35cu_model_init(&G, &M, n_ctx, R.kv.fmt)){ printf("cuda init failed\n"); return 77; }
    q35cu_align_win(&G, R.kv.chunk);
    q35cu_mark_resident_elsewhere(&G);

    Q35Resident RES;
    q35_reside_anon(&M, &RES, Q35_RES_FFN);
    q35cu_pin_weights(&G, &M);
    printf("  weights in RAM %.2f GiB\n", RES.resident/1073741824.0);
    q35cu_report(&G, stdout);

    Q35CuBatch Z;
    if(!q35cu_batch_init(&Z, &G, &R, 4)){ printf("batch init failed\n"); return 1; }
    printf("    batch scratch %.3f GiB extra VRAM\n\n", Z.vram/1073741824.0);

    const int prompt[] = { 760, 6511, 314, 9338, 369 };     /* "The capital of France is" */
    const int np = 5;
    float *l1 = (float*)malloc(sizeof(float)*c->vocab);
    float *l2 = (float*)malloc(sizeof(float)*2*(size_t)c->vocab);

    /* ---- gate 1: batch == sequential ---- */
    printf("== batch of 2 vs two sequential passes ==\n");
    q35_state_reset(&R); q35cu_state_reset(&G);
    for(int i = 0; i < np-1; i++) q35_forward_cu(&G, &R, prompt[i], i, NULL);
    q35_forward_cu(&G, &R, prompt[np-1], np-1, l1);
    const int nxt = q35_argmax(l1, c->vocab);
    float *seq = (float*)malloc(sizeof(float)*c->vocab);
    q35_forward_cu(&G, &R, nxt, np, seq);

    q35_state_reset(&R); q35cu_state_reset(&G);
    for(int i = 0; i < np-1; i++) q35_forward_cu(&G, &R, prompt[i], i, NULL);
    { const int pair[2] = { prompt[np-1], nxt };
      q35_forward_cu_batch(&Z, pair, 2, np-1, l2, 1); }
    {   const double s0 = score(l2, l1, c->vocab);
        const double s1 = score(l2 + c->vocab, seq, c->vocab);
        const int a0 = q35_argmax(l2, c->vocab), a1 = q35_argmax(l2 + c->vocab, c->vocab);
        printf("  pos 0  rel %.3e  argmax %d vs %d  %s\n", s0, a0, nxt, a0 == nxt ? "MATCH" : "DIFFER");
        printf("  pos 1  rel %.3e  argmax %d vs %d  %s\n", s1, a1, q35_argmax(seq, c->vocab),
               a1 == q35_argmax(seq, c->vocab) ? "MATCH" : "DIFFER");
        if(s0 > 1e-3 || s1 > 1e-3 || a0 != nxt || a1 != q35_argmax(seq, c->vocab)) g_fail++;
    }

    /* ---- gate 2: greedy speculation == greedy decode, token for token ---- */
    printf("\n== greedy decode vs greedy speculation, %d tokens ==\n", n_gen);
    int *ref = (int*)malloc(sizeof(int)*n_gen), *got = (int*)malloc(sizeof(int)*(n_gen+4));

    q35_state_reset(&R); q35cu_state_reset(&G);
    for(int i = 0; i < np; i++) q35_forward_cu(&G, &R, prompt[i], i, (i==np-1) ? l1 : NULL);
    int tok = q35_argmax(l1, c->vocab), pos = np;
    double t0 = now();                       /* DECODE only: the prefill is not what is compared */
    for(int i = 0; i < n_gen; i++){
        ref[i] = tok;
        q35_forward_cu(&G, &R, tok, pos++, l1);
        tok = q35_argmax(l1, c->vocab);
    }
    const double t_plain = now()-t0;

    Q35Mtp P;
    if(!q35_mtp_init(&P, &R)){ printf("  no MTP block in this model\n"); return 1; }
    q35_state_reset(&R); q35cu_state_reset(&G);
    for(int i = 0; i < np; i++) q35_forward_cu(&G, &R, prompt[i], i, (i==np-1) ? l1 : NULL);
    Q35SpecStats st; memset(&st, 0, sizeof st);
    t0 = now();
    const int ng = q35_spec_generate_cu(&Z, &P, q35_argmax(l1, c->vocab), np,
                                        got, n_gen, -1, &st, 0, 0, 0, NULL, NULL);
    const double t_spec = now()-t0;

    int mism = 0;
    for(int i = 0; i < n_gen && i < ng; i++) if(ref[i] != got[i]) mism++;
    printf("  produced %d of %d, %d mismatches vs greedy  %s\n",
           ng, n_gen, mism, mism ? "FAIL" : "ok");
    if(mism || ng < n_gen) g_fail++;
    printf("  rounds %lld, accepts %lld  -> acceptance %.1f%%\n",
           (long long)st.rounds, (long long)st.accepts,
           st.rounds ? 100.0*st.accepts/st.rounds : 0.0);
    printf("  draft %.3f s   verify %.3f s   rollback %.3f s\n",
           st.t_draft, st.t_verify, st.t_rollback);
    printf("  e = draft/verify-per-token = %.3f\n",
           st.rounds ? (st.t_draft/st.rounds)/(t_plain/n_gen) : 0.0);
    printf("\n  plain      %6.3f s  %d tok  -> %5.2f tok/s\n", t_plain, n_gen, n_gen/t_plain);
    printf("  speculative %6.3f s  %d tok  -> %5.2f tok/s   (%.2fx)\n",
           t_spec, ng, ng/t_spec, (n_gen/t_plain) > 0 ? (ng/t_spec)/(n_gen/t_plain) : 0.0);

    printf("\n%s  (%d checks failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    free(l1); free(l2); free(seq); free(ref); free(got);
    q35_mtp_free(&P); q35cu_batch_free(&Z); q35cu_model_free(&G);
    q35_state_free(&R); q35_close(&M);
    return g_fail ? 1 : 0;
}
