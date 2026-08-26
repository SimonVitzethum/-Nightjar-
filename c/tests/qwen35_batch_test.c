/* qwen35_batch_test.c — batched prefill must agree with sequential decode, token for token.
 *
 * A batch that leaks state across positions does not crash. It produces a model that is
 * subtly worse — attention that sees one token too many, a GDN state advanced out of order,
 * a conv ring shared between neighbours. All of that reads as "the model is a bit off", which
 * is indistinguishable from quantization noise until it is measured against the thing it is
 * supposed to reproduce.
 *
 * So: run the same tokens twice through the same weights — once one at a time, once in a
 * batch — and require the final logits to match. They will not match bitwise (a batched gemm
 * dequantizes the row and dots in fp32, decode dots the quantized row directly), so the bar
 * is agreement to fp32 reassociation, and identical argmax.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../models.h"
#include "../qwen35_batch.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : nj_model_path("Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf");
    const int  n_tok = argc > 2 ? atoi(argv[2]) : 24;
    const int  S_max = argc > 3 ? atoi(argv[3]) : 8;

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    Q35Resident RES;
    if(!q35_reside_anon(&M, &RES, Q35_RES_ALL)) return 1;
    q35_reside_report(&RES, stdout);

    /* a deterministic pseudo-prompt: real ids, no tokenizer dependency */
    int *toks = malloc(sizeof(int)*n_tok);
    uint32_t r = 7;
    for(int i = 0; i < n_tok; i++){ r = r*1664525u+1013904223u; toks[i] = 1000 + (r >> 18) % 40000; }

    const int V = M.c.vocab;
    float *lg_seq = malloc(sizeof(float)*V), *lg_bat = malloc(sizeof(float)*V);

    /* ---- 1. sequential, one token at a time ---- */
    Q35State R1; q35_state_init(&R1, &M, n_tok + 8); q35_state_reset(&R1);
    double t0 = now();
    for(int i = 0; i < n_tok; i++)
        q35_forward(&R1, toks[i], i, i == n_tok-1 ? lg_seq : NULL);
    const double t_seq = now()-t0;

    /* ---- 2. batched ---- */
    Q35State R2; q35_state_init(&R2, &M, n_tok + 8); q35_state_reset(&R2);
    Q35Batch B;  q35_batch_init(&B, &R2, S_max);
    t0 = now();
    q35_prefill(&B, toks, n_tok, 0, lg_bat);
    const double t_bat = now()-t0;

    /* ---- compare ---- */
    double worst = 0; int am_s = 0, am_b = 0;
    for(int i = 0; i < V; i++){
        if(lg_seq[i] > lg_seq[am_s]) am_s = i;
        if(lg_bat[i] > lg_bat[am_b]) am_b = i;
    }
    /* relative to the logit SPREAD, not to each value: logits near zero have unbounded
     * relative error and no influence on the softmax */
    double lo = 1e30, hi = -1e30;
    for(int i = 0; i < V; i++){ if(lg_seq[i]<lo) lo=lg_seq[i]; if(lg_seq[i]>hi) hi=lg_seq[i]; }
    const double span = hi - lo;
    for(int i = 0; i < V; i++){
        const double d = fabs(lg_seq[i]-lg_bat[i])/span;
        if(d > worst) worst = d;
    }

    int fails = 0;
    printf("\n%d tokens, batch %d\n", n_tok, S_max);
    printf("  sequential : %6.2f s  (%.2f tok/s)\n", t_seq, n_tok/t_seq);
    printf("  batched    : %6.2f s  (%.2f tok/s)   speedup %.2fx\n", t_bat, n_tok/t_bat, t_seq/t_bat);
    printf("  argmax     : seq %d, batch %d  %s\n", am_s, am_b, am_s == am_b ? "MATCH" : "DIFFER");
    printf("  worst logit deviation: %.3e of the %.1f-wide logit span\n", worst, span);
    if(am_s != am_b){ printf("  FAIL: batched prefill changes the prediction\n"); fails++; }
    if(worst > 1e-3){ printf("  FAIL: logits diverge beyond fp reassociation\n"); fails++; }

    printf("\n%s\n", fails ? "FAILED" : "PASS");
    q35_batch_free(&B); q35_state_free(&R1); q35_state_free(&R2);
    free(toks); free(lg_seq); free(lg_bat); q35_close(&M);
    return fails ? 1 : 0;
}
