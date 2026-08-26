/* qwen35_fwd_test.c — does the CPU forward pass agree with llama.cpp?
 *
 * This is the only test that can catch the failures this architecture specializes in. A
 * wrong q/gate split, a tiled-instead-of-interleaved GDN head map, an rmsnorm where an
 * l2norm belongs — none of them crash, none of them produce NaN, and all of them produce
 * text that reads fine for a sentence and then wanders. The only way to know is to put the
 * same token ids through both engines and compare the distribution.
 *
 * Usage:  qwen35_fwd_test <model.gguf> <tok0,tok1,...>  [--tile]
 * Prints the top-10 of the next-token distribution. Diff against llama-server's /completion
 * with n_probs=10 on the identical id list.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../models.h"
#include "../qwen35_cpu.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

typedef struct { float p; int id; } Cand;
static int cmp_cand(const void *a, const void *b){
    float x = ((const Cand*)a)->p, y = ((const Cand*)b)->p;
    return x < y ? 1 : (x > y ? -1 : 0);
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : nj_model_path("Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf");
    const char *idlist = argc > 2 ? argv[2] : "9707";     /* "Hello" */
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--tile")) q35_set_gdn_head_map(1);
        if(!strcmp(argv[i], "--dbg"))  g_q35_dbg = 1;
    }

    int toks[4096], nt = 0;
    { char *s = strdup(idlist), *p = strtok(s, ",");
      while(p && nt < 4096){ toks[nt++] = atoi(p); p = strtok(NULL, ","); }
      free(s); }
    if(nt == 0){ printf("no tokens\n"); return 1; }

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    /* Pull the model into RAM and PROVE it is there before timing anything. Without this the
     * first tokens measure the NVMe, not the engine. */
    Q35Resident RES;
    q35_reside_anon(&M, &RES, Q35_RES_ALL);   /* CPU-only run: nothing may come from disk */
    q35_reside_report(&RES, stdout);

    Q35State R;
    q35_state_init(&R, &M, nt + 8);
    q35_state_reset(&R);

    printf("gdn head map: %s\n", g_q35_gdn_tile ? "TILE (h %% n_k)" : "INTERLEAVE (h / (n_v/n_k))");
    printf("%d tokens:", nt);
    for(int i = 0; i < nt; i++) printf(" %d", toks[i]);
    printf("\n");

    float *logits = (float*)malloc(sizeof(float)*M.c.vocab);
    double t0 = now();
    for(int i = 0; i < nt; i++){
        int last = (i == nt-1);
        q35_forward(&R, toks[i], i, last ? logits : NULL);
        fprintf(stderr, "\r  pos %d/%d  %.1fs   ", i+1, nt, now()-t0);
    }
    fprintf(stderr, "\n");
    const double tot = now()-t0;
    printf("forward: %.2f s for %d tokens (%.2f s/token)\n", tot, nt, tot/nt);
    {   /* GiB moved by each stage, per token, so the rate is a bandwidth and not just a time */
        const double ffn_gib = 9.650, gdn_gib = 3.222, att_gib = 0.889, out_gib = 0.971;
        printf("\n  stage        s/token   share   weights      -> effective\n");
        printf("  ffn  (CPU)   %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_ffn/nt,  100*g_t_ffn/tot,  ffn_gib, ffn_gib*1.074/(g_t_ffn/nt));
        printf("  gdn  (CPU)   %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_gdn/nt,  100*g_t_gdn/tot,  gdn_gib, gdn_gib*1.074/(g_t_gdn/nt));
        printf("  attn (CPU)   %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_attn/nt, 100*g_t_attn/tot, att_gib, att_gib*1.074/(g_t_attn/nt));
        printf("  output       %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_out/nt,  100*g_t_out/tot,  out_gib, out_gib*1.074/(g_t_out/nt));
        printf("  embed        %7.3f  %5.1f%%\n", g_t_embd/nt, 100*g_t_embd/tot);
    }

    /* softmax the whole vocab so the numbers are directly comparable to llama-server's
     * n_probs output, which is post-softmax at temperature 1 */
    int V = M.c.vocab;
    float mx = -INFINITY;
    for(int i = 0; i < V; i++) if(logits[i] > mx) mx = logits[i];
    double sum = 0;
    for(int i = 0; i < V; i++) sum += exp((double)(logits[i]-mx));

    Cand *c = (Cand*)malloc(sizeof(Cand)*V);
    for(int i = 0; i < V; i++){ c[i].id = i; c[i].p = (float)(exp((double)(logits[i]-mx))/sum); }
    qsort(c, V, sizeof(Cand), cmp_cand);

    int nan = 0;
    for(int i = 0; i < V; i++) if(isnan(logits[i]) || isinf(logits[i])) nan++;
    if(nan) printf("WARNING: %d non-finite logits\n", nan);

    printf("\n  rank  token_id      prob     logit\n");
    for(int i = 0; i < 10; i++)
        printf("  %4d  %8d  %8.5f  %8.4f\n", i, c[i].id, c[i].p, logits[c[i].id]);

    free(c); free(logits);
    q35_state_free(&R);
    q35_close(&M);
    return 0;
}
