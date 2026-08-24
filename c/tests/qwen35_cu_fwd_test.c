/* qwen35_cu_fwd_test.c — the whole engine, GPU+CPU, against llama.cpp.
 *
 * THE GATE: the same token ids that make the CPU-only path print 11751 ' Paris' at p ~ 0.606
 * must do it here. Every kernel in the device path has already been diffed against its CPU
 * counterpart, but a per-kernel pass does not prove the SCHEDULE is right — a layer routed to
 * the wrong device, a residual added twice, a KV slot shared between the trunk and the draft
 * head. All of those produce fluent, wrong text.
 *
 * COLIBRI_KERNEL_LOG=1 prints where each component actually ran. "CUDA didn't help" and
 * "CUDA didn't run" look identical from the outside; that has already cost 1.84x once.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../qwen35_hetero.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
typedef struct { float p; int id; } Cand;
static int cmp_cand(const void *a, const void *b){
    float x = ((const Cand*)a)->p, y = ((const Cand*)b)->p;
    return x < y ? 1 : (x > y ? -1 : 0);
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : "/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf";
    const char *idlist = argc > 2 ? argv[2] : "9707";
    int n_ctx = 0;
    g_q35_stage_timing = 1;
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--dbg")) g_q35_dbg = 1;
        if(!strcmp(argv[i], "--no-timing")) g_q35_stage_timing = 0;
        if(!strncmp(argv[i], "--ctx=", 6)) n_ctx = atoi(argv[i]+6);
    }

    int toks[65536], nt = 0;
    { char *s = strdup(idlist), *p = strtok(s, ",");
      while(p && nt < 65536){ toks[nt++] = atoi(p); p = strtok(NULL, ","); }
      free(s); }
    if(!nt){ printf("no tokens\n"); return 1; }
    if(n_ctx < nt + 8) n_ctx = nt + 8;

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }

    /* Only the FFN and the embedding table need to be in RAM — the rest is going to VRAM.
     * 10.3 GiB instead of 15.6, and the difference is KV budget. */
    


    Q35State R;
    const int64_t kvb = q35_kv_bytes_per_token(&M.c, Q35_KV_Q8_0)*(int64_t)n_ctx + (16<<20);
    const char *spill = getenv("COLIBRI_KV_SPILL");
    if(!q35_state_init_ex(&R, &M, n_ctx, Q35_KV_Q8_0, kvb, spill, 4096)){
        printf("state init failed\n"); return 1; }
    q35_state_reset(&R);

    Q35Cu G;
    const double tu = now();
    if(!q35cu_model_init(&G, &M, n_ctx, R.kv.fmt)){ printf("cuda init failed\n"); return 77; }
    q35cu_align_win(&G, R.kv.chunk);
    q35cu_mark_resident_elsewhere(&G);
    q35cu_state_reset(&G);
    printf("  upload: %.2f s\n", now()-tu);
    /* residency AFTER the upload: layers already in VRAM must not be copied into RAM too,
     * and the streamed path needs the anonymous mapping pinned once it exists */
    Q35Resident RES;
    q35_reside_anon(&M, &RES, Q35_RES_FFN);
    q35_reside_report(&RES, stdout);
    q35cu_pin_weights(&G, &M);
    q35cu_report(&G, stdout);
    printf("    stream bandwidth %.1f GB/s measured\n", q35cu_measure_bw());
    printf("  gdn head map: %s\n%d tokens:", g_q35_gdn_tile ? "TILE (h %% n_k)" : "INTERLEAVE", nt);
    for(int i = 0; i < nt && i < 16; i++) printf(" %d", toks[i]);
    printf("%s\n", nt > 16 ? " ..." : "");

    float *logits = (float*)malloc(sizeof(float)*M.c.vocab);
    const double t0 = now();
    for(int i = 0; i < nt; i++){
        q35_forward_cu(&G, &R, toks[i], i, (i == nt-1) ? logits : NULL);
        fprintf(stderr, "\r  pos %d/%d  %.1fs   ", i+1, nt, now()-t0);
    }
    fprintf(stderr, "\n");
    const double tot = now()-t0;
    printf("forward: %.2f s for %d tokens (%.3f s/token, %.2f tok/s)\n",
           tot, nt, tot/nt, nt/tot);
    {
        const double ffn_gib = 9.650, gdn_gib = 3.222, att_gib = 0.889, out_gib = 0.971;
        const double cpu_share = G.n_ffn_gpu ? (double)(M.c.n_layer - G.n_ffn_gpu)/M.c.n_layer : 1.0;
        printf("\n  stage         s/token   share   weights      -> effective\n");
        printf("  ffn  %-7s %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               G.n_ffn_gpu ? "CPU+GPU" : "CPU", g_t_ffn/nt, 100*g_t_ffn/tot,
               ffn_gib, ffn_gib*1.074/(g_t_ffn/nt));
        printf("  gdn  (GPU)    %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_gdn/nt, 100*g_t_gdn/tot, gdn_gib, gdn_gib*1.074/(g_t_gdn/nt));
        printf("  attn (GPU)    %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_attn/nt, 100*g_t_attn/tot, att_gib, att_gib*1.074/(g_t_attn/nt));
        printf("  output (GPU)  %7.3f  %5.1f%%  %5.2f GiB  -> %5.2f GB/s\n",
               g_t_out/nt, 100*g_t_out/tot, out_gib, out_gib*1.074/(g_t_out/nt));
        printf("  embed (CPU)   %7.3f  %5.1f%%\n", g_t_embd/nt, 100*g_t_embd/tot);
        printf("  cpu ffn share of layers: %.0f%%\n", 100*cpu_share);
        if(G.stream_on)
            printf("  split ffn: gpu-side wall %.3f s/tok, of which cpu-half %.3f;"
                   " %.2f GiB/token over PCIe\n",
                   G.t_gpu_side/nt, G.t_cpu_side/nt, G.stream_bytes/1073741824.0/nt);
    }

    const int V = M.c.vocab;
    float mx = -INFINITY;
    for(int i = 0; i < V; i++) if(logits[i] > mx) mx = logits[i];
    double sum = 0;
    for(int i = 0; i < V; i++) sum += exp((double)(logits[i]-mx));
    Cand *cd = (Cand*)malloc(sizeof(Cand)*V);
    for(int i = 0; i < V; i++){ cd[i].id = i; cd[i].p = (float)(exp((double)(logits[i]-mx))/sum); }
    qsort(cd, V, sizeof(Cand), cmp_cand);
    int nan = 0;
    for(int i = 0; i < V; i++) if(isnan(logits[i]) || isinf(logits[i])) nan++;
    if(nan) printf("WARNING: %d non-finite logits\n", nan);
    printf("\n  rank  token_id      prob     logit\n");
    for(int i = 0; i < 10; i++)
        printf("  %4d  %8d  %8.5f  %8.4f\n", i, cd[i].id, cd[i].p, logits[cd[i].id]);

    free(cd); free(logits);
    q35cu_model_free(&G);
    q35_state_free(&R);
    q35_close(&M);
    return 0;
}
