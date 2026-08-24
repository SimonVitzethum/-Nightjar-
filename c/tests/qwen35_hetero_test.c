/* qwen35_hetero_test.c — every layer kind, GPU against the CPU reference.
 *
 * The gemv kernels are already proven elsewhere; what this catches is the wiring around them,
 * which on this architecture is where the silent failures live: a transposed conv1d, an
 * rmsnorm where an l2norm belongs, a tiled-vs-interleaved head map, a rope that rotates the
 * wrong 64 of 256 dims. None of them crash. All of them still produce fluent text.
 *
 * Both layer kinds are run TWICE on purpose. The first token exercises the arithmetic with a
 * zero state; the second is the only thing that can catch a wrong conv ring or a delta rule
 * that writes S back to the wrong place, because both are invisible until something has to
 * be carried forward.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../qwen35_hetero.h"

static int g_fail = 0;
static double score(const float *a, const float *b, int n){
    double d = 0, r = 0;
    for(int i = 0; i < n; i++){ const double e = (double)a[i]-b[i]; d += e*e; r += (double)b[i]*b[i]; }
    if(r == 0) return d == 0 ? 0 : INFINITY;
    return sqrt(d/r);
}
static void check(const char *what, const float *a, const float *b, int n, double tol){
    const double s = score(a, b, n);
    int bad = !(s <= tol);
    for(int i = 0; i < n && !bad; i++) if(isnan(a[i]) || isinf(a[i])) bad = 1;
    printf("  %-30s rel %.3e  (tol %.0e)  %s\n", what, s, tol, bad ? "FAIL" : "ok");
    if(bad) g_fail++;
}
static void frand(float *v, int n, unsigned seed){
    unsigned s = seed;
    for(int i = 0; i < n; i++){ s = s*1664525u + 1013904223u; v[i] = ((int)(s>>8)%2000 - 1000)/3000.0f; }
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : "/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf";
    const int n_ctx = argc > 2 ? atoi(argv[2]) : 512;
    int kv_fmt = Q35_KV_Q8_0;
    for(int i = 1; i < argc; i++) if(!strcmp(argv[i], "--kv-f16")) kv_fmt = Q35_KV_F16;

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    const Q35Cfg *c = &M.c;

    Q35State R;
    /* an explicit budget: the default sizes the RAM tier from n_ctx, and at a few hundred
     * tokens that is less than the one open chunk per attention layer the tier needs */
    if(!q35_state_init_ex(&R, &M, n_ctx, kv_fmt, 256<<20, NULL, 256)){
        printf("state init failed\n"); return 1; }
    q35_state_reset(&R);

    Q35Cu G;
    if(!q35cu_model_init(&G, &M, n_ctx, R.kv.fmt)){ printf("cuda init failed\n"); return 77; }
    q35cu_align_win(&G, R.kv.chunk);
    q35cu_report(&G, stdout);
    printf("  gdn head map: %s   kv: %s\n\n", g_q35_gdn_tile ? "TILE" : "INTERLEAVE",
           q35_kv_fmt_name(kv_fmt));
    /* The K vectors themselves, before anything quantizes them: this separates "the rope
     * disagrees" from "the quantizer disagrees". */

    float *xn  = (float*)malloc(4*c->d_model);
    float *cpu = (float*)malloc(4*c->d_model);
    float *gpu = (float*)malloc(4*c->d_model);

    /* ---- gdn ---- */
    int ig = -1, ia = -1;
    for(int i = 0; i < c->n_layer; i++){
        if(ig < 0 && c->kind[i] != Q35_LAYER_ATTN) ig = i;
        if(ia < 0 && c->kind[i] == Q35_LAYER_ATTN) ia = i;
    }
    printf("== gated delta net (layer %d) ==\n", ig);
    q35_state_reset(&R); q35cu_state_reset(&G);
    for(int step = 0; step < 2; step++){
        frand(xn, c->d_model, 42u + step*17u);
        q35_gdn(&R, &M.L[ig], ig, xn, cpu);
        q35cu_h2d(G.xn, xn, 4*c->d_model);
        q35cu_gdn_layer(&G, ig, R.gdn_slot[ig]);
        q35cu_sync();
        q35cu_d2h(gpu, G.t, 4*c->d_model);
        char lbl[64]; snprintf(lbl, sizeof lbl, "gdn out, token %d", step);
        check(lbl, gpu, cpu, c->d_model, 5e-4);
    }
    {   /* the recurrent state itself, not just what came out of it */
        const int nv = c->n_v_heads, dk = c->d_state, dv = c->d_head_v;
        const size_t n = (size_t)nv*dk*dv;
        float *sg = (float*)malloc(n*4);
        q35cu_d2h(sg, G.S + (size_t)R.gdn_slot[ig]*n, n*4);
        check("gdn recurrent state S", sg, R.S + (size_t)R.gdn_slot[ig]*n, (int)n, 5e-4);
        free(sg);
    }

    /* ---- attention ---- */
    printf("\n== full attention (layer %d) ==\n", ia);
    q35_state_reset(&R); q35cu_state_reset(&G);
    for(int pos = 0; pos < 3; pos++){
        frand(xn, c->d_model, 91u + pos*23u);
        q35_attn(&R, &M.L[ia], ia, xn, pos, cpu);
        q35cu_h2d(G.xn, xn, 4*c->d_model);
        q35cu_attn_layer(&G, &R, ia, R.attn_slot[ia], pos);
        q35cu_sync();
        q35cu_d2h(gpu, G.t, 4*c->d_model);
        char lbl[64]; snprintf(lbl, sizeof lbl, "attn out, pos %d", pos);
        check(lbl, gpu, cpu, c->d_model, 5e-4);
    }

    /* ---- the split-context merge ----
     * Force the window shut so every token is attended on the CPU half and merged back.
     * The result must be the same attention, because the flash combine is exact. */
    printf("\n== split context (device window forced to 0) ==\n");
    {
        const int save = G.win;
        G.win = 0;
        q35_state_reset(&R); q35cu_state_reset(&G);
        for(int pos = 0; pos < 3; pos++){
            frand(xn, c->d_model, 91u + pos*23u);
            q35_attn(&R, &M.L[ia], ia, xn, pos, cpu);
            q35cu_h2d(G.xn, xn, 4*c->d_model);
            q35cu_attn_layer(&G, &R, ia, R.attn_slot[ia], pos);
            q35cu_sync();
            q35cu_d2h(gpu, G.t, 4*c->d_model);
            char lbl[64]; snprintf(lbl, sizeof lbl, "host-half attn, pos %d", pos);
            check(lbl, gpu, cpu, c->d_model, 5e-4);
        }
        G.win = save;
    }

    printf("\n%s  (%d checks failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    free(xn); free(cpu); free(gpu);
    q35cu_model_free(&G);
    q35_state_free(&R);
    q35_close(&M);
    return g_fail ? 1 : 0;
}
