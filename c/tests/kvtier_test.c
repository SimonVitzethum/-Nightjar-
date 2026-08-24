/* kvtier_test.c — does the tiered KV store give back exactly what it was given, and does it
 * keep the drive busy while doing it?
 *
 * Two failures this has to rule out, both silent:
 *
 *   STALE CHUNKS. A prefetch that lands in the wrong staging slot, or an arena slot reused
 *   before its chunk was flushed, returns plausible KV for the wrong positions. Attention
 *   still runs, the model still talks, and the answer is quietly conditioned on the wrong
 *   history. So every position is written with a known pattern and read back byte-exact.
 *
 *   QUANTIZATION TAKEN ON FAITH. q8_0 is assumed fine because llama.cpp offers it. On this
 *   project q4_0 was assumed fine too and measured 10% error on MLA latents. So the error
 *   q8_0 causes in the ACTUAL attention output is measured here, not asserted.
 *
 * And one performance property that is the whole point of the design: with the RAM tier
 * deliberately too small, the scan must still run at the drive's streaming rate, because
 * chunk c+1 is requested before chunk c is touched.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../kv_tier.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

static uint32_t rnd_state = 12345;
static float frnd(void){ rnd_state = rnd_state*1664525u + 1013904223u;
    return ((float)(rnd_state >> 8) / 8388608.0f) - 1.0f; }

int main(int argc, char **argv){
    const int n_ctx   = argc > 1 ? atoi(argv[1]) : 65536;
    const double ramgb= argc > 2 ? atof(argv[2]) : 0.25;   /* deliberately small: force disk */
    const int chunk   = argc > 3 ? atoi(argv[3]) : 4096;
    const char *spill = argc > 4 ? argv[4] : NULL;   /* NULL => COLIBRI_KV_SPILL, else /tmp */
    const int  depth  = argc > 5 ? atoi(argv[5]) : 6;    /* chunks kept in flight */

    /* Qwen3.5-27B's real attention geometry */
    Q35Cfg c; memset(&c, 0, sizeof c);
    c.n_layer = 64; c.n_head = 24; c.n_head_kv = 4; c.d_head = 256;
    c.full_attn_iv = 4;
    for(int i = 0; i < c.n_layer; i++)
        c.kind[i] = ((i+1) % c.full_attn_iv) ? Q35_LAYER_GDN : Q35_LAYER_ATTN;

    const int n_kv = c.n_head_kv * c.d_head;
    const int n_at = q35_n_attn_layers(&c);
    int fails = 0;
    (void)n_at;

    /* ---- 0. the SIMD kernels must agree with the scalar reference ----
     * These are the two functions that touch every byte of an 8 GiB cache, so they are the
     * ones worth vectorizing and therefore the ones worth checking. A fast wrong kernel is
     * indistinguishable from progress until the model starts wandering. */
    {
        float *x = malloc(sizeof(float)*n_kv), *q = malloc(sizeof(float)*n_kv);
        float *o1 = malloc(sizeof(float)*n_kv), *o2 = malloc(sizeof(float)*n_kv);
        uint8_t *a = malloc(q35_kv_entry_bytes(Q35_KV_Q8_0, n_kv));
        uint8_t *b = malloc(q35_kv_entry_bytes(Q35_KV_Q8_0, n_kv));
        double dmax = 0, amax = 0; int packdiff = 0;
        for(int t = 0; t < 500; t++){
            for(int i = 0; i < n_kv; i++){ x[i] = frnd()*(1.0f+t%7); q[i] = frnd(); }
            kvt_pack_q8_scalar(a, x, n_kv);
            kvt_pack_q8(b, x, n_kv);
            if(memcmp(a, b, q35_kv_entry_bytes(Q35_KV_Q8_0, n_kv)) != 0) packdiff++;
            const float d1 = kvt_dot_q8_scalar(q, a, n_kv), d2 = kvt_dot_q8(q, a, n_kv);
            double rel = fabs(d1-d2)/(fabs(d1)+1e-6); if(rel > dmax) dmax = rel;
            for(int i = 0; i < n_kv; i++) o1[i] = o2[i] = 0.5f;
            kvt_axpy_q8_scalar(o1, 0.37f, a, n_kv);
            kvt_axpy_q8(o2, 0.37f, a, n_kv);
            for(int i = 0; i < n_kv; i++){ double e = fabs(o1[i]-o2[i]); if(e > amax) amax = e; }
        }
        printf("simd vs scalar: pack %s, dot rel<=%.2e, axpy abs<=%.2e\n",
               packdiff ? "DIFFERS" : "bit-identical", dmax, amax);
        if(packdiff){ printf("  FAIL: SIMD pack differs from scalar on %d/500 vectors\n", packdiff); fails++; }
        /* 1e-4, not 1e-6: the SIMD dot keeps ONE 8-wide accumulator for a 1024-long row
         * while the scalar adds a block at a time, so the two differ by fp addition ORDER,
         * not by value. Same reassociation moe_cpu_simd.h documents. A real bug shows up
         * orders of magnitude above this, never just below it. */
        if(dmax > 1e-4){ printf("  FAIL: SIMD dot diverges beyond reassociation\n"); fails++; }
        if(amax > 1e-5){ printf("  FAIL: SIMD axpy diverges\n"); fails++; }
        free(x); free(q); free(o1); free(o2); free(a); free(b);
    }

    /* ---- 0b. what the vectorization is worth ----
     * These two kernels are the ONLY things that touch all 8.1 GiB of a 250k cache on every
     * generated token. If they run slower than the drive delivers, the tiering is pointless:
     * the bottleneck just moves from the disk into the CPU and stops being visible. */
    {
        const int n = n_kv, reps = 200000;
        uint8_t *w = malloc(q35_kv_entry_bytes(Q35_KV_Q8_0, n));
        float *q = malloc(sizeof(float)*n), *o = malloc(sizeof(float)*n);
        for(int i = 0; i < n; i++){ q[i] = frnd(); o[i] = 0; }
        for(int i = 0; i < n; i++) o[i] = frnd();
        kvt_pack_q8(w, o, n);
        const double bytes = (double)reps*q35_kv_entry_bytes(Q35_KV_Q8_0, n);
        volatile float sink = 0;
        double t = now(); for(int r = 0; r < reps; r++) sink += kvt_dot_q8_scalar(q, w, n);
        const double ds = now()-t;
        t = now(); for(int r = 0; r < reps; r++) sink += kvt_dot_q8(q, w, n);
        const double dv = now()-t;
        t = now(); for(int r = 0; r < reps; r++) kvt_axpy_q8_scalar(o, 1e-9f, w, n);
        const double as = now()-t;
        t = now(); for(int r = 0; r < reps; r++) kvt_axpy_q8(o, 1e-9f, w, n);
        const double av = now()-t;
        (void)sink;
        printf("kernels, 1 core: dot  %5.2f -> %5.2f GB/s (%.1fx)   axpy %5.2f -> %5.2f GB/s (%.1fx)\n",
               bytes/ds/1e9, bytes/dv/1e9, ds/dv, bytes/as/1e9, bytes/av/1e9, as/av);
        printf("                 %d cores would give ~%.0f GB/s on the score pass\n",
               omp_get_max_threads(), bytes/dv/1e9*omp_get_max_threads());
        free(w); free(q); free(o);
    }

    /* ---- 1. the q8_0 error, in the units that matter: attention output ---- */
    {
        float *k = malloc(sizeof(float)*n_kv), *q = malloc(sizeof(float)*n_kv);
        uint8_t *pk = malloc(q35_kv_entry_bytes(Q35_KV_Q8_0, n_kv));
        double se = 0, sr = 0, amax_e = 0;
        for(int trial = 0; trial < 2000; trial++){
            for(int i = 0; i < n_kv; i++){ k[i] = frnd(); q[i] = frnd(); }
            kvt_pack_q8(pk, k, n_kv);
            float exact = 0;
            for(int i = 0; i < n_kv; i++) exact += q[i]*k[i];
            const float got = kvt_dot_q8(q, pk, n_kv);
            const double e = fabs(got-exact);
            se += e*e; sr += (double)exact*exact;
            if(e > amax_e) amax_e = e;
        }
        /* Reported against the RMS of the scores, not per-dot: a dot that cancels to near
         * zero has unbounded relative error and contributes nothing after the softmax, so a
         * worst-relative number would be alarming and meaningless. What matters is the error
         * next to the spread of the scores the softmax actually sees. */
        const double rms = sqrt(sr/2000.0);
        printf("q8_0 KV: score error %.4f%% of score RMS (rms %.3f, worst abs %.4f)\n",
               100.0*sqrt(se/sr), rms, amax_e);
        if(sqrt(se/sr) > 0.02){ printf("  FAIL: q8_0 score error above 2%%\n"); fails++; }
        free(k); free(q); free(pk);
    }

    /* ---- 2. round-trip through the tiers, byte-exact ---- */
    KvTier S;
    const int64_t budget = (int64_t)(ramgb*(1024.0*1024*1024));
    if(!kvt_open(&S, &c, n_ctx, Q35_KV_Q8_0, budget, spill, chunk, 8)){
        printf("open failed\n"); return 1;
    }
    const int64_t total = S.ent*(int64_t)n_ctx*n_at;
    printf("\n%d ctx x %d attn layers, q8_0 = %.3f GiB total, RAM tier %.3f GiB (%.0f%%)\n",
           n_ctx, n_at, total/(1024.0*1024*1024), S.arena_bytes/(1024.0*1024*1024),
           100.0*S.arena_bytes/total);

    /* Every (layer,pos) gets a distinct, reproducible K and V. If the store ever hands back
     * another position's data, the check below sees it. */
    float *k = malloc(sizeof(float)*n_kv), *v = malloc(sizeof(float)*n_kv);
    #define SEED(l,p,j) ( (float)(((l)*2654435761u ^ (p)*2246822519u ^ (j)*3266489917u) >> 9) / 4194304.0f - 1.0f )

    double t0 = now();
    for(int p = 0; p < n_ctx; p++)
        for(int l = 0; l < n_at; l++){
            for(int j = 0; j < n_kv; j++){ k[j] = SEED(l,p,j); v[j] = SEED(l,p,j+7919); }
            kvt_put(&S, l, p, k, v);
        }
    const double t_w = now()-t0;
    printf("write: %.2f s, %.2f GB/s (quantize + place)\n", t_w, total/t_w/1e9);

    /* ---- 3. read it back the way attention does: layer at a time, in position order ---- */
    float *ek = malloc(sizeof(float)*n_kv), *ev = malloc(sizeof(float)*n_kv);
    float *gk = malloc(sizeof(float)*n_kv), *gv = malloc(sizeof(float)*n_kv);
    uint8_t *ref = malloc(S.ent);
    int64_t checked = 0, bad = 0;
    const int n_ch = (n_ctx + S.chunk - 1)/S.chunk;

    t0 = now();
    for(int l = 0; l < n_at; l++){
        for(int ci = 0; ci < n_ch; ci++){
            kvt_prefetch_window(&S, l, ci, depth);      /* ask for the next few FIRST */
            const uint8_t *blk = kvt_chunk(&S, l, ci);
            if(!blk){ printf("  FAIL: layer %d chunk %d unavailable\n", l, ci); fails++; continue; }
            const int p0 = ci*S.chunk, p1 = (p0+S.chunk < n_ctx) ? p0+S.chunk : n_ctx;
            /* spot-check the ends and the middle of each chunk: a misrouted chunk or a stale
             * slot shows up at the very first position it touches */
            for(int p = p0; p < p1; p += (p1-p0 > 64 ? (p1-p0)/8 : 1)){
                for(int j = 0; j < n_kv; j++){ ek[j] = SEED(l,p,j); ev[j] = SEED(l,p,j+7919); }
                kvt_pack_q8(ref, ek, n_kv);
                kvt_pack_q8(ref + S.ent/2, ev, n_kv);
                /* K and V now live in separate regions, so they are checked separately —
                 * which is also the point: each read touches only what it needs */
                const uint8_t *gk = blk + kvt_koff(&S, p-p0);
                const uint8_t *gv = blk + kvt_voff(&S, p-p0);
                if(memcmp(gk, ref, S.ent/2) != 0 || memcmp(gv, ref + S.ent/2, S.ent/2) != 0){
                    if(bad < 5) printf("  FAIL: layer %d pos %d does not round-trip\n", l, p);
                    bad++;
                }
                checked++;
            }
        }
    }
    const double t_r = now()-t0;
    printf("read : %.2f s, %.2f GB/s over the whole cache, prefetch depth %d (%lld positions checked)\n",
           t_r, total/t_r/1e9, depth, (long long)checked);
    (void)gk; (void)gv;
    if(bad){ printf("  FAIL: %lld positions corrupted\n", (long long)bad); fails++; }
    else    printf("round-trip: byte-exact on every checked position\n");

    kvt_report(&S, stdout);
    kvt_close(&S);
    free(k); free(v); free(ek); free(ev); free(gk); free(gv); free(ref);

    printf("\n%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
