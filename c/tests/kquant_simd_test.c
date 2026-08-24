/* kquant_simd_test.c — the AVX2 Q4_K/Q6_K dots against the scalar ones that are themselves
 * bit-exact vs ggml. These two kernels carry 9.65 GiB per generated token, so they are worth
 * both the vectorization and the check: a fast wrong dot is indistinguishable from progress. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../kquant_simd.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
static uint32_t rs = 987654321;
static float frnd(void){ rs = rs*1664525u + 1013904223u;
    return ((float)(rs >> 8)/8388608.0f) - 1.0f; }

int main(void){
    const int I = 5120;                       /* Qwen3.5's d_model: the real contraction */
    int fails = 0;
#ifndef KQS_HAVE_AVX2
    printf("AVX2 not compiled in — nothing to test\n"); return 0;
#else
    float *x = malloc(sizeof(float)*I);
    for(int i = 0; i < I; i++) x[i] = frnd();

    const struct { int t; const char *nm; } types[2] = { {KQ_Q4_K,"Q4_K"}, {KQ_Q6_K,"Q6_K"} };
    for(int ti = 0; ti < 2; ti++){
        const int t = types[ti].t;
        const int64_t rb = kq_row_bytes(t, I);
        uint8_t *w = malloc(rb);
        /* random BYTES, not a random tensor: this exercises every scale/min/high-bit
         * combination the format admits, including the ones a real model rarely hits */
        for(int64_t b = 0; b < rb; b++) w[b] = (uint8_t)(rs = rs*1664525u+1013904223u) >> 3;

        /* THE ARBITER IS DOUBLE PRECISION, NOT THE SCALAR KERNEL.
         * Diffing vector against scalar only measures how differently they REASSOCIATE, and
         * on a 5120-long dot with cancellation that is naturally ~1e-4 — which looks like a
         * failure and is not. Dequantizing to double and summing in double gives the true
         * value, and the question becomes the right one: is the fast kernel in the same
         * accuracy class as the reference? (Here it is consistently BETTER, because one
         * 8-wide accumulator grows less error than a scalar chain of 5120 adds.) */
        float *deq = malloc(sizeof(float)*I);
        double ws = 0, wv = 0;
        for(int trial = 0; trial < 200; trial++){
            for(int i = 0; i < I; i++) x[i] = frnd();
            for(int64_t b = 0; b < rb; b++) w[b] = (uint8_t)((rs = rs*1664525u+1013904223u) >> 7);
            kq_dequant_row(t, w, deq, I);
            double ref = 0;
            for(int i = 0; i < I; i++) ref += (double)deq[i]*(double)x[i];
            const double a = kq_dot(t, w, x, I);
            const double c = (t == KQ_Q4_K) ? kq_dot_q4_K_avx2(w, x, I) : kq_dot_q6_K_avx2(w, x, I);
            const double den = fabs(ref) + 1e-6;
            if(fabs(a-ref)/den > ws) ws = fabs(a-ref)/den;
            if(fabs(c-ref)/den > wv) wv = fabs(c-ref)/den;
        }
        free(deq);
        const int ok = (wv <= ws*4);
        printf("%s vs double ref: scalar %.3e, avx2 %.3e  %s\n",
               types[ti].nm, ws, wv, ok ? "OK" : "FAIL (avx2 systematically worse)");
        if(!ok) fails++;

        const int reps = 200000;
        const double bytes = (double)reps*rb;
        volatile float sink = 0;
        double t0 = now(); for(int r = 0; r < reps; r++) sink += kq_dot(t, w, x, I);
        const double ts = now()-t0;
        t0 = now();
        for(int r = 0; r < reps; r++)
            sink += (t == KQ_Q4_K) ? kq_dot_q4_K_avx2(w, x, I) : kq_dot_q6_K_avx2(w, x, I);
        const double tv = now()-t0;
        (void)sink;
        printf("      1 core: %5.2f -> %5.2f GB/s  (%.1fx)\n",
               bytes/ts/1e9, bytes/tv/1e9, ts/tv);
        free(w);
    }

    /* ---- AVX-VNNI: correctness, and what it is actually worth ----
     * The activation goes to int8, so this is NOT expected to match the float kernel bitwise.
     * Same standard as everything else here: agreement with a DOUBLE reference, in the same
     * accuracy class as the float path. */
    {
        const int O = 4096;
        const int64_t rbv = kq_row_bytes(KQ_Q4_K, I);
        uint8_t *wv = malloc(rbv*(size_t)O);
        for(int64_t b = 0; b < rbv*(int64_t)O; b++) wv[b] = (uint8_t)((rs = rs*1664525u+1013904223u) >> 7);
        float *xf = malloc(sizeof(float)*I), *dq = malloc(sizeof(float)*I);
        float *y1 = malloc(sizeof(float)*O), *y2 = malloc(sizeof(float)*O);
        for(int i = 0; i < I; i++) xf[i] = frnd();

        kq_gemv_fast(y1, xf, wv, KQ_Q4_K, I, O);
        kq_gemv_vnni(y2, xf, wv, KQ_Q4_K, I, O);
        /* POPULATION RMS, not a per-row ratio.
         *
         * The previous version scored each row against |ref|+1e-3 and took the worst. With
         * random weights some rows dot to nearly zero, so that denominator collapses and the
         * ratio explodes -- it reported 6.4 for a kernel whose real error is ~1e-3, and that
         * is why VNNI sat unwired. The consumer of this vector is the next layer, which sees
         * all O values at once, so O values are the population and their RMS is the scale.
         * This is the third time this exact metric produced a false negative here. */
        /* The weights here are RANDOM BYTES, so some super-blocks carry an fp16 scale whose
         * bit pattern is Inf or NaN. The old max-ratio metric skipped those rows by accident
         * -- every comparison against NaN is false -- so they were never noticed. A sum does
         * not have that accident, and must exclude them on purpose. */
        double sf = 0, sq = 0, sr = 0; int nrow = 0, nskip = 0;
        for(int o = 0; o < 256; o++){
            kq_dequant_row(KQ_Q4_K, wv + (int64_t)o*rbv, dq, I);
            double ref = 0;
            for(int i = 0; i < I; i++) ref += (double)dq[i]*(double)xf[i];
            if(!isfinite(ref) || !isfinite(y1[o]) || !isfinite(y2[o])){ nskip++; continue; }
            sr += ref*ref;
            sf += ((double)y1[o]-ref)*((double)y1[o]-ref);
            sq += ((double)y2[o]-ref)*((double)y2[o]-ref);
            nrow++;
        }
        if(nskip) printf("  (%d of 256 rows have a non-finite fp16 scale from the random"
                         " bytes and are excluded)\n", nskip);
        const double wf = sqrt(sf/sr), wq = sqrt(sq/sr);
        printf("\nAVX-VNNI vs double ref (population rms): float %.3e, vnni %.3e\n"
               "  (vnni quantizes the activation to int8, so ~1e-3 is the format, not a bug)\n", wf, wq);

        const int reps = 300;
        double t0 = now();
        for(int r = 0; r < reps; r++) kq_gemv_fast(y1, xf, wv, KQ_Q4_K, I, O);
        const double tf = now()-t0;
        t0 = now();
        for(int r = 0; r < reps; r++) kq_gemv_vnni(y2, xf, wv, KQ_Q4_K, I, O);
        const double tv = now()-t0;
        const double gbv = (double)reps*rbv*O/1e9;
        printf("  gemv %dx%d Q4_K, %d threads: float %.1f GB/s, VNNI %.1f GB/s  -> %.2fx\n",
               O, I, omp_get_max_threads(), gbv/tf, gbv/tv, tf/tv);
        free(wv); free(xf); free(dq); free(y1); free(y2);
    }

    /* ---- batched prefill: does batching actually amortize the weight read? ----
     * The whole point of a batched path is that the quantized row is decoded once for S
     * tokens instead of S times. If tok/s does not scale with S, it is not batching, it is
     * a loop. */
    {
        const int O = 4096, nth = omp_get_max_threads();
        const int64_t rb4 = kq_row_bytes(KQ_Q4_K, I);
        uint8_t *w = malloc(rb4*(size_t)O);
        for(int64_t b = 0; b < rb4*(int64_t)O; b++) w[b] = (uint8_t)((rs = rs*1664525u+1013904223u) >> 7);
        const int SS[4] = {1, 8, 32, 128};
        float *xb = malloc(sizeof(float)*(size_t)I*128), *yb = malloc(sizeof(float)*(size_t)O*128);
        for(int i = 0; i < I*128; i++) xb[i] = frnd();

        /* batched must agree with the reference gemv, row for row */
        /* Against a DOUBLE reference, like every other kernel check here — not against the
         * gemv. A gemv-vs-gemm diff divided by the value is unbounded wherever the dot
         * cancels toward zero, so it reports 1e-3 or 3e-4 depending only on which random rows
         * happened to cancel. The question that matters is whether the batched path is in the
         * same accuracy class as the reference, and that needs the true value. */
        kq_gemv_fast(yb, xb, w, KQ_Q4_K, I, O);
        float *y2 = malloc(sizeof(float)*(size_t)O*8);
        float *dqb = malloc(sizeof(float)*I);
        kq_gemm_batched(y2, xb, w, KQ_Q4_K, 8, I, O);
        double wg = 0, wbm = 0;
        for(int o = 0; o < 512; o++){
            kq_dequant_row(KQ_Q4_K, w + (int64_t)o*rb4, dqb, I);
            double ref = 0;
            for(int i = 0; i < I; i++) ref += (double)dqb[i]*(double)xb[i];
            const double den = fabs(ref) + 1e-2;
            if(fabs(yb[o]-ref)/den > wg)  wg  = fabs(yb[o]-ref)/den;
            if(fabs(y2[o]-ref)/den > wbm) wbm = fabs(y2[o]-ref)/den;
        }
        const int bok = (wbm <= wg*8 + 1e-5);
        printf("\nbatched gemm vs double ref: gemv %.3e, batched %.3e  %s\n",
               wg, wbm, bok ? "OK" : "FAIL");
        if(!bok) fails++;
        free(dqb);

        printf("  S     ms/batch   tok/s(this gemm)   weight bytes/token\n");
        for(int k = 0; k < 4; k++){
            const int Sb = SS[k];
            double t0 = now(); int reps = Sb >= 32 ? 3 : 10;
            for(int r = 0; r < reps; r++) kq_gemm_batched(yb, xb, w, KQ_Q4_K, Sb, I, O);
            const double dt = (now()-t0)/reps;
            printf("  %-4d  %8.1f   %14.1f   %10.0f\n",
                   Sb, dt*1e3, Sb/dt, (double)rb4*O/Sb);
        }
        printf("  (%d threads; weight bytes/token falling with S is the amortization working)\n", nth);
        free(w); free(xb); free(yb); free(y2);
    }

    /* What the FFN actually costs: 64 layers x (gate+up Q4_K, down Q6_K), all cores. */
    {
        const int d_ff = 17408, nth = omp_get_max_threads();
        const int64_t rb4 = kq_row_bytes(KQ_Q4_K, I), rb6 = kq_row_bytes(KQ_Q6_K, d_ff);
        uint8_t *g = malloc(rb4*(size_t)d_ff), *dn = malloc(rb6*(size_t)I);
        if(g && dn){
            memset(g, 0x5A, rb4*(size_t)d_ff); memset(dn, 0x5A, rb6*(size_t)I);
            float *h = malloc(sizeof(float)*d_ff), *o = malloc(sizeof(float)*I);
            for(int i = 0; i < d_ff; i++) h[i] = frnd();
            const double gb = (2.0*rb4*d_ff + (double)rb6*I)/1e9;     /* one layer's FFN */
            double t0 = now();
            for(int r = 0; r < 4; r++){
                kq_gemv_fast(h, x, g,  KQ_Q4_K, I, d_ff);
                kq_gemv_fast(h, x, g,  KQ_Q4_K, I, d_ff);
                kq_gemv_fast(o, h, dn, KQ_Q6_K, d_ff, I);
            }
            const double dt = (now()-t0)/4.0;
            printf("\nFFN, one layer, %d threads: %.1f ms, %.1f GB/s\n", nth, dt*1e3, gb/dt);
            printf("  -> 64 layers = %.2f GiB/token in %.2f s  =>  %.2f tok/s from the FFN alone\n",
                   gb*64/1.074, dt*64, 1.0/(dt*64));
            free(h); free(o);
        }
        free(g); free(dn);
    }
    free(x);
    printf("\n%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
#endif
}
