/* qwen35_gemv_bench.c — the decode gemvs at their real shapes, on synthetic weights.
 *
 * WHY: CUPTI says the Q4_K gemv kernels reach ~85-145 GB/s while the Q6_K ones reach ~275,
 * across three different kernel variants. Q4_K carries gate/up in every layer plus half the
 * down projections, so it is ~42% of GPU busy time. If the gap is real and closable it is
 * the largest single item left in the decode budget.
 *
 * A benchmark that loads the 21 GiB model takes minutes per iteration, which is the wrong
 * loop for kernel work. Any bit pattern is a legal k-quant payload, so this synthesises
 * weights instead and checks the GPU against the CPU dequant on the same bytes -- fast
 * enough to iterate on the kernel, and still a correctness gate.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "../kquant.h"
#include "../kquant_simd.h"
#include "../qwen35_cuda.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
static uint32_t rs = 22222;
static uint32_t u32(void){ rs = rs*1664525u + 1013904223u; return rs; }
static float frnd(void){ return ((float)(u32() >> 8)/8388608.0f) - 1.0f; }

/* fp16 for a small positive scale: exponent kept in a sane range so no NaN/Inf appears */
static uint16_t h_small(void){ return (uint16_t)(0x1000 | (u32() & 0x03FF)); }

/* Random payload, sane scales. The values are meaningless; the byte layout is not. */
static void synth(uint8_t *w, int gtype, int I, int O){
    const int64_t rb = kq_row_bytes(gtype, I), nsb = I/KQ_QK_K;
    for(int64_t r = 0; r < O; r++){
        uint8_t *row = w + r*rb;
        for(int64_t b = 0; b < nsb; b++){
            if(gtype == KQ_Q4_K){
                kq_q4_K *q = (kq_q4_K*)(row + b*sizeof(kq_q4_K));
                q->d = h_small(); q->dmin = h_small();
                for(size_t i = 0; i < sizeof q->scales; i++) q->scales[i] = (uint8_t)u32();
                for(size_t i = 0; i < sizeof q->qs; i++)     q->qs[i]     = (uint8_t)u32();
            } else {
                kq_q6_K *q = (kq_q6_K*)(row + b*sizeof(kq_q6_K));
                q->d = h_small();
                for(size_t i = 0; i < sizeof q->ql; i++) q->ql[i] = (uint8_t)u32();
                for(size_t i = 0; i < sizeof q->qh; i++) q->qh[i] = (uint8_t)u32();
                for(size_t i = 0; i < sizeof q->scales; i++) q->scales[i] = (int8_t)(u32() & 0x7F);
            }
        }
    }
}

static int g_fail = 0;
static double relerr(const float *a, const float *b, int n){
    double num = 0, den = 0;
    for(int i = 0; i < n; i++){ const double d = (double)a[i]-b[i]; num += d*d; den += (double)a[i]*a[i]; }
    return den > 0 ? sqrt(num/den) : sqrt(num);
}

/* One shape, one quant type: correctness against the CPU, then GB/s.
 *
 * The pool is deliberately far larger than L2 and each timed call reads a different slice of
 * it. Without that, a 4.7 MiB weight set sits in L2 after the first iteration and the
 * benchmark reports cache bandwidth -- under which Q4_K and Q6_K take exactly the same time
 * for the same shape, which is a statement about L2, not about the decode. */
#define POOL_BYTES (512u<<20)
static void bench(const char *what, int gtype, int I, int O, int grouped, int n){
    const int64_t rb = kq_row_bytes(gtype, I);
    const int64_t wb = rb * O;                       /* bytes per expert */
    const int ng = grouped ? n : 1;
    uint8_t *W = (uint8_t*)malloc((size_t)wb*ng);
    float *x = (float*)malloc(sizeof(float)*(size_t)I*ng);
    float *yc = (float*)malloc(sizeof(float)*(size_t)O*ng);
    float *yg = (float*)malloc(sizeof(float)*(size_t)O*ng);
    for(int j = 0; j < ng; j++) synth(W + (size_t)j*wb, gtype, I, O);
    for(int64_t i = 0; i < (int64_t)I*ng; i++) x[i] = frnd();

    /* how many distinct copies of the whole group fit in the pool */
    int nrot = (int)(POOL_BYTES / (size_t)(wb*ng)); if(nrot < 1) nrot = 1; if(nrot > 64) nrot = 64;
    void *dW = q35cu_alloc((size_t)wb*ng*nrot), *dx = q35cu_alloc(sizeof(float)*(size_t)I*ng);
    void *dy = q35cu_alloc(sizeof(float)*(size_t)O*ng);
    void *dp = q35cu_alloc(sizeof(void*)*(size_t)ng);
    if(!dW || !dx || !dy || !dp){ printf("  %-26s alloc failed\n", what); g_fail++; return; }
    for(int r = 0; r < nrot; r++) q35cu_h2d((uint8_t*)dW + (size_t)r*wb*ng, W, (size_t)wb*ng);
    q35cu_h2d(dx, x, sizeof(float)*(size_t)I*ng);
    void *hp[64]; for(int j = 0; j < ng; j++) hp[j] = (uint8_t*)dW + (size_t)j*wb;
    q35cu_h2d(dp, hp, sizeof(void*)*(size_t)ng);

    for(int j = 0; j < ng; j++)
        kq_gemv_fast(yc + (size_t)j*O, x + (size_t)j*I, W + (size_t)j*wb, gtype, I, O);

    if(grouped) q35cu_gemv_grp(dy, dx, I, (const void* const*)dp, gtype, I, O, ng);
    else        q35cu_gemv(dy, dx, dW, gtype, I, O);
    q35cu_sync();
    q35cu_d2h(yg, dy, sizeof(float)*(size_t)O*ng);
    const double rel = relerr(yc, yg, O*ng);

    const int R = 200;
    for(int i = 0; i < 5; i++){
        if(grouped) q35cu_gemv_grp(dy, dx, I, (const void* const*)dp, gtype, I, O, ng);
        else        q35cu_gemv(dy, dx, dW, gtype, I, O);
    }
    q35cu_sync();
    const double t0 = now();
    for(int i = 0; i < R; i++){
        uint8_t *base = (uint8_t*)dW + (size_t)(i % nrot)*wb*ng;
        if(grouped){
            for(int j = 0; j < ng; j++) hp[j] = base + (size_t)j*wb;
            q35cu_h2d_compute(dp, hp, sizeof(void*)*(size_t)ng);
            q35cu_gemv_grp(dy, dx, I, (const void* const*)dp, gtype, I, O, ng);
        } else q35cu_gemv(dy, dx, base, gtype, I, O);
    }
    q35cu_sync();
    const double us = (now()-t0)/R*1e6;
    const double gbs = (double)wb*ng/(us*1e-6)/1e9;
    const int bad = !(rel < 2e-3);
    printf("  %-26s %-5s %8.1f us  %7.1f GB/s   x%-3d rel %.1e %s\n",
           what, kq_name(gtype), us, gbs, nrot, rel, bad ? "FAIL" : "");
    if(bad) g_fail++;
    q35cu_free(dW); q35cu_free(dx); q35cu_free(dy); q35cu_free(dp);
    free(W); free(x); free(yc); free(yg);
}

int main(void){
    if(!q35cu_init(0)){ printf("no CUDA: %s\n", q35cu_error()); return 0; }
    /* Anchor everything: an F32 gemv is a pure streaming read of its weight matrix, so this
     * is what the card actually delivers for this access pattern. Every k-quant number below
     * should be read as a fraction of THIS, not of a datasheet figure. */
    {
        const int I = 2048, O = 16384;
        const size_t wb = (size_t)I*O*sizeof(float);
        float *W = (float*)malloc(wb), *x = (float*)malloc(sizeof(float)*I);
        for(size_t i = 0; i < (size_t)I*O; i++) W[i] = frnd();
        for(int i = 0; i < I; i++) x[i] = frnd();
        void *dW = q35cu_alloc(wb), *dx = q35cu_alloc(sizeof(float)*I), *dy = q35cu_alloc(sizeof(float)*O);
        q35cu_h2d(dW, W, wb); q35cu_h2d(dx, x, sizeof(float)*I);
        for(int i = 0; i < 5; i++) q35cu_gemv(dy, dx, dW, KQ_F32, I, O);
        q35cu_sync();
        const double t0 = now();
        for(int i = 0; i < 50; i++) q35cu_gemv(dy, dx, dW, KQ_F32, I, O);
        q35cu_sync();
        const double us = (now()-t0)/50*1e6;
        printf("== achievable VRAM read bandwidth (f32 gemv, %.0f MiB) : %.0f GB/s ==\n\n",
               wb/1048576.0, wb/(us*1e-6)/1e9);
        q35cu_free(dW); q35cu_free(dx); q35cu_free(dy); free(W); free(x);
    }
    printf("== decode gemv shapes, synthetic weights ==\n");
    printf("  %-26s %-5s %10s %13s   %s\n", "what", "type", "time", "effective", "vs CPU");

    /* the direct comparison: same shape, same kernel structure, only the quant differs */
    bench("expert down (grouped n=8)", KQ_Q4_K, 512, 2048, 1, 8);
    bench("expert down (grouped n=8)", KQ_Q6_K, 512, 2048, 1, 8);
    printf("\n");
    bench("expert gate/up (grp n=8)",  KQ_Q4_K, 2048, 512, 1, 8);
    bench("expert gate/up (grp n=8)",  KQ_Q6_K, 2048, 512, 1, 8);
    printf("\n");
    bench("input proj O=8192",         KQ_Q4_K, 2048, 8192, 0, 1);
    bench("input proj O=8192",         KQ_Q6_K, 2048, 8192, 0, 1);
    printf("\n");
    bench("shared expert gate O=512",  KQ_Q4_K, 2048, 512, 0, 1);
    bench("  same, via grouped n=1",   KQ_Q4_K, 2048, 512, 1, 1);
    bench("shared expert down O=2048", KQ_Q4_K, 512, 2048, 0, 1);
    bench("  same, via grouped n=1",   KQ_Q4_K, 512, 2048, 1, 1);
    bench("gdn/attn out O=2048 I=4096",KQ_Q4_K, 4096, 2048, 0, 1);
    bench("  same, via grouped n=1",   KQ_Q4_K, 4096, 2048, 1, 1);
    printf("\n");
    bench("lm head O=248320",          KQ_Q6_K, 2048, 248320, 0, 1);

    /* rmsnorm: 81 calls per token, and CUPTI clocks it at 9.3 us for 8 KiB of data. */
    {
        const int n = 2048, R = 2000;
        float *x = (float*)malloc(sizeof(float)*n), *w = (float*)malloc(sizeof(float)*n);
        float *o = (float*)malloc(sizeof(float)*n), *oc = (float*)malloc(sizeof(float)*n);
        for(int i = 0; i < n; i++){ x[i] = frnd(); w[i] = frnd(); }
        void *dx = q35cu_alloc(sizeof(float)*n), *dw = q35cu_alloc(sizeof(float)*n);
        void *dо = q35cu_alloc(sizeof(float)*n);
        q35cu_h2d(dx, x, sizeof(float)*n); q35cu_h2d(dw, w, sizeof(float)*n);
        double ss = 0; for(int i = 0; i < n; i++) ss += (double)x[i]*x[i];
        const float r = 1.0f/sqrtf((float)(ss/n) + 1e-6f);
        for(int i = 0; i < n; i++) oc[i] = x[i]*r*w[i];
        q35cu_rmsnorm((float*)dо, (const float*)dx, (const float*)dw, n, 1e-6f);
        q35cu_sync(); q35cu_d2h(o, dо, sizeof(float)*n);
        for(int i = 0; i < 20; i++) q35cu_rmsnorm((float*)dо, (const float*)dx, (const float*)dw, n, 1e-6f);
        q35cu_sync();
        const double t0 = now();
        for(int i = 0; i < R; i++) q35cu_rmsnorm((float*)dо, (const float*)dx, (const float*)dw, n, 1e-6f);
        q35cu_sync();
        printf("\n== elementwise ==\n  %-26s %8.2f us/call   rel %.1e\n",
               "rmsnorm n=2048", (now()-t0)/R*1e6, relerr(oc, o, n));
        q35cu_free(dx); q35cu_free(dw); q35cu_free(dо); free(x); free(w); free(o); free(oc);
    }

    printf("\n%s\n", g_fail ? "FAIL" : "all shapes match the CPU dequant");
    q35cu_shutdown();
    return g_fail ? 1 : 0;
}
