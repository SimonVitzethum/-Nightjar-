/* qwen35_cuda_test.c — every device kernel against its CPU counterpart.
 *
 * THE SCORE IS THE POPULATION RMS, NEVER A PER-ROW RATIO.
 *
 * This project has produced three false negatives from per-row relative error, all with the
 * same shape: a kernel is correct, the per-element ratio reads 1.3, and the kernel gets
 * rewritten. Two uncorrelated vectors of equal norm sit at sqrt(2) = 1.414, so anything near
 * that number is measuring "these are unrelated", which is exactly what a per-row score of a
 * mostly-zero row reports about a perfectly good result. The denominator must match the
 * consumer: the next layer sees the whole vector, so the whole vector's RMS is the scale.
 *
 *   > 1.4      look at the metric first
 *   0.01..1.4  look at the arithmetic
 *   ~1e-6      reassociation; that is what a correct SIMD/warp kernel looks like
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../qwen35_cpu.h"
#include "../qwen35_cuda.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

static int g_fail = 0;

/* rms(a-b) / rms(b) — one number, scored against the population the consumer sees */
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
    printf("  %-26s rel %.3e  (tol %.0e)  %s\n", what, s, tol, bad ? "FAIL" : "ok");
    if(bad) g_fail++;
}

static void frand(float *v, int n, unsigned seed){
    unsigned s = seed;
    for(int i = 0; i < n; i++){ s = s*1664525u + 1013904223u; v[i] = ((int)(s>>8) % 20000 - 10000)/10000.0f; }
}

/* upload a tensor exactly as it lies in the file */
static void *up(const Q35Model *M, const gguf_tensor *T){
    void *d = q35cu_alloc(T->bytes);
    if(!d){ printf("  alloc %s failed: %s\n", T->name, q35cu_error()); exit(1); }
    if(!q35cu_h2d(d, q35_wdata(M, T), T->bytes)){ printf("  h2d failed: %s\n", q35cu_error()); exit(1); }
    return d;
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : "/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf";

    if(!q35cu_init(0)){ printf("cuda unavailable: %s\n", q35cu_error()); return 77; }
    printf("device: %s\n", q35cu_name());
    size_t fb, tb; q35cu_mem(&fb, &tb);
    printf("vram:   %.2f free / %.2f GiB\n", fb/1073741824.0, tb/1073741824.0);
    printf("stream: %.1f GB/s measured\n\n", q35cu_measure_bw());

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    const Q35Cfg *c = &M.c;

    /* ---- 1. bytes survive the trip ---- */
    printf("== upload round trip ==\n");
    {
        const gguf_tensor *T = M.L[0].ffn_gate;
        void *d = up(&M, T);
        uint8_t *back = (uint8_t*)malloc(T->bytes);
        q35cu_d2h(back, d, T->bytes);
        const int same = !memcmp(back, q35_wdata(&M, T), T->bytes);
        printf("  %-26s %lld bytes  %s\n", T->name, (long long)T->bytes, same ? "identical" : "CORRUPT");
        if(!same) g_fail++;
        free(back); q35cu_free(d);
    }

    /* ---- 2. gemv, both quant types, on real weights ---- */
    printf("\n== gemv vs CPU (real tensors) ==\n");
    {
        int ia = -1;
        for(int i = 0; i < c->n_layer; i++) if(c->kind[i] == Q35_LAYER_ATTN){ ia = i; break; }
        const gguf_tensor *tests[] = { M.L[0].wqkv, M.L[0].w_out,
                                       ia >= 0 ? M.L[ia].wq : NULL, M.output, M.L[0].w_beta };
        const char *nm[] = { "ssm_in Q4_K", "ssm_out Q4_K", "attn_q Q4_K", "output Q6_K", "ssm_beta Q4_K" };
        for(unsigned k = 0; k < sizeof tests/sizeof *tests; k++){
            const gguf_tensor *T = tests[k];
            if(!T){ printf("  %-26s MISSING\n", nm[k]); g_fail++; continue; }
            const int I = (int)T->ne[0], O = (int)T->ne[1];
            float *x = (float*)malloc(sizeof(float)*I);
            float *yc = (float*)malloc(sizeof(float)*O), *yg = (float*)malloc(sizeof(float)*O);
            frand(x, I, 1234u + k);
            kq_gemv_fast(yc, x, q35_wdata(&M, T), T->type, I, O);
            void *dW = up(&M, T);
            float *dx = (float*)q35cu_alloc(sizeof(float)*I), *dy = (float*)q35cu_alloc(sizeof(float)*O);
            q35cu_h2d(dx, x, sizeof(float)*I);
            if(!q35cu_gemv(dy, dx, dW, T->type, I, O)){ printf("  gemv: %s\n", q35cu_error()); g_fail++; }
            q35cu_sync();
            q35cu_d2h(yg, dy, sizeof(float)*O);
            char lbl[64]; snprintf(lbl, sizeof lbl, "%s %dx%d", nm[k], I, O);
            check(lbl, yg, yc, O, 2e-5);
            q35cu_free(dW); q35cu_free(dx); q35cu_free(dy);
            free(x); free(yc); free(yg);
        }
    }

    /* ---- 3. gemv bandwidth: is the decode keeping up with the memory? ---- */
    printf("\n== gemv throughput ==\n");
    {
        const gguf_tensor *T = M.output;                 /* 1.04 GB Q6_K, the biggest single read */
        const int I = (int)T->ne[0], O = (int)T->ne[1];
        void *dW = up(&M, T);
        float *dx = (float*)q35cu_alloc(sizeof(float)*I), *dy = (float*)q35cu_alloc(sizeof(float)*O);
        q35cu_zero(dx, sizeof(float)*I); q35cu_sync();
        for(int i = 0; i < 3; i++) q35cu_gemv(dy, dx, dW, T->type, I, O);
        q35cu_sync();
        const double t0 = now();
        const int R = 20;
        for(int i = 0; i < R; i++) q35cu_gemv(dy, dx, dW, T->type, I, O);
        q35cu_sync();
        const double dt = (now()-t0)/R;
        printf("  output proj  %6.2f ms/call  %.2f GiB  -> %.1f GB/s\n",
               dt*1e3, T->bytes/1073741824.0, T->bytes/dt/1e9);
        q35cu_free(dW); q35cu_free(dx); q35cu_free(dy);
    }

    /* ---- 4. elementwise ---- */
    printf("\n== elementwise ==\n");
    {
        const int n = c->d_model;
        float *x = (float*)malloc(4*n), *w = (float*)malloc(4*n);
        float *cpu = (float*)malloc(4*n), *gpu = (float*)malloc(4*n);
        frand(x, n, 7); frand(w, n, 9);
        for(int i = 0; i < n; i++) w[i] = 0.5f + 0.5f*w[i];
        q35_rmsnorm(cpu, x, w, n, c->eps);
        float *dx = (float*)q35cu_alloc(4*n), *dw = (float*)q35cu_alloc(4*n), *dy = (float*)q35cu_alloc(4*n);
        q35cu_h2d(dx, x, 4*n); q35cu_h2d(dw, w, 4*n);
        q35cu_rmsnorm(dy, dx, dw, n, c->eps);
        q35cu_sync(); q35cu_d2h(gpu, dy, 4*n);
        check("rmsnorm", gpu, cpu, n, 1e-6);

        float *g = (float*)malloc(4*n), *u = (float*)malloc(4*n);
        frand(g, n, 11); frand(u, n, 13);
        for(int i = 0; i < n; i++) cpu[i] = q35_silu(g[i])*u[i];
        q35cu_h2d(dx, g, 4*n); q35cu_h2d(dw, u, 4*n);
        q35cu_swiglu(dx, dw, n);
        q35cu_sync(); q35cu_d2h(gpu, dx, 4*n);
        check("swiglu", gpu, cpu, n, 1e-6);
        q35cu_free(dx); q35cu_free(dw); q35cu_free(dy);
        free(x); free(w); free(cpu); free(gpu); free(g); free(u);
    }

    /* ---- 4b. the S>1 path: prefill decodes each weight ONCE for the whole batch ---- */
    printf("\n== batched gemm (S>1) vs S=1 ==\n");
    {
        const gguf_tensor *T = M.L[0].wqkv;
        const int I = (int)T->ne[0], O = (int)T->ne[1];
        for(int S = 2; S <= 8; S += 3){
            float *x = (float*)malloc(sizeof(float)*(size_t)S*I);
            float *ref = (float*)malloc(sizeof(float)*(size_t)S*O);
            float *got = (float*)malloc(sizeof(float)*(size_t)S*O);
            for(int s = 0; s < S; s++) frand(x + (size_t)s*I, I, 500u + s);
            void *dW = up(&M, T);
            float *dx = (float*)q35cu_alloc(sizeof(float)*(size_t)S*I);
            float *dy = (float*)q35cu_alloc(sizeof(float)*(size_t)S*O);
            q35cu_h2d(dx, x, sizeof(float)*(size_t)S*I);
            /* the reference is the SAME device kernel at S=1, row by row: this isolates the
             * batching from the decode, which is already scored against the CPU above */
            for(int s = 0; s < S; s++) q35cu_gemv(dy + (size_t)s*O, dx + (size_t)s*I, dW, T->type, I, O);
            q35cu_sync(); q35cu_d2h(ref, dy, sizeof(float)*(size_t)S*O);
            q35cu_gemm(dy, dx, dW, T->type, I, O, S);
            q35cu_sync(); q35cu_d2h(got, dy, sizeof(float)*(size_t)S*O);
            char lbl[64]; snprintf(lbl, sizeof lbl, "gemm S=%d vs S=1 x%d", S, S);
            check(lbl, got, ref, S*O, 1e-6);
            q35cu_free(dW); q35cu_free(dx); q35cu_free(dy);
            free(x); free(ref); free(got);
        }
    }

    /* ---- 5. does the device pack KV to the SAME BYTES as the host? ----
     * The attention comparison is only a gate if both sides quantize identically. A one-LSB
     * disagreement in the fp16 scale is invisible per element and shows up downstream as a
     * few times 1e-4 in the attention output, which reads exactly like a wiring bug. */
    printf("\n== kv q8_0 pack, device vs host ==\n");
    {
        const int n = c->n_head_kv*c->d_head;
        const int64_t nb = q35_kv_entry_bytes(Q35_KV_Q8_0, n);
        float *x = (float*)malloc(4*n);
        frand(x, n, 4711u);
        uint8_t *hp = (uint8_t*)malloc(nb), *gp = (uint8_t*)malloc(nb);
        kvt_pack_q8(hp, x, n);
        float *dx = (float*)q35cu_alloc(4*n);
        void *dk = q35cu_alloc(nb), *dv = q35cu_alloc(nb);
        q35cu_h2d(dx, x, 4*n);
        q35cu_kv_store(dk, dv, dx, dx, 0, 0, n, Q35_KV_Q8_0);
        q35cu_sync();
        q35cu_d2h(gp, dk, nb);
        int diff = 0, dscale = 0;
        for(int64_t i = 0; i < nb; i++) if(hp[i] != gp[i]){ diff++; if((i % 34) < 2) dscale++; }
        printf("  %lld bytes  %d differ (%d of them scale bytes)  %s\n",
               (long long)nb, diff, dscale, diff ? "MISMATCH" : "identical");
        if(diff) g_fail++;
        free(x); free(hp); free(gp); q35cu_free(dx); q35cu_free(dk); q35cu_free(dv);
    }

    printf("\n%s  (%d checks failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    q35_close(&M);
    q35cu_shutdown();
    return g_fail ? 1 : 0;
}
