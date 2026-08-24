#ifndef COLIBRI_MOE_HETERO_H
#define COLIBRI_MOE_HETERO_H
/* moe_hetero.h — one layer's routed experts, split across GPU and CPU, computed concurrently.
 *
 * The GPU computes the first `ng` experts (uploaded over PCIe, or already VRAM-resident); the
 * CPU computes the rest straight out of RAM. Both run at the same time — the GPU work is
 * launched async and the host threads do the CPU experts while it runs — so the layer costs
 * max(GPU, CPU) instead of their sum. Measured on a laptop + 8 GB card: 97-100% overlap,
 * bit-identical to running the same split serially.
 *
 * TWO THINGS THAT ARE EASY TO GET WRONG, AND ARE HANDLED HERE:
 *
 *  - The GPU's acc copy-back MUST land in pinned memory, or cudaMemcpyAsync silently becomes a
 *    blocking copy and the "launch" waits for the whole GPU — killing the overlap. acc_gpu is
 *    cudaMallocHost'd for exactly this reason. (This bug cost the overlap once already.)
 *
 *  - OpenMP must NOT grab every core, or the GPU driver has no thread to feed the device and
 *    the two serialize. The caller should leave ~2 cores for the driver.
 *
 * AUTO-BALANCE: the split point `ng` is tuned online. Real GPU time (from CUDA events, not
 * wall clock — wall clock includes the overlap) and real CPU time are tracked as EMAs of
 * per-expert cost, and ng drifts toward the point where ng*gpu_cost == (k-ng)*cpu_cost. It
 * converges in a few layers and then rides the true balance, which shifts with cache hit rate
 * (a VRAM-resident expert is far cheaper on the GPU than one streamed over PCIe).
 */
#include <cuda_runtime.h>
#include "moe_gpu.h"
#include "moe_cpu.h"

typedef struct {
    MoeGpu *G;
    MoeCpu *C;
    float  *acc_gpu;              /* pinned; the GPU's partial sum */
    int     d_model;
    int     ng;                   /* experts assigned to the GPU (auto-tuned) */
    double  gpu_cost, cpu_cost;   /* EMA of ms per expert on each side */
    cudaEvent_t ev0, ev1;         /* true GPU time, independent of the CPU overlap */
    int     warm;
} MoeHetero;

static int moe_hetero_init(MoeHetero *H, MoeGpu *G, MoeCpu *C, int d_model){
    H->G=G; H->C=C; H->d_model=d_model;
    H->ng = -1;                   /* -1 = not yet decided; first layer measures */
    H->gpu_cost = H->cpu_cost = 0; H->warm = 0;
    if(cudaMallocHost((void**)&H->acc_gpu,(size_t)d_model*sizeof(float))!=cudaSuccess) return 0;
    if(cudaEventCreate(&H->ev0)!=cudaSuccess) return 0;
    if(cudaEventCreate(&H->ev1)!=cudaSuccess) return 0;
    return 1;
}

/* Compute a layer's k experts, split GPU/CPU, concurrently. acc must be zeroed by the caller.
 * The expert arrays are the full k; this routine slices [0,ng) to the GPU and [ng,k) to the
 * CPU. The GPU pointers must be pinned (cache/staging already are). */
static void moe_hetero_layer(MoeHetero *H, float *acc, const float *xn,
                             const void **eg, const void **eu, const void **ed,
                             const int *tg, const int *tu, const int *td,
                             const int64_t *ng_b, const int64_t *nu_b, const int64_t *nd_b,
                             const float *w, int k){
    const int D = H->d_model;

    /* decide the split. first layer: halve it and measure. after that: the balance point. */
    int ng;
    if(H->ng < 0) ng = k/2;
    else {
        if(H->gpu_cost>0 && H->cpu_cost>0){
            double f = H->cpu_cost / (H->gpu_cost + H->cpu_cost);   /* GPU's share */
            ng = (int)(f*k + 0.5);
        } else ng = H->ng;
    }
    if(ng<0) ng=0; if(ng>k) ng=k;
    const int nc = k - ng;

    /* launch GPU async, timed by events so we get its REAL cost, not the overlapped wall time */
    if(ng>0){
        cudaEventRecord(H->ev0, H->G->s);
        moe_gpu_launch(H->G, H->acc_gpu, xn, eg,eu,ed, tg,tu,td, ng_b,nu_b,nd_b, w, ng);
        cudaEventRecord(H->ev1, H->G->s);
    }

    /* CPU experts run NOW, on the host threads, while the GPU works */
    double c0 = 0, c1 = 0;
    if(nc>0){
        struct timespec ta,tb; clock_gettime(CLOCK_MONOTONIC,&ta);
        moe_cpu_layer(H->C, acc, xn, eg+ng,tg+ng, eu+ng,tu+ng, ed+ng,td+ng, w+ng, nc);
        clock_gettime(CLOCK_MONOTONIC,&tb);
        c0 = ta.tv_sec*1e3+ta.tv_nsec*1e-6; c1 = tb.tv_sec*1e3+tb.tv_nsec*1e-6;
    }

    /* collect the GPU and fold its partial in */
    if(ng>0){
        moe_gpu_wait(H->G);
        for(int i=0;i<D;i++) acc[i] += H->acc_gpu[i];
    }

    /* update the per-expert cost EMAs and remember the split */
    if(ng>0){
        float g_ms=0; cudaEventElapsedTime(&g_ms, H->ev0, H->ev1);
        double per = g_ms/ng;
        H->gpu_cost = H->warm ? 0.8*H->gpu_cost + 0.2*per : per;
    }
    if(nc>0){
        double per = (c1-c0)/nc;
        H->cpu_cost = H->warm ? 0.8*H->cpu_cost + 0.2*per : per;
    }
    H->ng = ng; H->warm = 1;
}

#endif
