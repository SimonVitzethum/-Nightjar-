#ifndef COLIBRI_MOE_CPU_H
#define COLIBRI_MOE_CPU_H
/* moe_cpu.h — routed experts on the CPU, computed IN RAM, no PCIe hop.
 *
 * WHY THIS EXISTS
 *
 * Once the experts live in RAM (not on the NVMe), the wall is PCIe: pushing a RAM-resident
 * expert to the GPU to be computed there costs the RAM->VRAM transfer, capped at ~55 GB/s on
 * PCIe 5.0 = ~6 tok/s for a 9 GB/token model. That transfer is pure overhead — the expert is
 * ALREADY in RAM, where the CPU can read it at 200-300 GB/s and compute it in place.
 *
 * So the design is heterogeneous: the GPU computes the experts that are VRAM-resident (or worth
 * the PCIe trip), the CPU computes the rest straight out of RAM, and the two run CONCURRENTLY.
 * Their bandwidths add, because the GPU's PCIe read and the CPU's RAM read both draw on DDR5
 * whose aggregate (~250 GB/s) exceeds their sum. The layer costs max(GPU, CPU), not the sum.
 *
 * STAGED, ON PURPOSE
 *
 * This file is stage 1: correct and multithreaded, built on the kq_dot_* kernels already
 * validated bit-for-bit against ggml. It parallelizes across output rows with OpenMP. Its job
 * is to establish the CORRECT answer and a HONEST throughput number. Only once both are in
 * hand does the SIMD (AVX2 + VNNI) inner loop go in — measured against this same reference, so
 * a fast-but-wrong kernel cannot masquerade as progress. Every silent-failure scar on this
 * project came from optimizing before validating; not here.
 */
#include <stdlib.h>
#include <math.h>
#include "kquant.h"
#include "moe_cpu_simd.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* Scratch for the CPU MoE: gate/up/hidden buffers, one set, reused across experts. */
typedef struct {
    float *g, *u, *out;    /* [d_hid], [d_hid], [d_model] */
    int    d_model, d_hid;
} MoeCpu;

static int moe_cpu_init(MoeCpu *M, int d_model, int d_hid){
    moe_simd_init();
    M->d_model = d_model; M->d_hid = d_hid;
    M->g   = (float*)malloc((size_t)d_hid*sizeof(float));
    M->u   = (float*)malloc((size_t)d_hid*sizeof(float));
    M->out = (float*)malloc((size_t)d_model*sizeof(float));
    return M->g && M->u && M->out;
}

static void moe_cpu_free(MoeCpu *M){ free(M->g); free(M->u); free(M->out); }

/* One expert: acc[d_model] += weight * down( swiglu( gate(x), up(x) ) ).
 *
 * gate/up contract d_model -> d_hid; down contracts d_hid -> d_model. Rows are independent,
 * so each gemv parallelizes across its output rows. The swiglu is a barrier between them (down
 * needs the full hidden vector), and the final accumulate is O(d_model) — cheap, kept serial.
 *
 * The byte layout matches kq_gemm exactly: row o of an [O,I] tensor of type `t` starts at
 * o*row_bytes(t,I) and is contracted with the I-length activation. */
static void moe_cpu_expert(MoeCpu *M, float *acc, const float *x,
                           const void *wg, int tg, const void *wu, int tu,
                           const void *wd, int td, float weight){
    const int D = M->d_model, F = M->d_hid;
    const int64_t rg = kq_row_bytes(tg, D), ru = kq_row_bytes(tu, D), rd = kq_row_bytes(td, F);
    const uint8_t *bg = (const uint8_t*)wg, *bu = (const uint8_t*)wu, *bd = (const uint8_t*)wd;

    /* gate and up share the same contraction (x over d_model); fuse their row spaces into one
     * parallel loop of 2F rows so every thread stays fed even when F is small. */
    #pragma omp parallel for schedule(static)
    for(int o=0;o<2*F;o++){
        if(o < F) M->g[o]     = kq_dot_fast(tg, bg + (int64_t)o    *rg, x, D);
        else      M->u[o-F]   = kq_dot_fast(tu, bu + (int64_t)(o-F)*ru, x, D);
    }

    /* SwiGLU — silu(g)*u, no clamp (glm-dsa). Cheap; parallelized for free. */
    #pragma omp parallel for schedule(static)
    for(int i=0;i<F;i++)
        M->g[i] = (M->g[i] / (1.0f + expf(-M->g[i]))) * M->u[i];

    /* down contracts the hidden vector; each output row is one dot over d_hid. */
    #pragma omp parallel for schedule(static)
    for(int o=0;o<D;o++)
        M->out[o] = kq_dot_fast(td, bd + (int64_t)o*rd, M->g, F);

    for(int o=0;o<D;o++) acc[o] += weight * M->out[o];
}

/* A batch of experts, accumulated. This is the CPU's share of a layer's routed experts; the
 * caller decides which experts land here vs on the GPU. acc must be pre-zeroed by the caller
 * if the CPU owns the whole layer, or left alone if it is adding to the GPU's partial sum
 * (the accumulation is += so both sides can target the same buffer under a barrier). */
static void moe_cpu_layer(MoeCpu *M, float *acc, const float *x,
                          const void **wg, const int *tg,
                          const void **wu, const int *tu,
                          const void **wd, const int *td,
                          const float *weight, int k){
    for(int e=0;e<k;e++)
        moe_cpu_expert(M, acc, x, wg[e], tg[e], wu[e], tu[e], wd[e], td[e], weight[e]);
}

#endif
