/* qwen35_cuda.cu — GDN, attention and the output projection on the GPU.
 *
 * Everything here decodes GGUF quant bytes IN REGISTERS. No tensor is ever expanded to fp16
 * or fp32 in VRAM: 5.08 GiB of Q4_K/Q6_K would become 15-20 GiB dequantized and this card has
 * 8. The bytes go to the device exactly as they came out of the file and are decoded inside
 * the dot product, which is also why the arithmetic below looks like kquant.h — it is the
 * same decode, transposed onto warps.
 *
 * THE ONE STRUCTURAL DECISION IN THE MATMULS
 *
 * The obvious layout is "one lane per super-block". It measures 20x below the card's memory
 * bandwidth, because consecutive lanes then touch addresses 144 bytes apart and nothing
 * coalesces. The whole WARP therefore cooperates on ONE super-block: a Q4_K super-block holds
 * four 64-weight groups of 32 bytes, so lane l takes byte l of each group, consecutive lanes
 * read consecutive bytes, and each group is a single 32-byte transaction. Same shape for Q6_K
 * (ql[l], ql[l+32], qh[l]).
 *
 * NUMERICS
 *
 * Reductions that the CPU reference does in double are done in double here too (rmsnorm,
 * l2norm). The dot products reassociate — the warp sums 32 partials in a tree instead of 256
 * terms in order — which lands at ~1e-6 relative. That is reassociation, not a bug; the gate
 * is the population RMS, never a per-row ratio (§6b: uncorrelated equal-norm vectors sit at
 * sqrt(2), so a per-row score cannot tell a wrong kernel from a right one).
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "qwen35_cuda.h"

#define QK_K 256
#define NWARP 8
#define NTHR  (NWARP*32)
#define FULL  0xffffffffu

/* ggml type ids, same numbering as kquant.h */
enum { CU_F32 = 0, CU_F16 = 1, CU_Q8_0 = 8, CU_Q4_K = 12, CU_Q5_K = 13, CU_Q6_K = 14 };

typedef struct { __half d, dmin; uint8_t scales[12]; uint8_t qs[128]; } b_q4_K;  /* 144 */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; } b_q6_K; /* 210 */
typedef struct { __half d; int8_t qs[32]; } b_q8_0;                              /* 34  */

/* ---------------- state ---------------- */

static int          g_ready  = 0;
static int          g_dev    = -1;
static cudaStream_t g_s      = 0;    /* compute */
static cudaStream_t g_cp     = 0;    /* copies, so DMA overlaps the CPU's FFN */
static char         g_err[512] = "";
static char         g_name[128] = "";
static double       g_bw     = 0;
static int          g_logon  = -1;

static void seterr(const char *what, const char *why){
    snprintf(g_err, sizeof g_err, "%s: %s", what, why);
}
extern "C" const char *q35cu_error(void){ return g_err; }
extern "C" const char *q35cu_name(void){ return g_name; }
extern "C" int q35cu_ok(void){ return g_ready; }
extern "C" double q35cu_bandwidth(void){ return g_bw; }

#define CK(call) do{ cudaError_t e_ = (call); if(e_ != cudaSuccess){ \
        seterr(#call, cudaGetErrorString(e_)); return 0; } }while(0)
#define CKV(call) do{ cudaError_t e_ = (call); if(e_ != cudaSuccess){ \
        seterr(#call, cudaGetErrorString(e_)); return; } }while(0)

extern "C" int q35cu_log_on(void){
    if(g_logon < 0){ const char *e = getenv("COLIBRI_KERNEL_LOG"); g_logon = (e && *e && *e != '0'); }
    return g_logon;
}
extern "C" void q35cu_log(const char *component, const char *detail){
    if(!q35cu_log_on()) return;
    fprintf(stderr, "  [kernel] %-14s %s\n", component, detail ? detail : "");
}

/* ---------------- lifecycle ---------------- */

extern "C" int q35cu_init(int device){
    if(g_ready) return 1;
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if(e != cudaSuccess){ seterr("cudaGetDeviceCount", cudaGetErrorString(e)); return 0; }
    if(n <= 0){ seterr("cuda", "no device"); return 0; }
    if(device < 0 || device >= n) device = 0;
    CK(cudaSetDevice(device));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, device));
    snprintf(g_name, sizeof g_name, "%s (sm_%d%d, %d SMs, %.2f GiB)",
             p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem/1073741824.0);
    CK(cudaStreamCreateWithFlags(&g_s,  cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&g_cp, cudaStreamNonBlocking));
    g_dev = device; g_ready = 1;
    return 1;
}

extern "C" void q35cu_shutdown(void){
    if(!g_ready) return;
    cudaStreamDestroy(g_s); cudaStreamDestroy(g_cp);
    g_s = g_cp = 0; g_ready = 0;
}

extern "C" int q35cu_mem(size_t *fb, size_t *tb){
    size_t f = 0, t = 0;
    if(!g_ready){ if(fb)*fb=0; if(tb)*tb=0; return 0; }
    CK(cudaMemGetInfo(&f, &t));
    if(fb)*fb=f; if(tb)*tb=t;
    return 1;
}

extern "C" void *q35cu_alloc(size_t n){
    void *p = NULL;
    if(!g_ready){ seterr("alloc", "cuda not initialized"); return NULL; }
    cudaError_t e = cudaMalloc(&p, n);
    if(e != cudaSuccess){ seterr("cudaMalloc", cudaGetErrorString(e)); return NULL; }
    return p;
}
extern "C" void q35cu_free(void *p){ if(p) cudaFree(p); }

extern "C" void *q35cu_host_alloc(size_t n){
    void *p = NULL;
    if(!g_ready) return NULL;
    if(cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess) return NULL;
    return p;
}
extern "C" void q35cu_host_free(void *p){ if(p) cudaFreeHost(p); }

extern "C" int q35cu_h2d(void *d, const void *s, size_t n){ CK(cudaMemcpy(d,s,n,cudaMemcpyHostToDevice)); return 1; }
extern "C" int q35cu_d2h(void *d, const void *s, size_t n){ CK(cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost)); return 1; }
extern "C" int q35cu_h2d_async(void *d, const void *s, size_t n){ CK(cudaMemcpyAsync(d,s,n,cudaMemcpyHostToDevice,g_cp)); return 1; }
extern "C" int q35cu_d2h_async(void *d, const void *s, size_t n){ CK(cudaMemcpyAsync(d,s,n,cudaMemcpyDeviceToHost,g_cp)); return 1; }
extern "C" int q35cu_copy_sync(void){ CK(cudaStreamSynchronize(g_cp)); return 1; }
extern "C" int q35cu_sync(void){
    CK(cudaStreamSynchronize(g_s));
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){ seterr("kernel", cudaGetErrorString(e)); return 0; }
    return 1;
}
extern "C" int q35cu_zero(void *p, size_t n){ CK(cudaMemsetAsync(p, 0, n, g_s)); return 1; }

/* ---------------- quant decode helpers ---------------- */

/* ggml get_scale_min_k4: the 6-bit scales of the last four groups are split across two bytes,
 * with the high 2 bits living in the TOP of an earlier byte. Reading it as a flat 6-bit array
 * is in-bounds and wrong by a few percent per group. */
__device__ __forceinline__ void gsm4(const uint8_t *q, int j, float *sc, float *mn){
    uint8_t d, m;
    if(j < 4){ d = q[j] & 63; m = q[j+4] & 63; }
    else     { d = (uint8_t)((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4));
               m = (uint8_t)((q[j+4] >> 4)  | ((q[j  ] >> 6) << 4)); }
    *sc = (float)d; *mn = (float)m;
}

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ double warp_sum(double v){
    #pragma unroll
    for(int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v){
    #pragma unroll
    for(int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
    return v;
}

/* block reduction in double — matches the CPU reference's accumulator, so a norm mismatch
 * means a real bug and not the summation order. */
__device__ __forceinline__ double block_sum(double v, double *sh){
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if(lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    if(wid == 0){
        double t = (lane < nw) ? sh[lane] : 0.0;
        t = warp_sum(t);
        if(lane == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}

/* ---------------- gemv: y[S,O] = x[S,I] . W[O,I]^T ---------------- */

#define SMAX 8
#define RPW  4          /* rows per warp on the S==1 path */
#define NARROW 2048     /* below this O, use one block per row */

/* WHY ROWS ARE BLOCKED AND x IS NOT STAGED IN SHARED
 *
 * The obvious optimization is to copy x into shared memory once per block. Measured, it made
 * the output projection SLOWER: 8 rows per block over 248320 rows is 31040 blocks, each
 * re-reading all 20 KiB of x, which is 0.59 GiB of extra global traffic against a 0.97 GiB
 * tensor — a 61% surcharge — and 20 KiB of shared caps the SM at 5 blocks. x is 20 KiB; it
 * lives in L1 without being asked.
 *
 * What does pay is reuse in REGISTERS. A lane loads its four x values once and dots them
 * against RPW rows, so x traffic drops RPW-fold and the four rows' weight loads are mutually
 * independent — which is what actually fills the memory pipeline. */

/* WHY THE Q4_K SCALES ARE UNPACKED FROM REGISTERS
 *
 * Measured: Q6_K reaches 253 GB/s on this card and the first Q4_K kernel reached 104 on the
 * same shape, while reading FEWER bytes. Q4_K is not memory-bound, it is decode-bound. A
 * super-block needs eight 6-bit scale/min pairs and ggml splits the last four across two
 * bytes each, so the naive form does ~26 single-byte loads per super-block per row and then
 * branches on j<4. Pulling d, dmin and all twelve scale bytes in as four 32-bit loads and
 * unpacking with constant shifts leaves only the quant bytes going to memory, which is what
 * the kernel is supposed to be waiting on. */
__device__ __forceinline__ void gsm4_reg(uint32_t s0, uint32_t s1, uint32_t s2, int j,
                                         float *sc, float *mn){
    const uint32_t w[3] = { s0, s1, s2 };
    #define SB(i) ((uint32_t)((w[(i)>>2] >> (8*((i)&3))) & 0xFFu))
    if(j < 4){ *sc = (float)(SB(j) & 63u); *mn = (float)(SB(j+4) & 63u); }
    else     { *sc = (float)((SB(j+4) & 0xFu) | ((SB(j-4) >> 6) << 4));
               *mn = (float)((SB(j+4) >> 4)   | ((SB(j)   >> 6) << 4)); }
    #undef SB
}

__global__ static void k_gemv_q4k_r(float * __restrict__ y, const float * __restrict__ x,
                                    const uint8_t * __restrict__ W, int I, int O){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x*NWARP + wid)*RPW;
    if(row0 >= O) return;
    const int nr = (O - row0 < RPW) ? (O - row0) : RPW;
    const int nsb = I >> 8;
    const uint8_t *w0 = W + (size_t)row0*(size_t)nsb*144;
    const size_t rstride = (size_t)nsb*144;

    float acc[RPW];
    #pragma unroll
    for(int r = 0; r < RPW; r++) acc[r] = 0.f;

    for(int sb = 0; sb < nsb; ++sb){
        const float *xs = x + sb*256;
        float xv[8];
        #pragma unroll
        for(int u = 0; u < 8; ++u) xv[u] = xs[u*32 + lane];
        #pragma unroll
        for(int r = 0; r < RPW; ++r){
            if(r >= nr) break;
            const uint8_t *bp = w0 + r*rstride + (size_t)sb*144;
            const uint32_t dd = __ldg((const uint32_t*)bp);
            const float d  = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFFu)));
            const float mn = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
            const uint32_t s0 = __ldg((const uint32_t*)(bp + 4));
            const uint32_t s1 = __ldg((const uint32_t*)(bp + 8));
            const uint32_t s2 = __ldg((const uint32_t*)(bp + 12));
            #pragma unroll
            for(int g = 0; g < 4; ++g){
                float c1, m1, c2, m2;
                gsm4_reg(s0, s1, s2, 2*g,   &c1, &m1);
                gsm4_reg(s0, s1, s2, 2*g+1, &c2, &m2);
                const uint8_t q = bp[16 + g*32 + lane];
                acc[r] += (d*c1*(float)(q & 0xF) - mn*m1)*xv[2*g]
                        + (d*c2*(float)(q >> 4)  - mn*m2)*xv[2*g+1];
            }
        }
    }
    #pragma unroll
    for(int r = 0; r < RPW; ++r){
        if(r >= nr) break;
        const float t = warp_sum(acc[r]);
        if(lane == 0) y[row0 + r] = t;
    }
}

/* THE NARROW SHAPE
 *
 * ssm_alpha and ssm_beta are [5120, 48]: 132 KiB of weights, and the row-blocked kernel above
 * put twelve warps on the whole card to read them. Measured 50 us — per layer, twice, across
 * 48 layers that is 4.8 ms a token spent on 12 MiB. It is not bandwidth, it is that there is
 * nothing to hide the load latency behind.
 *
 * So for narrow outputs the decomposition flips: one BLOCK per row, warps split the row's
 * super-blocks, and the block reduces at the end. 48 rows becomes 48 blocks of 8 warps
 * instead of 12 warps total. */
__global__ static void k_gemv_q4k_narrow(float * __restrict__ y, const float * __restrict__ x,
                                         const uint8_t * __restrict__ W, int I, int O){
    __shared__ float red[NWARP];
    const int row = blockIdx.x;
    if(row >= O) return;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nsb = I >> 8;
    const uint8_t *w = W + (size_t)row*(size_t)nsb*144;

    float acc = 0.f;
    for(int sb = wid; sb < nsb; sb += NWARP){
        const uint8_t *bp = w + (size_t)sb*144;
        const float *xs = x + sb*256;
        const uint32_t dd = __ldg((const uint32_t*)bp);
        const float d  = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFFu)));
        const float mn = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
        const uint32_t s0 = __ldg((const uint32_t*)(bp + 4));
        const uint32_t s1 = __ldg((const uint32_t*)(bp + 8));
        const uint32_t s2 = __ldg((const uint32_t*)(bp + 12));
        #pragma unroll
        for(int g = 0; g < 4; ++g){
            float c1, m1, c2, m2;
            gsm4_reg(s0, s1, s2, 2*g,   &c1, &m1);
            gsm4_reg(s0, s1, s2, 2*g+1, &c2, &m2);
            const uint8_t q = bp[16 + g*32 + lane];
            const float xa = xs[g*64 + lane], xc = xs[g*64 + 32 + lane];
            acc += (d*c1*(float)(q & 0xF) - mn*m1)*xa
                 + (d*c2*(float)(q >> 4)  - mn*m2)*xc;
        }
    }
    acc = warp_sum(acc);
    if(lane == 0) red[wid] = acc;
    __syncthreads();
    if(threadIdx.x == 0){
        float t = 0;
        #pragma unroll
        for(int i = 0; i < NWARP; i++) t += red[i];
        y[row] = t;
    }
}

__global__ static void k_gemv_q6k_narrow(float * __restrict__ y, const float * __restrict__ x,
                                         const uint8_t * __restrict__ W, int I, int O){
    __shared__ float red[NWARP];
    const int row = blockIdx.x;
    if(row >= O) return;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nsb = I >> 8, is = lane >> 4;
    const uint8_t *w = W + (size_t)row*(size_t)nsb*210;
    float acc = 0.f;
    for(int sb = wid; sb < nsb; sb += NWARP){
        const b_q6_K *b = (const b_q6_K*)(w + (size_t)sb*210);
        const float d = __half2float(b->d);
        #pragma unroll
        for(int n = 0; n < 2; ++n){
            const uint8_t *ql = b->ql + n*64, *qh = b->qh + n*32;
            const int8_t *sc = b->scales + n*8;
            const float *xb = x + sb*256 + n*128;
            const uint8_t l0 = ql[lane], l1 = ql[lane+32], h = qh[lane];
            const float q1 = (float)((int)((l0 & 0xF) | (((h >> 0) & 3) << 4)) - 32);
            const float q2 = (float)((int)((l1 & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            const float q3 = (float)((int)((l0 >>  4) | (((h >> 4) & 3) << 4)) - 32);
            const float q4 = (float)((int)((l1 >>  4) | (((h >> 6) & 3) << 4)) - 32);
            acc += d*sc[is+0]*q1*xb[lane] + d*sc[is+2]*q2*xb[lane+32]
                 + d*sc[is+4]*q3*xb[lane+64] + d*sc[is+6]*q4*xb[lane+96];
        }
    }
    acc = warp_sum(acc);
    if(lane == 0) red[wid] = acc;
    __syncthreads();
    if(threadIdx.x == 0){
        float t = 0;
        #pragma unroll
        for(int i = 0; i < NWARP; i++) t += red[i];
        y[row] = t;
    }
}

__global__ static void k_gemv_q6k_r(float *y, const float *x, const uint8_t *W, int I, int O){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x*NWARP + wid)*RPW;
    if(row0 >= O) return;
    const int nr = (O - row0 < RPW) ? (O - row0) : RPW;
    const int nsb = I >> 8;
    const uint8_t *w0 = W + (size_t)row0*(size_t)nsb*210;
    const size_t rstride = (size_t)nsb*210;
    const int is = lane >> 4;

    float acc[RPW];
    #pragma unroll
    for(int r = 0; r < RPW; r++) acc[r] = 0.f;

    for(int sb = 0; sb < nsb; ++sb){
        #pragma unroll
        for(int n = 0; n < 2; ++n){
            const float *xb = x + sb*256 + n*128;
            const float x0 = xb[lane], x1 = xb[lane+32], x2 = xb[lane+64], x3 = xb[lane+96];
            #pragma unroll
            for(int r = 0; r < RPW; ++r){
                if(r >= nr) break;
                const b_q6_K *b = (const b_q6_K*)(w0 + r*rstride + (size_t)sb*210);
                const uint8_t *ql = b->ql + n*64;
                const uint8_t *qh = b->qh + n*32;
                const int8_t  *sc = b->scales + n*8;
                const float d = __half2float(b->d);
                const uint8_t l0 = ql[lane], l1 = ql[lane+32], h = qh[lane];
                const float q1 = (float)((int)((l0 & 0xF) | (((h >> 0) & 3) << 4)) - 32);
                const float q2 = (float)((int)((l1 & 0xF) | (((h >> 2) & 3) << 4)) - 32);
                const float q3 = (float)((int)((l0 >>  4) | (((h >> 4) & 3) << 4)) - 32);
                const float q4 = (float)((int)((l1 >>  4) | (((h >> 6) & 3) << 4)) - 32);
                acc[r] += d*sc[is+0]*q1*x0 + d*sc[is+2]*q2*x1
                        + d*sc[is+4]*q3*x2 + d*sc[is+6]*q4*x3;
            }
        }
    }
    #pragma unroll
    for(int r = 0; r < RPW; ++r){
        if(r >= nr) break;
        const float t = warp_sum(acc[r]);
        if(lane == 0) y[row0 + r] = t;
    }
}

/* The S > 1 path: prefill. The weight bytes are decoded ONCE and reused for every token in
 * the batch, which is the whole point — at S=8 a super-block is fetched and unpacked once and
 * feeds 8 dot products instead of 1. Rows are not blocked here because acc[RPW][SMAX] would
 * spill. */
__global__ static void k_gemv_q4k(float *y, const float *x, const uint8_t *W,
                                  int I, int O, int S, int unused){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x*NWARP + wid;
    if(row >= O) return;
    const int nsb = I >> 8;
    const uint8_t *w = W + (size_t)row*(size_t)nsb*144;
    float acc[SMAX];
    #pragma unroll
    for(int s = 0; s < SMAX; s++) acc[s] = 0.f;
    for(int sb = 0; sb < nsb; ++sb){
        const b_q4_K *b = (const b_q4_K*)(w + (size_t)sb*144);
        const float d = __half2float(b->d), mn = __half2float(b->dmin);
        #pragma unroll
        for(int g = 0; g < 4; ++g){
            float s1, m1, s2, m2;
            gsm4(b->scales, 2*g,   &s1, &m1);
            gsm4(b->scales, 2*g+1, &s2, &m2);
            const float d1 = d*s1, mm1 = mn*m1, d2 = d*s2, mm2 = mn*m2;
            const uint8_t q = b->qs[g*32 + lane];
            const float lo = (float)(q & 0xF), hi = (float)(q >> 4);
            for(int s = 0; s < S; ++s){
                const float *xb = x + (size_t)s*I + sb*256 + g*64;
                const float xa = xb[lane], xc = xb[32 + lane];
                acc[s] += (d1*lo - mm1)*xa + (d2*hi - mm2)*xc;
            }
        }
    }
    for(int s = 0; s < S; ++s){
        const float t = warp_sum(acc[s]);
        if(lane == 0) y[(size_t)s*O + row] = t;
    }
}

__global__ static void k_gemv_q6k(float *y, const float *x, const uint8_t *W,
                                  int I, int O, int S, int unused){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x*NWARP + wid;
    if(row >= O) return;
    const int nsb = I >> 8;
    const uint8_t *w = W + (size_t)row*(size_t)nsb*210;
    const int is = lane >> 4;
    float acc[SMAX];
    #pragma unroll
    for(int s = 0; s < SMAX; s++) acc[s] = 0.f;
    for(int sb = 0; sb < nsb; ++sb){
        const b_q6_K *b = (const b_q6_K*)(w + (size_t)sb*210);
        const float d = __half2float(b->d);
        #pragma unroll
        for(int n = 0; n < 2; ++n){
            const uint8_t *ql = b->ql + n*64;
            const uint8_t *qh = b->qh + n*32;
            const int8_t  *sc = b->scales + n*8;
            const uint8_t l0 = ql[lane], l1 = ql[lane+32], h = qh[lane];
            const float q1 = (float)((int)((l0 & 0xF) | (((h >> 0) & 3) << 4)) - 32);
            const float q2 = (float)((int)((l1 & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            const float q3 = (float)((int)((l0 >>  4) | (((h >> 4) & 3) << 4)) - 32);
            const float q4 = (float)((int)((l1 >>  4) | (((h >> 6) & 3) << 4)) - 32);
            const float a1 = d*sc[is+0]*q1, a2 = d*sc[is+2]*q2;
            const float a3 = d*sc[is+4]*q3, a4 = d*sc[is+6]*q4;
            for(int s = 0; s < S; ++s){
                const float *xb = x + (size_t)s*I + sb*256 + n*128;
                acc[s] += a1*xb[lane] + a2*xb[lane+32] + a3*xb[lane+64] + a4*xb[lane+96];
            }
        }
    }
    for(int s = 0; s < S; ++s){
        const float t = warp_sum(acc[s]);
        if(lane == 0) y[(size_t)s*O + row] = t;
    }
}

__global__ static void k_gemv_f32(float *y, const float *x, const float *W, int I, int O, int S){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x*NWARP + wid;
    if(row >= O) return;
    const float *w = W + (size_t)row*I;
    for(int s = 0; s < S; ++s){
        float a = 0.f;
        for(int i = lane; i < I; i += 32) a += w[i]*x[(size_t)s*I + i];
        a = warp_sum(a);
        if(lane == 0) y[(size_t)s*O + row] = a;
    }
}

__global__ static void k_gemv_f16(float *y, const float *x, const __half *W, int I, int O, int S){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x*NWARP + wid;
    if(row >= O) return;
    const __half *w = W + (size_t)row*I;
    for(int s = 0; s < S; ++s){
        float a = 0.f;
        for(int i = lane; i < I; i += 32) a += __half2float(w[i])*x[(size_t)s*I + i];
        a = warp_sum(a);
        if(lane == 0) y[(size_t)s*O + row] = a;
    }
}

__global__ static void k_gemv_q80(float *y, const float *x, const uint8_t *W, int I, int O, int S){
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x*NWARP + wid;
    if(row >= O) return;
    const int nb = I >> 5;
    const uint8_t *w = W + (size_t)row*(size_t)nb*34;
    for(int s = 0; s < S; ++s){
        float a = 0.f;
        for(int b = 0; b < nb; ++b){
            const b_q8_0 *q = (const b_q8_0*)(w + (size_t)b*34);
            a += __half2float(q->d) * (float)q->qs[lane] * x[(size_t)s*I + b*32 + lane];
        }
        a = warp_sum(a);
        if(lane == 0) y[(size_t)s*O + row] = a;
    }
}

extern "C" int q35cu_type_ok(int t){
    return t == CU_F32 || t == CU_F16 || t == CU_Q8_0 || t == CU_Q4_K || t == CU_Q6_K;
}

/* Staging x in shared cuts the L2 traffic of a gemv by the block count. It only pays for
 * S == 1 (a batch would need S*I floats and blow the 100 KB budget) and only when x fits. */
extern "C" int q35cu_gemm(float *y, const float *x, const void *W, int gt, int I, int O, int S){
    if(!g_ready){ seterr("gemm", "cuda not initialized"); return 0; }
    if(S < 1 || S > SMAX){ seterr("gemm", "S out of range"); return 0; }
    const int grid = (O + NWARP - 1)/NWARP;
    if(gt == CU_Q4_K || gt == CU_Q6_K){
        if(I & 255){ seterr("gemm", "k-quant row length must be a multiple of 256"); return 0; }
        if(S == 1){
            /* Below this the card is not busy: O rows spread over O/(8*RPW) blocks stops
             * filling 36 SMs long before it stops being memory-bound. */
            if(O <= NARROW){
                if(gt == CU_Q4_K) k_gemv_q4k_narrow<<<O, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O);
                else              k_gemv_q6k_narrow<<<O, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O);
            } else {
                const int gr = (O + NWARP*RPW - 1)/(NWARP*RPW);
                if(gt == CU_Q4_K) k_gemv_q4k_r<<<gr, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O);
                else              k_gemv_q6k_r<<<gr, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O);
            }
        } else if(gt == CU_Q4_K) k_gemv_q4k<<<grid, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O, S, 0);
        else                     k_gemv_q6k<<<grid, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O, S, 0);
    } else if(gt == CU_F32){
        k_gemv_f32<<<grid, NTHR, 0, g_s>>>(y, x, (const float*)W, I, O, S);
    } else if(gt == CU_F16){
        k_gemv_f16<<<grid, NTHR, 0, g_s>>>(y, x, (const __half*)W, I, O, S);
    } else if(gt == CU_Q8_0){
        if(I & 31){ seterr("gemm", "q8_0 row length must be a multiple of 32"); return 0; }
        k_gemv_q80<<<grid, NTHR, 0, g_s>>>(y, x, (const uint8_t*)W, I, O, S);
    } else {
        snprintf(g_err, sizeof g_err, "gemm: type %d has no device kernel", gt);
        return 0;
    }
    return 1;
}

extern "C" int q35cu_gemv(float *y, const float *x, const void *W, int gt, int I, int O){
    return q35cu_gemm(y, x, W, gt, I, O, 1);
}

/* ---------------- elementwise ---------------- */

__global__ static void k_rmsnorm(float *out, const float *x, const float *w, int n, float eps){
    __shared__ double sh[32];
    double s = 0;
    for(int i = threadIdx.x; i < n; i += blockDim.x){ const double v = x[i]; s += v*v; }
    s = block_sum(s, sh);
    const float r = 1.0f/sqrtf((float)(s/n) + eps);
    for(int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i]*r*w[i];
}

__global__ static void k_add(float *x, const float *t, int n){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] += t[i];
}
__global__ static void k_scale_add(float *x, const float *t, float a, int n){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] += a*t[i];
}
__global__ static void k_swiglu(float *g, const float *u, int n){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n){ const float v = g[i]; g[i] = (v/(1.0f + expf(-v)))*u[i]; }
}

extern "C" void q35cu_rmsnorm(float *o, const float *x, const float *w, int n, float eps){
    const int thr = n >= 1024 ? 1024 : ((n + 31)/32)*32;
    k_rmsnorm<<<1, thr, 0, g_s>>>(o, x, w, n, eps);
}
extern "C" void q35cu_add(float *x, const float *t, int n){
    k_add<<<(n+255)/256, 256, 0, g_s>>>(x, t, n);
}
extern "C" void q35cu_scale_add(float *x, const float *t, float a, int n){
    k_scale_add<<<(n+255)/256, 256, 0, g_s>>>(x, t, a, n);
}
extern "C" void q35cu_swiglu(float *g, const float *u, int n){
    k_swiglu<<<(n+255)/256, 256, 0, g_s>>>(g, u, n);
}

/* ---------------- gated delta net ---------------- */

__global__ static void k_gdn_conv(float *qkv, float *cs, const float *w, int cd, int d_conv){
    const int ch = blockIdx.x*blockDim.x + threadIdx.x;
    if(ch >= cd) return;
    const int K1 = d_conv - 1;
    const float xin = qkv[ch];                       /* PRE-conv: this is what the ring keeps */
    const float *kk = w + (size_t)ch*d_conv;         /* [d_conv, cd], d_conv fastest */
    float acc = 0.f;
    for(int j = 0; j < K1; j++) acc += kk[j]*cs[(size_t)j*cd + ch];
    acc += kk[K1]*xin;
    for(int j = 0; j < K1-1; j++) cs[(size_t)j*cd + ch] = cs[(size_t)(j+1)*cd + ch];
    cs[(size_t)(K1-1)*cd + ch] = xin;
    qkv[ch] = acc/(1.0f + expf(-acc));               /* silu */
}

__global__ static void k_gdn_gates(float *beta, float *gdec, const float *dt, const float *a, int nv){
    const int h = blockIdx.x*blockDim.x + threadIdx.x;
    if(h >= nv) return;
    beta[h] = 1.0f/(1.0f + expf(-beta[h]));
    const float x = gdec[h] + dt[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));   /* softplus, guarded like the CPU */
    gdec[h] = expf(sp * a[h]);                          /* a already holds -exp(A_log) < 0 */
}

/* ggml_l2_norm: x / max(sqrt(sum x^2), eps). The sum is NOT divided by n — that missing 1/n
 * is the whole difference from rmsnorm and it is worth sqrt(128) here. */
__global__ static void k_gdn_l2norm(float *v, int dk, float eps){
    __shared__ double sh[32];
    float *p = v + (size_t)blockIdx.x*dk;
    double s = 0;
    for(int i = threadIdx.x; i < dk; i += blockDim.x){ const double t = p[i]; s += t*t; }
    s = block_sum(s, sh);
    const float r = 1.0f/fmaxf(sqrtf((float)s), eps);
    for(int i = threadIdx.x; i < dk; i += blockDim.x) p[i] *= r;
}

/* One block per v-head, thread j owns COLUMN j of S. S[i*dv + j] with consecutive j in
 * consecutive addresses makes every state access a coalesced 512-byte transaction. */
__global__ static void k_gdn_delta(float *S, float *o, const float *q, const float *k,
                                   const float *v, const float *beta, const float *gdec,
                                   int nv, int nk, int dk, int dv, int tile){
    extern __shared__ float sh[];
    float *skh = sh, *sqh = sh + dk;
    const int h = blockIdx.x, j = threadIdx.x;
    const int hk = tile ? (h % nk) : (h / (nv/nk));
    const float qscale = 1.0f/sqrtf((float)dk);
    for(int i = j; i < dk; i += blockDim.x){
        skh[i] = k[(size_t)hk*dk + i];
        sqh[i] = q[(size_t)hk*dk + i]*qscale;        /* pre-scaled: matches the CPU's order */
    }
    __syncthreads();

    float *Sh = S + (size_t)h*dk*dv;
    const float g = gdec[h], b = beta[h];

    float kv = 0.f;
    for(int i = 0; i < dk; i++){
        const size_t p = (size_t)i*dv + j;
        const float s = Sh[p]*g;
        Sh[p] = s;
        kv += s*skh[i];
    }
    const float d = b*(v[(size_t)h*dv + j] - kv);
    float acc = 0.f;
    for(int i = 0; i < dk; i++){
        const size_t p = (size_t)i*dv + j;
        const float s = Sh[p] + skh[i]*d;
        Sh[p] = s;
        acc += s*sqh[i];
    }
    o[(size_t)h*dv + j] = acc;
}

__global__ static void k_gdn_norm_gate(float *o, const float *z, const float *w, int dv, float eps){
    __shared__ double sh[32];
    float *p = o + (size_t)blockIdx.x*dv;
    const float *zh = z + (size_t)blockIdx.x*dv;
    double s = 0;
    for(int i = threadIdx.x; i < dv; i += blockDim.x){ const double t = p[i]; s += t*t; }
    s = block_sum(s, sh);
    const float r = 1.0f/sqrtf((float)(s/dv) + eps);
    for(int i = threadIdx.x; i < dv; i += blockDim.x){
        const float zv = zh[i];
        p[i] = p[i]*r*w[i] * (zv/(1.0f + expf(-zv)));
    }
}

extern "C" void q35cu_gdn_conv(float *qkv, float *cs, const float *w, int cd, int d_conv){
    k_gdn_conv<<<(cd+255)/256, 256, 0, g_s>>>(qkv, cs, w, cd, d_conv);
}
extern "C" void q35cu_gdn_gates(float *b, float *g, const float *dt, const float *a, int nv){
    k_gdn_gates<<<(nv+63)/64, 64, 0, g_s>>>(b, g, dt, a, nv);
}
extern "C" void q35cu_gdn_l2norm(float *qk, int n_head, int dk, float eps){
    const int thr = dk >= 256 ? 256 : ((dk + 31)/32)*32;
    k_gdn_l2norm<<<n_head, thr, 0, g_s>>>(qk, dk, eps);
}
extern "C" void q35cu_gdn_delta(float *S, float *o, const float *q, const float *k, const float *v,
                                const float *beta, const float *gdec,
                                int nv, int nk, int dk, int dv, int tile){
    k_gdn_delta<<<nv, dv, (size_t)2*dk*sizeof(float), g_s>>>(S, o, q, k, v, beta, gdec,
                                                             nv, nk, dk, dv, tile);
}
extern "C" void q35cu_gdn_norm_gate(float *o, const float *z, const float *w, int nv, int dv, float eps){
    const int thr = dv >= 256 ? 256 : ((dv + 31)/32)*32;
    k_gdn_norm_gate<<<nv, thr, 0, g_s>>>(o, z, w, dv, eps);
}

/* ---------------- full attention ---------------- */

/* Per-head QK-rmsnorm then PARTIAL NeoX rope: only dims [0, n_rot) rotate, pairing i with
 * i + n_rot/2; [n_rot, d_head) passes through untouched.
 *
 * theta is accumulated in DOUBLE here where the CPU multiplies a float by a float ratio
 * n_rot/2 times. At pos=250000 that difference is real: the CPU's theta_1 carries ~1e-2 rad
 * of accumulated rounding. Being more accurate than the reference is safe — the short-context
 * gate compares at pos<10 where both are exact to 1e-7. */
__global__ static void k_attn_qk(float *qg, float *k, const float *qn, const float *kn,
                                 int nh, int dh, int n_rot, float base, int pos, float eps){
    __shared__ double sh[32];
    const int b = blockIdx.x;
    float *p; const float *w;
    if(b < nh){ p = qg + (size_t)b*2*dh; w = qn; }
    else      { p = k  + (size_t)(b-nh)*dh; w = kn; }

    double s = 0;
    for(int i = threadIdx.x; i < dh; i += blockDim.x){ const double t = p[i]; s += t*t; }
    s = block_sum(s, sh);
    const float r = 1.0f/sqrtf((float)(s/dh) + eps);
    for(int i = threadIdx.x; i < dh; i += blockDim.x) p[i] = p[i]*r*w[i];
    __syncthreads();

    const int half = n_rot/2;
    for(int i = threadIdx.x; i < half; i += blockDim.x){
        const double th = (double)pos * pow((double)base, -2.0*(double)i/(double)n_rot);
        const float c = (float)cos(th), sn = (float)sin(th);
        const float x0 = p[i], x1 = p[i+half];
        p[i]      = x0*c - x1*sn;
        p[i+half] = x0*sn + x1*c;
    }
}

extern "C" void q35cu_attn_qk(float *qg, float *k, const float *qn, const float *kn,
                              int nh, int nkv, int dh, int n_rot, float base, int pos, float eps){
    const int thr = dh >= 256 ? 256 : ((dh + 31)/32)*32;
    k_attn_qk<<<nh + nkv, thr, 0, g_s>>>(qg, k, qn, kn, nh, dh, n_rot, base, pos, eps);
}

/* fp32 -> fp16 EXACTLY as kv_tier_simd.h's kvt_f2h_bits does it: the mantissa is TRUNCATED
 * (m >> 13), not rounded to nearest. __float2half rounds, and that one-ulp disagreement on
 * the block scale is invisible per element but reads as 5e-4 in the attention output — which
 * looks exactly like a wiring bug and is not one.
 *
 * The host format wins, not because truncation is better (it is not), but because the two
 * tiers must be INTERCHANGEABLE: a spilled chunk gets DMA'd from RAM into the device window
 * and back, and a token whose value changes when it migrates is a real inconsistency. */
__device__ __forceinline__ uint16_t f2h_trunc(float f){
    const uint32_t b = __float_as_uint(f);
    const uint32_t sg = (b >> 16) & 0x8000u;
    const int32_t  e  = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    const uint32_t m  = b & 0x7FFFFFu;
    if(e <= 0)  return (uint16_t)sg;
    if(e >= 31) return (uint16_t)(sg | 0x7C00u);
    return (uint16_t)(sg | ((uint32_t)e << 10) | (m >> 13));
}

/* q8_0 pack, byte-identical to kvt_pack_q8: fp16 scale then 32 int8, amax/127, RNE rounding. */
__global__ static void k_kv_store_q8(uint8_t *dst, const float *src, int n){
    const int b = blockIdx.x*(blockDim.x/32) + (threadIdx.x >> 5);
    if(b >= n/32) return;
    const int lane = threadIdx.x & 31;
    const float x = src[b*32 + lane];
    float a = warp_max(fabsf(x));
    a = __shfl_sync(FULL, a, 0);
    const float d = a/127.0f, id = d ? 1.0f/d : 0.0f;
    int q = __float2int_rn(x*id);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    uint8_t *p = dst + (size_t)b*34;
    if(lane == 0) *(uint16_t*)p = f2h_trunc(d);
    p[2 + lane] = (uint8_t)(int8_t)q;
}
__global__ static void k_kv_store_f16(uint16_t *dst, const float *src, int n){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) dst[i] = f2h_trunc(src[i]);
}

extern "C" void q35cu_kv_store(void *Kc, void *Vc, const float *k, const float *v,
                               int64_t koff, int64_t voff, int n_kv, int fmt){
    uint8_t *kp = (uint8_t*)Kc + koff, *vp = (uint8_t*)Vc + voff;
    if(fmt == 2){
        const int nb = n_kv/32, grid = (nb + NWARP - 1)/NWARP;
        k_kv_store_q8<<<grid, NTHR, 0, g_s>>>(kp, k, n_kv);
        k_kv_store_q8<<<grid, NTHR, 0, g_s>>>(vp, v, n_kv);
    } else {
        k_kv_store_f16<<<(n_kv+255)/256, 256, 0, g_s>>>((uint16_t*)kp, k, n_kv);
        k_kv_store_f16<<<(n_kv+255)/256, 256, 0, g_s>>>((uint16_t*)vp, v, n_kv);
    }
}

/* ---- flash decode ----
 *
 * ONE pass over the window with an online softmax. The two-pass form (scores, then a second
 * sweep for the weighted V sum) reads K once and V once, which is the same total bytes — but
 * it needs an att[nh][T] array, and at 250k that is 24 MB of VRAM that the KV window wants.
 *
 * Shape: block = 8 warps. In the score phase warp w takes timestep t0+w, so eight scores are
 * produced per round; in the accumulate phase all 256 threads switch to owning one output
 * element each. m and l are kept in registers, computed redundantly and identically by every
 * thread, which removes a shared-memory round trip per round. */
#define NSPLIT_MAX 128
#define DHMAX 4                 /* dh <= 4*NTHR */

__device__ __forceinline__ float kv_dot(const uint8_t *base, const float *q, int dh, int fmt, int lane){
    float a = 0.f;
    if(fmt == 2){
        const int nb = dh >> 5;
        for(int b = 0; b < nb; ++b){
            const b_q8_0 *blk = (const b_q8_0*)(base + (size_t)b*34);
            a += __half2float(blk->d) * (float)blk->qs[lane] * q[b*32 + lane];
        }
    } else {
        const __half *h = (const __half*)base;
        for(int i = lane; i < dh; i += 32) a += __half2float(h[i])*q[i];
    }
    return warp_sum(a);
}
__device__ __forceinline__ float kv_get(const uint8_t *base, int j, int fmt){
    if(fmt == 2){
        const b_q8_0 *blk = (const b_q8_0*)(base + (size_t)(j >> 5)*34);
        return __half2float(blk->d) * (float)blk->qs[j & 31];
    }
    return __half2float(((const __half*)base)[j]);
}

__global__ static void k_attn_flash(float *opart, float *mpart, float *lpart,
                                    const float *qg, const uint8_t *Kc, const uint8_t *Vc,
                                    int nh, int nkv, int dh, int T,
                                    long long kstride, long long vstride,
                                    float scale, int fmt, int nsplit, int hb){
    extern __shared__ float shq[];
    __shared__ float sc[NWARP];
    const int h = blockIdx.x, sp = blockIdx.y;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int grp = nh/nkv, hk = h/grp;

    const float *qh = qg + (size_t)h*2*dh;
    for(int i = threadIdx.x; i < dh; i += blockDim.x) shq[i] = qh[i];
    __syncthreads();

    const int per = (T + nsplit - 1)/nsplit;
    const int tb = sp*per, te = (tb + per < T) ? tb + per : T;

    float acc[DHMAX];
    #pragma unroll
    for(int u = 0; u < DHMAX; u++) acc[u] = 0.f;
    float m = -INFINITY, l = 0.f;

    for(int t0 = tb; t0 < te; t0 += NWARP){
        const int t = t0 + wid;
        float s = -INFINITY;
        if(t < te)
            s = kv_dot(Kc + (size_t)t*kstride + (size_t)hk*hb, shq, dh, fmt, lane)*scale;
        if(lane == 0) sc[wid] = s;
        __syncthreads();

        float mn = m;
        #pragma unroll
        for(int w = 0; w < NWARP; w++) mn = fmaxf(mn, sc[w]);
        const float corr = (m == -INFINITY) ? 0.f : __expf(m - mn);
        #pragma unroll
        for(int u = 0; u < DHMAX; u++) acc[u] *= corr;
        l *= corr;
        #pragma unroll
        for(int w = 0; w < NWARP; w++){
            const float sw = sc[w];
            if(sw == -INFINITY) continue;
            const float p = __expf(sw - mn);
            l += p;
            const uint8_t *vp = Vc + (size_t)(t0+w)*vstride + (size_t)hk*hb;
            for(int u = 0, j = threadIdx.x; j < dh; j += blockDim.x, ++u)
                acc[u] += p*kv_get(vp, j, fmt);
        }
        m = mn;
        __syncthreads();
    }

    float *op = opart + ((size_t)h*nsplit + sp)*dh;
    for(int u = 0, j = threadIdx.x; j < dh; j += blockDim.x, ++u) op[j] = acc[u];
    if(threadIdx.x == 0){
        mpart[(size_t)h*nsplit + sp] = (te > tb) ? m : -INFINITY;
        lpart[(size_t)h*nsplit + sp] = (te > tb) ? l : 0.f;
    }
}

__global__ static void k_attn_merge(float *o, float *lse, const float *opart,
                                    const float *mpart, const float *lpart,
                                    int dh, int nsplit){
    const int h = blockIdx.x;
    const float *mp = mpart + (size_t)h*nsplit, *lp = lpart + (size_t)h*nsplit;
    float M = -INFINITY;
    for(int s = 0; s < nsplit; s++) M = fmaxf(M, mp[s]);
    float sum = 0.f;
    for(int s = 0; s < nsplit; s++)
        if(mp[s] != -INFINITY) sum += __expf(mp[s] - M)*lp[s];
    const float inv = sum > 0.f ? 1.0f/sum : 0.f;
    for(int j = threadIdx.x; j < dh; j += blockDim.x){
        float a = 0.f;
        for(int s = 0; s < nsplit; s++)
            if(mp[s] != -INFINITY) a += __expf(mp[s] - M)*opart[((size_t)h*nsplit + s)*dh + j];
        o[(size_t)h*dh + j] = a*inv;
    }
    if(threadIdx.x == 0) lse[h] = M + logf(sum > 0.f ? sum : 1.0f);
}

extern "C" size_t q35cu_attn_scratch(int nh, int dh, int T){
    (void)T;
    return (size_t)nh*NSPLIT_MAX*(size_t)(dh + 2);
}

static int attn_nsplit(int T){
    int n = (T + 511)/512;
    if(n < 1) n = 1;
    if(n > NSPLIT_MAX) n = NSPLIT_MAX;
    return n;
}

extern "C" void q35cu_attn_flash(float *o, float *lse, const float *qg,
                                 const void *Kc, const void *Vc,
                                 int nh, int nkv, int dh, int T, int64_t kstride, int64_t vstride,
                                 float scale, int fmt, float *scratch){
    if(T <= 0) return;
    const int nsplit = attn_nsplit(T);
    const int hb = (fmt == 2) ? (dh/32)*34 : dh*2;     /* bytes of ONE kv-head's K (or V) */
    float *opart = scratch;
    float *mpart = scratch + (size_t)nh*nsplit*dh;
    float *lpart = mpart + (size_t)nh*nsplit;
    dim3 grid(nh, nsplit);
    k_attn_flash<<<grid, NTHR, (size_t)dh*sizeof(float), g_s>>>(
        opart, mpart, lpart, qg, (const uint8_t*)Kc, (const uint8_t*)Vc,
        nh, nkv, dh, T, (long long)kstride, (long long)vstride, scale, fmt, nsplit, hb);
    const int thr = dh >= 256 ? 256 : ((dh + 31)/32)*32;
    k_attn_merge<<<nh, thr, 0, g_s>>>(o, lse, opart, mpart, lpart, dh, nsplit);
}

/* The per-head output gate: sigmoid of the SECOND half of this head's q slice.
 * attn_q is [d_model, n_head*d_head*2] interleaved per head — q_h0, gate_h0, q_h1, ... —
 * not [all q | all gates]. Reading it as the latter still produces grammatical English. */
__global__ static void k_attn_gate(float *o, const float *qg, int dh){
    const int h = blockIdx.x;
    const float *g = qg + (size_t)h*2*dh + dh;
    float *oh = o + (size_t)h*dh;
    for(int i = threadIdx.x; i < dh; i += blockDim.x)
        oh[i] *= 1.0f/(1.0f + expf(-g[i]));
}
extern "C" void q35cu_attn_gate(float *o, const float *qg, int nh, int dh){
    const int thr = dh >= 256 ? 256 : ((dh + 31)/32)*32;
    k_attn_gate<<<nh, thr, 0, g_s>>>(o, qg, dh);
}

/* Flash-decoding combine for a split context: (o,lse) absorbs (o2,lse2). Exact up to
 * rounding, which is what makes the CPU-path comparison stay a hard gate. */
__global__ static void k_lse_merge(float *o, float *lse, const float *o2, const float *l2, int dh){
    const int h = blockIdx.x;
    const float a = lse[h], b = l2[h];
    const float M = fmaxf(a, b);
    if(M == -INFINITY) return;
    const float ea = (a == -INFINITY) ? 0.f : __expf(a - M);
    const float eb = (b == -INFINITY) ? 0.f : __expf(b - M);
    const float s = ea + eb, inv = s > 0.f ? 1.0f/s : 0.f;
    float *oh = o + (size_t)h*dh;
    const float *o2h = o2 + (size_t)h*dh;
    for(int j = threadIdx.x; j < dh; j += blockDim.x)
        oh[j] = (oh[j]*ea + o2h[j]*eb)*inv;
    if(threadIdx.x == 0) lse[h] = M + logf(s > 0.f ? s : 1.0f);
}
extern "C" void q35cu_lse_merge(float *o, float *lse, const float *o2, const float *l2, int nh, int dh){
    const int thr = dh >= 256 ? 256 : ((dh + 31)/32)*32;
    k_lse_merge<<<nh, thr, 0, g_s>>>(o, lse, o2, l2, dh);
}

/* ---------------- bandwidth ---------------- */

__global__ static void k_stream(const float4 *src, float *sink, size_t n4){
    float4 a = make_float4(0,0,0,0);
    for(size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n4; i += gridDim.x*(size_t)blockDim.x){
        const float4 v = src[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if(a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;   /* never true; keeps the loads alive */
}

extern "C" double q35cu_measure_bw(void){
    if(!g_ready) return 0;
    const size_t bytes = 512u<<20;
    void *p = q35cu_alloc(bytes);
    if(!p) return 0;
    float *sink = (float*)q35cu_alloc(64);
    cudaMemsetAsync(p, 1, bytes, g_s);
    const size_t n4 = bytes/16;
    k_stream<<<1024, 256, 0, g_s>>>((const float4*)p, sink, n4);   /* warm up */
    cudaStreamSynchronize(g_s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, g_s);
    for(int r = 0; r < 8; r++) k_stream<<<1024, 256, 0, g_s>>>((const float4*)p, sink, n4);
    cudaEventRecord(b, g_s);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    q35cu_free(p); q35cu_free(sink);
    g_bw = ms > 0 ? (8.0*bytes/1e9)/(ms/1e3) : 0;
    return g_bw;
}

/* ---------------- in-sequence kernel timing ----------------
 * Timing a kernel in a loop by itself and timing it inside the layer are different
 * measurements: alone it gets the whole card and a warm L2, in the layer it inherits
 * whatever the previous kernel left. Only the second one predicts a token. */
#define NMARK 32
static cudaEvent_t g_ev[NMARK+1];
static const char *g_mark[NMARK];
static int g_nmark = 0, g_mark_on = 0;

extern "C" void q35cu_mark_begin(void){
    if(!g_ev[0]) for(int i = 0; i <= NMARK; i++) cudaEventCreate(&g_ev[i]);
    g_nmark = 0; g_mark_on = 1;
    cudaEventRecord(g_ev[0], g_s);
}
extern "C" void q35cu_mark(const char *name){
    if(!g_mark_on || g_nmark >= NMARK) return;
    g_mark[g_nmark] = name;
    cudaEventRecord(g_ev[g_nmark+1], g_s);
    g_nmark++;
}
extern "C" int q35cu_mark_read(int i, const char **name, float *ms){
    if(i >= g_nmark) return 0;
    cudaEventSynchronize(g_ev[i+1]);
    cudaEventElapsedTime(ms, g_ev[i], g_ev[i+1]);
    *name = g_mark[i];
    return 1;
}
extern "C" void q35cu_mark_end(void){ g_mark_on = 0; }

/* ---------------- batched helpers ---------------- */

__global__ static void k_rmsnorm_rows(float *out, const float *x, const float *w,
                                      int n, float eps){
    __shared__ double sh[32];
    const size_t off = (size_t)blockIdx.x*n;
    const float *xr = x + off; float *o = out + off;
    double s = 0;
    for(int i = threadIdx.x; i < n; i += blockDim.x){ const double v = xr[i]; s += v*v; }
    s = block_sum(s, sh);
    const float r = 1.0f/sqrtf((float)(s/n) + eps);
    for(int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i]*r*w[i];
}
extern "C" void q35cu_rmsnorm_rows(float *o, const float *x, const float *w,
                                   int n, int S, float eps){
    const int thr = n >= 1024 ? 1024 : ((n + 31)/32)*32;
    k_rmsnorm_rows<<<S, thr, 0, g_s>>>(o, x, w, n, eps);
}

extern "C" int q35cu_d2d(void *dst, const void *src, size_t n){
    CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_s)); return 1;
}

/* ---------------- pinning host weights for real async DMA ----------------
 * Pageable H2D measured 6.8 GB/s on this machine, pinned 27.7 — a 4x difference, and the
 * whole streaming design depends on the second number. The weights already live in one
 * anonymous mapping, so it is one registration, not one per tensor. */
extern "C" int q35cu_host_register(void *p, size_t n){
    cudaError_t e = cudaHostRegister(p, n, cudaHostRegisterPortable);
    if(e == cudaErrorHostMemoryAlreadyRegistered) return 1;
    if(e != cudaSuccess){ seterr("cudaHostRegister", cudaGetErrorString(e)); return 0; }
    return 1;
}
extern "C" void q35cu_host_unregister(void *p){ if(p) cudaHostUnregister(p); }

/* A copy issued on the compute stream, so it orders against the kernels rather than racing
 * them. Used for the small activation exchanges inside a split FFN layer. */
extern "C" int q35cu_h2d_compute(void *d, const void *s, size_t n){
    CK(cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, g_s)); return 1;
}
extern "C" int q35cu_d2h_compute(void *d, const void *s, size_t n){
    CK(cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToHost, g_s)); return 1;
}

/* Strided host->device copy. The down projection is [d_ff, d_model] with the CONTRACTED dim
 * fastest, so giving the GPU a share of the contraction means a column band of every row --
 * 5120 rows of ~6 KB. cudaMemcpy2DAsync does exactly that in one descriptor. */
extern "C" int q35cu_h2d_2d_async(void *dst, size_t dpitch, const void *src, size_t spitch,
                                  size_t width, size_t height){
    CK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                         cudaMemcpyHostToDevice, g_cp));
    return 1;
}

/* Copy-stream events. The GPU must wait for its weight slice, but the HOST must not -- if the
 * host blocked here it would stop computing the CPU's half of the same layer, which is the
 * whole point. cudaStreamWaitEvent makes the dependency a GPU-side one and returns instantly. */
#define NCPEV 4
static cudaEvent_t g_cpev[NCPEV];     /* copy done  -> compute may read  */
static cudaEvent_t g_csev[NCPEV];     /* compute done -> copy may overwrite */
extern "C" void q35cu_copy_mark(int slot){
    if(slot < 0 || slot >= NCPEV) return;
    if(!g_cpev[slot]) cudaEventCreateWithFlags(&g_cpev[slot], cudaEventDisableTiming);
    cudaEventRecord(g_cpev[slot], g_cp);
}
extern "C" void q35cu_stream_join(int slot){
    if(slot < 0 || slot >= NCPEV || !g_cpev[slot]) return;
    cudaStreamWaitEvent(g_s, g_cpev[slot], 0);
}

/* The other half of the double buffer, and the one that was missing.
 *
 * A staging half is written by the copy stream and read by the compute stream. Making compute
 * wait for the copy is obvious and was there. Making the COPY wait for compute is not, and
 * without it the host can issue the next layer's DMA into a half whose gemms are still
 * running — the weights change underneath a kernel that is mid-flight.
 *
 * It is timing-dependent, so it hides: at a 104 MiB slice the DMA was slow enough that the
 * gemms had always finished, and at 145 MiB it was not. The symptom was a batch that
 * disagreed with sequential decode by 1e-1 while still producing fluent text. */
extern "C" void q35cu_compute_mark(int slot){
    if(slot < 0 || slot >= NCPEV) return;
    if(!g_csev[slot]) cudaEventCreateWithFlags(&g_csev[slot], cudaEventDisableTiming);
    cudaEventRecord(g_csev[slot], g_s);
}
extern "C" void q35cu_copy_join(int slot){
    if(slot < 0 || slot >= NCPEV || !g_csev[slot]) return;
    cudaStreamWaitEvent(g_cp, g_csev[slot], 0);
}
