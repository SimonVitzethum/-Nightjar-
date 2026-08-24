/* kquant_cuda.cu — GEMV straight off GGUF quant bytes, on the GPU.
 *
 * Why this exists: streaming GLM-5.2-UD-Q2_K_XL measured 5.26 s/token of disk and
 * 10.52 s/token of CPU matmul. The 2-bit weights are half the bytes of colibri's int4,
 * but decoding an i-quant lattice in scalar C costs more than the disk time it saves —
 * so the compression was a net LOSS. The decode has to move somewhere with spare ALUs.
 *
 * The GPU is the natural home for it: the expert arrives from NVMe as ~12-16 MB of
 * quantized bytes, goes over PCIe AS IS (no dequant on the host, no fp16 blow-up), and
 * the lattice decode happens in registers right before the dot product. Nothing is ever
 * materialized as a weight matrix — not in RAM, not in VRAM.
 *
 * Layout mirrors kquant.h exactly; the codebooks live in __constant__ (they total ~5 KB).
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kquant_cuda.h"

#define QK_K 256

/* ---- codebooks in constant memory (identical values to iq_grids.h) ---- */
__constant__ uint64_t c_iq2xs_grid[512];
__constant__ uint32_t c_iq3xxs_grid[256];
__constant__ uint8_t  c_ksigns[128];
__constant__ uint8_t  c_kmask[8];
__constant__ int8_t   c_kvalues_iq4nl[16];

/* ---- block layouts (byte-identical to kquant.h / ggml-common.h) ---- */
typedef struct { uint8_t  scales[16]; uint8_t qs[64];  __half d, dmin; } b_q2_K;    /* 84  */
typedef struct { uint8_t  hmask[32];  uint8_t qs[64];  uint8_t scales[12]; __half d; } b_q3_K; /* 110 */
typedef struct { __half d, dmin; uint8_t scales[12];   uint8_t qs[128]; } b_q4_K;   /* 144 */
typedef struct { __half d, dmin; uint8_t scales[12];   uint8_t qh[32]; uint8_t qs[128]; } b_q5_K; /* 176 */
typedef struct { uint8_t  ql[128];    uint8_t qh[64];  int8_t scales[16]; __half d; } b_q6_K;  /* 210 */
typedef struct { __half d; int8_t qs[32]; } b_q8_0;                                  /* 34  */
typedef struct { __half d; uint16_t qs[32]; uint8_t scales[8]; } b_iq2_xs;           /* 74  */
typedef struct { __half d; uint8_t qs[96]; } b_iq3_xxs;                              /* 98  */
typedef struct { __half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } b_iq4_xs; /* 136 */

/* Dot ONE super-block (256 weights) of type `t` against x[0..255].
 * Decode happens in registers; no shared memory, no dequant buffer. */
__device__ __forceinline__ float dot_sb(int t, const uint8_t *w, const float *x){
    float acc = 0.f;

    if(t == KQ_CU_IQ2_XS){
        const b_iq2_xs *b = (const b_iq2_xs*)w;
        const float d = __half2float(b->d);
        for(int ib32 = 0; ib32 < 8; ++ib32){
            const float db0 = d * (0.5f + (b->scales[ib32] & 0xF)) * 0.25f;
            const float db1 = d * (0.5f + (b->scales[ib32] >> 4))  * 0.25f;
            for(int l = 0; l < 4; ++l){
                const uint16_t q = b->qs[4*ib32 + l];
                const uint8_t *g = (const uint8_t*)(c_iq2xs_grid + (q & 511));
                const uint8_t  s = c_ksigns[q >> 9];
                const float   db = (l < 2) ? db0 : db1;
                const float  *xx = x + ib32*32 + l*8;
                float p = 0.f;
                #pragma unroll
                for(int j = 0; j < 8; ++j)
                    p += (float)g[j] * ((s & c_kmask[j]) ? -1.f : 1.f) * xx[j];
                acc += db * p;
            }
        }
        return acc;
    }

    if(t == KQ_CU_IQ3_XXS){
        const b_iq3_xxs *b = (const b_iq3_xxs*)w;
        const float d = __half2float(b->d);
        const uint8_t *qs = b->qs;
        const uint8_t *ss = qs + 64;
        for(int ib32 = 0; ib32 < 8; ++ib32){
            uint32_t aux32;
            memcpy(&aux32, ss + 4*ib32, 4);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for(int l = 0; l < 4; ++l){
                const uint8_t s  = c_ksigns[(aux32 >> (7*l)) & 127];
                const uint8_t *g1 = (const uint8_t*)(c_iq3xxs_grid + qs[8*ib32 + 2*l + 0]);
                const uint8_t *g2 = (const uint8_t*)(c_iq3xxs_grid + qs[8*ib32 + 2*l + 1]);
                const float  *xx = x + ib32*32 + l*8;
                float p = 0.f;
                #pragma unroll
                for(int j = 0; j < 4; ++j){
                    p += (float)g1[j] * ((s & c_kmask[j+0]) ? -1.f : 1.f) * xx[j+0];
                    p += (float)g2[j] * ((s & c_kmask[j+4]) ? -1.f : 1.f) * xx[j+4];
                }
                acc += db * p;
            }
        }
        return acc;
    }

    if(t == KQ_CU_Q2_K){
        const b_q2_K *b = (const b_q2_K*)w;
        const float d = __half2float(b->d), mn = __half2float(b->dmin);
        int is = 0;
        for(int n = 0; n < 256; n += 128){
            int shift = 0;
            const uint8_t *q = b->qs + (n/128)*32;
            for(int j = 0; j < 4; ++j){
                for(int half = 0; half < 2; ++half){
                    const uint8_t sc = b->scales[is++];
                    const uint8_t *qq = q + 16*half;
                    const float   *xx = x + n + j*32 + 16*half;
                    float sq = 0.f, sx = 0.f;
                    #pragma unroll
                    for(int l = 0; l < 16; ++l){
                        const float xv = xx[l];
                        sq += (float)((qq[l] >> shift) & 3) * xv;
                        sx += xv;
                    }
                    acc += d * (float)(sc & 0xF) * sq - mn * (float)(sc >> 4) * sx;
                }
                shift += 2;
            }
        }
        return acc;
    }

    if(t == KQ_CU_Q3_K){
        const b_q3_K *b = (const b_q3_K*)w;
        const float d_all = __half2float(b->d);
        uint32_t aux[4];
        memcpy(aux, b->scales, 12);
        const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        aux[0] = ( aux[0]       & km2) | (((tmp >> 0) & km1) << 4);
        aux[1] = ( aux[1]       & km2) | (((tmp >> 2) & km1) << 4);
        const int8_t *sc = (const int8_t*)aux;
        int is = 0; uint8_t m = 1;
        for(int n = 0; n < 256; n += 128){
            int shift = 0;
            const uint8_t *q = b->qs + (n/128)*32;
            for(int j = 0; j < 4; ++j){
                float dl = d_all * (float)(sc[is++] - 32);
                #pragma unroll
                for(int l = 0; l < 16; ++l)
                    acc += dl * (float)((int8_t)((q[l] >> shift) & 3) - ((b->hmask[l] & m) ? 0 : 4)) * x[n + j*32 + l];
                dl = d_all * (float)(sc[is++] - 32);
                #pragma unroll
                for(int l = 0; l < 16; ++l)
                    acc += dl * (float)((int8_t)((q[l+16] >> shift) & 3) - ((b->hmask[l+16] & m) ? 0 : 4)) * x[n + j*32 + 16 + l];
                shift += 2; m <<= 1;
            }
        }
        return acc;
    }

    if(t == KQ_CU_IQ4_XS){
        const b_iq4_xs *b = (const b_iq4_xs*)w;
        const float d = __half2float(b->d);
        for(int ib = 0; ib < 8; ++ib){
            const int ls = ((b->scales_l[ib/2] >> (4*(ib%2))) & 0xF) | (((b->scales_h >> (2*ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            const uint8_t *qs = b->qs + ib*16;
            const float   *xx = x + ib*32;
            #pragma unroll
            for(int j = 0; j < 16; ++j){
                acc += dl * (float)c_kvalues_iq4nl[qs[j] & 0xF] * xx[j];
                acc += dl * (float)c_kvalues_iq4nl[qs[j] >> 4]  * xx[j + 16];
            }
        }
        return acc;
    }

    /* k-quants with a shared 6-bit scale/min layout */
    if(t == KQ_CU_Q4_K || t == KQ_CU_Q5_K){
        const b_q4_K *b4 = (const b_q4_K*)w;
        const b_q5_K *b5 = (const b_q5_K*)w;
        const uint8_t *sc_arr = (t == KQ_CU_Q4_K) ? b4->scales : b5->scales;
        const float d  = __half2float((t == KQ_CU_Q4_K) ? b4->d    : b5->d);
        const float mn = __half2float((t == KQ_CU_Q4_K) ? b4->dmin : b5->dmin);
        const uint8_t *qs = (t == KQ_CU_Q4_K) ? b4->qs : b5->qs;
        const uint8_t *qh = (t == KQ_CU_Q5_K) ? b5->qh : nullptr;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for(int j = 0; j < 256; j += 64){
            uint8_t sc, m;
            /* get_scale_min_k4 */
            #define GSM(jj) do {                                                        \
                if((jj) < 4){ sc = sc_arr[jj] & 63; m = sc_arr[(jj)+4] & 63; }          \
                else { sc = (uint8_t)((sc_arr[(jj)+4] & 0xF) | ((sc_arr[(jj)-4] >> 6) << 4)); \
                       m  = (uint8_t)((sc_arr[(jj)+4] >> 4)  | ((sc_arr[(jj)-0] >> 6) << 4)); } \
            } while(0)
            GSM(is + 0); const float d1 = d*sc, m1 = mn*m;
            GSM(is + 1); const float d2 = d*sc, m2 = mn*m;
            #undef GSM
            const uint8_t *q = qs + (j/64)*32;
            float s1 = 0, a1 = 0, s2 = 0, a2 = 0;
            #pragma unroll
            for(int l = 0; l < 32; ++l){
                const float xa = x[j + l], xb = x[j + 32 + l];
                int lo = q[l] & 0xF, hi = q[l] >> 4;
                if(t == KQ_CU_Q5_K){
                    /* qh is indexed by l and NEVER advanced — the 5th bit for the next
                     * 64-weight group is selected by shifting the MASKS (u1,u2 <<= 2), not
                     * by walking the pointer. Advancing qh here reads the high bits of the
                     * wrong weights: no crash, no NaN, just a 59% wrong projection. */
                    lo += (qh[l] & u1) ? 16 : 0;
                    hi += (qh[l] & u2) ? 16 : 0;
                }
                s1 += (float)lo * xa; a1 += xa;
                s2 += (float)hi * xb; a2 += xb;
            }
            acc += d1*s1 - m1*a1 + d2*s2 - m2*a2;
            is += 2; u1 <<= 2; u2 <<= 2;
        }
        return acc;
    }

    if(t == KQ_CU_Q6_K){
        const b_q6_K *b = (const b_q6_K*)w;
        const float d = __half2float(b->d);
        for(int n = 0; n < 256; n += 128){
            const uint8_t *ql = b->ql + (n/128)*64;
            const uint8_t *qh = b->qh + (n/128)*32;
            const int8_t  *sc = b->scales + (n/128)*8;
            #pragma unroll
            for(int l = 0; l < 32; ++l){
                const int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += d*sc[is+0]*q1*x[n+l+ 0];  acc += d*sc[is+2]*q2*x[n+l+32];
                acc += d*sc[is+4]*q3*x[n+l+64];  acc += d*sc[is+6]*q4*x[n+l+96];
            }
        }
        return acc;
    }

    if(t == KQ_CU_Q8_0){
        /* Q8_0's block is 32, not 256 — the caller passes ONE 32-block here. */
        const b_q8_0 *b = (const b_q8_0*)w;
        const float d = __half2float(b->d);
        float s = 0.f;
        #pragma unroll
        for(int i = 0; i < 32; ++i) s += (float)b->qs[i] * x[i];
        return d * s;
    }
    return 0.f;   /* unreachable: the host validates the type before launching */
}

/* Generic path: one warp per row, lanes stride over super-blocks. Correct for every type,
 * but the loads are scattered — each lane pulls its own 74..210 byte block, so nothing
 * coalesces. Kept for the rare types; the hot ones get the specialized kernels below. */
/* Generic path. `bs` is the TYPE's block size, not QK_K: Q8_0 blocks 32 values, the k- and
 * i-quants block 256. Hardcoding 256 made every Q8_0 tensor decode from the wrong offsets —
 * and made kq_cu_gemv silently refuse any row length that is not a multiple of 256, which
 * is every k_b in the model (192). It returned 0 and left the output buffer as it found it. */
__global__ static void gemv_q(float *y, const float *x, const uint8_t *W,
                              int gtype, int I, int O, int ts, int bs, int S){
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if(row >= O) return;
    const int lane = threadIdx.x & 31;
    const int nsb  = I / bs;
    const uint8_t *w = W + (size_t)row * (size_t)nsb * (size_t)ts;

    for(int s = 0; s < S; ++s){
        float acc = 0.f;
        for(int sb = lane; sb < nsb; sb += 32)
            acc += dot_sb(gtype, w + (size_t)sb * ts, x + (size_t)s*I + sb * bs);
        #pragma unroll
        for(int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if(lane == 0) y[(size_t)s * O + row] = acc;
    }
}

/* ---- specialized kernels for the two types that carry every routed expert ----
 *
 * The generic kernel above measured 127 GFLOP/s while moving 7.44 GB/token — 20x short of
 * this card's memory bandwidth. Two reasons, both structural:
 *   - each lane fetched a whole super-block on its own, so consecutive lanes touched
 *     addresses 74 bytes apart: no coalescing at all;
 *   - a 6144-long row has only 24 super-blocks, so 8 of the 32 lanes sat idle.
 *
 * Fix: the WHOLE WARP cooperates on ONE super-block. A super-block of IQ2_XS is exactly
 * 32 uint16 quants, so lane j takes quant j — consecutive lanes read consecutive halfwords,
 * which is a single coalesced transaction — decodes its 8 lattice weights, and the warp
 * reduces once at the end of the row. Same shape for IQ3_XXS (32 groups of 2 lattice bytes).
 */
/* The codebooks CANNOT live in __constant__ for this access pattern. Constant memory
 * broadcasts one address per cycle: it is fast only when every lane reads the SAME entry.
 * Here each of the 32 lanes decodes a different quant, so it looks up a different lattice
 * point, and the hardware serializes into 32 separate reads — which is why the first
 * version of this kernel ran at 127 GFLOP/s while moving data 20x below the card's memory
 * bandwidth. In shared memory a divergent lookup is one banked access, not 32.
 * 512*8 + 128 bytes = 4.2 KB per block, cheap at 8 warps. */
/* BATCHED over S tokens. This is not just a convenience — it is the single biggest lever
 * left, because decoding an i-quant lattice point is far more expensive than using it:
 *
 *   S=1  : decode 8 weights, do 8 multiply-adds. Decode dominates.
 *   S=8  : decode 8 weights ONCE, do 64 multiply-adds. The decode is amortized 8-fold.
 *
 * On top of that, S tokens routed through the SAME expert pay for that expert's ~12 MB
 * read once instead of S times. Batching therefore multiplies both the compute and the
 * I/O side. x is [S, I] row-major; y is [S, O].
 *
 * S is capped at KQ_CU_SMAX because each lane keeps one accumulator per token in registers.
 */
#define SMAX 8

__global__ static void gemv_iq2xs(float *y, const float *x, const uint8_t *W,
                                  int I, int O, int S){
    __shared__ uint64_t grid[512];
    __shared__ uint8_t  sign[128];
    for(int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = c_iq2xs_grid[i];
    for(int i = threadIdx.x; i < 128; i += blockDim.x) sign[i] = c_ksigns[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if(row >= O) return;
    const int nsb  = I / QK_K;
    const b_iq2_xs *b = (const b_iq2_xs*)(W + (size_t)row * (size_t)nsb * sizeof(b_iq2_xs));

    float acc[SMAX];
    #pragma unroll
    for(int s = 0; s < SMAX; ++s) acc[s] = 0.f;

    for(int sb = 0; sb < nsb; ++sb){
        const float d = __half2float(b[sb].d);
        const uint16_t q  = b[sb].qs[lane];                /* coalesced: 32 consecutive u16 */
        const uint8_t  sc = b[sb].scales[lane >> 2];
        /* lane j owns weights [8j, 8j+8); which 4-bit half of the scale byte is bit 1 of j */
        const float db = d * (0.5f + (float)((lane & 2) ? (sc >> 4) : (sc & 0xF))) * 0.25f;

        const uint8_t *g = (const uint8_t*)(grid + (q & 511));
        const uint8_t  sg = sign[q >> 9];

        /* decode the 8 weights ONCE, then reuse them for every token in the batch */
        float w8[8];
        #pragma unroll
        for(int j = 0; j < 8; ++j)
            w8[j] = db * (float)g[j] * ((sg >> j) & 1 ? -1.f : 1.f);

        const int base = sb * QK_K + lane * 8;
        for(int s = 0; s < S; ++s){
            const float *xx = x + (size_t)s * I + base;
            float p = 0.f;
            #pragma unroll
            for(int j = 0; j < 8; ++j) p += w8[j] * xx[j];
            acc[s] += p;
        }
    }
    for(int s = 0; s < S; ++s){
        float v = acc[s];
        #pragma unroll
        for(int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if(lane == 0) y[(size_t)s * O + row] = v;
    }
}

__global__ static void gemv_iq3xxs(float *y, const float *x, const uint8_t *W,
                                   int I, int O, int S){
    __shared__ uint32_t grid[256];
    __shared__ uint8_t  sign[128];
    for(int i = threadIdx.x; i < 256; i += blockDim.x) grid[i] = c_iq3xxs_grid[i];
    for(int i = threadIdx.x; i < 128; i += blockDim.x) sign[i] = c_ksigns[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if(row >= O) return;
    const int nsb  = I / QK_K;
    const b_iq3_xxs *b = (const b_iq3_xxs*)(W + (size_t)row * (size_t)nsb * sizeof(b_iq3_xxs));

    float acc[SMAX];
    #pragma unroll
    for(int s = 0; s < SMAX; ++s) acc[s] = 0.f;

    for(int sb = 0; sb < nsb; ++sb){
        const float d = __half2float(b[sb].d);
        const uint8_t *qs = b[sb].qs;                      /* 64 lattice bytes */
        const uint8_t *ss = qs + 64;                       /* then 8 x uint32 scale+signs */

        uint32_t aux32;
        memcpy(&aux32, ss + 4*(lane >> 2), 4);             /* 4 lanes share one aux32 */
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const uint8_t sg = sign[(aux32 >> (7*(lane & 3))) & 127];

        const uint8_t *g1 = (const uint8_t*)(grid + qs[2*lane + 0]);
        const uint8_t *g2 = (const uint8_t*)(grid + qs[2*lane + 1]);

        float w8[8];
        #pragma unroll
        for(int j = 0; j < 4; ++j){
            w8[j+0] = db * (float)g1[j] * ((sg >> (j+0)) & 1 ? -1.f : 1.f);
            w8[j+4] = db * (float)g2[j] * ((sg >> (j+4)) & 1 ? -1.f : 1.f);
        }

        const int base = sb * QK_K + lane * 8;
        for(int s = 0; s < S; ++s){
            const float *xx = x + (size_t)s * I + base;
            float p = 0.f;
            #pragma unroll
            for(int j = 0; j < 8; ++j) p += w8[j] * xx[j];
            acc[s] += p;
        }
    }
    for(int s = 0; s < S; ++s){
        float v = acc[s];
        #pragma unroll
        for(int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if(lane == 0) y[(size_t)s * O + row] = v;
    }
}

/* ================== int8 path: 4 multiply-adds per instruction ==================
 *
 * The float kernels above reach ~490 GFLOP/s; ggml's reach ~1300 on the identical tensor.
 * The whole gap is the inner product. A lattice weight is a signed uint8 and the activation
 * can be quantized to int8 with a scale per 32 values — so the dot is an INTEGER dot, and
 * __dp4a does four of them in one instruction. The float version was doing them one at a
 * time.
 *
 * The sign flip is also SIMD: unpack_ksigns rebuilds the 8th sign bit from the parity of
 * the other seven (that is how the format encodes it), broadcasts the byte, and then
 *   __vsub4(g ^ mask, mask)
 * negates exactly the bytes whose mask is 0xFF — two's complement, four bytes at a time,
 * no branches. Straight out of ggml/vecdotq.cuh; the trick is theirs.
 */
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v){
    const uint32_t p = __popc(v) & 1;      /* 7 stored bits; the 8th is their parity */
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;                /* broadcast so 0x08040201 can select bit-per-byte */
}

/* Activation -> int8 with one scale per 32 values (Q8_0's layout, which is what the
 * i-quant vec_dots expect). One block per 32-value group. */
__global__ static void quant_act_q8(int8_t *qs, float *d, const float *x, int n){
    const int b = blockIdx.x;
    const int i = threadIdx.x;                /* 0..31 */
    const float v = x[b*32 + i];

    float amax = fabsf(v);
    #pragma unroll
    for(int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    const float sc = amax / 127.0f;
    if(i == 0) d[b] = sc;
    qs[b*32 + i] = (int8_t)(sc > 0.f ? __float2int_rn(v / sc) : 0);
    (void)n;
}

__global__ static void gemv_iq2xs_i8(float *y, const int8_t *xq, const float *xd,
                                     const uint8_t *W, int I, int O, int S){
    __shared__ uint64_t grid[512];
    for(int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = c_iq2xs_grid[i];
    __syncthreads();

    const int row  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if(row >= O) return;
    const int nsb = I / QK_K;
    const b_iq2_xs *b = (const b_iq2_xs*)(W + (size_t)row * (size_t)nsb * sizeof(b_iq2_xs));

    for(int s = 0; s < S; ++s){
        const int8_t *xq_s = xq + (size_t)s * I;
        const float  *xd_s = xd + (size_t)s * (I/32);
        float acc = 0.f;

        for(int sb = 0; sb < nsb; ++sb){
            const float d = __half2float(b[sb].d);
            const uint16_t q  = b[sb].qs[lane];          /* coalesced: 32 consecutive u16 */
            const uint8_t  sc = b[sb].scales[lane >> 2];
            const int      ls = (lane & 2) ? (sc >> 4) : (sc & 0xF);

            const uint2 g = ((const uint2*)grid)[q & 511];      /* the 8 lattice bytes */
            const uint32_t sg = unpack_ksigns((uint8_t)(q >> 9));

            const int m0 = __vcmpne4(sg & 0x08040201u, 0);      /* 0xFF per negated byte */
            const int m1 = __vcmpne4(sg & 0x80402010u, 0);
            const int g0 = __vsub4(g.x ^ m0, m0);               /* two's complement, x4 */
            const int g1 = __vsub4(g.y ^ m1, m1);

            const int *u = (const int*)(xq_s + sb*QK_K + lane*8);
            int sumi = __dp4a(g0, u[0], 0);
            sumi     = __dp4a(g1, u[1], sumi);

            acc += d * (0.5f + (float)ls) * 0.25f
                     * xd_s[sb*8 + (lane >> 2)] * (float)sumi;
        }
        #pragma unroll
        for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if(lane == 0) y[(size_t)s * O + row] = acc;
    }
}

__global__ static void gemv_iq3xxs_i8(float *y, const int8_t *xq, const float *xd,
                                      const uint8_t *W, int I, int O, int S){
    __shared__ uint32_t grid[256];
    for(int i = threadIdx.x; i < 256; i += blockDim.x) grid[i] = c_iq3xxs_grid[i];
    __syncthreads();

    const int row  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if(row >= O) return;
    const int nsb = I / QK_K;
    const b_iq3_xxs *b = (const b_iq3_xxs*)(W + (size_t)row * (size_t)nsb * sizeof(b_iq3_xxs));

    for(int s = 0; s < S; ++s){
        const int8_t *xq_s = xq + (size_t)s * I;
        const float  *xd_s = xd + (size_t)s * (I/32);
        float acc = 0.f;

        for(int sb = 0; sb < nsb; ++sb){
            const float d = __half2float(b[sb].d);
            const uint8_t *qs = b[sb].qs;
            const uint8_t *ss = qs + 64;

            uint32_t aux32;
            memcpy(&aux32, ss + 4*(lane >> 2), 4);
            const uint32_t sg = unpack_ksigns((uint8_t)((aux32 >> (7*(lane & 3))) & 127));

            /* two 4-byte lattice points; each is a uint32 in the 256-entry grid */
            const int gx = (int)grid[qs[2*lane + 0]];
            const int gy = (int)grid[qs[2*lane + 1]];

            const int m0 = __vcmpne4(sg & 0x08040201u, 0);
            const int m1 = __vcmpne4(sg & 0x80402010u, 0);
            const int g0 = __vsub4(gx ^ m0, m0);
            const int g1 = __vsub4(gy ^ m1, m1);

            const int *u = (const int*)(xq_s + sb*QK_K + lane*8);
            int sumi = __dp4a(g0, u[0], 0);
            sumi     = __dp4a(g1, u[1], sumi);

            acc += d * (0.5f + (float)(aux32 >> 28)) * 0.5f
                     * xd_s[sb*8 + (lane >> 2)] * (float)sumi;
        }
        #pragma unroll
        for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if(lane == 0) y[(size_t)s * O + row] = acc;
    }
}

/* ---- the small ops that keep an expert's whole FFN on the device ----
 * Without these, every expert would round-trip gate/up back to the host just to apply a
 * sigmoid and a multiply, and the PCIe latency would dwarf the matmuls. */
__global__ static void k_swiglu(float *g, const float *u, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    const float x = g[i];
    g[i] = (x / (1.0f + __expf(-x))) * u[i];
}
__global__ static void k_axpy(float *acc, const float *v, float a, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) acc[i] += a * v[i];
}
__global__ static void k_zero(float *v, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) v[i] = 0.f;
}

extern "C" void kq_cu_swiglu(float *g, const float *u, int n, cudaStream_t s){
    k_swiglu<<<(n+255)/256, 256, 0, s>>>(g, u, n);
}
extern "C" void kq_cu_axpy(float *acc, const float *v, float a, int n, cudaStream_t s){
    k_axpy<<<(n+255)/256, 256, 0, s>>>(acc, v, a, n);
}
extern "C" void kq_cu_zero(float *v, int n, cudaStream_t s){
    k_zero<<<(n+255)/256, 256, 0, s>>>(v, n);
}

/* ------------------------------------------------------------------ host side */

extern "C" int kq_cu_upload_tables(const uint64_t *iq2xs, const uint32_t *iq3xxs,
                                   const uint8_t *ksigns, const uint8_t *kmask,
                                   const int8_t *kv4nl){
    cudaError_t e;
    e = cudaMemcpyToSymbol(c_iq2xs_grid,     iq2xs,  512*sizeof(uint64_t)); if(e) return 0;
    e = cudaMemcpyToSymbol(c_iq3xxs_grid,    iq3xxs, 256*sizeof(uint32_t)); if(e) return 0;
    e = cudaMemcpyToSymbol(c_ksigns,         ksigns, 128);                  if(e) return 0;
    e = cudaMemcpyToSymbol(c_kmask,          kmask,  8);                    if(e) return 0;
    e = cudaMemcpyToSymbol(c_kvalues_iq4nl,  kv4nl,  16);                   if(e) return 0;
    return 1;
}

/* W_dev: the expert's quantized bytes, already on the device. Never dequantized.
 * x is [S,I], y is [S,O]. S=1 is the decode case; S>1 is speculative decoding or prefill,
 * where the same decoded weight serves every token in the batch. */
/* Quantize the activation ONCE per layer. Every routed expert of that layer contracts
 * against the SAME x, so re-quantizing per expert would repeat the work 8 times. */
extern "C" int kq_cu_quant_act(int8_t *qs_dev, float *d_dev, const float *x_dev,
                               int S, int I, cudaStream_t s){
    if(I % 32) return 0;
    quant_act_q8<<<S*(I/32), 32, 0, s>>>(qs_dev, d_dev, x_dev, S*I);
    return cudaGetLastError() == cudaSuccess;
}

/* int8 path, for the two types that carry every routed expert. Needs the activation
 * pre-quantized by kq_cu_quant_act. Everything else keeps the fp32 kernels. */
extern "C" int kq_cu_gemm_i8(float *y_dev, const int8_t *xq_dev, const float *xd_dev,
                             const uint8_t *W_dev, int gtype, int S, int I, int O,
                             cudaStream_t s){
    if(I % QK_K || S < 1 || S > SMAX) return 0;
    const int warps = 8;
    const int blocks = (O + warps - 1) / warps;
    if(gtype == KQ_CU_IQ2_XS)
        gemv_iq2xs_i8<<<blocks, warps*32, 0, s>>>(y_dev, xq_dev, xd_dev, W_dev, I, O, S);
    else if(gtype == KQ_CU_IQ3_XXS)
        gemv_iq3xxs_i8<<<blocks, warps*32, 0, s>>>(y_dev, xq_dev, xd_dev, W_dev, I, O, S);
    else
        return 0;
    return cudaGetLastError() == cudaSuccess;
}

/* BATCHED over heads: y[h] = W[h] . x[h], for h in [0, nb).
 *
 * MLA does two of these per layer — k_b absorbs each head's query into the latent, v_b
 * projects each head's output back — and there are 64 heads. Launching one kernel per head,
 * with a memcpy and a stream sync around each, means ~15000 round trips per token: the GPU
 * spends its life waiting on ~50 us of launch latency and the drive it was supposed to be
 * hiding behind goes idle. One launch for all 64 heads instead. */
__global__ static void gemv_batched(float *y, const float *x, const uint8_t *W,
                                    int gtype, int I, int O, int ts, int bs,
                                    int64_t wstride, int xstride, int ystride){
    const int h   = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if(row >= O) return;
    const int lane = threadIdx.x & 31;
    const int nsb  = I / bs;

    const uint8_t *w = W + (size_t)h*wstride + (size_t)row*nsb*ts;
    const float   *xx = x + (size_t)h*xstride;

    float acc = 0.f;
    for(int sb = lane; sb < nsb; sb += 32)
        acc += dot_sb(gtype, w + (size_t)sb*ts, xx + sb*bs);
    #pragma unroll
    for(int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if(lane == 0) y[(size_t)h*ystride + row] = acc;
}

static int cu_typesize(int t){
    switch(t){ case KQ_CU_Q8_0: return 34; case KQ_CU_Q2_K: return 84;
               case KQ_CU_Q3_K: return 110; case KQ_CU_Q4_K: return 144;
               case KQ_CU_Q5_K: return 176; case KQ_CU_Q6_K: return 210;
               case KQ_CU_IQ2_XS: return 74; case KQ_CU_IQ3_XXS: return 98;
               case KQ_CU_IQ4_XS: return 136; default: return 0; }
}
static int cu_blocksize(int t){ return (t == KQ_CU_Q8_0) ? 32 : QK_K; }

extern "C" int kq_cu_gemv_heads(float *y, const float *x, const uint8_t *W,
                                int gtype, int I, int O, int n_head,
                                long wstride, int xstride, int ystride,
                                cudaStream_t s){
    const int bs = cu_blocksize(gtype);
    if(I % bs) return 0;
    const int warps = 8;
    dim3 grid((O + warps - 1)/warps, n_head);
    gemv_batched<<<grid, warps*32, 0, s>>>(y, x, W, gtype, I, O, cu_typesize(gtype), bs,
                                           wstride, xstride, ystride);
    return cudaGetLastError() == cudaSuccess;
}

/* cudaGetLastError() CLEARS the error it reports. So a caller that sees kq_cu_gemm return 0
 * and then asks CUDA what went wrong is told "no error" — the check inside already ate it.
 * That cost an hour: a launch failure looked exactly like a missing kernel. Stash it. */
static char cu_err[256] = "";
extern "C" const char *kq_cu_last_error(void){ return cu_err; }

extern "C" int kq_cu_gemm(float *y_dev, const float *x_dev, const uint8_t *W_dev,
                          int gtype, int S, int I, int O, int type_size, cudaStream_t s){
    const int bs = cu_blocksize(gtype);
    if(I % bs || S < 1 || S > SMAX){
        snprintf(cu_err, sizeof(cu_err),
                 "declined to launch: I=%d not a multiple of block %d, or S=%d outside 1..%d",
                 I, bs, S, SMAX);
        return 0;
    }
    const int warps = 8;                       /* 256 threads */
    const int blocks = (O + warps - 1) / warps;
    if(gtype == KQ_CU_IQ2_XS)
        gemv_iq2xs<<<blocks, warps*32, 0, s>>>(y_dev, x_dev, W_dev, I, O, S);
    else if(gtype == KQ_CU_IQ3_XXS)
        gemv_iq3xxs<<<blocks, warps*32, 0, s>>>(y_dev, x_dev, W_dev, I, O, S);
    else
        gemv_q<<<blocks, warps*32, 0, s>>>(y_dev, x_dev, W_dev, gtype, I, O, type_size, bs, S);
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){
        snprintf(cu_err, sizeof(cu_err), "launch failed: %s", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int kq_cu_gemv(float *y_dev, const float *x_dev, const uint8_t *W_dev,
                          int gtype, int I, int O, int type_size, cudaStream_t s){
    return kq_cu_gemm(y_dev, x_dev, W_dev, gtype, 1, I, O, type_size, s);
}
