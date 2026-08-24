#ifndef COLIBRI_MOE_CPU_SIMD_H
#define COLIBRI_MOE_CPU_SIMD_H
/* moe_cpu_simd.h — AVX2+FMA dot products for the two lattice quants the routed experts use.
 *
 * The scalar kq_dot_iq2_xs / kq_dot_iq3_xxs run at ~0.1 GB/s per core: a grid gather per 8
 * weights, then eight scalar multiplies each gated by a branch on a sign bit. That branch and
 * the one-float-at-a-time multiply are what kill it.
 *
 * The vector form keeps ONE 8-wide float accumulator for the whole row and folds four things
 * into a handful of instructions per 8 weights:
 *   - the 8 grid bytes widen straight to 8 floats (cvtepu8 -> cvtepi32 -> cvtps),
 *   - the sign bits become a +/-1 float vector by table lookup, no branch,
 *   - the per-group scale multiplies the weight vector before the FMA,
 *   - one horizontal sum per row instead of per group.
 *
 * Validated against the scalar kernels (which are themselves bit-exact vs ggml). Reassociation
 * makes it match to ~1e-6 relative, not bit-for-bit — that is fp addition order, not a defect,
 * and the moecpu test holds it to 1e-5. Everything on this project that skipped that check
 * shipped a silent bug; this one is checked.
 */
#include <stdint.h>
#include <string.h>
#include "kquant.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

/* signf[s] = 8 floats, element j = (s & (1<<j)) ? -1 : +1.  Built once. */
static float g_signf[256][8];
static int   g_signf_ready = 0;
static void moe_simd_init(void){
    if(g_signf_ready) return;
    for(int s=0;s<256;s++)
        for(int j=0;j<8;j++)
            g_signf[s][j] = (s & (1<<j)) ? -1.0f : 1.0f;
    g_signf_ready = 1;
}

static inline float hsum256(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v,1);
    lo = _mm_add_ps(lo,hi);
    __m128 sh = _mm_movehl_ps(lo,lo);
    lo = _mm_add_ps(lo,sh);
    sh = _mm_shuffle_ps(lo,lo,0x1);
    lo = _mm_add_ss(lo,sh);
    return _mm_cvtss_f32(lo);
}

/* 8 packed uint8 (low 64 bits of a uint64) -> __m256 of 8 floats */
static inline __m256 bytes8_to_ps(uint64_t packed){
    __m128i b = _mm_cvtsi64_si128((long long)packed);      /* 8 bytes in low lane */
    __m256i i = _mm256_cvtepu8_epi32(b);                   /* zero-extend to 8x int32 */
    return _mm256_cvtepi32_ps(i);
}

static float kq_dot_iq2_xs_avx2(const void *vw, const float *x, int64_t n){
    const kq_iq2_xs *w = (const kq_iq2_xs*)vw;
    __m256 acc = _mm256_setzero_ps();
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const float d = kq_half(w[i].d);
        const float *xb = x + i*KQ_QK_K;
        for(int ib32=0; ib32<KQ_QK_K/32; ib32++){
            const float db0 = d*(0.5f+(float)(w[i].scales[ib32]&0xF))*0.25f;
            const float db1 = d*(0.5f+(float)(w[i].scales[ib32]>>4)) *0.25f;
            for(int l=0;l<4;l++){
                const uint16_t q = w[i].qs[4*ib32+l];
                const __m256 gridf = bytes8_to_ps(iq2xs_grid[q & 511]);
                const __m256 signv = _mm256_loadu_ps(g_signf[ksigns_iq2xs[q >> 9]]);
                const __m256 sc    = _mm256_set1_ps(l < 2 ? db0 : db1);
                const __m256 wv    = _mm256_mul_ps(_mm256_mul_ps(gridf, signv), sc);
                const __m256 xv    = _mm256_loadu_ps(xb + ib32*32 + l*8);
                acc = _mm256_fmadd_ps(wv, xv, acc);
            }
        }
    }
    return hsum256(acc);
}

static float kq_dot_iq3_xxs_avx2(const void *vw, const float *x, int64_t n){
    const kq_iq3_xxs *w = (const kq_iq3_xxs*)vw;
    __m256 acc = _mm256_setzero_ps();
    uint32_t aux32;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const float d = kq_half(w[i].d);
        const uint8_t *qs = w[i].qs, *ss = qs + KQ_QK_K/4;
        const float *xb = x + i*KQ_QK_K;
        for(int ib32=0; ib32<KQ_QK_K/32; ib32++){
            memcpy(&aux32, ss + 4*ib32, 4);
            const float db = d*(0.5f+(float)(aux32>>28))*0.5f;
            const __m256 sc = _mm256_set1_ps(db);
            for(int l=0;l<4;l++){
                /* two 4-byte grid entries packed into 8 bytes: g1 low, g2 high */
                const uint64_t packed = (uint64_t)iq3xxs_grid[qs[2*l+0]]
                                      | ((uint64_t)iq3xxs_grid[qs[2*l+1]] << 32);
                const __m256 gridf = bytes8_to_ps(packed);
                const __m256 signv = _mm256_loadu_ps(g_signf[ksigns_iq2xs[(aux32>>(7*l))&127]]);
                const __m256 wv    = _mm256_mul_ps(_mm256_mul_ps(gridf, signv), sc);
                const __m256 xv    = _mm256_loadu_ps(xb + ib32*32 + l*8);
                acc = _mm256_fmadd_ps(wv, xv, acc);
            }
            qs += 8;
        }
    }
    return hsum256(acc);
}

/* Fast dot: vectorized path for the two hot lattice types, scalar for everything else. */
static inline float kq_dot_fast(int t, const void *w, const float *x, int64_t n){
    if(t == KQ_IQ2_XS)  return kq_dot_iq2_xs_avx2(w, x, n);
    if(t == KQ_IQ3_XXS) return kq_dot_iq3_xxs_avx2(w, x, n);
    return kq_dot(t, w, x, n);
}

#else   /* no AVX2 — fall back to the scalar path, unchanged */
static void moe_simd_init(void){}
static inline float kq_dot_fast(int t, const void *w, const float *x, int64_t n){
    return kq_dot(t, w, x, n);
}
#endif

#endif
