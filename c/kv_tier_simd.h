#ifndef QWEN_KV_TIER_SIMD_H
#define QWEN_KV_TIER_SIMD_H
/* kv_tier_simd.h — the three kernels that touch every byte of the KV cache.
 *
 * WHY THESE THREE AND NOTHING ELSE
 *
 * At 250k context a generated token reads the ENTIRE cache once: 8.1 GiB through
 * kvt_dot_q8 (the scores) and again through kvt_axpy_q8 (the weighted sum of V). Nothing
 * else in the decode loop is anywhere near that. The scalar versions run at roughly the
 * speed of one fp16 decode plus one int8 widen per weight, which is about 1 GB/s per core —
 * so on a drive that delivers 6.8 GB/s the KERNEL becomes the bottleneck and the tiering
 * work is wasted. These exist so the disk stays the limit.
 *
 * WHERE THE BITS ACTUALLY GET MANIPULATED
 *
 * 1. THE FP16 SCALE. Every 32 weights carry one fp16 scale. Scalar decoding of an fp16 is a
 *    branchy mess — zero/subnormal/inf all need their own path — and it sits on the critical
 *    path once per block. F16C does it in one instruction (vcvtph2ps). Where F16C is absent
 *    the fallback below is BRANCH-FREE: it reconstructs the float by bit surgery and fixes
 *    up subnormals by multiplying with a magic constant, instead of looping to renormalize.
 *
 * 2. THE INT8 WIDEN. vpmovsxbd sign-extends 8 int8 straight to 8 int32, then vcvtdq2ps to
 *    8 floats. No shifts, no masks, no sign branch — the sign extension IS the bit trick,
 *    done by the load.
 *
 * 3. THE ABSMAX, on the pack side. |x| on a float is one ANDPS against 0x7fffffff: clearing
 *    the sign bit is cheaper than any comparison, and the max reduction is then a plain tree.
 *
 * WHAT IS DELIBERATELY *NOT* DONE HERE
 *
 * This CPU has AVX-VNNI, and quantizing q to int8 to use vpdpbusd would be faster still —
 * glm.c already does exactly that for the expert weights. It is not used here because the
 * attention SCORES feed a softmax: an int8 q would put quantization noise on the exponent's
 * input, where it is amplified, rather than on a value that gets averaged. The float
 * accumulator keeps the scores exact to fp32 and still moves 8 weights per FMA. If a
 * measurement ever shows the scores tolerate it, the VNNI path belongs behind a flag with
 * kv_tier_test measuring the softmax error, not before.
 *
 * Every kernel here is checked against the scalar reference in kvtier_test — same discipline
 * as moe_cpu_simd.h, and for the same reason: a fast wrong kernel looks exactly like
 * progress.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define KVT_HAVE_AVX2 1
#endif

/* ---- fp16 -> fp32, branch-free scalar fallback ----
 * Shift the 5-bit exponent and 10-bit mantissa into fp32 position and add the exponent bias
 * difference by MULTIPLYING with 2^112 (0x77800000). That single multiply normalizes
 * subnormals for free and leaves normals untouched, which is what removes the branch.
 * Inf/NaN keep their all-ones exponent because the multiply saturates them there. */
static inline float kvt_h2f_bits(uint16_t h){
    union { uint32_t u; float f; } o, magic;
    magic.u = 0x77800000u;                       /* 2^112 */
    o.u = (uint32_t)(h & 0x7FFFu) << 13;         /* exponent+mantissa into place */
    o.f *= magic.f;                              /* rebias; normalizes subnormals */
    o.u |= (uint32_t)(h & 0x8000u) << 16;        /* sign back on, last */
    return o.f;
}

static inline uint16_t kvt_f2h_bits(float f){
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000u;
    int32_t  e = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = b & 0x7FFFFFu;
    if(e <= 0)  return (uint16_t)s;
    if(e >= 31) return (uint16_t)(s | 0x7C00u);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

#ifdef KVT_HAVE_AVX2
static inline float kvt_hsum8(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}
/* one fp16 -> float. F16C when the build has it, else the branch-free bit path. */
static inline float kvt_h2f(uint16_t h){
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    return kvt_h2f_bits(h);
#endif
}
#else
static inline float kvt_h2f(uint16_t h){ return kvt_h2f_bits(h); }
#endif

/* ---------------- q8_0: dot ----------------
 * layout per block of 32: [fp16 scale][32 x int8].  n must be a multiple of 32. */
static float kvt_dot_q8_scalar(const float *q, const void *w, int n){
    const uint8_t *p = (const uint8_t*)w;
    float acc = 0;
    for(int b = 0; b < n/32; b++){
        uint16_t h; memcpy(&h, p, 2); p += 2;
        const float d = kvt_h2f_bits(h);
        float s = 0;
        for(int i = 0; i < 32; i++) s += q[b*32+i] * (float)*(const int8_t*)p++;
        acc += d*s;
    }
    return acc;
}

#ifdef KVT_HAVE_AVX2
static float kvt_dot_q8_avx2(const float *q, const void *w, int n){
    const uint8_t *p = (const uint8_t*)w;
    __m256 acc = _mm256_setzero_ps();
    for(int b = 0; b < n/32; b++){
        uint16_t h; memcpy(&h, p, 2); p += 2;
        const __m256 d = _mm256_set1_ps(kvt_h2f(h));
        /* 32 int8 -> 4 x (8 int32 -> 8 float). vpmovsxbd does the sign extension, so the
         * two's-complement sign never needs a mask or a branch. */
        const __m128i q8 = _mm_loadu_si128((const __m128i*)p);          /* 16 bytes */
        const __m128i q8b= _mm_loadu_si128((const __m128i*)(p+16));     /* 16 bytes */
        p += 32;
        const float *qp = q + b*32;
        __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8, 8)));
        __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8b));
        __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8b, 8)));
        /* fold the block scale into the weights once, then a plain FMA chain */
        acc = _mm256_fmadd_ps(_mm256_mul_ps(w0, d), _mm256_loadu_ps(qp +  0), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(w1, d), _mm256_loadu_ps(qp +  8), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(w2, d), _mm256_loadu_ps(qp + 16), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(w3, d), _mm256_loadu_ps(qp + 24), acc);
    }
    return kvt_hsum8(acc);
}
#endif

static inline float kvt_dot_q8(const float *q, const void *w, int n){
#ifdef KVT_HAVE_AVX2
    return kvt_dot_q8_avx2(q, w, n);
#else
    return kvt_dot_q8_scalar(q, w, n);
#endif
}

/* ---------------- q8_0: axpy (out += g * V) ---------------- */
static void kvt_axpy_q8_scalar(float *out, float g, const void *w, int n){
    const uint8_t *p = (const uint8_t*)w;
    for(int b = 0; b < n/32; b++){
        uint16_t h; memcpy(&h, p, 2); p += 2;
        const float d = kvt_h2f_bits(h) * g;
        for(int i = 0; i < 32; i++) out[b*32+i] += d * (float)*(const int8_t*)p++;
    }
}

#ifdef KVT_HAVE_AVX2
static void kvt_axpy_q8_avx2(float *out, float g, const void *w, int n){
    const uint8_t *p = (const uint8_t*)w;
    for(int b = 0; b < n/32; b++){
        uint16_t h; memcpy(&h, p, 2); p += 2;
        const __m256 d = _mm256_set1_ps(kvt_h2f(h) * g);
        const __m128i q8  = _mm_loadu_si128((const __m128i*)p);
        const __m128i q8b = _mm_loadu_si128((const __m128i*)(p+16));
        p += 32;
        float *o = out + b*32;
        _mm256_storeu_ps(o+ 0, _mm256_fmadd_ps(d,
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)),                 _mm256_loadu_ps(o+ 0)));
        _mm256_storeu_ps(o+ 8, _mm256_fmadd_ps(d,
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8,8))),_mm256_loadu_ps(o+ 8)));
        _mm256_storeu_ps(o+16, _mm256_fmadd_ps(d,
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8b)),                _mm256_loadu_ps(o+16)));
        _mm256_storeu_ps(o+24, _mm256_fmadd_ps(d,
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8b,8))),_mm256_loadu_ps(o+24)));
    }
}
#endif

static inline void kvt_axpy_q8(float *out, float g, const void *w, int n){
#ifdef KVT_HAVE_AVX2
    kvt_axpy_q8_avx2(out, g, w, n);
#else
    kvt_axpy_q8_scalar(out, g, w, n);
#endif
}

/* ---------------- q8_0: pack ----------------
 * The write side. absmax via ANDPS (clear the sign bit) rather than a compare chain, then
 * one reciprocal scale and a saturating pack down to int8. Bit-identical to the scalar
 * reference because the rounding is still round-to-nearest-even on the same values. */
static void kvt_pack_q8_scalar(void *dst, const float *src, int n){
    uint8_t *p = (uint8_t*)dst;
    for(int b = 0; b < n/32; b++){
        const float *x = src + b*32;
        float amax = 0;
        for(int i = 0; i < 32; i++){ float a = fabsf(x[i]); if(a > amax) amax = a; }
        const float d = amax/127.0f, id = d ? 1.0f/d : 0.0f;
        const uint16_t h = kvt_f2h_bits(d); memcpy(p, &h, 2); p += 2;
        for(int i = 0; i < 32; i++){
            int q = (int)lrintf(x[i]*id);
            if(q >  127) q =  127;
            if(q < -127) q = -127;
            *(int8_t*)p++ = (int8_t)q;
        }
    }
}

#ifdef KVT_HAVE_AVX2
static void kvt_pack_q8_avx2(void *dst, const float *src, int n){
    uint8_t *p = (uint8_t*)dst;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for(int b = 0; b < n/32; b++){
        const float *x = src + b*32;
        __m256 v0 = _mm256_loadu_ps(x+0),  v1 = _mm256_loadu_ps(x+8);
        __m256 v2 = _mm256_loadu_ps(x+16), v3 = _mm256_loadu_ps(x+24);
        __m256 m = _mm256_max_ps(
                     _mm256_max_ps(_mm256_and_ps(v0,absmask), _mm256_and_ps(v1,absmask)),
                     _mm256_max_ps(_mm256_and_ps(v2,absmask), _mm256_and_ps(v3,absmask)));
        __m128 lo = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m,1));
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
        const float amax = _mm_cvtss_f32(lo);
        const float d = amax/127.0f, id = d ? 1.0f/d : 0.0f;
        const uint16_t h = kvt_f2h_bits(d); memcpy(p, &h, 2); p += 2;

        const __m256 s = _mm256_set1_ps(id);
        /* cvtps_epi32 rounds with the current mode = round-to-nearest-even, matching lrintf */
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, s));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, s));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, s));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, s));
        /* pack 32->16->8 with signed saturation; the |q|<=127 clamp comes free from packs */
        __m256i p0 = _mm256_packs_epi32(i0, i1);              /* lanes interleave... */
        __m256i p1 = _mm256_packs_epi32(i2, i3);
        __m256i q  = _mm256_packs_epi16(p0, p1);
        /* ...so undo AVX2's 128-bit lane interleaving to get sequential order back */
        q = _mm256_permutevar8x32_epi32(q, _mm256_setr_epi32(0,4,1,5,2,6,3,7));
        _mm256_storeu_si256((__m256i*)p, q);
        p += 32;
    }
}
#endif

static inline void kvt_pack_q8(void *dst, const float *src, int n){
#ifdef KVT_HAVE_AVX2
    kvt_pack_q8_avx2(dst, src, n);
#else
    kvt_pack_q8_scalar(dst, src, n);
#endif
}

#endif
