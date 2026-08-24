#ifndef QWEN_KQUANT_SIMD_H
#define QWEN_KQUANT_SIMD_H
/* kquant_simd.h — AVX2 dot products for Q4_K and Q6_K, the two quants Qwen3.5's FFN is
 * made of.
 *
 * WHY THIS WAS THE BOTTLENECK
 *
 * moe_cpu_simd.h vectorized IQ2_XS and IQ3_XXS because those were what GLM-5.2's experts
 * used, and it took them from 1.4 to 21 GB/s. Q4_K and Q6_K never got the same treatment,
 * because nothing on this project depended on them at scale. Qwen3.5 does: its FFN is
 * 9.65 GiB of Q4_K gate/up and Q6_K down, read in full for every generated token, and
 * measured at 2.89 GB/s out of RAM that does ~60. That is 3.6 s/token spent decoding
 * nibbles, and no amount of tiering, prefetching or GPU offload elsewhere can hide it.
 *
 * WHAT THE SCALAR VERSIONS DO WRONG
 *
 * kq_dot_q6_K accumulates ONE weight at a time into ONE scalar. Per weight that is four
 * byte loads, three shifts, two masks, an ADD, and a multiply into a serial dependency
 * chain whose latency the CPU cannot hide. The weights are 6 bits; the machine is moving
 * them at the speed of the dependency chain, not the speed of memory.
 *
 * THE BIT-LEVEL SHAPE OF THE FIX
 *
 * Q4_K, per 32-byte load: the low nibbles ARE one 32-weight group and the high nibbles ARE
 * the next. One AND with 0x0F and one shift produce both groups from a single load — the
 * scalar code re-reads the same byte twice for the same reason it re-derives them twice.
 * The group's contribution is
 *
 *      d*sc * sum(q_i * x_i)  -  dmin*m * sum(x_i)
 *
 * so the min term needs only the SUM of the activations, kept in a second accumulator and
 * folded in once per group instead of subtracted per weight.
 *
 * Q6_K: the 6 bits are split across two arrays, 4 low in ql and 2 high in qh, with the high
 * pairs of four different weights packed into one qh byte at shifts 0/2/4/6. Extracting all
 * four with one load of qh and four (shift, AND 3, shift-left 4) chains lets a single 16-byte
 * load feed four 16-weight groups. The -32 bias is applied on the BYTES, before widening, so
 * vpmovsxbd's sign extension does the two's-complement work for free.
 *
 * Both keep ONE 8-wide float accumulator per row and do one horizontal sum at the end, which
 * is what removes the latency chain the scalar version is stuck in.
 *
 * VALIDATED, NOT ASSUMED. kquant_simd_test diffs every kernel against the scalar one that is
 * itself bit-exact against ggml. Reassociation makes the match ~1e-6 relative, not bitwise —
 * that is fp addition order, the same caveat moe_cpu_simd.h carries.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>   /* lrintf in kqs_quant_act; do not rely on the caller including it */

#include "kquant.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define KQS_HAVE_AVX2 1

/* forward: the VNNI gemv falls back to this for types it does not handle */
static void kq_gemv_fast(float *y, const float *x, const void *W, int gtype, int I, int O);

static inline float kqs_hsum(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

/* 16 unsigned bytes -> two 8-wide float vectors. vpmovzxbd does the widen; no masking. */
static inline void kqs_u8x16(__m128i b, __m256 *a, __m256 *c){
    *a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
    *c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8)));
}
/* 16 SIGNED bytes -> two 8-wide float vectors. vpmovsxbd sign-extends, which is exactly the
 * two's-complement fixup the scalar path spells out with a cast. */
static inline void kqs_i8x16(__m128i b, __m256 *a, __m256 *c){
    *a = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
    *c = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8)));
}

/* ---------------- Q4_K ----------------
 * 256 weights per super-block in 8 groups of 32. Group g has a 6-bit scale and a 6-bit min,
 * both bit-sliced into scales[12]; the value of weight i in group g is
 *     d*sc[g]*q_i - dmin*m[g]
 * The qs bytes pair group 2k (low nibbles) with group 2k+1 (high nibbles). */
static float kq_dot_q4_K_avx2(const void *vw, const float *x, int64_t n){
    const kq_q4_K *w = (const kq_q4_K*)vw;
    const __m256i lomask = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();

    for(int64_t i = 0; i < n/KQ_QK_K; i++){
        const uint8_t *q = w[i].qs;
        const float d = kq_half(w[i].d), mn = kq_half(w[i].dmin);
        const float *xb = x + i*KQ_QK_K;
        int is = 0; uint8_t sc, m;

        for(int j = 0; j < KQ_QK_K; j += 64){
            kq_sc_min_k4(is+0, w[i].scales, &sc, &m); const float d1 = d*sc, m1 = mn*m;
            kq_sc_min_k4(is+1, w[i].scales, &sc, &m); const float d2 = d*sc, m2 = mn*m;

            /* one 32-byte load feeds BOTH groups: low nibbles are group is+0, high are is+1 */
            const __m256i qb = _mm256_loadu_si256((const __m256i*)q);
            const __m256i lo = _mm256_and_si256(qb, lomask);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), lomask);

            __m256 s1 = _mm256_setzero_ps(), x1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), x2 = _mm256_setzero_ps();
            const float *xa = xb + j, *xc = xb + j + 32;

            __m256 f0, f1, f2, f3;
            kqs_u8x16(_mm256_castsi256_si128(lo),        &f0, &f1);
            kqs_u8x16(_mm256_extracti128_si256(lo, 1),   &f2, &f3);
            { __m256 v0=_mm256_loadu_ps(xa+0), v1=_mm256_loadu_ps(xa+8),
                     v2=_mm256_loadu_ps(xa+16),v3=_mm256_loadu_ps(xa+24);
              s1 = _mm256_fmadd_ps(f0,v0, s1); s1 = _mm256_fmadd_ps(f1,v1, s1);
              s1 = _mm256_fmadd_ps(f2,v2, s1); s1 = _mm256_fmadd_ps(f3,v3, s1);
              /* the min term needs only sum(x) — one accumulator, folded in once per group
               * instead of a subtract per weight */
              x1 = _mm256_add_ps(_mm256_add_ps(v0,v1), _mm256_add_ps(v2,v3)); }

            kqs_u8x16(_mm256_castsi256_si128(hi),        &f0, &f1);
            kqs_u8x16(_mm256_extracti128_si256(hi, 1),   &f2, &f3);
            { __m256 v0=_mm256_loadu_ps(xc+0), v1=_mm256_loadu_ps(xc+8),
                     v2=_mm256_loadu_ps(xc+16),v3=_mm256_loadu_ps(xc+24);
              s2 = _mm256_fmadd_ps(f0,v0, s2); s2 = _mm256_fmadd_ps(f1,v1, s2);
              s2 = _mm256_fmadd_ps(f2,v2, s2); s2 = _mm256_fmadd_ps(f3,v3, s2);
              x2 = _mm256_add_ps(_mm256_add_ps(v0,v1), _mm256_add_ps(v2,v3)); }

            acc = _mm256_fmadd_ps(_mm256_set1_ps(d1), s1, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(-m1), x1, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d2), s2, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(-m2), x2, acc);
            q += 32; is += 2;
        }
    }
    return kqs_hsum(acc);
}

/* ---------------- Q6_K ----------------
 * 6 bits per weight: 4 low in ql, 2 high in qh. One qh byte carries the high pairs of FOUR
 * weights at shifts 0/2/4/6, so a single 16-byte qh load feeds four 16-weight groups. The
 * -32 bias is applied on the bytes so the later sign-extend does the two's-complement work. */
static float kq_dot_q6_K_avx2(const void *vw, const float *x, int64_t n){
    const kq_q6_K *w = (const kq_q6_K*)vw;
    const __m128i m4  = _mm_set1_epi8(0x0F);
    const __m128i m3  = _mm_set1_epi8(0x03);
    const __m128i b32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for(int64_t i = 0; i < n/KQ_QK_K; i++){
        const float d = kq_half(w[i].d);
        const uint8_t *ql = w[i].ql, *qh = w[i].qh;
        const int8_t  *sc = w[i].scales;
        const float   *y  = x + i*KQ_QK_K;

        for(int nn = 0; nn < KQ_QK_K; nn += 128){
            /* l0 = 0 and 16: the scale index is l/16, so it is constant across each half */
            for(int l0 = 0; l0 < 32; l0 += 16){
                const int is = l0/16;
                const __m128i lA = _mm_loadu_si128((const __m128i*)(ql + l0));
                const __m128i lB = _mm_loadu_si128((const __m128i*)(ql + l0 + 32));
                const __m128i hh = _mm_loadu_si128((const __m128i*)(qh + l0));

                /* four weights out of one qh byte: shift down, keep 2 bits, put them at 4..5 */
                const __m128i q1 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(lA, m4),
                        _mm_slli_epi16(_mm_and_si128(hh, m3), 4)), b32);
                const __m128i q2 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(lB, m4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hh,2), m3), 4)), b32);
                const __m128i q3 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(lA,4), m4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hh,4), m3), 4)), b32);
                const __m128i q4 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(lB,4), m4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hh,6), m3), 4)), b32);

                const float s1 = d*sc[is+0], s2 = d*sc[is+2];
                const float s3 = d*sc[is+4], s4 = d*sc[is+6];
                __m256 a, b;
                #define KQS_Q6_ACC(QQ, SS, OFF) do {                                   \
                    kqs_i8x16((QQ), &a, &b);                                            \
                    const __m256 sv = _mm256_set1_ps(SS);                               \
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(a, sv),                         \
                                          _mm256_loadu_ps(y + (OFF) + 0), acc);         \
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(b, sv),                         \
                                          _mm256_loadu_ps(y + (OFF) + 8), acc);         \
                } while(0)
                KQS_Q6_ACC(q1, s1, l0 +  0);
                KQS_Q6_ACC(q2, s2, l0 + 32);
                KQS_Q6_ACC(q3, s3, l0 + 64);
                KQS_Q6_ACC(q4, s4, l0 + 96);
                #undef KQS_Q6_ACC
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    return kqs_hsum(acc);
}

/* ---------------- Q6_K, TWO activations, ONE decode ----------------
 * The other half of the model. Q4_K alone was not enough: ffn_down, attn_qkv, attn_v and the
 * output projection are all Q6_K — about 7.8 of the 15.6 GiB — so with only the Q4_K pair
 * kernel a batch of 2 still re-read half the weights and measured 0.74x, i.e. slower than
 * not batching at all. The 6-bit unpack (ql nibble + 2 bits out of qh at shift 0/2/4/6,
 * minus 32, sign-extended) is the expensive part and is exactly what gets shared here. */
/* Q4_K, two activations, one decode. Companion to the Q6_K version below — it was lost in a
 * file rewrite once, and its absence is what made a batch of 2 run 1.34x SLOWER than not
 * batching: without it kq_gemm_batched fell through to the dequantize-once path, whose
 * kq_dequant_row is SCALAR and whose fixed cost only pays off from about S=6. */
static void kq_dot_q4_K_avx2_x2(const void *vw, const float *xa, const float *xb,
                                int64_t n, float *ra, float *rb){
    const kq_q4_K *w = (const kq_q4_K*)vw;
    const __m256i lomask = _mm256_set1_epi8(0x0F);
    __m256 acca = _mm256_setzero_ps(), accb = _mm256_setzero_ps();

    for(int64_t i = 0; i < n/KQ_QK_K; i++){
        const uint8_t *q = w[i].qs;
        const float d = kq_half(w[i].d), mn = kq_half(w[i].dmin);
        const float *pa = xa + i*KQ_QK_K, *pb = xb + i*KQ_QK_K;
        int is = 0; uint8_t sc, m;

        for(int j = 0; j < KQ_QK_K; j += 64){
            kq_sc_min_k4(is+0, w[i].scales, &sc, &m); const float d1 = d*sc, m1 = mn*m;
            kq_sc_min_k4(is+1, w[i].scales, &sc, &m); const float d2 = d*sc, m2 = mn*m;

            const __m256i qb = _mm256_loadu_si256((const __m256i*)q);
            const __m256i grp[2] = { _mm256_and_si256(qb, lomask),
                                     _mm256_and_si256(_mm256_srli_epi16(qb, 4), lomask) };
            const float dd[2] = { d1, d2 }, mm[2] = { m1, m2 };

            for(int g = 0; g < 2; g++){
                const float *ca = pa + j + 32*g, *cb = pb + j + 32*g;
                __m256 f[4];
                kqs_u8x16(_mm256_castsi256_si128(grp[g]),      &f[0], &f[1]);
                kqs_u8x16(_mm256_extracti128_si256(grp[g], 1), &f[2], &f[3]);
                __m256 sa = _mm256_setzero_ps(), sb = _mm256_setzero_ps();
                __m256 ta = _mm256_setzero_ps(), tb = _mm256_setzero_ps();
                for(int k = 0; k < 4; k++){
                    const __m256 va = _mm256_loadu_ps(ca + 8*k);
                    const __m256 vb = _mm256_loadu_ps(cb + 8*k);
                    sa = _mm256_fmadd_ps(f[k], va, sa);
                    sb = _mm256_fmadd_ps(f[k], vb, sb);
                    ta = _mm256_add_ps(ta, va);
                    tb = _mm256_add_ps(tb, vb);
                }
                acca = _mm256_fmadd_ps(_mm256_set1_ps(dd[g]),  sa, acca);
                acca = _mm256_fmadd_ps(_mm256_set1_ps(-mm[g]), ta, acca);
                accb = _mm256_fmadd_ps(_mm256_set1_ps(dd[g]),  sb, accb);
                accb = _mm256_fmadd_ps(_mm256_set1_ps(-mm[g]), tb, accb);
            }
            q += 32; is += 2;
        }
    }
    *ra = kqs_hsum(acca);
    *rb = kqs_hsum(accb);
}

static void kq_dot_q6_K_avx2_x2(const void *vw, const float *xa, const float *xb,
                                int64_t n, float *ra, float *rb){
    const kq_q6_K *w = (const kq_q6_K*)vw;
    const __m128i m4  = _mm_set1_epi8(0x0F);
    const __m128i m3  = _mm_set1_epi8(0x03);
    const __m128i b32 = _mm_set1_epi8(32);
    __m256 acca = _mm256_setzero_ps(), accb = _mm256_setzero_ps();

    for(int64_t i = 0; i < n/KQ_QK_K; i++){
        const float d = kq_half(w[i].d);
        const uint8_t *ql = w[i].ql, *qh = w[i].qh;
        const int8_t  *sc = w[i].scales;
        const float *ya = xa + i*KQ_QK_K, *yb = xb + i*KQ_QK_K;

        for(int nn = 0; nn < KQ_QK_K; nn += 128){
            for(int l0 = 0; l0 < 32; l0 += 16){
                const int is = l0/16;
                const __m128i lA = _mm_loadu_si128((const __m128i*)(ql + l0));
                const __m128i lB = _mm_loadu_si128((const __m128i*)(ql + l0 + 32));
                const __m128i hh = _mm_loadu_si128((const __m128i*)(qh + l0));

                const __m128i q1 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(lA, m4),
                        _mm_slli_epi16(_mm_and_si128(hh, m3), 4)), b32);
                const __m128i q2 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(lB, m4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hh,2), m3), 4)), b32);
                const __m128i q3 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(lA,4), m4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hh,4), m3), 4)), b32);
                const __m128i q4 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(lB,4), m4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hh,6), m3), 4)), b32);

                const float s1 = d*sc[is+0], s2 = d*sc[is+2];
                const float s3 = d*sc[is+4], s4 = d*sc[is+6];
                __m256 a, b;
                #define KQS_Q6X2(QQ, SS, OFF) do {                                        \
                    kqs_i8x16((QQ), &a, &b);                                               \
                    const __m256 sv = _mm256_set1_ps(SS);                                  \
                    const __m256 wa = _mm256_mul_ps(a, sv), wb = _mm256_mul_ps(b, sv);     \
                    acca = _mm256_fmadd_ps(wa, _mm256_loadu_ps(ya+(OFF)+0), acca);         \
                    acca = _mm256_fmadd_ps(wb, _mm256_loadu_ps(ya+(OFF)+8), acca);         \
                    accb = _mm256_fmadd_ps(wa, _mm256_loadu_ps(yb+(OFF)+0), accb);         \
                    accb = _mm256_fmadd_ps(wb, _mm256_loadu_ps(yb+(OFF)+8), accb);         \
                } while(0)
                KQS_Q6X2(q1, s1, l0 +  0);
                KQS_Q6X2(q2, s2, l0 + 32);
                KQS_Q6X2(q3, s3, l0 + 64);
                KQS_Q6X2(q4, s4, l0 + 96);
                #undef KQS_Q6X2
            }
            ya += 128; yb += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    *ra = kqs_hsum(acca);
    *rb = kqs_hsum(accb);
}
#endif /* AVX2 */

/* ---------------- AVX-VNNI: integer dot, activation quantized ONCE ----------------
 *
 * This CPU has AVX-VNNI (`vpdpbusd`: 32 u8*i8 products accumulated into 8 i32 lanes in ONE
 * instruction). The float kernels above spend most of their work turning 4- and 6-bit
 * weights into floats so they can be multiplied by float activations. VNNI removes that
 * conversion entirely — the quantized weights ARE the operands.
 *
 * The trick that makes it worth it is amortization, and it is the same idea as the batched
 * gemm: the ACTIVATION is quantized once per gemv and reused across every output row. For
 * ffn_gate that is 17408 rows sharing one quantization of a 5120-long vector, so its cost is
 * 1/17408 of the work. Doing it per row instead would cost more than it saves — which is
 * exactly the mistake the batched path made with kq_dequant_row.
 *
 * Numerics: the activation goes to int8 with a per-32-block scale, so this is llama.cpp's
 * q8_K arrangement and carries the same error — around 0.3% on a dot, which is why
 * kquant_simd_test measures it against a DOUBLE reference rather than against the float
 * kernel. It is enabled per call site, not globally: attention scores feed a softmax that
 * amplifies error, so they keep the exact float path.
 *
 * WHERE IT ACTUALLY HELPS: prefill and any compute-bound path. Decode is BANDWIDTH bound
 * (36 GB/s achieved of 57.5 available), so a faster inner loop there has less to win — the
 * bytes still have to arrive. The measurement below says which.
 */
#ifdef KQS_HAVE_AVX2

/* int8 activation with one fp32 scale per 32 values. */
typedef struct { int8_t *q; float *d; int n; } KqsQ8;

static void kqs_quant_act(KqsQ8 *Q, const float *x, int n){
    const int nb = n/32;
    for(int b = 0; b < nb; b++){
        const float *p = x + b*32;
        __m256 m = _mm256_setzero_ps();
        const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for(int k = 0; k < 32; k += 8)
            m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(p+k), absmask));
        __m128 lo = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m,1));
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
        const float amax = _mm_cvtss_f32(lo);
        const float d = amax/127.0f, id = d ? 1.0f/d : 0.0f;
        Q->d[b] = d;
        for(int k = 0; k < 32; k++){
            int v = (int)lrintf(p[k]*id);
            if(v >  127) v =  127;
            if(v < -127) v = -127;
            Q->q[b*32+k] = (int8_t)v;
        }
    }
    Q->n = n;
}

/* vpdpbusd wrapper: u8 weights x i8 activations -> i32 lanes. Falls back to the AVX2
 * two-instruction form where the compiler has no VNNI. */
static inline __m256i kqs_dpbusd(__m256i acc, __m256i u, __m256i i){
#ifdef __AVXVNNI__
    return _mm256_dpbusd_avx_epi32(acc, u, i);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, i), ones));
#endif
}
static inline int kqs_hsum_i32(__m256i v){
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v,1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
    return _mm_cvtsi128_si32(lo);
}

/* Q4_K against a pre-quantized activation. Per 32-weight group g:
 *     d*sc[g]*dx[b] * sum(qw*qx)  -  dmin*m[g]*dx[b] * sum(qx)
 * Both sums come from vpdpbusd — the second by dotting the activation with all-ones. */
static float kq_dot_q4_K_vnni(const void *vw, const KqsQ8 *X, int64_t n){
    const kq_q4_K *w = (const kq_q4_K*)vw;
    const __m256i lomask = _mm256_set1_epi8(0x0F);
    const __m256i ones8  = _mm256_set1_epi8(1);
    float acc = 0;

    for(int64_t i = 0; i < n/KQ_QK_K; i++){
        const uint8_t *q = w[i].qs;
        const float d = kq_half(w[i].d), mn = kq_half(w[i].dmin);
        const int8_t *qx = X->q + i*KQ_QK_K;
        const float  *dx = X->d + i*(KQ_QK_K/32);
        int is = 0; uint8_t sc, m;

        for(int j = 0; j < KQ_QK_K; j += 64){
            kq_sc_min_k4(is+0, w[i].scales, &sc, &m); const float d1 = d*sc, m1 = mn*m;
            kq_sc_min_k4(is+1, w[i].scales, &sc, &m); const float d2 = d*sc, m2 = mn*m;

            const __m256i qb = _mm256_loadu_si256((const __m256i*)q);
            const __m256i lo = _mm256_and_si256(qb, lomask);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), lomask);

            const __m256i xa = _mm256_loadu_si256((const __m256i*)(qx + j));
            const __m256i xb = _mm256_loadu_si256((const __m256i*)(qx + j + 32));

            const int pa = kqs_hsum_i32(kqs_dpbusd(_mm256_setzero_si256(), lo, xa));
            const int sa = kqs_hsum_i32(kqs_dpbusd(_mm256_setzero_si256(), ones8, xa));
            const int pb = kqs_hsum_i32(kqs_dpbusd(_mm256_setzero_si256(), hi, xb));
            const int sb = kqs_hsum_i32(kqs_dpbusd(_mm256_setzero_si256(), ones8, xb));

            acc += dx[j/32]     * (d1*(float)pa - m1*(float)sa);
            acc += dx[j/32 + 1] * (d2*(float)pb - m2*(float)sb);
            q += 32; is += 2;
        }
    }
    return acc;
}

/* gemv with the activation quantized once for the whole matrix */
static void kq_gemv_vnni(float *y, const float *x, const void *W, int gtype, int I, int O){
    if(gtype != KQ_Q4_K || (I % KQ_QK_K) != 0){ kq_gemv_fast(y, x, W, gtype, I, O); return; }
    const int64_t rb = kq_row_bytes(gtype, I);
    KqsQ8 X;
    X.q = (int8_t*)malloc((size_t)I);
    X.d = (float*)malloc((size_t)(I/32)*sizeof(float));
    if(!X.q || !X.d){ free(X.q); free(X.d); kq_gemv_fast(y, x, W, gtype, I, O); return; }
    kqs_quant_act(&X, x, I);            /* ONCE — amortized over all O rows */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for(int o = 0; o < O; o++)
        y[o] = kq_dot_q4_K_vnni((const uint8_t*)W + (int64_t)o*rb, &X, I);
    free(X.q); free(X.d);
}
#endif /* AVX2 */

/* ---------------- observable dispatch ----------------
 *
 * WHY THIS EXISTS, AND WHY IT IS WORTH A GLOBAL TABLE
 *
 * An intervention measurement has TWO premises: that the measurement is sound, and that the
 * intervention was actually in the binary. Only the first is ever checked. This project has
 * now paid for the second twice — a pair-kernel dispatch was lost in a file rewrite, and the
 * conclusion drawn was "adding the pair kernels did not help, therefore the cost is not weight
 * decode." The measurement was real; the code being credited was not there. That closed a
 * branch wrongly, and cost a 1.84x speedup for a while.
 *
 * Negative results are precisely the ones acted on by abandoning a direction, so the premise
 * that is invisible is the one that must be made observable. QWEN_KERNEL_LOG=1 prints, once
 * per (quant, S-class) pair, which kernel actually ran. "Which path executed" then becomes an
 * observation rather than an inference, for every future negative result. */
static void kqs_log(const char *kernel, int gtype, int S){
    static const char *seen[64];
    static int n_seen = 0;
    static int enabled = -1;
    if(enabled < 0) enabled = (getenv("QWEN_KERNEL_LOG") != NULL);
    if(!enabled) return;
    char key[96];
    snprintf(key, sizeof key, "%s/%s/S%s", kernel, kq_name(gtype),
             S <= 1 ? "1" : (S <= 4 ? "2-4" : "5+"));
    for(int i = 0; i < n_seen; i++) if(!strcmp(seen[i], key)) return;
    if(n_seen < 64){
        seen[n_seen] = strdup(key);
        n_seen++;
        fprintf(stderr, "[kernel] %s\n", key);
    }
}

/* Dispatch. Everything not accelerated falls through to the scalar kernels, which stay the
 * reference the fast paths are diffed against. */
static inline float kq_dot_fast(int t, const void *w, const float *x, int64_t n){
#ifdef KQS_HAVE_AVX2
    if(t == KQ_Q4_K) return kq_dot_q4_K_avx2(w, x, n);
    if(t == KQ_Q6_K) return kq_dot_q6_K_avx2(w, x, n);
#endif
    return kq_dot(t, w, x, n);
}

/* ---------------- keeping the right things in cache ----------------
 *
 * The instinct is to pin the hot data IN cache. On x86 you cannot: there is no cache-pin
 * instruction on consumer parts (Xeon CAT can partition L3, Raptor Lake cannot), and the
 * genuinely hot metadata is tiny anyway — the whole weight-pointer table is 866 pointers,
 * 7 KB, resident in L2 after the first touch and never a cost again.
 *
 * The lever that DOES exist is the inverse: keep the cold data OUT.
 *
 * A gemv reads one 20 KB activation vector and streams 50-100 MB of weights past it. The
 * activation is reused by every one of the 17408 output rows and wants to stay in L1; the
 * weights are read exactly once per token and will never be looked at again. Left alone, the
 * weight stream evicts the activation from L1 continuously and then has to be re-fetched.
 *
 * So: prefetch the NEXT row with _MM_HINT_NTA. "Non-temporal" tells the hardware this line
 * has no reuse — it lands in L1 without displacing the L2/L3 working set behind it. The
 * activation stays put, and the prefetch also covers the row-start latency the hardware
 * prefetcher misses because consecutive rows are `rb` apart, not contiguous.
 *
 * MEASURED, AND IT LOSES — off by default. A/B on the real model, 16 tokens:
 *     without NTA : 0.43 s/token, FFN 36.34 GB/s
 *     with NTA    : 0.45 s/token, FFN 34.15 GB/s
 * The hardware prefetcher already covers a stream this regular, so the explicit hints add
 * nothing to fetch and cost the issue slots they occupy. That is the general shape of this
 * class of idea on a sequential workload: the cache is not being used wrongly, it is being
 * used the only way a single-pass stream can use it. QWEN_NTA=1 to retry. */
static int kqs_use_nta = -1;
static inline int kqs_nta(void){
    if(kqs_use_nta < 0) kqs_use_nta = (getenv("QWEN_NTA") != NULL);
    return kqs_use_nta;
}

/* Row-parallel gemv over a [O, I] tensor, using the fast dot. Same contract as kq_gemm. */
static void kq_gemv_fast(float *y, const float *x, const void *W, int gtype, int I, int O){
    kqs_log("gemv_fast:fused", gtype, 1);
    const int64_t rb = kq_row_bytes(gtype, I);
    const int nta = kqs_nta();
#ifdef _OPENMP
    if(!omp_in_parallel()){
        #pragma omp parallel for schedule(static)
        for(int o = 0; o < O; o++){
            const uint8_t *w = (const uint8_t*)W + (int64_t)o*rb;
#if defined(KQS_HAVE_AVX2)
            if(nta && o + 1 < O){
                const uint8_t *nx = w + rb;
                /* one hint per 64-byte line, capped: the point is to start the row, not to
                 * issue thousands of prefetches that themselves cost issue slots */
                for(int64_t b = 0; b < rb && b < 1024; b += 64)
                    _mm_prefetch((const char*)nx + b, _MM_HINT_NTA);
            }
#endif
            y[o] = kq_dot_fast(gtype, w, x, I);
        }
        return;
    }
#endif
    for(int o = 0; o < O; o++){
        const uint8_t *w = (const uint8_t*)W + (int64_t)o*rb;
#if defined(KQS_HAVE_AVX2)
        if(nta && o + 1 < O)
            for(int64_t b = 0; b < rb && b < 1024; b += 64)
                _mm_prefetch((const char*)(w + rb) + b, _MM_HINT_NTA);
#endif
        y[o] = kq_dot_fast(gtype, w, x, I);
    }
}

/* ---------------- batched: decode the row ONCE, dot it S times ----------------
 *
 * kq_gemm's batched path calls kq_dot once per (row, token), so a batch of 64 decodes every
 * weight row 64 times over. Batching then buys exactly nothing: prefill costs as much per
 * token as decode does, and prefilling 8k tokens takes an hour and a half.
 *
 * The fix separates two costs that were fused. Decoding a Q4_K row of 5120 weights touches
 * 2880 quantized bytes and produces 20 KB of floats — which fits in L1 (48 KB on Raptor
 * Lake). So dequantize ONCE into that scratch, then run S plain fp32 dots against it. The
 * quantized read is amortized by S; what remains is FMA throughput.
 *
 * That changes which wall you hit. Decode is bandwidth-bound: 15.6 GiB of weights per token
 * against 57.5 GB/s of RAM. Prefill at S=64 reads those 15.6 GiB once per BATCH — 0.24 GiB
 * per token-equivalent — and becomes compute-bound instead. A different wall, several times
 * further out.
 *
 * The scratch is per-thread and allocated once per parallel region: a malloc per output row
 * would cost more than the dequantization it exists to serve. */
static float kqs_dot_f32(const float *a, const float *b, int n){
#ifdef KQS_HAVE_AVX2
    /* Four accumulators, not one: a single 8-wide chain is FMA-LATENCY bound (4-5 cycles per
     * step) rather than throughput bound (2 per cycle). Four independent chains keep both
     * FMA ports fed. */
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for(; i + 32 <= n; i += 32){
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+ 0), _mm256_loadu_ps(b+i+ 0), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+ 8), _mm256_loadu_ps(b+i+ 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), a3);
    }
    for(; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), a0);
    float acc = kqs_hsum(_mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3)));
    for(; i < n; i++) acc += a[i]*b[i];
    return acc;
#else
    float acc = 0;
    for(int i = 0; i < n; i++) acc += a[i]*b[i];
    return acc;
#endif
}

/* THE SECOND HALF OF THE PROBLEM: THE ACTIVATIONS.
 *
 * Dequantizing the row once fixes the weight read, and then the naive inner loop re-reads
 * ALL of x for every output row: with O=4096 and S=32 that is 4096 x 640 KB = 2.7 GB of
 * activation traffic per gemm, and batching measures only 1.5x instead of the 4x it should.
 *
 * So the loop is blocked: NR output rows against NS tokens at once, accumulating NR*NS
 * partial sums in registers while I streams past. Each loaded weight vector is used NS times
 * and each loaded activation vector NR times, so the load:FMA ratio goes from 2:1 to 6:8.
 * NR=4, NS=2 is chosen to fit 8 accumulators + 4 weight + 2 activation vectors inside the
 * 16 YMM registers — one more of either and the compiler starts spilling, which costs more
 * than the blocking saves. */
#define KQS_NR 4
#define KQS_NS 2

/* y is [S, O] row-major, x is [S, I] row-major, W is [O, I] quantized. */
/* Two activations against one decoded row. Q4_K and Q6_K share the decode; anything else
 * falls back to two fused dots, which is still the 2.0x floor rather than worse. */
static void kq_dot_fast_x2(int t, const void *w, const float *xa, const float *xb,
                           int64_t n, float *ra, float *rb){
#ifdef KQS_HAVE_AVX2
    if(t == KQ_Q4_K){ kq_dot_q4_K_avx2_x2(w, xa, xb, n, ra, rb); return; }
    if(t == KQ_Q6_K){ kq_dot_q6_K_avx2_x2(w, xa, xb, n, ra, rb); return; }
#endif
    *ra = kq_dot_fast(t, w, xa, n);
    *rb = kq_dot_fast(t, w, xb, n);
}

static void kq_gemm_batched(float *y, const float *x, const void *W, int gtype,
                            int S, int I, int O){
    if(S <= 1){ kq_gemv_fast(y, x, W, gtype, I, O); return; }
    const int64_t rb = kq_row_bytes(gtype, I);

    /* SMALL S MUST NOT TAKE THE DEQUANTIZE-ONCE PATH. That path materializes a float row via
     * kq_dequant_row, which is scalar; its fixed cost only pays from about S=6. Below that,
     * share the DECODE instead of the dequantized row. Getting this wrong is not a missed
     * optimization — measured, a batch of 2 came out 1.34x slower than two single passes,
     * i.e. worse than not batching at all, and uniformly across every component because
     * every component goes through this one function. */
    if(S <= 4){
        kqs_log("gemm_batched:shared-decode", gtype, S);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
        for(int o = 0; o < O; o++){
            const uint8_t *w = (const uint8_t*)W + (int64_t)o*rb;
            int s = 0;
            for(; s + 2 <= S; s += 2)
                kq_dot_fast_x2(gtype, w, x + (int64_t)s*I, x + (int64_t)(s+1)*I, I,
                               &y[(int64_t)s*O + o], &y[(int64_t)(s+1)*O + o]);
            for(; s < S; s++)
                y[(int64_t)s*O + o] = kq_dot_fast(gtype, w, x + (int64_t)s*I, I);
        }
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel if(!omp_in_parallel())
#endif
    {
        float *rows = (float*)malloc((size_t)KQS_NR*I*sizeof(float));
        if(rows){
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for(int ob = 0; ob < O; ob += KQS_NR){
                const int nr = (ob + KQS_NR <= O) ? KQS_NR : (O - ob);
                for(int r = 0; r < nr; r++)
                    kq_dequant_row(gtype, (const uint8_t*)W + (int64_t)(ob+r)*rb,
                                   rows + (size_t)r*I, I);

                int sb = 0;
#ifdef KQS_HAVE_AVX2
                for(; sb + KQS_NS <= S; sb += KQS_NS){
                    __m256 acc[KQS_NR][KQS_NS];
                    for(int r = 0; r < nr; r++)
                        for(int t = 0; t < KQS_NS; t++) acc[r][t] = _mm256_setzero_ps();
                    int i = 0;
                    for(; i + 8 <= I; i += 8){
                        __m256 xv[KQS_NS];
                        for(int t = 0; t < KQS_NS; t++)
                            xv[t] = _mm256_loadu_ps(x + (int64_t)(sb+t)*I + i);
                        for(int r = 0; r < nr; r++){
                            const __m256 wv = _mm256_loadu_ps(rows + (size_t)r*I + i);
                            for(int t = 0; t < KQS_NS; t++)
                                acc[r][t] = _mm256_fmadd_ps(wv, xv[t], acc[r][t]);
                        }
                    }
                    for(int r = 0; r < nr; r++)
                        for(int t = 0; t < KQS_NS; t++){
                            float v = kqs_hsum(acc[r][t]);
                            for(int k = i; k < I; k++)
                                v += rows[(size_t)r*I + k] * x[(int64_t)(sb+t)*I + k];
                            y[(int64_t)(sb+t)*O + ob + r] = v;
                        }
                }
#endif
                for(; sb < S; sb++)                       /* the tail, and the no-AVX2 path */
                    for(int r = 0; r < nr; r++)
                        y[(int64_t)sb*O + ob + r] =
                            kqs_dot_f32(rows + (size_t)r*I, x + (int64_t)sb*I, I);
            }
            free(rows);
        }
    }
}

#endif
