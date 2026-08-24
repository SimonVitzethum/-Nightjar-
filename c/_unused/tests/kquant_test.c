/* kquant_test.c — assert colibri's k-quant decoders are BIT-IDENTICAL to ggml's.
 *
 * The on-disk k-quant layouts are not documented anywhere authoritative; ggml IS
 * the specification. So we do not eyeball the shifts — we quantize with ggml's
 * reference quantizer, dequantize with both ggml and kquant.h, and require the
 * float bits to match exactly. Any drift here is silent model corruption.
 *
 * Build: see tests/Makefile.kquant (needs llama.cpp's ggml sources for the oracle).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../kquant.h"

/* ggml-quants.c calls this on malformed input; we never feed it any, but it must link. */
void ggml_abort(const char *file, int line, const char *fmt, ...);
void ggml_abort(const char *file, int line, const char *fmt, ...){
    (void)fmt; fprintf(stderr, "ggml_abort at %s:%d\n", file, line); exit(1);
}
/* Pulled in by ggml's imatrix-aware quantizers, which this test does not call.
 * Defined against kquant.h's own tables so it stays honest if it ever is. */
size_t ggml_row_size(int type, int64_t ne);
size_t ggml_row_size(int type, int64_t ne){
    if(!kq_supported(type)){ fprintf(stderr, "ggml_row_size: type %d\n", type); exit(1); }
    return (size_t)kq_row_bytes(type, ne);
}
size_t ggml_type_size(int type);
size_t ggml_type_size(int type){ return (size_t)kq_typesize(type); }
const char *ggml_type_name(int type);
const char *ggml_type_name(int type){ return kq_name(type); }

/* ggml oracle (compiled in from llama.cpp; declared here to avoid its headers) */
void quantize_row_q2_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q4_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q5_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q6_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q8_0_ref(const float *x, void *y, int64_t k);
void dequantize_row_q2_K(const void *x, float *y, int64_t k);
void dequantize_row_q4_K(const void *x, float *y, int64_t k);
void dequantize_row_q5_K(const void *x, float *y, int64_t k);
void dequantize_row_q6_K(const void *x, float *y, int64_t k);
void dequantize_row_q8_0(const void *x, float *y, int64_t k);
void dequantize_row_iq2_xs(const void *x, float *y, int64_t k);
void dequantize_row_iq3_xxs(const void *x, float *y, int64_t k);
void dequantize_row_q3_K(const void *x, float *y, int64_t k);
void dequantize_row_iq4_xs(const void *x, float *y, int64_t k);
void quantize_row_q3_K_ref(const float *x, void *y, int64_t k);
size_t quantize_iq4_xs(const float *src, void *dst, int64_t nrow, int64_t n_per_row, const float *imatrix);
/* i-quants have no plain _ref quantizer. The lattice search needs (a) precomputed
 * neighbour tables and (b) a non-NULL importance vector — ggml asserts on both. We give
 * it uniform importance: we are conformance-testing a DECODER, not reproducing Unsloth's
 * imatrix. The resulting bytes are valid IQ2_XS/IQ3_XXS either way, which is all the
 * decoder cares about. */
size_t quantize_iq2_xs (const float *src, void *dst, int64_t nrow, int64_t n_per_row, const float *imatrix);
size_t quantize_iq3_xxs(const float *src, void *dst, int64_t nrow, int64_t n_per_row, const float *imatrix);
void iq2xs_init_impl(int type);      /* enum ggml_type; IQ2_XS == 17 */
void iq3xs_init_impl(int grid_size); /* IQ3_XXS uses the 256-point grid */

static int fails = 0;

static void check_size(const char *nm, size_t got, size_t want){
    if(got != want){ printf("  FAIL %-6s sizeof = %zu, ggml says %zu\n", nm, got, want); fails++; }
    else            printf("  ok   %-6s sizeof = %zu B\n", nm, got);
}

/* bit-exact comparison: k-quant dequant is pure fp32 arithmetic in a fixed order,
 * so "close enough" is not the bar — identical bits are. */
static void cmp_rows(const char *nm, const float *a, const float *b, int64_t n){
    int64_t bad = 0; float worst = 0.f;
    for(int64_t i = 0; i < n; i++){
        if(memcmp(&a[i], &b[i], sizeof(float)) != 0){
            bad++;
            float d = fabsf(a[i] - b[i]);
            if(d > worst) worst = d;
        }
    }
    if(bad){ printf("  FAIL %-6s dequant: %lld/%lld floats differ (max |d| = %g)\n",
                    nm, (long long)bad, (long long)n, worst); fails++; }
    else   printf("  ok   %-6s dequant bit-identical over %lld weights\n", nm, (long long)n);
}

/* the streamed path never materializes the row, so the fused dot must agree with
 * dequantize-then-dot. fp32 reassociation makes this a tolerance check, not a bit check. */
static void cmp_dot(const char *nm, int type, const void *q, const float *x,
                    const float *deq, int64_t n){
    float fused = kq_dot(type, q, x, n);
    double ref = 0.0;
    for(int64_t i = 0; i < n; i++) ref += (double)deq[i] * (double)x[i];
    float rel = (float)(fabs(fused - ref) / (fabs(ref) + 1e-9));
    if(rel > 1e-5f){ printf("  FAIL %-6s fused dot %.6f vs ref %.6f (rel %.2e)\n",
                            nm, fused, ref, rel); fails++; }
    else           printf("  ok   %-6s fused dot matches dequant+dot (rel %.1e)\n", nm, rel);
}

int main(void){
    const int64_t N = 256 * 64;          /* 64 super-blocks */

    printf("block layouts vs ggml-common.h\n");
    check_size("Q2_K", sizeof(kq_q2_K), 84);
    check_size("Q4_K", sizeof(kq_q4_K), 144);
    check_size("Q5_K", sizeof(kq_q5_K), 176);
    check_size("Q6_K", sizeof(kq_q6_K), 210);
    check_size("Q8_0", sizeof(kq_q8_0), 34);
    check_size("IQ2_XS",  sizeof(kq_iq2_xs),  74);
    check_size("IQ3_XXS", sizeof(kq_iq3_xxs), 98);
    check_size("Q3_K",   sizeof(kq_q3_K),   110);
    check_size("IQ4_XS", sizeof(kq_iq4_xs), 136);

    float *x    = malloc(N * sizeof(float));
    float *ours = malloc(N * sizeof(float));
    float *ggml = malloc(N * sizeof(float));
    void  *q    = malloc(N * 2);          /* generous: largest type is 6.5 bpw */
    if(!x || !ours || !ggml || !q){ printf("OOM\n"); return 1; }

    /* trained weights are ~gaussian around zero; a couple of outliers exercise the
     * per-sub-block scales, which is where a wrong shift would hide. */
    srand(1234);
    for(int64_t i = 0; i < N; i++){
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0), u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        x[i] = (float)(0.02 * sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2));
    }
    x[0] = 0.9f; x[137] = -0.75f; x[4001] = 0.5f;

    printf("\ndequant, bit-for-bit against ggml\n");

    quantize_row_q2_K_ref(x, q, N);
    dequantize_row_q2_K(q, ggml, N);  kq_dequant_row(KQ_Q2_K, q, ours, N);
    cmp_rows("Q2_K", ours, ggml, N);
    cmp_dot ("Q2_K", KQ_Q2_K, q, x, ggml, N);

    quantize_row_q4_K_ref(x, q, N);
    dequantize_row_q4_K(q, ggml, N);  kq_dequant_row(KQ_Q4_K, q, ours, N);
    cmp_rows("Q4_K", ours, ggml, N);
    cmp_dot ("Q4_K", KQ_Q4_K, q, x, ggml, N);

    quantize_row_q6_K_ref(x, q, N);
    dequantize_row_q6_K(q, ggml, N);  kq_dequant_row(KQ_Q6_K, q, ours, N);
    cmp_rows("Q6_K", ours, ggml, N);
    cmp_dot ("Q6_K", KQ_Q6_K, q, x, ggml, N);

    quantize_row_q5_K_ref(x, q, N);
    dequantize_row_q5_K(q, ggml, N);  kq_dequant_row(KQ_Q5_K, q, ours, N);
    cmp_rows("Q5_K", ours, ggml, N);
    cmp_dot ("Q5_K", KQ_Q5_K, q, x, ggml, N);

    quantize_row_q8_0_ref(x, q, N);
    dequantize_row_q8_0(q, ggml, N);  kq_dequant_row(KQ_Q8_0, q, ours, N);
    cmp_rows("Q8_0", ours, ggml, N);
    cmp_dot ("Q8_0", KQ_Q8_0, q, x, ggml, N);

    /* The ones that actually carry the routed experts. */
    iq2xs_init_impl(KQ_IQ2_XS);
    iq3xs_init_impl(256);
    float *imp = malloc(N * sizeof(float));
    for(int64_t i = 0; i < N; i++) imp[i] = 1.0f;

    quantize_iq2_xs(x, q, 1, N, imp);
    dequantize_row_iq2_xs(q, ggml, N);  kq_dequant_row(KQ_IQ2_XS, q, ours, N);
    cmp_rows("IQ2XS", ours, ggml, N);
    cmp_dot ("IQ2XS", KQ_IQ2_XS, q, x, ggml, N);

    quantize_iq3_xxs(x, q, 1, N, imp);
    dequantize_row_iq3_xxs(q, ggml, N);  kq_dequant_row(KQ_IQ3_XXS, q, ours, N);
    cmp_rows("IQ3XX", ours, ggml, N);
    cmp_dot ("IQ3XX", KQ_IQ3_XXS, q, x, ggml, N);

    /* The rare types Unsloth sprinkles in where the imatrix demands more bits. */
    quantize_row_q3_K_ref(x, q, N);
    dequantize_row_q3_K(q, ggml, N);  kq_dequant_row(KQ_Q3_K, q, ours, N);
    cmp_rows("Q3_K", ours, ggml, N);
    cmp_dot ("Q3_K", KQ_Q3_K, q, x, ggml, N);

    quantize_iq4_xs(x, q, 1, N, imp);
    dequantize_row_iq4_xs(q, ggml, N);  kq_dequant_row(KQ_IQ4_XS, q, ours, N);
    cmp_rows("IQ4XS", ours, ggml, N);
    cmp_dot ("IQ4XS", KQ_IQ4_XS, q, x, ggml, N);

    /* what each format actually costs us, for the record */
    printf("\nbits/weight and reconstruction error (gaussian sigma=0.02)\n");
    struct { int t; const char *n; } T[] = {
        {KQ_IQ2_XS,"IQ2_XS"}, {KQ_Q2_K,"Q2_K"}, {KQ_IQ3_XXS,"IQ3_XXS"},
        {KQ_Q4_K,"Q4_K"}, {KQ_Q5_K,"Q5_K"}, {KQ_Q6_K,"Q6_K"}, {KQ_Q8_0,"Q8_0"}
    };
    for(unsigned t = 0; t < sizeof(T)/sizeof(T[0]); t++){
        switch(T[t].t){
            case KQ_Q2_K:    quantize_row_q2_K_ref(x, q, N); break;
            case KQ_Q4_K:    quantize_row_q4_K_ref(x, q, N); break;
            case KQ_Q5_K:    quantize_row_q5_K_ref(x, q, N); break;
            case KQ_Q6_K:    quantize_row_q6_K_ref(x, q, N); break;
            case KQ_Q8_0:    quantize_row_q8_0_ref(x, q, N); break;
            case KQ_IQ2_XS:  quantize_iq2_xs (x, q, 1, N, imp); break;
            case KQ_IQ3_XXS: quantize_iq3_xxs(x, q, 1, N, imp); break;
        }
        kq_dequant_row(T[t].t, q, ours, N);
        double se = 0, sr = 0;
        for(int64_t i = 0; i < N; i++){ double d = x[i] - ours[i]; se += d*d; sr += (double)x[i]*x[i]; }
        double bpw = 8.0 * (double)kq_row_bytes(T[t].t, N) / (double)N;
        printf("  %-7s %5.3f bit/W   rel. RMSE %6.2f%%\n", T[t].n, bpw, 100.0 * sqrt(se/sr));
    }

    /* kq_gemm's row stride is the one place a silent corruption could hide: get
     * kq_row_bytes wrong and every row but the first reads from the wrong offset,
     * which still produces plausible-looking floats. Check a real O x I matrix. */
    printf("\nkq_gemm (the streamed-expert path): W[O,I] contracted in place\n");
    {
        const int O = 12, I = 512, S = 2;
        float *W  = malloc((size_t)O * I * sizeof(float));
        float *xs = malloc((size_t)S * I * sizeof(float));
        void  *Wq = malloc((size_t)O * 2 * I);
        float *row = malloc((size_t)I * sizeof(float));
        float *got = malloc((size_t)S * O * sizeof(float));

        for(int i = 0; i < O * I; i++) W[i] = (float)(0.02 * sin(i * 0.37));
        for(int i = 0; i < S * I; i++) xs[i] = (float)cos(i * 0.11);

        int types[] = { KQ_Q2_K, KQ_Q4_K, KQ_Q6_K, KQ_IQ2_XS, KQ_IQ3_XXS };
        const char *tn[] = { "Q2_K", "Q4_K", "Q6_K", "IQ2_XS", "IQ3_XXS" };
        for(unsigned t = 0; t < sizeof(types)/sizeof(types[0]); t++){
            int ty = types[t];
            int64_t rb = kq_row_bytes(ty, I);
            for(int o = 0; o < O; o++){
                const float *src = W + (size_t)o * I;
                void *dst = (uint8_t*)Wq + (size_t)o * rb;
                switch(ty){
                    case KQ_Q2_K: quantize_row_q2_K_ref(src, dst, I); break;
                    case KQ_Q4_K: quantize_row_q4_K_ref(src, dst, I); break;
                    case KQ_Q6_K: quantize_row_q6_K_ref(src, dst, I); break;
                    case KQ_IQ2_XS:  quantize_iq2_xs (src, dst, 1, I, imp); break;
                    case KQ_IQ3_XXS: quantize_iq3_xxs(src, dst, 1, I, imp); break;
                }
            }
            kq_gemm(got, xs, Wq, ty, S, I, O);

            /* reference: dequantize each row, then plain dot */
            double worst = 0;
            for(int s = 0; s < S; s++) for(int o = 0; o < O; o++){
                kq_dequant_row(ty, (uint8_t*)Wq + (size_t)o * rb, row, I);
                double ref = 0;
                for(int i = 0; i < I; i++) ref += (double)row[i] * (double)xs[s*I + i];
                double rel = fabs(got[s*O + o] - ref) / (fabs(ref) + 1e-6);
                if(rel > worst) worst = rel;
            }
            if(worst > 1e-4){ printf("  FAIL %-7s gemm: worst rel error %.2e (row stride wrong?)\n",
                                     tn[t], worst); fails++; }
            else            printf("  ok   %-7s gemm matches dequant+dot on all %d rows (rel %.1e)\n",
                                   tn[t], O, worst);
        }
        free(W); free(xs); free(Wq); free(row); free(got);
    }

    free(imp);
    free(x); free(ours); free(ggml); free(q);
    printf("\n%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
