/* dsv4_test.c — check colibri's hyper-connections against ggml, op for op.
 *
 * Sinkhorn is the one part of DeepSeek-V4 that fails SILENTLY. Swap a row normalization
 * for a column one and nothing crashes: the combination matrix is still a plausible
 * 4x4 of positive numbers, the model still emits fluent text, and it is quietly wrong
 * across all 43 layers. So we do not trust a careful reading — we build the same graph
 * with real ggml ops (exactly as llama.cpp's build_hc_sinkhorn does) and compare.
 *
 * Also checks the property Sinkhorn exists to enforce: the result must be
 * doubly stochastic, i.e. every row AND every column sums to ~1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../dsv4.h"

#include "ggml.h"
#include "ggml-cpu.h"

#define HC 4
#define EPS 1e-6f

static int fails = 0;
static void ck(int ok, const char *msg){
    printf("  %s %s\n", ok ? "ok  " : "FAIL", msg);
    if(!ok) fails++;
}

/* ggml oracle: literally build_hc_sinkhorn from llama.cpp/src/models/deepseek4.cpp */
static void sinkhorn_ggml(const float *in, float *out, int hc, float eps, int iters){
    size_t bufsz = 64u*1024u*1024u;
    struct ggml_init_params ip = { bufsz, malloc(bufsz), false };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, 1);
    memcpy(comb->data, in, (size_t)hc*hc*sizeof(float));

    comb = ggml_soft_max(ctx, comb);

    struct ggml_tensor *e = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    e = ggml_fill(ctx, e, eps);
    comb = ggml_add(ctx, comb, e);

    /* norm_cols(): permute to [src,dst], sum along ne0 (src), permute back, divide */
    #define NORM_COLS() do {                                                         \
        struct ggml_tensor *sd = ggml_cont(ctx, ggml_permute(ctx, comb, 1,0,2,3));   \
        struct ggml_tensor *cs = ggml_sum_rows(ctx, sd);                             \
        cs = ggml_add(ctx, cs, e);                                                   \
        cs = ggml_permute(ctx, cs, 1,0,2,3);                                          \
        comb = ggml_div(ctx, comb, cs);                                              \
    } while(0)
    /* norm_rows(): sum along ne0 (dst), divide */
    #define NORM_ROWS() do {                                                         \
        struct ggml_tensor *rs = ggml_sum_rows(ctx, comb);                           \
        rs = ggml_add(ctx, rs, e);                                                   \
        comb = ggml_div(ctx, comb, rs);                                              \
    } while(0)

    NORM_COLS();
    for(int i = 1; i < iters; i++){ NORM_ROWS(); NORM_COLS(); }
    #undef NORM_COLS
    #undef NORM_ROWS

    struct ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, comb);
    ggml_graph_compute_with_ctx(ctx, g, 1);

    memcpy(out, comb->data, (size_t)hc*hc*sizeof(float));
    ggml_free(ctx);
    free(ip.mem_buffer);
}

int main(void){
    srand(99);
    printf("Sinkhorn: colibri (dsv4.h) vs ggml (llama.cpp build_hc_sinkhorn)\n");

    float worst = 0.f;
    for(int trial = 0; trial < 200; trial++){
        float in[HC*HC], mine[HC*HC], ref[HC*HC];
        for(int i = 0; i < HC*HC; i++)
            in[i] = (float)(4.0 * ((double)rand()/RAND_MAX - 0.5));   /* logits, both signs */

        memcpy(mine, in, sizeof(in));
        dsv4_sinkhorn(mine, HC, EPS, 20);          /* 20 iters, as the GGUF metadata says */
        sinkhorn_ggml(in, ref, HC, EPS, 20);

        for(int i = 0; i < HC*HC; i++){
            float d = fabsf(mine[i] - ref[i]) / (fabsf(ref[i]) + 1e-6f);
            if(d > worst) worst = d;
        }
    }
    printf("  worst relative deviation over 200 random matrices: %.2e\n", worst);
    ck(worst < 1e-4f, "matches ggml elementwise");

    /* the invariant Sinkhorn is there to produce */
    float m[HC*HC];
    for(int i = 0; i < HC*HC; i++) m[i] = (float)(3.0*((double)rand()/RAND_MAX - 0.5));
    dsv4_sinkhorn(m, HC, EPS, 20);

    float rmin = 1e9f, rmax = -1e9f, cmin = 1e9f, cmax = -1e9f;
    for(int s = 0; s < HC; s++){                     /* sum over dst, per src */
        float sum = 0; for(int d = 0; d < HC; d++) sum += m[d + s*HC];
        if(sum < rmin) rmin = sum; if(sum > rmax) rmax = sum;
    }
    for(int d = 0; d < HC; d++){                     /* sum over src, per dst */
        float sum = 0; for(int s = 0; s < HC; s++) sum += m[d + s*HC];
        if(sum < cmin) cmin = sum; if(sum > cmax) cmax = sum;
    }
    printf("  row sums in [%.4f, %.4f], column sums in [%.4f, %.4f]\n", rmin, rmax, cmin, cmax);
    ck(fabsf(cmin-1.f) < 1e-3f && fabsf(cmax-1.f) < 1e-3f,
       "columns are normalized (the last Sinkhorn step)");
    ck(rmin > 0.5f && rmax < 1.6f, "rows stay near 1 (doubly stochastic, not collapsed)");

    /* hc_pre / hc_post round trip: with post=0 and comb=I the streams must survive intact */
    printf("\nhc_pre / hc_post plumbing\n");
    {
        const int D = 8;
        float x[HC*8], y[8], scratch[HC*8];
        for(int i = 0; i < HC*D; i++) x[i] = (float)(i % 7) - 3.f;

        dsv4_hc st;
        for(int h = 0; h < HC; h++){ st.pre[h] = 0.f; st.post[h] = 0.f; }
        for(int i = 0; i < HC*HC; i++) st.comb[i] = 0.f;
        for(int h = 0; h < HC; h++) st.comb[h + h*HC] = 1.f;      /* identity */
        for(int e = 0; e < D; e++) y[e] = 999.f;                   /* must be ignored (post=0) */

        float before[HC*8]; memcpy(before, x, sizeof(before));
        dsv4_hc_post(&st, x, y, HC, D, scratch);
        ck(memcmp(before, x, sizeof(before)) == 0,
           "post=0, comb=I leaves the streams untouched");

        /* comb=I, post=1 must add y to every stream exactly once */
        for(int h = 0; h < HC; h++) st.post[h] = 1.f;
        for(int e = 0; e < D; e++) y[e] = 2.f;
        dsv4_hc_post(&st, x, y, HC, D, scratch);
        int ok = 1;
        for(int h = 0; h < HC; h++) for(int e = 0; e < D; e++)
            if(fabsf(x[h*D+e] - (before[h*D+e] + 2.f)) > 1e-5f) ok = 0;
        ck(ok, "post=1, comb=I adds the block output to each stream");
    }

    /* ---- router: sqrt(softplus(logits)), biased selection, unbiased weights ---- */
    printf("\nMoE router (SQRT_SOFTPLUS gating, DeepSeek-V4)\n");
    {
        const int NE = 32, K = 6;
        float logits[32], bias[32], probs[32], w[6];
        int idx[6];
        for(int e = 0; e < NE; e++){
            logits[e] = (float)(6.0*((double)rand()/RAND_MAX - 0.5));
            bias[e]   = (float)(1.0*((double)rand()/RAND_MAX - 0.5));
        }

        /* ggml oracle for probs = sqrt(softplus(logits)) */
        size_t bufsz = 4u*1024u*1024u;
        struct ggml_init_params ip = { bufsz, malloc(bufsz), false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *lg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NE);
        memcpy(lg->data, logits, sizeof(float)*NE);
        struct ggml_tensor *pr = ggml_sqrt(ctx, ggml_softplus(ctx, lg));
        struct ggml_cgraph *g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, pr);
        ggml_graph_compute_with_ctx(ctx, g, 1);

        float wp = 0.f;
        for(int e = 0; e < NE; e++){
            float mine = sqrtf(dsv4_softplus(logits[e]));
            float ref  = ((float*)pr->data)[e];
            float d = fabsf(mine - ref) / (fabsf(ref) + 1e-6f);
            if(d > wp) wp = d;
            probs[e] = ref;
        }
        ggml_free(ctx); free(ip.mem_buffer);
        printf("  probs: worst relative deviation vs ggml sqrt(softplus): %.2e\n", wp);
        ck(wp < 1e-5f, "gating function matches ggml");

        dsv4_route(logits, bias, NE, K, idx, w, 1.5f, 1, probs);

        /* selection must follow probs+bias, ranked descending */
        float sel[32];
        for(int e = 0; e < NE; e++) sel[e] = sqrtf(dsv4_softplus(logits[e])) + bias[e];
        int ok_sel = 1;
        for(int i = 1; i < K; i++) if(sel[idx[i-1]] < sel[idx[i]] - 1e-6f) ok_sel = 0;
        for(int i = 0; i < K; i++) for(int e = 0; e < NE; e++){
            int chosen = 0;
            for(int j = 0; j < K; j++) if(idx[j] == e) chosen = 1;
            if(!chosen && sel[e] > sel[idx[i]] + 1e-6f) ok_sel = 0;   /* an unpicked expert outranks a picked one */
        }
        ck(ok_sel, "top-k selection ranks by probs + selection bias");

        /* weights must come from the UNBIASED probs, normalized, then scaled by 1.5 */
        float raw[6], sum = 0.f;
        for(int i = 0; i < K; i++){ raw[i] = sqrtf(dsv4_softplus(logits[idx[i]])); sum += raw[i]; }
        int ok_w = 1;
        for(int i = 0; i < K; i++)
            if(fabsf(w[i] - 1.5f*raw[i]/sum) > 1e-5f) ok_w = 0;
        ck(ok_w, "weights use UNBIASED probs, renormalized, x expert_weights_scale");

        float wsum = 0.f;
        for(int i = 0; i < K; i++) wsum += w[i];
        ck(fabsf(wsum - 1.5f) < 1e-4f, "weights sum to expert_weights_scale (1.5)");
    }

    /* ---- SwiGLU with the DeepSeek-V4 clamp order ---- */
    printf("\nSwiGLU (gate clamped BEFORE silu — the DeepSeek-V4 order)\n");
    {
        const int N = 64;
        const float limit = 7.0f;
        float gate[64], up[64], gsave[64], usave[64];
        for(int i = 0; i < N; i++){
            gate[i] = (float)(30.0*((double)rand()/RAND_MAX - 0.5));   /* wide: must hit the clamp */
            up[i]   = (float)(30.0*((double)rand()/RAND_MAX - 0.5));
        }
        memcpy(gsave, gate, sizeof(gate)); memcpy(usave, up, sizeof(up));

        size_t bufsz = 4u*1024u*1024u;
        struct ggml_init_params ip = { bufsz, malloc(bufsz), false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *tg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        struct ggml_tensor *tu = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        memcpy(tg->data, gsave, sizeof(gsave));
        memcpy(tu->data, usave, sizeof(usave));
        struct ggml_tensor *cu = ggml_clamp(ctx, tu, -limit, limit);
        struct ggml_tensor *cg = ggml_clamp(ctx, tg, -INFINITY, limit);
        struct ggml_tensor *out = ggml_swiglu_split(ctx, cg, cu);
        struct ggml_cgraph *g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_graph_compute_with_ctx(ctx, g, 1);

        dsv4_swiglu(gate, up, N, limit);

        float wd = 0.f;
        int clamped = 0;
        for(int i = 0; i < N; i++){
            if(fabsf(gsave[i]) > limit || fabsf(usave[i]) > limit) clamped++;
            float ref = ((float*)out->data)[i];
            float d = fabsf(gate[i] - ref) / (fabsf(ref) + 1e-4f);
            if(d > wd) wd = d;
        }
        ggml_free(ctx); free(ip.mem_buffer);
        printf("  %d/%d values actually hit the clamp; worst relative deviation %.2e\n",
               clamped, N, wd);
        ck(clamped > 0, "the test data exercises the clamp at all");
        ck(wd < 1e-5f, "matches ggml clamp + swiglu_split");
    }

    /* ---- attention sinks ---- */
    printf("\nAttention sinks (extra logit in the denominator, no value behind it)\n");
    {
        const int N = 48;
        const float scale = 0.125f, sink = 0.7f;
        float sc[48], mine[48];
        for(int i = 0; i < N; i++) sc[i] = (float)(8.0*((double)rand()/RAND_MAX - 0.5));
        memcpy(mine, sc, sizeof(sc));

        size_t bufsz = 4u*1024u*1024u;
        struct ggml_init_params ip = { bufsz, malloc(bufsz), false };
        struct ggml_context *ctx = ggml_init(ip);
        /* ne = [N, 1, 1, 1]: one row, one "head", so sinks[0] applies */
        struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, 1);
        memcpy(t->data, sc, sizeof(sc));
        struct ggml_tensor *sk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ((float*)sk->data)[0] = sink;
        struct ggml_tensor *sm = ggml_soft_max_ext(ctx, t, NULL, scale, 0.0f);
        ggml_soft_max_add_sinks(sm, sk);
        struct ggml_cgraph *g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, sm);
        ggml_graph_compute_with_ctx(ctx, g, 1);

        dsv4_softmax_sinks(mine, N, sink, scale);

        float wd = 0.f, sum = 0.f;
        for(int i = 0; i < N; i++){
            float ref = ((float*)sm->data)[i];
            float d = fabsf(mine[i] - ref) / (fabsf(ref) + 1e-8f);
            if(d > wd) wd = d;
            sum += mine[i];
        }
        ggml_free(ctx); free(ip.mem_buffer);
        printf("  weights sum to %.4f (< 1: the sink keeps the rest)\n", sum);
        ck(wd < 1e-5f, "matches ggml soft_max_ext + sinks");
        ck(sum < 0.9999f, "mass is genuinely withheld — the sink is not a no-op");
    }

    /* ---- grouped output LoRA ---- */
    printf("\nGrouped output projection (8 groups -> rank 1024 -> n_embd)\n");
    {
        const int GD = 16, R = 4, NG = 3, ND = 10;      /* small stand-in for 4096/1024/8/4096 */
        float heads[16*3], wa[3*4*16], wb[10*12], mid[12], out[10], ref[10];
        for(int i = 0; i < GD*NG; i++)  heads[i] = (float)(0.5*sin(i*1.7));
        for(int i = 0; i < NG*R*GD; i++) wa[i]   = (float)(0.3*cos(i*0.9));
        for(int i = 0; i < ND*NG*R; i++) wb[i]   = (float)(0.2*sin(i*0.4));

        dsv4_o_lora(out, heads, wa, wb, GD, R, NG, ND, mid);

        /* independent reference: per-group down-projection, concat, then one up-projection */
        float m2[12];
        for(int g = 0; g < NG; g++)
            for(int r = 0; r < R; r++){
                double a = 0;
                for(int i = 0; i < GD; i++) a += wa[(g*R + r)*GD + i] * heads[g*GD + i];
                m2[g*R + r] = (float)a;
            }
        for(int o = 0; o < ND; o++){
            double a = 0;
            for(int i = 0; i < NG*R; i++) a += wb[o*(NG*R) + i] * m2[i];
            ref[o] = (float)a;
        }
        float wd = 0.f;
        for(int o = 0; o < ND; o++){
            float d = fabsf(out[o] - ref[o]) / (fabsf(ref[o]) + 1e-6f);
            if(d > wd) wd = d;
        }
        ck(wd < 1e-5f, "each group uses its own wo_a slab, then a shared wo_b");

        /* a group must NOT see another group's activations */
        float h2[16*3]; memcpy(h2, heads, sizeof(h2));
        for(int i = 0; i < GD; i++) h2[i] += 10.f;         /* perturb group 0 only */
        float o2[10], m3[12];
        dsv4_o_lora(o2, h2, wa, wb, GD, R, NG, ND, m3);
        int isolated = 1;
        for(int r = 0; r < R; r++) if(fabsf(m3[R + r] - mid[R + r]) > 1e-5f) isolated = 0;  /* group 1 */
        ck(isolated, "perturbing group 0 leaves group 1's rank vector untouched");
    }

    printf("\n%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
