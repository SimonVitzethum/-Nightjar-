#ifndef COLIBRI_DSV4_H
#define COLIBRI_DSV4_H
/* dsv4.h — DeepSeek-V4 hyper-connections, in plain C.
 *
 * DeepSeek-V4 replaces the single residual stream with HC=4 parallel streams. Every block
 * reads a learned mixture of them, and writes back a learned recombination. The mixing
 * weights are produced per token from the streams themselves:
 *
 *   flat   = [x_0 .. x_{HC-1}] flattened        (HC*n_embd)
 *   mixes  = hc_fn @ rms_norm(flat)             ((2+HC)*HC values)
 *   pre    = sigmoid(mixes[0:HC]   * s0 + b_pre)  + eps      -> read weights
 *   post   = sigmoid(mixes[HC:2HC] * s1 + b_post) * 2        -> write weights
 *   comb   = sinkhorn(mixes[2HC:]  * s2 + b_comb)            -> stream mixing matrix
 *
 *   input to attn/ffn : sum_h  x[h] * pre[h]
 *   new streams       : x'[dst] = y * post[dst] + sum_src x[src] * comb[dst][src]
 *
 * `comb` is pushed toward doubly-stochastic by Sinkhorn, so the streams neither blow up
 * nor collapse across 43 layers. Getting the normalization order wrong does NOT crash —
 * it quietly degrades the model, which is why dsv4_test.c checks every step against ggml
 * rather than against my reading of the paper.
 *
 * Index convention follows ggml exactly: comb is [ne0 = dst, ne1 = src], i.e.
 * comb[dst + src*HC]. ggml_soft_max and ggml_sum_rows both work along ne0 (dst).
 * Reference: llama.cpp src/models/deepseek4.cpp, build_hc_{sinkhorn,pre,post,head}.
 */
#include <math.h>
#include <string.h>

#define DSV4_HC_MAX 8

static inline float dsv4_sigmoid(float v){ return 1.0f/(1.0f+expf(-v)); }

/* rms_norm over the full HC*n_embd vector (ggml_rms_norm, no weight) */
static void dsv4_rms_norm(float *dst,const float *src,int n,float eps){
    double ss=0.0;
    for(int i=0;i<n;i++) ss+=(double)src[i]*src[i];
    float sc=1.0f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) dst[i]=src[i]*sc;
}

/* Sinkhorn on comb[dst + src*hc], in place.
 * ggml order: soft_max over dst; +eps; norm_cols; then (iters-1) x (norm_rows, norm_cols).
 *   norm_rows -> divide by the sum over dst   (ggml_sum_rows, ne0)
 *   norm_cols -> divide by the sum over src   (permute, sum_rows, permute back)
 * The eps is added to every denominator too, exactly as ggml does. */
static void dsv4_sinkhorn(float *comb,int hc,float eps,int iters){
    /* softmax along dst, for each src column */
    for(int s=0;s<hc;s++){
        float *c=comb+(size_t)s*hc, mx=c[0];
        for(int d=1;d<hc;d++) if(c[d]>mx) mx=c[d];
        float sum=0.0f;
        for(int d=0;d<hc;d++){ c[d]=expf(c[d]-mx); sum+=c[d]; }
        for(int d=0;d<hc;d++) c[d]/=sum;
    }
    for(int i=0;i<hc*hc;i++) comb[i]+=eps;

    float acc[DSV4_HC_MAX];
    /* norm_cols: for each dst, divide by sum over src */
    #define DSV4_NORM_COLS() do{                                   \
        for(int d=0;d<hc;d++){                                     \
            float sum=eps;                                         \
            for(int s=0;s<hc;s++) sum+=comb[d+(size_t)s*hc];       \
            acc[d]=sum;                                            \
        }                                                          \
        for(int s=0;s<hc;s++) for(int d=0;d<hc;d++)                \
            comb[d+(size_t)s*hc]/=acc[d];                          \
    }while(0)
    /* norm_rows: for each src, divide by sum over dst */
    #define DSV4_NORM_ROWS() do{                                   \
        for(int s=0;s<hc;s++){                                     \
            float *c=comb+(size_t)s*hc, sum=eps;                   \
            for(int d=0;d<hc;d++) sum+=c[d];                       \
            for(int d=0;d<hc;d++) c[d]/=sum;                       \
        }                                                          \
    }while(0)

    DSV4_NORM_COLS();
    for(int i=1;i<iters;i++){ DSV4_NORM_ROWS(); DSV4_NORM_COLS(); }
    #undef DSV4_NORM_COLS
    #undef DSV4_NORM_ROWS
}

/* Per-token HC state. x is [hc][n_embd], row h at x + h*n_embd. */
typedef struct {
    float pre[DSV4_HC_MAX];
    float post[DSV4_HC_MAX];
    float comb[DSV4_HC_MAX*DSV4_HC_MAX];    /* [dst + src*hc] */
} dsv4_hc;

/* build_hc_pre: derive pre/post/comb from the streams, and return the mixed input.
 *   hc_fn    [hc_dim, hc_mix_dim] row-major, one row per output  (mixes = hc_fn @ flat_norm)
 *   hc_scale [3]  : scale_pre, scale_post, scale_comb
 *   hc_base  [hc_mix_dim] : base_pre[hc], base_post[hc], base_comb[hc*hc]
 *   scratch  [hc*n_embd] caller-owned, avoids a malloc per token per layer
 * `out` receives n_embd floats: the input the attention or FFN block actually sees. */
static void dsv4_hc_pre(dsv4_hc *st, float *out,
                        const float *x, int hc, int n_embd,
                        const float *hc_fn, const float *hc_scale, const float *hc_base,
                        float eps, float rms_eps, int sink_iters, float *scratch){
    const int hc_dim = hc*n_embd;
    const int hc_mix = (2+hc)*hc;

    dsv4_rms_norm(scratch, x, hc_dim, rms_eps);

    /* mixes = hc_fn @ flat_norm */
    float mixes[(2+DSV4_HC_MAX)*DSV4_HC_MAX];
    for(int o=0;o<hc_mix;o++){
        const float *w = hc_fn + (size_t)o*hc_dim;
        float acc=0.0f;
        for(int i=0;i<hc_dim;i++) acc += w[i]*scratch[i];
        mixes[o]=acc;
    }

    const float *b_pre  = hc_base;
    const float *b_post = hc_base + hc;
    const float *b_comb = hc_base + 2*hc;

    for(int h=0;h<hc;h++)
        st->pre[h]  = dsv4_sigmoid(mixes[h]*hc_scale[0] + b_pre[h]) + eps;
    for(int h=0;h<hc;h++)
        st->post[h] = dsv4_sigmoid(mixes[hc+h]*hc_scale[1] + b_post[h]) * 2.0f;
    for(int i=0;i<hc*hc;i++)
        st->comb[i] = mixes[2*hc+i]*hc_scale[2] + b_comb[i];
    dsv4_sinkhorn(st->comb, hc, eps, sink_iters);

    /* weighted sum of the streams -> what the block reads */
    for(int e=0;e<n_embd;e++) out[e]=0.0f;
    for(int h=0;h<hc;h++){
        const float *xh = x + (size_t)h*n_embd;
        float w = st->pre[h];
        for(int e=0;e<n_embd;e++) out[e] += xh[e]*w;
    }
}

/* build_hc_post: fold the block output y back into the streams, in place on x. */
static void dsv4_hc_post(const dsv4_hc *st, float *x, const float *y, int hc, int n_embd,
                         float *scratch){
    memcpy(scratch, x, (size_t)hc*n_embd*sizeof(float));    /* old streams */
    for(int dst=0;dst<hc;dst++){
        float *o = x + (size_t)dst*n_embd;
        float pw = st->post[dst];
        for(int e=0;e<n_embd;e++) o[e] = y[e]*pw;
        for(int src=0;src<hc;src++){
            const float *r = scratch + (size_t)src*n_embd;
            float cw = st->comb[dst + (size_t)src*hc];
            for(int e=0;e<n_embd;e++) o[e] += r[e]*cw;
        }
    }
}

/* build_hc_head: collapse the streams back to one vector before the final norm/lm_head.
 * Same shape as hc_pre but hc_fn is [hc_dim, hc] and there is no post/comb. */
static void dsv4_hc_head(float *out, const float *x, int hc, int n_embd,
                         const float *hc_fn, const float *hc_scale, const float *hc_base,
                         float eps, float rms_eps, float *scratch){
    const int hc_dim = hc*n_embd;
    dsv4_rms_norm(scratch, x, hc_dim, rms_eps);

    float pre[DSV4_HC_MAX];
    for(int h=0;h<hc;h++){
        const float *w = hc_fn + (size_t)h*hc_dim;
        float acc=0.0f;
        for(int i=0;i<hc_dim;i++) acc += w[i]*scratch[i];
        pre[h] = dsv4_sigmoid(acc*hc_scale[0] + hc_base[h]) + eps;
    }
    for(int e=0;e<n_embd;e++) out[e]=0.0f;
    for(int h=0;h<hc;h++){
        const float *xh = x + (size_t)h*n_embd;
        for(int e=0;e<n_embd;e++) out[e] += xh[e]*pre[h];
    }
}

/* ---------------------------------------------------------------- MoE routing
 *
 * DeepSeek-V4's router differs from V3/GLM in three ways that all matter:
 *
 *  1. gating is SQRT_SOFTPLUS:  probs = sqrt(log(1 + e^logit))   (not sigmoid)
 *  2. the selection bias `exp_probs_b` steers WHICH experts are picked, but the
 *     mixing weights are read from the UNBIASED probs. Using the biased ones is a
 *     classic silent bug: output stays fluent, quality drifts.
 *  3. weights are renormalized to sum 1 (with the sum floored at F16_MIN to avoid a
 *     divide-by-zero), then scaled by expert_weights_scale (1.5 for this model).
 *
 * Reference: llama.cpp build_moe_ffn (llama-graph.cpp) with arch == LLM_ARCH_DEEPSEEK4.
 */
static inline float dsv4_softplus(float x){
    /* log1p(exp(x)) without overflowing for large x */
    return x > 20.0f ? x : log1pf(expf(x));
}

/* Pick the top-k experts and produce their mixing weights.
 * logits[n_expert] -> idx[k], w[k]. bias may be NULL (hash layers pass NULL). */
static void dsv4_route(const float *logits, const float *bias, int n_expert, int k,
                       int *idx, float *w, float w_scale, int norm_w, float *probs_scratch){
    float *probs = probs_scratch;                       /* n_expert */
    for(int e=0;e<n_expert;e++) probs[e]=sqrtf(dsv4_softplus(logits[e]));

    /* selection uses probs+bias; the weights below deliberately do NOT. */
    for(int i=0;i<k;i++){
        int best=-1; float bv=-INFINITY;
        for(int e=0;e<n_expert;e++){
            int taken=0;
            for(int j=0;j<i;j++) if(idx[j]==e){ taken=1; break; }
            if(taken) continue;
            float s = bias ? probs[e]+bias[e] : probs[e];
            if(s>bv){ bv=s; best=e; }
        }
        idx[i]=best;
        w[i]=probs[best];                                /* unbiased */
    }

    if(norm_w){
        float sum=0.0f;
        for(int i=0;i<k;i++) sum+=w[i];
        if(sum < 6.103515625e-5f) sum = 6.103515625e-5f;   /* ggml_clamp to F16 min */
        for(int i=0;i<k;i++) w[i]/=sum;
    }
    if(w_scale!=0.0f && w_scale!=1.0f)
        for(int i=0;i<k;i++) w[i]*=w_scale;
}

/* DeepSeek-V4 SwiGLU, in place on `gate`:
 *      up   = clamp(up,   -limit, +limit)
 *      gate = clamp(gate,   -inf, +limit)     <- clamped BEFORE silu, unlike other archs
 *      out  = silu(gate) * up
 * limit <= 1e-6 disables the clamp entirely (that is ggml's own gate). */
static void dsv4_swiglu(float *gate, float *up, int n, float limit){
    const int clamp = (limit > 1e-6f);
    for(int i=0;i<n;i++){
        float u = up[i], g = gate[i];
        if(clamp){
            if(u >  limit) u =  limit;
            if(u < -limit) u = -limit;
            if(g >  limit) g =  limit;      /* no lower bound: clamp(-INFINITY, limit) */
        }
        gate[i] = (g / (1.0f + expf(-g))) * u;          /* silu(g) * u */
    }
}

/* -------------------------------------------------------- attention sinks
 *
 * A sink is one extra logit per head that enters the softmax DENOMINATOR but has no
 * value vector behind it. The attention weights therefore sum to LESS than 1, and the
 * leftover mass is simply discarded — that is the point: a head can decline to attend.
 *
 * Implemented exactly as ggml does it (ggml-cpu/ops.cpp, soft_max with sinks):
 * the running max includes the sink, and exp(sink - max) is added to the sum.
 * Forgetting the sink in the max is fine numerically here, but forgetting it in the
 * SUM silently renormalizes the weights back to 1 and removes the mechanism. */
static void dsv4_softmax_sinks(float *scores, int n, float sink, float scale){
    float mx = -INFINITY;
    for(int i=0;i<n;i++){ scores[i]*=scale; if(scores[i]>mx) mx=scores[i]; }
    if(sink > mx) mx = sink;

    double sum = 0.0;
    for(int i=0;i<n;i++){ scores[i]=expf(scores[i]-mx); sum += scores[i]; }
    sum += expf(sink - mx);                    /* the sink absorbs its share */

    float inv = (float)(1.0/sum);
    for(int i=0;i<n;i++) scores[i]*=inv;
}

/* ------------------------------------------------- grouped output projection
 *
 * DeepSeek-V4 does not use a single o_proj. The concatenated head output is cut into
 * `n_groups` slices; each slice gets its OWN down-projection to `rank` (wo_a), the
 * results are concatenated (n_groups*rank), and one shared up-projection (wo_b) maps
 * that back to n_embd. For this model: 64 heads x 512 = 32768, cut into 8 groups of
 * 4096, each -> 1024, giving 8192, then 8192 -> 4096.
 *
 *   wo_a: [group_dim, rank, n_groups]  (row-major per group: rank rows of group_dim)
 *   wo_b: [n_groups*rank, n_embd]      (row-major: n_embd rows)
 *
 * `mid` is caller-owned scratch of n_groups*rank floats. */
static void dsv4_o_lora(float *out, const float *heads,
                        const float *wo_a, const float *wo_b,
                        int group_dim, int rank, int n_groups, int n_embd,
                        float *mid){
    for(int g=0; g<n_groups; g++){
        const float *xg = heads + (size_t)g*group_dim;
        const float *Wg = wo_a  + (size_t)g*rank*group_dim;      /* this group's slab */
        float *mg = mid + (size_t)g*rank;
        for(int r=0; r<rank; r++){
            const float *w = Wg + (size_t)r*group_dim;
            float acc=0.0f;
            for(int i=0;i<group_dim;i++) acc += w[i]*xg[i];
            mg[r]=acc;
        }
    }
    const int mid_dim = n_groups*rank;
    for(int o=0;o<n_embd;o++){
        const float *w = wo_b + (size_t)o*mid_dim;
        float acc=0.0f;
        for(int i=0;i<mid_dim;i++) acc += w[i]*mid[i];
        out[o]=acc;
    }
}

#endif /* COLIBRI_DSV4_H */
