/* qwen35_sample.h — top-k / top-p sampling that does not sort the vocabulary.
 *
 * WHY THIS FILE EXISTS
 *   The obvious implementation softmaxes all V logits into a Cand array and qsorts it. For
 *   this model V = 248320, so that is a 4.4M-comparison sort behind a function pointer plus
 *   2 MiB of stores, once per token, on the CPU, with the GPU idle throughout. CUPTI put the
 *   cost at 18.4 ms/token against 16.7 ms of actual GPU work — the sampler was the largest
 *   single item in the decode budget, bigger than every CUDA kernel combined.
 *
 *   Nothing downstream needs the vocabulary sorted. top_k keeps 20 entries; top_p then keeps
 *   fewer. Everything below the cut is read once and discarded. So the work splits in two:
 *
 *     - the normaliser s = Σ exp((l-mx)/temp) genuinely needs every entry, but only as a
 *       scalar. No per-entry store, and it cancels out of the final draw entirely, surviving
 *       only in the top_p threshold test.
 *     - the candidates themselves are a SELECTION problem, not a sorting one. softmax is
 *       strictly monotone for temp > 0, so the top-k by probability is the top-k by logit:
 *       one compare per entry against the running k-th best, ~k·ln(V/k) ≈ 190 insertions for
 *       the whole vocabulary.
 *
 *   With top_k disabled there is no fixed k to select, and how deep top_p reaches depends on
 *   the distribution — on a flat one it is thousands of entries. Guessing a depth and
 *   growing it is wrong in a way that silently truncates the tail, so that case instead
 *   histograms the logits during the sum pass and reads the cut point straight off the
 *   cumulative mass. Two passes, then a sort of only what survives the cut.
 */
#ifndef QWEN35_SAMPLE_H
#define QWEN35_SAMPLE_H

#include <math.h>
#include <float.h>
#include <stdlib.h>

typedef struct { float p; int id; } Cand;

/* Entries more than this many temperature-units below the max underflow to <1e-34 in float
 * and cannot move a sum that is >= 1. Also the histogram's range. */
#define Q35_SAMPLE_FLOOR 80.0f
#define Q35_SAMPLE_NBUCK 2048

static int q35_cand_desc(const void *a, const void *b){
    const float x = ((const Cand*)a)->p, y = ((const Cand*)b)->p;
    return x < y ? 1 : (x > y ? -1 : 0);
}

static float q35_logit_max(const float *logits, int V){
    float mx = -INFINITY;
    for(int i = 0; i < V; i++) if(logits[i] > mx) mx = logits[i];
    return mx;
}

/* s = Σ exp((l-mx)/temp) over the whole vocabulary. */
static double q35_softmax_denom(const float *logits, int V, float mx, float temp){
    const float inv = 1.0f/temp, lim = mx - Q35_SAMPLE_FLOOR*temp;
    double s = 0;
#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:s) schedule(static) if(V > 65536)
#endif
    for(int i = 0; i < V; i++)
        if(logits[i] > lim) s += (double)expf((logits[i]-mx)*inv);
    return s;
}

/* The K largest logits, descending. sel[cnt-1] is the bar a new entry must clear, so the
 * common case is a single predictable compare and nothing else. */
typedef struct { float l; int id; } Q35Sel;
static int q35_select_topk(const float *logits, int V, int K, Q35Sel *sel){
    float thr = -INFINITY;
    int cnt = 0;
    for(int i = 0; i < V; i++){
        const float l = logits[i];
        if(cnt < K || l > thr){
            int j = (cnt < K) ? cnt++ : K-1;
            while(j > 0 && sel[j-1].l < l){ sel[j] = sel[j-1]; j--; }
            sel[j].l = l; sel[j].id = i;
            if(cnt == K) thr = sel[K-1].l;
        }
    }
    return cnt;
}

/* No top_k to select on: bin the logits by how far they sit below the max, accumulating
 * both count and probability mass per bin, then walk the bins down from the top until the
 * mass covers top_p. Everything at or above that bin's lower edge is a superset of the
 * prefix top_p needs, and it is the only thing that gets sorted. */
static int q35_collect_by_mass(const float *logits, int V, float mx, float temp,
                               float top_p, int top_k, double *out_s, Cand *buf){
    const float inv = 1.0f/temp, lim = mx - Q35_SAMPLE_FLOOR*temp;
    const float bscale = (float)Q35_SAMPLE_NBUCK / Q35_SAMPLE_FLOOR;
    static double mass[Q35_SAMPLE_NBUCK];
    static int    cnts[Q35_SAMPLE_NBUCK];
    for(int i = 0; i < Q35_SAMPLE_NBUCK; i++){ mass[i] = 0; cnts[i] = 0; }

    double s = 0;
    for(int i = 0; i < V; i++){
        if(logits[i] <= lim) continue;
        const float d = (logits[i]-mx)*inv;              /* in (-80, 0] */
        const double e = (double)expf(d);
        s += e;
        int b = (int)(-d * bscale);
        if(b < 0) b = 0; else if(b >= Q35_SAMPLE_NBUCK) b = Q35_SAMPLE_NBUCK-1;
        mass[b] += e; cnts[b]++;
    }
    *out_s = s;

    /* Enough is enough as soon as EITHER truncation is settled: the kept count is
     * min(top_k, top_p-index), so whichever bound is reached first bounds the other. */
    const int    want_p = (top_p > 0 && top_p < 1.0f);
    const double need   = want_p ? (double)top_p * s : 0.0;
    int b = 0, cnt = 0; double acc = 0;
    for(; b < Q35_SAMPLE_NBUCK; b++){
        acc += mass[b]; cnt += cnts[b];
        if(top_k > 0 && cnt >= top_k) break;
        if(want_p && acc >= need) break;
    }
    if(b >= Q35_SAMPLE_NBUCK) b = Q35_SAMPLE_NBUCK-1;
    const float cut = mx + temp * (-(float)(b+1) / bscale);

    int n = 0;
    for(int i = 0; i < V && n < V; i++){
        if(logits[i] < cut) continue;
        buf[n].p = (float)expf((logits[i]-mx)*inv); buf[n].id = i; n++;
    }
    qsort(buf, (size_t)n, sizeof(Cand), q35_cand_desc);
    const float sinv = (float)(1.0/s);
    for(int i = 0; i < n; i++) buf[i].p *= sinv;
    return n;
}

/* temperature, then top-k, then top-p — the order llama.cpp uses, and the order the model's
 * own shipped defaults were tuned against. `buf` is caller scratch; it must hold V entries
 * for the no-top-k path, which is what both callers already allocate. */
static int q35_sample(const float *logits, int V, float temp, float top_p, int top_k,
                      Cand *buf, double (*rng)(void)){
    if(temp <= 0.0f){
        int best = 0;
        for(int i = 1; i < V; i++) if(logits[i] > logits[best]) best = i;
        return best;
    }

    const float mx = q35_logit_max(logits, V);
    int cnt;

    if(top_k > 0 && top_k < V){
        Q35Sel sel[1024];
        const int K = top_k > 1024 ? 1024 : top_k;
        cnt = q35_select_topk(logits, V, K, sel);
        const double s = q35_softmax_denom(logits, V, mx, temp);
        const float inv = (float)(1.0/s), it = 1.0f/temp;
        for(int i = 0; i < cnt; i++){
            buf[i].p  = expf((sel[i].l-mx)*it) * inv;
            buf[i].id = sel[i].id;
        }
    } else {
        double s = 0;
        cnt = q35_collect_by_mass(logits, V, mx, temp, top_p, top_k, &s, buf);
    }

    int n = cnt;
    if(top_k > 0 && top_k < n) n = top_k;
    if(top_p > 0 && top_p < 1.0f){
        double acc = 0; int m = 0;
        for(; m < n; m++){ acc += buf[m].p; if(acc >= top_p){ m++; break; } }
        n = m ? m : 1;
    }
    double tot = 0;
    for(int i = 0; i < n; i++) tot += buf[i].p;
    const double r = rng()*tot; double c = 0;
    for(int i = 0; i < n; i++){ c += buf[i].p; if(r <= c) return buf[i].id; }
    return buf[0].id;
}

#endif
