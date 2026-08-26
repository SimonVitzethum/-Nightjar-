/* qwen35_sample_test.c — the fast sampler against the sort-the-whole-vocabulary one it
 * replaces. The fast path exists because the naive one cost 18.4 ms/token (CUPTI, decode of
 * Ornith-35B): more than every CUDA kernel in the forward pass put together. A sampler that
 * is fast but draws from a different distribution is not an optimization, so this compares
 * the actual token drawn, over the real vocabulary size, on distributions ranging from
 * sharply peaked to nearly flat. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "../qwen35_sample.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* the implementation this replaces, verbatim */
static int cmp_desc(const void *a, const void *b){
    const float x = ((const Cand*)a)->p, y = ((const Cand*)b)->p;
    return x < y ? 1 : (x > y ? -1 : 0);
}
static int sample_ref(float *logits, int V, float temp, float top_p, int top_k, Cand *buf,
                      double (*rng)(void)){
    if(temp <= 0.0f){
        int best = 0;
        for(int i = 1; i < V; i++) if(logits[i] > logits[best]) best = i;
        return best;
    }
    float mx = -INFINITY;
    for(int i = 0; i < V; i++) if(logits[i] > mx) mx = logits[i];
    double s = 0;
    for(int i = 0; i < V; i++){
        const double e = exp((double)(logits[i]-mx)/temp);
        buf[i].p = (float)e; buf[i].id = i; s += e;
    }
    const float inv = (float)(1.0/s);
    for(int i = 0; i < V; i++) buf[i].p *= inv;
    qsort(buf, V, sizeof(Cand), cmp_desc);
    int n = V;
    if(top_k > 0 && top_k < n) n = top_k;
    if(top_p > 0 && top_p < 1.0f){
        double acc = 0; int m = 0;
        for(; m < n; m++){ acc += buf[m].p; if(acc >= top_p){ m++; break; } }
        n = m ? m : 1;
    }
    double tot = 0;
    for(int i = 0; i < n; i++) tot += buf[i].p;
    double r = rng()*tot, c = 0;
    for(int i = 0; i < n; i++){ c += buf[i].p; if(r <= c) return buf[i].id; }
    return buf[0].id;
}

static uint64_t g_rng = 0;
static double urand(void){
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11)/9007199254740992.0;
}
static uint32_t rs = 12345;
static float frnd(void){ rs = rs*1664525u + 1013904223u; return (float)(rs >> 8)/8388608.0f; }

/* a logit vector shaped like a real one: a few strong candidates over a long weak tail */
static void make_logits(float *l, int V, float peak, float spread){
    for(int i = 0; i < V; i++) l[i] = (frnd()-0.5f)*spread;
    for(int i = 0; i < 40; i++) l[(rs>>3) % V] = peak * (frnd()+0.2f);
}

int main(void){
    const int V = 248320;                     /* Ornith's vocabulary, the size that hurts */
    float *l   = (float*)malloc(sizeof(float)*V);
    Cand  *b1  = (Cand*)malloc(sizeof(Cand)*V);
    Cand  *b2  = (Cand*)malloc(sizeof(Cand)*V);
    int fails = 0, cases = 0;

    const struct { float temp, top_p; int top_k; const char *what; } cfg[] = {
        {0.7f, 0.8f,  20, "shipped defaults"},
        {1.0f, 0.95f, 40, "creative"},
        {0.6f, 1.0f,  50, "top-k only"},
        {0.8f, 0.9f,   0, "top-p only, no k"},
        {1.5f, 0.99f,  0, "hot and wide"},
        {0.1f, 0.8f,  20, "near-greedy"},
        {0.0f, 0.8f,  20, "greedy"},
    };
    const struct { float peak, spread; const char *what; } shape[] = {
        {18.0f, 6.0f,  "peaked"},
        { 6.0f, 8.0f,  "soft"},
        { 1.0f, 20.0f, "nearly flat"},
    };

    for(size_t si = 0; si < sizeof shape/sizeof *shape; si++){
        for(size_t ci = 0; ci < sizeof cfg/sizeof *cfg; ci++){
            int mism = 0;
            for(int trial = 0; trial < 12; trial++){
                rs = 12345 + (uint32_t)(si*100 + ci*10 + trial);
                make_logits(l, V, shape[si].peak, shape[si].spread);
                g_rng = 0x243F6A8885A308D3ull ^ (uint64_t)(trial+1)*0x9E3779B97F4A7C15ull;
                const int a = sample_ref(l, V, cfg[ci].temp, cfg[ci].top_p, cfg[ci].top_k, b1, urand);
                g_rng = 0x243F6A8885A308D3ull ^ (uint64_t)(trial+1)*0x9E3779B97F4A7C15ull;
                const int c = q35_sample(l, V, cfg[ci].temp, cfg[ci].top_p, cfg[ci].top_k, b2, urand);
                if(a != c) mism++;
                cases++;
            }
            printf("  %-14s %-18s temp %.1f top_p %.2f top_k %-3d : %s\n",
                   shape[si].what, cfg[ci].what, cfg[ci].temp, cfg[ci].top_p, cfg[ci].top_k,
                   mism ? "MISMATCH" : "same token");
            if(mism){ printf("      %d/12 draws differed\n", mism); fails++; }
        }
    }

    /* what the whole exercise was for */
    make_logits(l, V, 18.0f, 6.0f);
    const int R = 200;
    double t0 = now();
    for(int i = 0; i < R; i++) sample_ref(l, V, 0.7f, 0.8f, 20, b1, urand);
    const double t_ref = (now()-t0)/R;
    t0 = now();
    for(int i = 0; i < R; i++) q35_sample(l, V, 0.7f, 0.8f, 20, b2, urand);
    const double t_new = (now()-t0)/R;
    printf("\n  sort-the-vocabulary %.2f ms/token   selection %.2f ms/token   %.0fx\n",
           t_ref*1e3, t_new*1e3, t_ref/t_new);

    printf("\n%s (%d cases)\n", fails ? "FAIL" : "all configurations agree", cases);
    free(l); free(b1); free(b2);
    return fails ? 1 : 0;
}
