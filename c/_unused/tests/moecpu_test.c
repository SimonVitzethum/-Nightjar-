/* moecpu_test.c — the CPU expert kernel against the reference, and how fast it really is.
 *
 * Correctness: moe_cpu_layer must equal glm_ffn summed over the same experts. They call the
 * same validated kq_dot, so this must be bit-identical, not merely close — anything else means
 * the threading or the byte-addressing is wrong. Throughput: the number that decides whether
 * heterogeneous GPU+CPU is worth building. If the CPU cannot move experts at a rate comparable
 * to PCIe (~55 GB/s), splitting work onto it buys little.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../gguf.h"
#include "../kquant.h"
#include "../moe_cpu.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* the reference: gate/up/swiglu/down exactly as glm_engine does it */
static void ref_expert(float *acc, const float *x, int D, int F,
                       const void *wg,int tg,const void *wu,int tu,const void *wd,int td,
                       float weight, float *g, float *u, float *o){
    kq_gemm(g, x, wg, tg, 1, D, F);
    kq_gemm(u, x, wu, tu, 1, D, F);
    for(int i=0;i<F;i++) g[i] = (g[i]/(1.0f+expf(-g[i]))) * u[i];
    kq_gemm(o, g, wd, td, 1, F, D);
    for(int i=0;i<D;i++) acc[i] += weight*o[i];
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }

    const char *arch = gguf_str(&m,"general.architecture","");
    char k[128];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    const int D=KVI("embedding_length"), F=KVI("expert_feed_forward_length");
    const int NEXP=KVI("expert_count"), NDENSE=KVI("leading_dense_block_count");
    #undef KVI

    /* a real MoE layer's expert tensors (3-D: [D,F,NEXP]) */
    const int L = NDENSE;             /* first MoE layer */
    char nm[128];
    snprintf(nm,sizeof(nm),"blk.%d.ffn_gate_exps.weight",L); const gguf_tensor *EG=gguf_find(&m,nm);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_up_exps.weight",L);   const gguf_tensor *EU=gguf_find(&m,nm);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_down_exps.weight",L); const gguf_tensor *ED=gguf_find(&m,nm);
    if(!EG||!EU||!ED){ printf("no experts at layer %d\n", L); return 1; }

    const int64_t gstride = kq_row_bytes(EG->type, (int64_t)D*F);   /* one expert slice */
    const int64_t ustride = kq_row_bytes(EU->type, (int64_t)D*F);
    const int64_t dstride = kq_row_bytes(ED->type, (int64_t)F*D);
    printf("layer %d experts: gate %s up %s down %s   (D=%d F=%d, %d experts)\n",
           L, kq_name(EG->type), kq_name(EU->type), kq_name(ED->type), D, F, NEXP);
    printf("per expert: %.2f MB   (gate %.2f + up %.2f + down %.2f)\n",
           (gstride+ustride+dstride)/1e6, gstride/1e6, ustride/1e6, dstride/1e6);

    /* a real activation: a token embedding, RMS-normed the way the router input is */
    const gguf_tensor *emb = gguf_find(&m,"token_embd.weight");
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);
    float *x = malloc((size_t)D*sizeof(float));
    kq_dequant_row(emb->type,(const uint8_t*)gguf_data(&m,emb)+(size_t)1234*erb, x, D);
    /* rms-norm to a realistic scale */
    double ss=0; for(int i=0;i<D;i++) ss+=(double)x[i]*x[i];
    float inv=1.0f/sqrtf((float)ss/D + 1e-6f);
    for(int i=0;i<D;i++) x[i]*=inv;

    const int K = 8;                              /* a top-8 layer's worth */
    int    eid[8] = { 5, 17, 42, 88, 130, 171, 200, 251 };
    float  w[8]   = { 0.3f,0.2f,0.15f,0.1f,0.09f,0.07f,0.05f,0.04f };
    const void *wg[8],*wu[8],*wd[8]; int tg[8],tu[8],td[8];
    for(int e=0;e<K;e++){
        wg[e]=(const uint8_t*)gguf_data(&m,EG)+(int64_t)eid[e]*gstride; tg[e]=EG->type;
        wu[e]=(const uint8_t*)gguf_data(&m,EU)+(int64_t)eid[e]*ustride; tu[e]=EU->type;
        wd[e]=(const uint8_t*)gguf_data(&m,ED)+(int64_t)eid[e]*dstride; td[e]=ED->type;
    }

    /* reference */
    float *acc_ref = calloc(D,sizeof(float));
    float *rg=malloc((size_t)F*sizeof(float)),*ru=malloc((size_t)F*sizeof(float)),
          *ro=malloc((size_t)D*sizeof(float));
    for(int e=0;e<K;e++)
        ref_expert(acc_ref,x,D,F,wg[e],tg[e],wu[e],tu[e],wd[e],td[e],w[e],rg,ru,ro);

    /* the CPU kernel */
    MoeCpu M; moe_cpu_init(&M,D,F);
    float *acc_cpu = calloc(D,sizeof(float));
    moe_cpu_layer(&M, acc_cpu, x, wg,tg, wu,tu, wd,td, w, K);

    /* compare — must be bit-identical (same kq_dot, deterministic order) */
    double worst=0, den=0; int nbad=0;
    for(int i=0;i<D;i++){
        double d=fabs((double)acc_ref[i]-acc_cpu[i]);
        if(d>worst) worst=d;
        if(fabs(acc_ref[i])>den) den=fabs(acc_ref[i]);
        if(d>1e-4) nbad++;
    }
    printf("\ncorrectness vs reference: max abs diff %.3e (rel %.3e), %d/%d elems off\n",
           worst, worst/(den+1e-9), nbad, D);
    int ok = worst/(den+1e-9) < 1e-5;
    printf("  %s\n", ok ? "MATCHES the reference" : "DIFFERS — kernel is wrong");

    /* throughput: average several runs, report GB/s of expert bytes moved */
    const double bytes = (double)K*(gstride+ustride+dstride);
    const int REP=8;
    double best=1e9;
    for(int r=0;r<REP;r++){
        memset(acc_cpu,0,(size_t)D*sizeof(float));
        double t0=now();
        moe_cpu_layer(&M, acc_cpu, x, wg,tg, wu,tu, wd,td, w, K);
        double dt=now()-t0; if(dt<best) best=dt;
    }
    printf("\nthroughput (%d experts, best of %d):\n", K, REP);
    printf("  %.2f ms/layer   %.1f GB/s   %.1f experts/s\n",
           best*1e3, bytes/best/1e9, K/best);
    #ifdef _OPENMP
    printf("  (%d threads)\n", omp_get_max_threads());
    #endif
    printf("\n  For reference: PCIe 5.0 x16 ~55 GB/s. Above that, the CPU is worth a share.\n");

    return ok ? 0 : 1;
}
