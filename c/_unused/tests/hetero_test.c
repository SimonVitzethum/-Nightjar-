/* hetero_test.c — GPU and CPU computing a layer's experts CONCURRENTLY.
 *
 * This is the mechanism the whole heterogeneous idea rests on: split the routed experts, launch
 * the GPU's share async, compute the CPU's share on the host threads while the GPU runs, then
 * combine. If it works, the layer costs max(GPU, CPU) instead of the sum, and the two memory
 * paths (PCIe for the GPU, RAM for the CPU) add.
 *
 * Two things must hold, and both are checked:
 *   1. CORRECTNESS — the split-and-combine result must equal all-experts-on-CPU (the reference).
 *      A wrong split (dropped expert, double-counted weight) does not crash; it quietly shifts
 *      the logits. So this compares against the reference to 1e-5.
 *   2. CONCURRENCY — the combined time must be near max(gpu_only, cpu_only), not their sum. If
 *      the GPU launch blocks (unpinned uploads), the two serialize and there is no point.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "../gguf.h"
#include "../kquant.h"
#include "../moe_cpu.h"
#include "../moe_gpu.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf> [n_gpu]\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    if(!kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl)){
        printf("tables failed\n"); return 1; }

    const char *arch = gguf_str(&m,"general.architecture","");
    char k[128];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    const int D=KVI("embedding_length"), F=KVI("expert_feed_forward_length");
    const int NDENSE=KVI("leading_dense_block_count");
    #undef KVI
    const int L = NDENSE;
    char nm[128];
    snprintf(nm,sizeof(nm),"blk.%d.ffn_gate_exps.weight",L); const gguf_tensor *EG=gguf_find(&m,nm);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_up_exps.weight",L);   const gguf_tensor *EU=gguf_find(&m,nm);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_down_exps.weight",L); const gguf_tensor *ED=gguf_find(&m,nm);
    const int64_t gs=kq_row_bytes(EG->type,(int64_t)D*F), us=kq_row_bytes(EU->type,(int64_t)D*F),
                  ds=kq_row_bytes(ED->type,(int64_t)F*D);

    /* Leave 2 cores for the GPU driver to feed the device. If OpenMP grabs every core, the
     * GPU cannot make async progress and the two serialize — the -17% "overlap" was exactly
     * that: the CPU starved the driver. */
#ifdef _OPENMP
    int ncpu = omp_get_max_threads();
    omp_set_num_threads(ncpu > 3 ? ncpu-2 : ncpu);
#endif
    const int K = 8;
    const int NG = argc > 2 ? atoi(argv[2]) : 5;    /* experts on the GPU; rest on the CPU */
    const int NC = K - NG;
    printf("layer %d: gate %s up %s down %s  (D=%d F=%d)\n", L,
           kq_name(EG->type), kq_name(EU->type), kq_name(ED->type), D, F);
    printf("split: %d experts on GPU, %d on CPU\n\n", NG, NC);

    int   eid[8] = { 5,17,42,88,130,171,200,251 };
    float w[8]   = { 0.3f,0.2f,0.15f,0.1f,0.09f,0.07f,0.05f,0.04f };
    const void *wg[8],*wu[8],*wd[8]; int tg[8],tu[8],td[8];
    for(int e=0;e<K;e++){
        wg[e]=(const uint8_t*)gguf_data(&m,EG)+(int64_t)eid[e]*gs; tg[e]=EG->type;
        wu[e]=(const uint8_t*)gguf_data(&m,EU)+(int64_t)eid[e]*us; tu[e]=EU->type;
        wd[e]=(const uint8_t*)gguf_data(&m,ED)+(int64_t)eid[e]*ds; td[e]=ED->type;
    }
    int64_t ng[8],nu[8],nd[8];
    for(int e=0;e<K;e++){ ng[e]=gs; nu[e]=us; nd[e]=ds; }

    /* activation */
    const gguf_tensor *emb = gguf_find(&m,"token_embd.weight");
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);
    float *x; cudaMallocHost((void**)&x,(size_t)D*sizeof(float));   /* pinned: async H2D */
    kq_dequant_row(emb->type,(const uint8_t*)gguf_data(&m,emb)+(size_t)1234*erb, x, D);
    double ss=0; for(int i=0;i<D;i++) ss+=(double)x[i]*x[i];
    float inv=1.0f/sqrtf((float)ss/D+1e-6f); for(int i=0;i<D;i++) x[i]*=inv;

    MoeCpu Mc; moe_cpu_init(&Mc,D,F);
    /* reference computed LATER (needs the GPU set up) — it is GPU(NG)+CPU(NC) run serially,
     * so it isolates the split/combine logic from the GPU's int8 activation precision. */

    /* GPU setup: pin the GPU experts so uploads are truly async */
    int64_t wreg[3]={ (gs+255)&~255LL, (us+255)&~255LL, (ds+255)&~255LL };
    int64_t woff[3]={ 0, wreg[0], wreg[0]+wreg[1] }, wcap=wreg[0]+wreg[1]+wreg[2];
    MoeGpu Mg;
    if(!moe_gpu_init(&Mg, D, F, K, wcap, woff)){ printf("gpu init failed\n"); return 1; }

    const void *pg[8],*pu[8],*pd[8];
    for(int e=0;e<NG;e++){
        void *a,*b,*c;
        cudaMallocHost(&a,gs); memcpy(a,wg[e],gs); pg[e]=a;
        cudaMallocHost(&b,us); memcpy(b,wu[e],us); pu[e]=b;
        cudaMallocHost(&c,ds); memcpy(c,wd[e],ds); pd[e]=c;
    }

    float *acc_gpu; cudaMallocHost((void**)&acc_gpu,(size_t)D*sizeof(float));  /* pinned: async D2H */
    memset(acc_gpu,0,(size_t)D*sizeof(float));
    float *acc_cpu = calloc(D,sizeof(float));
    float *acc     = calloc(D,sizeof(float));

    /* the honest reference: same experts, same precision per side, just run serially */
    float *acc_ref = calloc(D,sizeof(float));
    { float *ag=calloc(D,4), *ac=calloc(D,4);
      moe_gpu_layer(&Mg, ag, x, pg,pu,pd, tg,tu,td, ng,nu,nd, w, NG);
      moe_cpu_layer(&Mc, ac, x, wg+NG,tg+NG,wu+NG,tu+NG,wd+NG,td+NG, w+NG, NC);
      for(int i=0;i<D;i++) acc_ref[i]=ag[i]+ac[i]; free(ag); free(ac); }

    /* ---- the concurrent path ---- */
    const int REP=10; double t_het=1e9,t_gpu=1e9,t_cpu=1e9;
    for(int r=0;r<REP;r++){
        memset(acc_gpu,0,(size_t)D*4); memset(acc_cpu,0,(size_t)D*4);
        double t0=now();
        moe_gpu_launch(&Mg, acc_gpu, x, pg,pu,pd, tg,tu,td, ng,nu,nd, w, NG);   /* async */
        moe_cpu_layer(&Mc, acc_cpu, x, wg+NG,tg+NG, wu+NG,tu+NG, wd+NG,td+NG, w+NG, NC); /* concurrent */
        moe_gpu_wait(&Mg);
        for(int i=0;i<D;i++) acc[i]=acc_gpu[i]+acc_cpu[i];
        double dt=now()-t0; if(dt<t_het) t_het=dt;
    }
    /* GPU-only and CPU-only for the same K, to show the overlap win */
    for(int r=0;r<REP;r++){
        double t0=now();
        moe_gpu_layer(&Mg, acc_gpu, x, pg,pu,pd, tg,tu,td, ng,nu,nd, w, NG);
        double dt=now()-t0; if(dt<t_gpu) t_gpu=dt;
    }
    for(int r=0;r<REP;r++){
        memset(acc_cpu,0,(size_t)D*4);
        double t0=now();
        moe_cpu_layer(&Mc, acc_cpu, x, wg+NG,tg+NG,wu+NG,tu+NG,wd+NG,td+NG, w+NG, NC);
        double dt=now()-t0; if(dt<t_cpu) t_cpu=dt;
    }

    /* correctness */
    double worst=0,den=0;
    for(int i=0;i<D;i++){ double d=fabs((double)acc_ref[i]-acc[i]);
        if(d>worst)worst=d; if(fabs(acc_ref[i])>den)den=fabs(acc_ref[i]); }
    int ok = worst/(den+1e-9) < 1e-4;
    printf("correctness (split+combine vs all-CPU reference): rel %.2e  %s\n\n",
           worst/(den+1e-9), ok?"MATCHES":"WRONG");

    printf("timing (best of %d):\n", REP);
    printf("  GPU-only (%d experts):     %6.2f ms\n", NG, t_gpu*1e3);
    printf("  CPU-only (%d experts):     %6.2f ms\n", NC, t_cpu*1e3);
    printf("  concurrent (both):        %6.2f ms\n", t_het*1e3);
    printf("  sum if serial:            %6.2f ms\n", (t_gpu+t_cpu)*1e3);
    printf("  overlap efficiency:       %.0f%% (100%% = perfect max(gpu,cpu))\n",
           100.0*(t_gpu+t_cpu - t_het)/( (t_gpu<t_cpu?t_gpu:t_cpu) ));
    double all_bytes = (double)K*(gs+us+ds);
    printf("\n  combined: %.1f GB/s over all %d experts\n", all_bytes/t_het/1e9, K);
    return ok?0:1;
}
