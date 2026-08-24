/* hetauto_test.c — does the auto-balancer find the split that minimizes layer time? */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../gguf.h"
#include "../kquant.h"
#include "../moe_hetero.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec; }

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s <shard1.gguf>\n",argv[0]);return 2;}
    gguf_model m; if(!gguf_open(&m,argv[1])){printf("open failed\n");return 1;}
    kq_cu_upload_tables(iq2xs_grid,iq3xxs_grid,ksigns_iq2xs,kmask_iq2xs,kvalues_iq4nl);
    const char*arch=gguf_str(&m,"general.architecture","");char k[128];
    #define KVI(s)(snprintf(k,sizeof(k),"%s." s,arch),(int)gguf_u64(&m,k,0))
    const int D=KVI("embedding_length"),F=KVI("expert_feed_forward_length"),ND=KVI("leading_dense_block_count");
    #undef KVI
#ifdef _OPENMP
    int nc=omp_get_max_threads(); omp_set_num_threads(nc>3?nc-2:nc);
#endif
    char nm[128];
    snprintf(nm,sizeof(nm),"blk.%d.ffn_gate_exps.weight",ND);const gguf_tensor*EG=gguf_find(&m,nm);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_up_exps.weight",ND);  const gguf_tensor*EU=gguf_find(&m,nm);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_down_exps.weight",ND);const gguf_tensor*ED=gguf_find(&m,nm);
    const int64_t gs=kq_row_bytes(EG->type,(int64_t)D*F),us=kq_row_bytes(EU->type,(int64_t)D*F),
                  ds=kq_row_bytes(ED->type,(int64_t)F*D);
    const int K=8; int eid[8]={5,17,42,88,130,171,200,251};
    float w[8]={0.3f,0.2f,0.15f,0.1f,0.09f,0.07f,0.05f,0.04f};
    /* pin ALL experts (as the real cache would) so GPU uploads are async */
    const void *pg[8],*pu[8],*pd[8];int tg[8],tu[8],td[8];int64_t ng[8],nu[8],nd[8];
    for(int e=0;e<K;e++){
        void*a,*b,*c;
        cudaMallocHost(&a,gs);memcpy(a,(uint8_t*)gguf_data(&m,EG)+(int64_t)eid[e]*gs,gs);pg[e]=a;
        cudaMallocHost(&b,us);memcpy(b,(uint8_t*)gguf_data(&m,EU)+(int64_t)eid[e]*us,us);pu[e]=b;
        cudaMallocHost(&c,ds);memcpy(c,(uint8_t*)gguf_data(&m,ED)+(int64_t)eid[e]*ds,ds);pd[e]=c;
        tg[e]=EG->type;tu[e]=EU->type;td[e]=ED->type;ng[e]=gs;nu[e]=us;nd[e]=ds;
    }
    const gguf_tensor*emb=gguf_find(&m,"token_embd.weight");
    const int64_t erb=kq_row_bytes(emb->type,emb->ne[0]);
    float*x;cudaMallocHost((void**)&x,(size_t)D*4);
    kq_dequant_row(emb->type,(uint8_t*)gguf_data(&m,emb)+(size_t)1234*erb,x,D);
    double ssum=0;for(int i=0;i<D;i++)ssum+=(double)x[i]*x[i];
    float invn=1.0f/sqrtf((float)ssum/D+1e-6f);for(int i=0;i<D;i++)x[i]*=invn;

    int64_t wreg[3]={(gs+255)&~255LL,(us+255)&~255LL,(ds+255)&~255LL};
    int64_t woff[3]={0,wreg[0],wreg[0]+wreg[1]},wcap=wreg[0]+wreg[1]+wreg[2];
    MoeGpu G;moe_gpu_init(&G,D,F,K,wcap,woff);
    MoeCpu C;moe_cpu_init(&C,D,F);
    MoeHetero H;moe_hetero_init(&H,&G,&C,D);

    /* reference: all-on-CPU (for a stable correctness anchor per split we recompute below) */
    float*acc=calloc(D,4);
    printf("auto-balancing split over 24 simulated layers:\n");
    printf("  %-5s %-4s %10s %10s %10s\n","layer","ng","gpu ms/exp","cpu ms/exp","layer ms");
    double best=1e9;
    for(int L=0;L<24;L++){
        memset(acc,0,(size_t)D*4);
        double t0=now();
        moe_hetero_layer(&H,acc,x,pg,pu,pd,tg,tu,td,ng,nu,nd,w,K);
        double dt=now()-t0; if(L>=4&&dt<best)best=dt;
        if(L<6||L%4==0)
            printf("  %-5d %-4d %10.4f %10.4f %10.3f\n",L,H.ng,H.gpu_cost,H.cpu_cost,dt*1e3);
    }
    /* correctness of the final split vs serial GPU(ng)+CPU(nc) */
    float*ar=calloc(D,4),*ag=calloc(D,4);
    int fng=H.ng,fnc=K-fng;
    moe_gpu_layer(&G,ag,x,pg,pu,pd,tg,tu,td,ng,nu,nd,w,fng);
    MoeCpu C2;moe_cpu_init(&C2,D,F);
    moe_cpu_layer(&C2,ar,x,pg+fng,tg+fng,pu+fng,tu+fng,pd+fng,td+fng,w+fng,fnc);
    for(int i=0;i<D;i++)ar[i]+=ag[i];
    double wr=0,dn=0;for(int i=0;i<D;i++){double d=fabs((double)ar[i]-acc[i]);if(d>wr)wr=d;if(fabs(ar[i])>dn)dn=fabs(ar[i]);}
    printf("\nconverged split: %d GPU / %d CPU\n",fng,fnc);
    printf("best layer time: %.3f ms  ->  %.1f GB/s over 8 experts\n",best*1e3,(double)K*(gs+us+ds)/best/1e9);
    printf("correctness at converged split: rel %.2e  %s\n",wr/(dn+1e-9),wr/(dn+1e-9)<1e-4?"MATCHES":"WRONG");
    return 0;
}
