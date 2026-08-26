/* qwen35_cu_split_test.c — does a batch of S reproduce S sequential passes, at EVERY S?
 *
 * WHY THIS GATE EXISTS
 *
 * The speculation test checks a batch of 2, because that is the shape speculation uses. That
 * left S=1, 4 and 8 unchecked — and prefill runs at 8. A buffer race in the split FFN's
 * batched path (the CPU's partial was written into the layer INPUT while the GPU was still
 * reading it) was exact at S=2 and wrong by 1e-1 at S=4 and S=8. The engine still produced
 * fluent text; a long prompt came back as "!!!!!!!!". Nothing failed, nothing warned.
 *
 * The lesson is narrower than "test more": a test that pins one value of a parameter proves
 * nothing about the others, and the parameter here is the batch width the fast path uses.
 * Every S the engine can emit is checked here, against the sequential path, at whatever split
 * fraction the environment sets.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../models.h"
#include "../qwen35_cu_spec.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static double score(const float*a,const float*b,int n){ double d=0,r=0;
    for(int i=0;i<n;i++){ double e=(double)a[i]-b[i]; d+=e*e; r+=(double)b[i]*b[i]; }
    return r==0?(d==0?0:INFINITY):sqrt(d/r); }
int main(int argc,char**argv){
    const char *path = argc>1?argv[1]:nj_model_path("Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf");
    Q35Model M; if(!q35_open(&M,path)) return 1;
    Q35State R; const int n_ctx=4096;
    q35_state_init_ex(&R,&M,n_ctx,Q35_KV_Q8_0,q35_kv_bytes_per_token(&M.c,Q35_KV_Q8_0)*n_ctx+(32<<20),getenv("QWEN_KV_SPILL"),4096);
    Q35Cu G; if(!q35cu_model_init(&G,&M,n_ctx,R.kv.fmt)) return 77;
    q35cu_align_win(&G,R.kv.chunk); q35cu_mark_resident_elsewhere(&G);
    Q35Resident RES; q35_reside_anon(&M,&RES,Q35_RES_FFN); q35cu_pin_weights(&G,&M);
    q35cu_report(&G,stdout);
    Q35CuBatch Z; if(!q35cu_batch_init(&Z,&G,&R,8)) return 1;
    const int V=M.c.vocab;
    int toks[16]; for(int i=0;i<16;i++) toks[i]=1000+i*37;
    float *seq=(float*)malloc(4*(size_t)V), *bat=(float*)malloc(4*(size_t)V);
    int bad=0;
    printf("\n  %3s %10s %10s %12s  %s\n","S","s/batch","tok/s","rel vs seq","verdict");
    for(int S=1;S<=8;S*=2){
        /* reference: S sequential single-token passes */
        q35_state_reset(&R); q35cu_state_reset(&G);
        for(int i=0;i<S;i++) q35_forward_cu(&G,&R,toks[i],i,(i==S-1)?seq:NULL);
        /* the batch */
        q35_state_reset(&R); q35cu_state_reset(&G);
        q35_forward_cu_batch(&Z,toks,S,0,bat,0);
        const double rel=score(bat,seq,V);
        q35_state_reset(&R); q35cu_state_reset(&G);
        const double t0=now(); const int REP=3;
        for(int r=0;r<REP;r++){ q35_state_reset(&R); q35cu_state_reset(&G);
            q35_forward_cu_batch(&Z,toks,S,0,bat,0); }
        const double dt=(now()-t0)/REP;
        const int ok = (rel<=2e-3);
        if(!ok) bad++;
        printf("  %3d %10.3f %10.2f %12.3e  %s\n",S,dt,S/dt,rel,ok?"ok":"WRONG");
        fflush(stdout);
    }
    printf("\n%s\n", bad?"FAIL":"PASS");
    q35cu_batch_free(&Z); q35cu_model_free(&G); q35_state_free(&R); q35_close(&M);
    return bad?1:0;
}
