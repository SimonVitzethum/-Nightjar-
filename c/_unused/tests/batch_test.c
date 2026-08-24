/* batch_test.c — what does batching S tokens per forward actually buy?
 *
 * Two independent effects, and they should be measured separately or they get confused:
 *
 *   A. DECODE AMORTIZATION (pure compute). Decoding an IQ2_XS lattice point is much more
 *      expensive than using it: 8 weights cost a shared-memory lookup, a sign unpack and a
 *      scale multiply, then only 8 multiply-adds. At S=8 the decode happens once and feeds
 *      64 multiply-adds. This is a property of the kernel and does not depend on routing.
 *
 *   B. EXPERT OVERLAP (I/O). S tokens each pick top-8 of 256. If two tokens pick the same
 *      expert, its ~12 MB is read once, not twice. How much overlap there is depends
 *      entirely on the routing distribution — so this half is reported as a function of
 *      the skew, and is a MODEL, not a measurement (no real forward pass yet).
 *
 * A is real regardless of what the router does. B is real only if routing is skewed.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "../gguf.h"
#include "../kquant_cuda.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
#define CU(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#c,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    struct cudaDeviceProp pr; CU(cudaGetDeviceProperties(&pr,0));
    if(!kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl)){
        printf("tables failed\n"); return 1; }

    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    const char *arch = gguf_str(&m,"general.architecture","");
    char k[128];
    #define KV(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    int n_layer=KV("block_count"), topk=KV("expert_used_count");
    int n_exp=KV("expert_count"), d_model=KV("embedding_length");
    #undef KV

    const gguf_tensor *G=NULL,*D=NULL;
    for(int i=0;i<n_layer && !G;i++){
        char a[128],c[128];
        snprintf(a,sizeof(a),"blk.%d.ffn_gate_exps.weight",i);
        snprintf(c,sizeof(c),"blk.%d.ffn_down_exps.weight",i);
        G=gguf_find(&m,a); D=gguf_find(&m,c);
    }
    const int64_t d_ff = G->ne[1];

    /* ---------- A. decode amortization: pure GPU, one expert, S tokens ---------- */
    printf("A. decode amortization (compute only, no I/O, no routing assumptions)\n\n");
    int fd; uint64_t off; int64_t nb, rows, cols;
    gguf_expert_slice(&m, G, 7, &fd, &off, &nb, &rows, &cols);
    uint8_t *h = malloc(nb);
    if(pread(fd, h, nb, off) != nb){ perror("pread"); return 1; }

    uint8_t *dW; CU(cudaMalloc((void**)&dW, nb));
    CU(cudaMemcpy(dW, h, nb, cudaMemcpyHostToDevice));
    float *dx, *dy;
    CU(cudaMalloc((void**)&dx, (size_t)KQ_CU_SMAX*d_model*sizeof(float)));
    CU(cudaMalloc((void**)&dy, (size_t)KQ_CU_SMAX*d_ff   *sizeof(float)));
    float *hx = malloc((size_t)KQ_CU_SMAX*d_model*sizeof(float));
    for(int i=0;i<KQ_CU_SMAX*d_model;i++) hx[i]=(float)(0.05*sin(i*0.011));
    CU(cudaMemcpy(dx, hx, (size_t)KQ_CU_SMAX*d_model*sizeof(float), cudaMemcpyHostToDevice));

    /* one expert = 3 matrices; here we time the gate matmul and scale up */
    const double flop_expert = 2.0 * 3.0 * (double)d_model * d_ff;   /* per token */
    printf("  %-3s %10s %12s %14s\n", "S", "ms/call", "GFLOP/s", "us per token");
    for(int S=1; S<=KQ_CU_SMAX; S*=2){
        const int R = 200;
        CU(cudaDeviceSynchronize());
        double t0 = now();
        for(int r=0;r<R;r++)
            kq_cu_gemm(dy, dx, dW, G->type, S, (int)cols, (int)rows, kq_typesize(G->type), 0);
        CU(cudaDeviceSynchronize());
        double dt = (now()-t0)/R;
        /* the timed call is 1 of the 3 matrices of an expert */
        double gflops = 2.0*S*(double)cols*rows/1e9/dt;
        printf("  %-3d %10.3f %12.1f %14.1f\n", S, dt*1e3, gflops, dt*1e6/S);
    }

    /* ---------- B. expert overlap across a batch ---------- */
    printf("\nB. expert overlap: unique experts a batch of S tokens actually needs\n");
    printf("   (a MODEL of routing skew, not a measurement — see the header)\n\n");
    printf("  %-6s", "zipf");
    for(int S=1;S<=8;S*=2) printf(" %8s", "S=");
    printf("\n  %-6s", "");
    for(int S=1;S<=8;S*=2) printf(" %8d", S);
    printf("\n");

    double *cdf = malloc((size_t)n_exp*sizeof(double));
    for(double zs=0.0; zs<=1.6; zs+=0.5){
        double s=0;
        for(int r=0;r<n_exp;r++){ s += 1.0/pow(r+1,zs); cdf[r]=s; }
        for(int r=0;r<n_exp;r++) cdf[r]/=s;

        printf("  %-6.1f", zs);
        for(int S=1;S<=8;S*=2){
            unsigned sd = 4242;
            double uniq_tot = 0; int trials = 400;
            char *seen = calloc(n_exp,1);
            char *tok  = calloc(n_exp,1);
            for(int t=0;t<trials;t++){
                memset(seen,0,n_exp);
                int uniq=0;
                for(int b=0;b<S;b++){
                    /* top-k picks k DISTINCT experts. Drawing k times WITH replacement
                     * would let one token "collide with itself" and fake overlap even at
                     * S=1 — which is how the first version of this table reported 1.58x
                     * for a single token, an impossibility. */
                    memset(tok,0,n_exp);
                    for(int got=0; got<topk; ){
                        double u=(double)rand_r(&sd)/RAND_MAX;
                        int lo=0,hi=n_exp-1;
                        while(lo<hi){ int mid=(lo+hi)/2; if(cdf[mid]<u) lo=mid+1; else hi=mid; }
                        if(tok[lo]) continue;            /* already this token's, redraw */
                        tok[lo]=1; got++;
                        if(!seen[lo]){ seen[lo]=1; uniq++; }
                    }
                }
                uniq_tot += uniq;
            }
            free(seen); free(tok);
            double uniq_avg = uniq_tot/trials;
            /* bytes per token scale as unique/(S*topk) */
            printf(" %7.2fx", (double)(S*topk)/uniq_avg);
        }
        printf("\n");
    }
    printf("\n  (the number is how much LESS expert data per token the batch reads;\n");
    printf("   1.00x means no overlap at all)\n");
    return 0;
}
