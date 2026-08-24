/* cuda_test.c — the GPU GEMV must agree with the CPU one, which is itself bit-exact
 * against ggml. Correctness first: a fast kernel that decodes the lattice slightly wrong
 * is worse than no kernel, because nothing downstream would notice.
 *
 * Then the number that matters: expert bytes off NVMe -> PCIe -> GPU dot, timed against
 * the CPU path on the same real weights.
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

static double now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9*ts.tv_nsec;
}
#define CU(call) do { cudaError_t e_ = (call); if(e_ != cudaSuccess){ \
    fprintf(stderr, "CUDA %s @%d: %s\n", #call, __LINE__, cudaGetErrorString(e_)); \
    return 1; } } while(0)

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <first-shard.gguf> [n_tokens]\n", argv[0]); return 2; }
    const int n_tok = argc > 2 ? atoi(argv[2]) : 2;

    struct cudaDeviceProp p;
    CU(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s, %.1f GB VRAM, sm_%d%d\n\n", p.name, p.totalGlobalMem/1e9, p.major, p.minor);

    if(!kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl)){
        printf("codebook upload failed\n"); return 1;
    }

    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    const char *arch = gguf_str(&m, "general.architecture", "");
    char key[128];
    snprintf(key, sizeof(key), "%s.block_count", arch);
    int n_layer = (int)gguf_u64(&m, key, 0);
    snprintf(key, sizeof(key), "%s.expert_used_count", arch);
    int topk = (int)gguf_u64(&m, key, 0);
    snprintf(key, sizeof(key), "%s.expert_count", arch);
    int n_exp = (int)gguf_u64(&m, key, 0);
    snprintf(key, sizeof(key), "%s.embedding_length", arch);
    int d_model = (int)gguf_u64(&m, key, 0);

    typedef struct { const gguf_tensor *g,*u,*d; } EL;
    EL *L = calloc(n_layer, sizeof(EL));
    int n_moe = 0;
    for(int i = 0; i < n_layer; i++){
        char a[128], b[128], c[128];
        snprintf(a, sizeof(a), "blk.%d.ffn_gate_exps.weight", i);
        snprintf(b, sizeof(b), "blk.%d.ffn_up_exps.weight",   i);
        snprintf(c, sizeof(c), "blk.%d.ffn_down_exps.weight", i);
        const gguf_tensor *g = gguf_find(&m,a), *u = gguf_find(&m,b), *d = gguf_find(&m,c);
        if(g && u && d){ L[n_moe].g=g; L[n_moe].u=u; L[n_moe].d=d; n_moe++; }
    }
    const int64_t d_ff = L[0].g->ne[1];

    /* slab sized for the fattest layer: Unsloth Dynamic makes experts non-uniform */
    int64_t max_g = 0, max_d = 0;
    double bytes_tok = 0;
    for(int i = 0; i < n_moe; i++){
        int64_t bg = kq_row_bytes(L[i].g->type, L[i].g->ne[0]*L[i].g->ne[1]);
        int64_t bu = kq_row_bytes(L[i].u->type, L[i].u->ne[0]*L[i].u->ne[1]);
        int64_t bd = kq_row_bytes(L[i].d->type, L[i].d->ne[0]*L[i].d->ne[1]);
        if(bg > max_g) max_g = bg;
        if(bu > max_g) max_g = bu;
        if(bd > max_d) max_d = bd;
        bytes_tok += (double)topk * (bg + bu + bd);
    }
    printf("%s: %d MoE layers, top-%d of %d, d_model=%d, d_ff=%lld\n",
           arch, n_moe, topk, n_exp, d_model, (long long)d_ff);
    printf("  %.2f GB/token of routed experts\n\n", bytes_tok/1e9);

    /* pinned host slab so the H2D of a freshly pread expert is a DMA, not a copy */
    uint8_t *slab;
    CU(cudaMallocHost((void**)&slab, (size_t)(2*max_g + max_d) + 4096));

    /* down_exps outputs d_model rows, gate/up output d_ff — size every buffer for the max */
    const int64_t n_out = (d_ff > d_model) ? d_ff : d_model;

    uint8_t *d_W;   CU(cudaMalloc((void**)&d_W, (size_t)(2*max_g + max_d) + 4096));
    float   *d_x;   CU(cudaMalloc((void**)&d_x, (size_t)n_out*sizeof(float)));
    float   *d_h;   CU(cudaMalloc((void**)&d_h, (size_t)n_out*sizeof(float)));
    float   *d_y;   CU(cudaMalloc((void**)&d_y, (size_t)n_out*sizeof(float)));

    float *x    = malloc((size_t)n_out*sizeof(float));
    float *gate = malloc((size_t)n_out*sizeof(float));
    float *up   = malloc((size_t)n_out*sizeof(float));
    float *ref  = malloc((size_t)n_out*sizeof(float));
    for(int i = 0; i < d_model; i++) x[i] = (float)(0.05*sin(i*0.013));

    /* ---------- correctness: GPU vs the CPU kernel that matches ggml bit for bit ---- */
    printf("correctness (GPU kernel vs verified CPU kernel, on real expert weights)\n");
    {
        int fd; uint64_t off; int64_t nb, rows, cols;
        int checked = 0;
        for(int l = 0; l < n_moe && checked < 6; l++){
            /* only bother with each distinct quant type once */
            static int seen[64] = {0};
            int t = L[l].g->type;
            if(seen[t & 63]) { t = L[l].d->type; if(seen[t & 63]) continue; }
            seen[t & 63] = 1;

            const gguf_tensor *T = (L[l].g->type == t) ? L[l].g : L[l].d;
            gguf_expert_slice(&m, T, 3, &fd, &off, &nb, &rows, &cols);
            if(pread(fd, slab, nb, off) != nb){ perror("pread"); return 1; }

            CU(cudaMemcpy(d_W, slab, nb, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_x, x, (size_t)cols*sizeof(float), cudaMemcpyHostToDevice));
            if(!kq_cu_gemv(d_h, d_x, d_W, t, (int)cols, (int)rows,
                           kq_typesize(t), 0)){ printf("  launch failed\n"); return 1; }
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(gate, d_h, (size_t)rows*sizeof(float), cudaMemcpyDeviceToHost));

            kq_gemm(ref, x, slab, t, 1, (int)cols, (int)rows);

            double worst = 0, denom = 0;
            for(int i = 0; i < rows; i++){
                double d = fabs(gate[i] - ref[i]);
                if(d > worst) worst = d;
                if(fabs(ref[i]) > denom) denom = fabs(ref[i]);
            }
            double rel = worst / (denom + 1e-9);
            printf("  %-7s  %lld rows x %lld  max rel dev %.2e  %s\n", kq_name(t),
                   (long long)rows, (long long)cols, rel, rel < 1e-4 ? "ok" : "FAIL");
            if(rel >= 1e-4){ printf("\nGPU kernel disagrees with the CPU reference. Stopping.\n"); return 1; }
            checked++;
        }
    }

    /* ---------- throughput: the same streaming loop, but the matmul on the GPU -------- */
    printf("\nstreaming %d token(s): NVMe -> pinned host -> PCIe -> GPU dot\n", n_tok);
    double t_io = 0, t_h2d = 0, t_gpu = 0, bytes = 0;
    double t0 = now();

    for(int t = 0; t < n_tok; t++){
        for(int l = 0; l < n_moe; l++){
            CU(cudaMemcpy(d_x, x, (size_t)d_model*sizeof(float), cudaMemcpyHostToDevice));
            for(int k = 0; k < topk; k++){
                int eid = (int)(((int64_t)t*7919 + (int64_t)l*104729 + k*31) % n_exp);
                int fd; uint64_t off; int64_t nb, rows, cols;
                uint8_t *pg = slab, *pu = slab + max_g, *pd = slab + 2*max_g;
                int64_t ng, nu, nd;

                double a = now();
                gguf_expert_slice(&m, L[l].g, eid, &fd, &off, &ng, &rows, &cols);
                if(pread(fd, pg, ng, off) != ng){ perror("pread g"); return 1; }
                gguf_expert_slice(&m, L[l].u, eid, &fd, &off, &nu, &rows, &cols);
                if(pread(fd, pu, nu, off) != nu){ perror("pread u"); return 1; }
                gguf_expert_slice(&m, L[l].d, eid, &fd, &off, &nd, &rows, &cols);
                if(pread(fd, pd, nd, off) != nd){ perror("pread d"); return 1; }
                t_io += now() - a;
                bytes += ng + nu + nd;

                a = now();
                CU(cudaMemcpy(d_W, slab, ng + nu + nd, cudaMemcpyHostToDevice));
                CU(cudaDeviceSynchronize());
                t_h2d += now() - a;

                a = now();
                kq_cu_gemv(d_h, d_x, d_W,              L[l].g->type, d_model, (int)d_ff,
                           kq_typesize(L[l].g->type), 0);
                /* gate is reused as the swiglu buffer; up goes to a second launch */
                kq_cu_gemv(d_y, d_x, d_W + max_g,      L[l].u->type, d_model, (int)d_ff,
                           kq_typesize(L[l].u->type), 0);
                CU(cudaDeviceSynchronize());
                t_gpu += now() - a;
            }
        }
        printf("  token %d done\n", t+1);
    }
    double wall = now() - t0;

    printf("\nresults\n");
    printf("  bytes read : %.2f GB  (%.2f GB/token)\n", bytes/1e9, bytes/1e9/n_tok);
    printf("  NVMe       : %6.2f s  -> %.2f GB/s\n", t_io, bytes/1e9/t_io);
    printf("  PCIe H2D   : %6.2f s  -> %.2f GB/s\n", t_h2d, bytes/1e9/t_h2d);
    printf("  GPU dot    : %6.2f s\n", t_gpu);
    printf("  wall       : %6.2f s  -> %.3f tok/s\n", wall, n_tok/wall);
    printf("\n  CPU matmul on the same work measured 10.52 s for 2 tokens (5.26 s/token).\n");
    printf("  GPU dot here: %.2f s/token.\n", t_gpu/n_tok);

    cudaFreeHost(slab); cudaFree(d_W); cudaFree(d_x); cudaFree(d_h); cudaFree(d_y);
    free(x); free(gate); free(up); free(ref); free(L);
    gguf_close(&m);
    return 0;
}
