/* kernel_bench.c — my CUDA kernel vs ggml's, on the same real expert weights.
 *
 * My kernel measured 596 GFLOP/s, which is ~4% of this card's fp32 peak and ~5% of its
 * memory bandwidth. That gap is either (a) inherent to decoding a lattice quant, or (b) my
 * kernel being naive. ggml's CUDA i-quant kernels are mature and heavily tuned, so they
 * settle the question: if ggml is 3x faster on the identical tensor, the headroom is real
 * and worth chasing. If it is not, the gap is the format's floor and I should stop.
 *
 * Same weights, same shapes, same device. No modelling, no extrapolation.
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

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }

    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }

    /* one real expert: gate is IQ2_XS [6144 -> 2048], down is IQ3_XXS [2048 -> 6144] */
    struct { const char *name; const gguf_tensor *t; } W[2] = {
        { "blk.10.ffn_gate_exps.weight", NULL },
        { "blk.10.ffn_down_exps.weight", NULL },
    };
    for(int i=0;i<2;i++){
        W[i].t = gguf_find(&m, W[i].name);
        if(!W[i].t){ printf("missing %s\n", W[i].name); return 1; }
    }

    if(!kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl)){
        printf("table upload failed\n"); return 1; }

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if(!be){ printf("ggml CUDA backend init failed\n"); return 1; }
    printf("ggml CUDA backend: %s\n\n", ggml_backend_name(be));

    printf("  %-9s %6s %6s   %11s   %11s   %11s   %6s\n",
           "type", "I", "O", "ggml", "ours fp32", "ours int8", "gain");

    for(int w = 0; w < 2; w++){
        const gguf_tensor *T = W[w].t;
        const int I = (int)T->ne[0], O = (int)T->ne[1];

        /* pull expert 5's slice off disk — the exact bytes both kernels will chew on */
        int fd; uint64_t off; int64_t nb, rows, cols;
        gguf_expert_slice(&m, T, 5, &fd, &off, &nb, &rows, &cols);
        uint8_t *h = malloc(nb);
        if(pread(fd, h, nb, off) != nb){ perror("pread"); return 1; }

        float *hx = malloc((size_t)I*sizeof(float));
        for(int i=0;i<I;i++) hx[i] = (float)(0.05*sin(i*0.011));

        /* ---------------- ggml ---------------- */
        struct ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(),
                                       NULL, /*no_alloc*/ true };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *gw = ggml_new_tensor_2d(ctx, (enum ggml_type)T->type, I, O);
        struct ggml_tensor *gx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I, 1);
        struct ggml_tensor *gy = ggml_mul_mat(ctx, gw, gx);
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gy);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if(!buf){ printf("alloc failed\n"); return 1; }
        ggml_backend_tensor_set(gw, h,  0, nb);
        ggml_backend_tensor_set(gx, hx, 0, (size_t)I*sizeof(float));

        ggml_backend_graph_compute(be, gf);          /* warm up */
        ggml_backend_synchronize(be);

        const int R = 300;
        double t0 = now();
        for(int r=0;r<R;r++) ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        double t_ggml = (now()-t0)/R;

        float *y_ggml = malloc((size_t)O*sizeof(float));
        ggml_backend_tensor_get(gy, y_ggml, 0, (size_t)O*sizeof(float));

        /* ---------------- ours: fp32 ---------------- */
        uint8_t *dW; cudaMalloc((void**)&dW, nb);
        cudaMemcpy(dW, h, nb, cudaMemcpyHostToDevice);
        float *dx, *dy;
        cudaMalloc((void**)&dx, (size_t)I*sizeof(float));
        cudaMalloc((void**)&dy, (size_t)O*sizeof(float));
        cudaMemcpy(dx, hx, (size_t)I*sizeof(float), cudaMemcpyHostToDevice);

        kq_cu_gemv(dy, dx, dW, T->type, I, O, kq_typesize(T->type), 0);
        cudaDeviceSynchronize();

        t0 = now();
        for(int r=0;r<R;r++) kq_cu_gemv(dy, dx, dW, T->type, I, O, kq_typesize(T->type), 0);
        cudaDeviceSynchronize();
        double t_f32 = (now()-t0)/R;

        float *y_f32 = malloc((size_t)O*sizeof(float));
        cudaMemcpy(y_f32, dy, (size_t)O*sizeof(float), cudaMemcpyDeviceToHost);

        /* ---------------- ours: int8 / dp4a ---------------- */
        signed char *dxq; float *dxd;
        cudaMalloc((void**)&dxq, (size_t)I);
        cudaMalloc((void**)&dxd, (size_t)(I/32)*sizeof(float));
        kq_cu_quant_act(dxq, dxd, dx, 1, I, 0);
        cudaDeviceSynchronize();

        if(!kq_cu_gemm_i8(dy, dxq, dxd, dW, T->type, 1, I, O, 0)){
            printf("  %-9s int8 path unavailable\n", kq_name(T->type)); return 1; }
        cudaDeviceSynchronize();

        t0 = now();
        for(int r=0;r<R;r++){
            kq_cu_quant_act(dxq, dxd, dx, 1, I, 0);      /* in the engine this is once per
                                                          * LAYER, not per expert — counted
                                                          * here anyway, so the number is
                                                          * pessimistic */
            kq_cu_gemm_i8(dy, dxq, dxd, dW, T->type, 1, I, O, 0);
        }
        cudaDeviceSynchronize();
        double t_i8 = (now()-t0)/R;

        float *y_i8 = malloc((size_t)O*sizeof(float));
        cudaMemcpy(y_i8, dy, (size_t)O*sizeof(float), cudaMemcpyDeviceToHost);

        /* both of ours must agree with ggml, or speed means nothing */
        double w32=0, w8=0, den=0;
        for(int i=0;i<O;i++){
            double a = fabs(y_ggml[i] - y_f32[i]);
            double b = fabs(y_ggml[i] - y_i8[i]);
            if(a > w32) w32 = a;
            if(b > w8)  w8  = b;
            if(fabs(y_ggml[i]) > den) den = fabs(y_ggml[i]);
        }

        const double fl = 2.0*(double)I*O;
        printf("  %-9s %6d %6d   %11.1f   %11.1f   %11.1f   %6.2fx\n",
               kq_name(T->type), I, O,
               fl/1e9/t_ggml, fl/1e9/t_f32, fl/1e9/t_i8, t_f32/t_i8);
        printf("  %-9s %6s %6s   %11s   %10.1e   %10.1e   (deviation from ggml)\n",
               "", "", "", "", w32/(den+1e-9), w8/(den+1e-9));
        free(y_f32); free(y_i8);
        cudaFree(dxq); cudaFree(dxd);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        cudaFree(dW); cudaFree(dx); cudaFree(dy);
        free(h); free(hx); free(y_ggml);
    }

    printf("\n  GFLOP/s. 'gain' is int8 over our own fp32 kernel.\n");
    ggml_backend_free(be);
    return 0;
}
