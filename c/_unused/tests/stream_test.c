/* stream_test.c — the real thing: pull routed experts off NVMe and run the FFN on them
 * WITHOUT ever materializing a weight matrix.
 *
 * This is the whole thesis in one binary. For each simulated token we:
 *   1. pread each routed expert's gate/up/down slice straight out of the GGUF (no mmap,
 *      no page-cache reliance — that is what a 254 GB model on a 31 GB box forces)
 *   2. contract x against the quantized bytes in place (kq_gemm)
 *   3. never allocate a dequantized expert
 *
 * and report the only number that matters for a disk-bound engine: GB/token and tok/s.
 *
 * Usage: stream_test <first-shard.gguf> [n_tokens]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "../gguf.h"
#include "../dsv4.h"

static double now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9*ts.tv_nsec;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <first-shard.gguf> [n_tokens]\n", argv[0]); return 2; }
    const int n_tok = argc > 2 ? atoi(argv[2]) : 4;

    gguf_model m;
    double t0 = now();
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    printf("model: %s, %d tensors, header parsed in %.1f ms\n",
           gguf_str(&m, "general.architecture", "?"), m.n_tensors, (now()-t0)*1e3);

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
    snprintf(key, sizeof(key), "%s.leading_dense_block_count", arch);
    int dense0 = (int)gguf_u64(&m, key, 0);

    printf("  %d layers (%d dense), %d experts, top-%d, d_model=%d\n\n",
           n_layer, dense0, n_exp, topk, d_model);

    /* collect every MoE layer that actually has expert tensors */
    typedef struct { const gguf_tensor *g,*u,*d; } EL;
    EL *L = calloc(n_layer, sizeof(EL));
    int n_moe = 0;
    for(int i = 0; i < n_layer; i++){
        char ng[128], nu[128], nd[128];
        snprintf(ng, sizeof(ng), "blk.%d.ffn_gate_exps.weight", i);
        snprintf(nu, sizeof(nu), "blk.%d.ffn_up_exps.weight",   i);
        snprintf(nd, sizeof(nd), "blk.%d.ffn_down_exps.weight", i);
        const gguf_tensor *g = gguf_find(&m, ng), *u = gguf_find(&m, nu), *d = gguf_find(&m, nd);
        if(g && u && d){ L[n_moe].g = g; L[n_moe].u = u; L[n_moe].d = d; n_moe++; }
    }
    printf("  %d MoE layers found\n", n_moe);
    if(!n_moe){ printf("no expert tensors\n"); return 1; }

    const gguf_tensor *g0 = L[0].g, *d0 = L[0].d;
    const int64_t d_ff = g0->ne[1];
    printf("  expert FFN: %lld x %lld, gate/up %s, down %s (layer 0)\n",
           (long long)g0->ne[0], (long long)d_ff, kq_name(g0->type), kq_name(d0->type));

    /* Expert size is NOT uniform across layers. "Dynamic" means Unsloth bumps individual
     * layers to a fatter type where the imatrix says it matters (in GLM-5.2-UD-Q2_K_XL:
     * blk.8 down is IQ4_XS, blk.78 is Q2_K/Q3_K, the rest IQ2_XS/IQ3_XXS). Sizing a slab
     * from layer 0 gives a short pread on exactly those layers — so size for the max, and
     * total the real per-token bytes layer by layer. */
    int64_t max_g = 0, max_d = 0;
    double  bytes_tok = 0;
    int n_odd = 0;
    for(int i = 0; i < n_moe; i++){
        int64_t bg = kq_row_bytes(L[i].g->type, L[i].g->ne[0]*L[i].g->ne[1]);
        int64_t bu = kq_row_bytes(L[i].u->type, L[i].u->ne[0]*L[i].u->ne[1]);
        int64_t bd = kq_row_bytes(L[i].d->type, L[i].d->ne[0]*L[i].d->ne[1]);
        if(bg > max_g) max_g = bg;
        if(bu > max_g) max_g = bu;
        if(bd > max_d) max_d = bd;
        bytes_tok += (double)topk * (bg + bu + bd);
        if(L[i].g->type != g0->type || L[i].d->type != d0->type) n_odd++;
    }
    double gb_tok = bytes_tok / 1e9;
    printf("  %d of %d layers use a different quant than layer 0 (Unsloth Dynamic)\n\n",
           n_odd, n_moe);

    /* buffers: ONE expert in flight, sized for the fattest layer. No dequantized matrix. */
    const int64_t bg = max_g, bd = max_d;
    size_t cap = (size_t)(2*bg + bd) + 4096;
    uint8_t *slab = aligned_alloc(4096, cap);
    float *x    = malloc((size_t)d_model*sizeof(float));
    float *gate = malloc((size_t)d_ff*sizeof(float));
    float *up   = malloc((size_t)d_ff*sizeof(float));
    float *out  = malloc((size_t)d_model*sizeof(float));
    float *acc  = malloc((size_t)d_model*sizeof(float));
    if(!slab||!x||!gate||!up||!out||!acc){ printf("OOM\n"); return 1; }

    for(int i = 0; i < d_model; i++) x[i] = (float)(0.05*sin(i*0.013));

    printf("streaming %d token(s), %d layers x top-%d experts, cold reads\n", n_tok, n_moe, topk);
    printf("  RAM for weights: one %.1f MB expert slab. Nothing else is resident.\n\n",
           (2.0*bg + bd)/1e6);

    double t_io = 0, t_mm = 0, bytes = 0;
    double tstart = now();

    for(int t = 0; t < n_tok; t++){
        for(int l = 0; l < n_moe; l++){
            for(int e = 0; e < d_model; e++) acc[e] = 0.f;
            for(int k = 0; k < topk; k++){
                /* a different expert every time: defeat the page cache, like real routing */
                int eid = (int)(((int64_t)t*7919 + (int64_t)l*104729 + k*31) % n_exp);

                int fd; uint64_t off; int64_t nb, rows, cols;
                double a = now();
                uint8_t *pg = slab, *pu = slab + bg, *pd = slab + 2*bg;

                gguf_expert_slice(&m, L[l].g, eid, &fd, &off, &nb, &rows, &cols);
                if(pread(fd, pg, nb, off) != nb){ perror("pread gate"); return 1; }
                bytes += nb;
                gguf_expert_slice(&m, L[l].u, eid, &fd, &off, &nb, &rows, &cols);
                if(pread(fd, pu, nb, off) != nb){ perror("pread up"); return 1; }
                bytes += nb;
                gguf_expert_slice(&m, L[l].d, eid, &fd, &off, &nb, &rows, &cols);
                if(pread(fd, pd, nb, off) != nb){ perror("pread down"); return 1; }
                bytes += nb;
                t_io += now() - a;

                a = now();
                /* gate/up: [d_ff, d_model] contracted against x, straight off the quant bytes */
                kq_gemm(gate, x, pg, L[l].g->type, 1, d_model, (int)d_ff);
                kq_gemm(up,   x, pu, L[l].u->type, 1, d_model, (int)d_ff);
                dsv4_swiglu(gate, up, (int)d_ff, 0.f);          /* GLM has no clamp */
                kq_gemm(out, gate, pd, L[l].d->type, 1, (int)d_ff, d_model);
                for(int e = 0; e < d_model; e++) acc[e] += out[e];
                t_mm += now() - a;
            }
            /* feed the layer output forward so nothing is optimized away */
            for(int e = 0; e < d_model; e++) x[e] = 0.05f*tanhf(acc[e]);
        }
        printf("  token %d done\n", t+1);
    }
    double wall = now() - tstart;

    printf("\nresults\n");
    printf("  bytes read        : %.2f GB  (%.2f GB/token; table says %.2f)\n",
           bytes/1e9, bytes/1e9/n_tok, gb_tok);
    printf("  disk time         : %6.2f s  -> %.2f GB/s effective\n", t_io, bytes/1e9/t_io);
    printf("  matmul time       : %6.2f s  (on the quantized bytes, no dequant buffer)\n", t_mm);
    printf("  wall              : %6.2f s  -> %.3f tok/s\n", wall, n_tok/wall);
    printf("  disk share of wall: %5.1f %%\n", 100.0*t_io/wall);

    printf("\n  colibri today (int4, 11.5 GB/token) would need %.1f s/token on this disk\n",
           11.5/(bytes/1e9/t_io));
    free(slab); free(x); free(gate); free(up); free(out); free(acc); free(L);
    gguf_close(&m);
    return 0;
}
