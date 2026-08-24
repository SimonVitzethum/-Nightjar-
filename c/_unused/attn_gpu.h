#ifndef COLIBRI_ATTN_GPU_H
#define COLIBRI_ATTN_GPU_H
/* attn_gpu.h — MLA attention on the GPU, weights streamed from RAM.
 *
 * WHY THIS HAD TO HAPPEN
 *
 * Moving the routed experts to the GPU cut them from 60 ms to 0.4 ms per layer, and the
 * drive went from 870 MiB/s to 1150. That is all it went, because the bottleneck simply
 * moved. Measured, one attention layer on the CPU:
 *
 *      q_b          Q8_0   35.7 MB    35.0 ms   1.02 GB/s
 *      attn_output  Q5_K   69.2 MB    69.9 ms   0.99 GB/s
 *      k_b (x64)    Q8_0    6.7 MB    18.5 ms   0.36 GB/s
 *      ----------------------------------------------------
 *      one layer          133 MB     156 ms    0.85 GB/s
 *      x 76 layers                   11.8 s/token
 *
 * 0.85 GB/s, out of RAM that does 62. The reason is the same one the GPU kernel had before
 * dp4a: kq_gemm decodes a k-quant scalar, one weight at a time. On the GPU I fixed that; on
 * the CPU I never touched it. So attention was 9x more expensive than the disk read it was
 * supposed to be hiding behind, and no amount of I/O tuning could matter.
 *
 * THE SHAPE OF THE FIX
 *
 * The attention weights are 10.4 GB — they do not fit in 8 GB of VRAM alongside an expert
 * cache, so they cannot live there. But they do not need to: PCIe moves 24 GB/s, four times
 * what the NVMe does. Pushing a layer's 133 MB across costs 5.5 ms, against the 17.9 ms the
 * drive needs for that same layer's experts. It hides completely behind the read.
 *
 * So the weights stay in RAM (mmap, page-cache backed), and each layer's are uploaded into
 * a double buffer while the previous layer computes. The GPU decodes them in registers, the
 * same way it decodes the experts. Nothing is ever dequantized on the host.
 *
 * What stays on the CPU: RoPE, the softmax, and the score loop. They are O(T * 576) per
 * head — kilobytes, not megabytes — and moving them would buy nothing but latency.
 */
#include <cuda_runtime.h>

#include "mla.h"
#include "kquant_cuda.h"

/* kq_cu_gemv returns 0 when it refuses to launch — a row length it cannot block, a type it
 * cannot decode. Ignoring that return is how the GPU attention came to produce garbage:
 * the kernel never ran, the output buffer kept whatever was in it, x went to NaN, and the
 * router picked expert -1. The failure surfaced 400 lines away as "estream: read failed".
 * A kernel that declines to run must be loud. */
#define AG_GEMV(...) do{ if(!kq_cu_gemv(__VA_ARGS__)){ \
    fprintf(stderr, "attn_gpu: gemv refused at %s:%d\n", __FILE__, __LINE__); exit(1); } }while(0)

typedef struct {
    int d_model, q_lora, kv_lora, n_head, qk_nope, qk_rope, v_head;

    uint8_t *W[2];          /* double-buffered layer weights on the device (streaming mode) */
    uint8_t *Wall;          /* ALL layers, resident on the device (residency mode) */
    int      resident, n_layer;
    int64_t  cap, off[6];   /* q_a, q_b, kv_a, k_b, v_b, o */
    int      cur;

    float   *x, *qa, *q, *kva, *qlat, *heads_d, *out;
    cudaStream_t s_cpy, s_cmp;
    cudaEvent_t  ev[2];
    int      staged;        /* which layer is sitting in W[cur], -1 if none */

} AttnGpu;

/* Byte sizes of one layer's six attention tensors, in a fixed order. */
static void ag_sizes(const MLACfg *c, const MLALayer *L, int64_t *n){
    const int qk = c->qk_nope + c->qk_rope;
    n[0] = kq_row_bytes(L->tq_a,  (int64_t)c->d_model * c->q_lora);
    n[1] = kq_row_bytes(L->tq_b,  (int64_t)c->q_lora  * c->n_head*qk);
    n[2] = kq_row_bytes(L->tkv_a, (int64_t)c->d_model * (c->kv_lora + c->qk_rope));
    n[3] = kq_row_bytes(L->tk_b,  (int64_t)c->qk_nope * c->kv_lora) * c->n_head;
    n[4] = kq_row_bytes(L->tv_b,  (int64_t)c->kv_lora * c->v_head)  * c->n_head;
    n[5] = kq_row_bytes(L->to,    (int64_t)c->n_head*c->v_head * c->d_model);
}

/* The regions must be sized for the FATTEST layer, not for layer 0.
 *
 * "Dynamic" means exactly one of GLM's 77 layers (blk.8) carries attn_q_a and attn_output as
 * Q6_K where every other layer has Q5_K — and Q6_K is 210 bytes per 256 weights against
 * Q5_K's 176. Deriving the offsets from layer 0, as the first version of this did, makes the
 * upload for that one layer run past its region and into the next. It does not crash: it
 * corrupts q_b's bytes, layer 8's attention comes out NaN, the router picks expert -1, and
 * the whole thing surfaces 400 lines away as "estream: read failed".
 *
 * `mn` is the per-tensor maximum across every layer; the caller scans for it. */
static int attn_gpu_init(AttnGpu *A, const MLACfg *c, const int64_t *mn, int max_t){
    memset(A, 0, sizeof(*A));
    A->d_model=c->d_model; A->q_lora=c->q_lora; A->kv_lora=c->kv_lora;
    A->n_head=c->n_head; A->qk_nope=c->qk_nope; A->qk_rope=c->qk_rope; A->v_head=c->v_head;
    A->staged = -1;

    int64_t o = 0;
    for(int i=0;i<6;i++){ A->off[i] = o; o += (mn[i] + 255) & ~255LL; }
    A->cap = o;

    const int qk = c->qk_nope + c->qk_rope;
    #define A_(p,b) if(cudaMalloc((void**)&(p),(b)) != cudaSuccess) return 0
    A_(A->W[0], (size_t)A->cap);
    A_(A->W[1], (size_t)A->cap);
    A_(A->x,     (size_t)c->d_model*sizeof(float));
    A_(A->qa,    (size_t)c->q_lora *sizeof(float));
    A_(A->q,     (size_t)c->n_head*qk*sizeof(float));
    A_(A->kva,   (size_t)(c->kv_lora + c->qk_rope)*sizeof(float));
    A_(A->qlat,   (size_t)c->n_head*c->kv_lora*sizeof(float));
    A_(A->heads_d,(size_t)c->n_head*c->v_head*sizeof(float));
    A_(A->out,   (size_t)c->d_model*sizeof(float));
    #undef A_
    if(cudaStreamCreate(&A->s_cpy) != cudaSuccess) return 0;
    if(cudaStreamCreate(&A->s_cmp) != cudaSuccess) return 0;
    for(int i=0;i<2;i++)
        if(cudaEventCreateWithFlags(&A->ev[i], cudaEventDisableTiming) != cudaSuccess) return 0;
    (void)max_t;
    fprintf(stderr, "[gpu] attention on the GPU, %.0f MB/layer streamed over PCIe\n", A->cap/1e6);
    return 1;
}

/* PIN the attention weights so the GPU can actually DMA them.
 *
 * THE BUG THIS FIXES. The weights live in the GGUF mmap, which is PAGEABLE — and
 * cudaMemcpyAsync cannot DMA out of pageable memory. It does not fail; it silently degrades
 * to a synchronous staged copy. So the "async" prefetch overlaps with nothing, and each
 * layer's 133 MB costs its full transfer time on the critical path instead of hiding behind
 * the disk read. That is why moving attention to the GPU made the drive SLOWER (530 MiB/s)
 * than leaving it on the CPU (870): the prefetch thread spent its life blocked in memcpy.
 *
 * WHY WE COPY INSTEAD OF REGISTERING. The obvious fix is cudaHostRegister on the mmap — pin
 * the pages where they already are, no copy, no second 10 GB. CUDA refuses it: "operation
 * not supported". It will not pin a private file mapping. So we make our own copy in memory
 * that IS pinnable.
 *
 * WHY THAT COSTS NO RAM. It looks like we are duplicating 10.4 GB. We are not. Those pages
 * are touched on every single token, so they are permanently resident in the page cache
 * already — we are trading evictable page cache for unevictable pinned memory, one for one.
 * The kernel drops the now-unused mmap pages under pressure on its own.
 *
 * MEASURED (pinbench, one 140 MB layer):
 *     pageable mmap    4.42 GB/s   issue 463 ms   <- the "async" copy fully BLOCKS
 *     pinned            27.3 GB/s  issue 0.0 ms   <- a real background DMA
 * The 463 ms issue time is the whole bug: the prefetch thread never prefetched, it sat in a
 * synchronous memcpy while the drive idled.
 *
 * The caller does the pinning (it owns the model), and repoints MLALayer's six pointers at
 * the pinned copy — so this file needs no host mirror of its own. */
/* Hand the device ONE layer's requantized weights, to keep. The caller requantizes into a
 * small pinned staging buffer and calls this per layer, so there is never a 10 GB host copy:
 * the weights go straight from the mmap through the quantizer to VRAM.
 *
 * Residency is the whole point of requantizing attention. Streaming it cost 2.8 ms/layer —
 * nothing. What it cost was RAM: 12.5 GB pinned, unevictable, taken from the only thing the
 * disk actually cares about, which is the expert cache. Resident attention gives all of it
 * back and drops its PCIe traffic to zero at the same time. */
static int attn_gpu_reside(AttnGpu *A, int n_layer){
    if(cudaMalloc((void**)&A->Wall, (size_t)n_layer * (size_t)A->cap) != cudaSuccess){
        fprintf(stderr, "[gpu] attention will not fit in VRAM (%.2f GB): %s\n",
                (double)n_layer*A->cap/1e9, cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    A->resident = 1; A->n_layer = n_layer;
    fprintf(stderr, "[gpu] attention RESIDENT in VRAM: %.2f GB, zero PCIe per token\n",
            (double)n_layer*A->cap/1e9);
    return 1;
}

static int attn_gpu_put(AttnGpu *A, int l, const void *host_layer){
    return cudaMemcpy(A->Wall + (size_t)l*A->cap, host_layer, (size_t)A->cap,
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

/* Start layer `l`'s weights moving while the caller is still busy with layer l-1. */
static void attn_gpu_prefetch(AttnGpu *A, const MLACfg *c, const MLALayer *L, int l){
    if(A->resident) return;              /* already there, and staying */
    const int slot = l & 1;

    int64_t n[6]; ag_sizes(c, L, n);
    const void *src[6] = { L->pq_a, L->pq_b, L->pkv_a, L->pk_b, L->pv_b, L->po };
    for(int i=0;i<6;i++)
        cudaMemcpyAsync(A->W[slot] + A->off[i], src[i], n[i],
                        cudaMemcpyHostToDevice, A->s_cpy);
    cudaEventRecord(A->ev[slot], A->s_cpy);
}

/* One decode step, absorbed MLA. Same math as mla_step — the projections run on the device,
 * the score loop and RoPE stay on the host where they are cheap. */
static void mla_step_gpu(AttnGpu *A, float *out, const float *x, int pos,
                         const MLACfg *c, const MLALayer *L, int l,
                         MLACache *kv, MLAScratch *s, float rope_theta){
    const int nope=c->qk_nope, rope=c->qk_rope, qk=nope+rope;
    const int H=c->n_head, DL=c->kv_lora, DV=c->v_head;
    const int slot = l & 1;
    uint8_t *W = A->resident ? A->Wall + (size_t)l*A->cap : A->W[slot];

    mla_rms(s->xn, x, L->attn_norm, c->d_model, c->eps);

    if(!A->resident)
        cudaStreamWaitEvent(A->s_cmp, A->ev[slot], 0);    /* the weights have landed */
    cudaMemcpyAsync(A->x, s->xn, (size_t)c->d_model*sizeof(float),
                    cudaMemcpyHostToDevice, A->s_cmp);
    AG_GEMV(A->qa, A->x, W + A->off[0], L->tq_a, c->d_model, c->q_lora,
               kq_typesize(L->tq_a), A->s_cmp);
    cudaMemcpyAsync(s->qa, A->qa, (size_t)c->q_lora*sizeof(float),
                    cudaMemcpyDeviceToHost, A->s_cmp);
    cudaStreamSynchronize(A->s_cmp);

    mla_rms(s->qa, s->qa, L->q_a_norm, c->q_lora, c->eps);

    cudaMemcpyAsync(A->qa, s->qa, (size_t)c->q_lora*sizeof(float),
                    cudaMemcpyHostToDevice, A->s_cmp);
    AG_GEMV(A->q, A->qa, W + A->off[1], L->tq_b, c->q_lora, H*qk,
               kq_typesize(L->tq_b), A->s_cmp);
    AG_GEMV(A->kva, A->x, W + A->off[2], L->tkv_a, c->d_model, DL + rope,
               kq_typesize(L->tkv_a), A->s_cmp);
    cudaMemcpyAsync(s->q,   A->q,   (size_t)H*qk*sizeof(float), cudaMemcpyDeviceToHost, A->s_cmp);
    cudaMemcpyAsync(s->kva, A->kva, (size_t)(DL+rope)*sizeof(float), cudaMemcpyDeviceToHost, A->s_cmp);
    cudaStreamSynchronize(A->s_cmp);

    float *c_t  = kv->c  + (size_t)pos*DL;
    float *kr_t = kv->kr + (size_t)pos*rope;
    mla_rms(c_t, s->kva, L->kv_a_norm, DL, c->eps);
    memcpy(kr_t, s->kva + DL, (size_t)rope*sizeof(float));
    mla_rope(kr_t, rope, pos, rope_theta);
    if(pos + 1 > kv->n) kv->n = pos + 1;
    const int T = kv->n;

    const int64_t kb_row = kq_row_bytes(L->tk_b, (int64_t)nope*DL);
    const int64_t vb_row = kq_row_bytes(L->tv_b, (int64_t)DL*DV);

    /* RoPE every head's query, then absorb ALL of them into the latent in ONE launch. */
    for(int h=0; h<H; h++) mla_rope(s->q + (size_t)h*qk + nope, rope, pos, rope_theta);

    /* pack the nope halves contiguously: the batched kernel wants a stride, not a gather */
    for(int h=0; h<H; h++)
        memcpy(s->heads + (size_t)h*nope, s->q + (size_t)h*qk, (size_t)nope*sizeof(float));
    cudaMemcpyAsync(A->q, s->heads, (size_t)H*nope*sizeof(float),
                    cudaMemcpyHostToDevice, A->s_cmp);
    if(!kq_cu_gemv_heads(A->qlat, A->q, W + A->off[3], L->tk_b, nope, DL, H,
                         kb_row, nope, DL, A->s_cmp)){
        fprintf(stderr, "attn_gpu: batched k_b refused\n"); exit(1); }
    cudaMemcpyAsync(s->qlat_all, A->qlat, (size_t)H*DL*sizeof(float),
                    cudaMemcpyDeviceToHost, A->s_cmp);
    cudaStreamSynchronize(A->s_cmp);

    /* the score loop stays on the host: it is O(T * 576) per head, kilobytes of work */
    for(int h=0; h<H; h++){
        const float *ql = s->qlat_all + (size_t)h*DL;
        const float *qh = s->q + (size_t)h*qk;
        float mx = -INFINITY;
        for(int t=0;t<T;t++){
            const float *ct = kv->c + (size_t)t*DL, *krt = kv->kr + (size_t)t*rope;
            float sc = 0.f;
            for(int i=0;i<DL;i++)   sc += ql[i] * ct[i];
            for(int i=0;i<rope;i++) sc += qh[nope+i] * krt[i];
            sc *= c->attn_scale;
            s->scores[t] = sc;
            if(sc > mx) mx = sc;
        }
        float sum = 0.f;
        for(int t=0;t<T;t++){ s->scores[t] = expf(s->scores[t]-mx); sum += s->scores[t]; }
        const float inv = 1.0f/sum;

        float *ol = s->olat_all + (size_t)h*DL;
        for(int i=0;i<DL;i++) ol[i] = 0.f;
        for(int t=0;t<T;t++){
            const float w = s->scores[t]*inv;
            const float *ct = kv->c + (size_t)t*DL;
            for(int i=0;i<DL;i++) ol[i] += w*ct[i];
        }
    }

    /* and project all 64 heads back out through v_b, again in one launch */
    cudaMemcpyAsync(A->qlat, s->olat_all, (size_t)H*DL*sizeof(float),
                    cudaMemcpyHostToDevice, A->s_cmp);
    if(!kq_cu_gemv_heads(A->heads_d, A->qlat, W + A->off[4], L->tv_b, DL, DV, H,
                         vb_row, DL, DV, A->s_cmp)){
        fprintf(stderr, "attn_gpu: batched v_b refused\n"); exit(1); }

    AG_GEMV(A->out, A->heads_d, W + A->off[5], L->to, H*DV, c->d_model,
               kq_typesize(L->to), A->s_cmp);
    cudaMemcpyAsync(out, A->out, (size_t)c->d_model*sizeof(float),
                    cudaMemcpyDeviceToHost, A->s_cmp);
    cudaStreamSynchronize(A->s_cmp);
}

#endif /* COLIBRI_ATTN_GPU_H */
