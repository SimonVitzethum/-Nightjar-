#ifndef COLIBRI_MOE_GPU_H
#define COLIBRI_MOE_GPU_H
/* moe_gpu.h — the routed experts, on the GPU, and an NVMe that never idles.
 *
 * WHERE THE TIME ACTUALLY GOES (per MoE layer, measured on this box)
 *
 *      disk, 8 experts        17.9 ms
 *      expert FFN on the CPU  60.0 ms   <- dwarfs everything
 *      expert FFN on the GPU   0.4 ms
 *      attention + dense, CPU  3.3 ms
 *
 * That 60 ms is why the drive was observed at ~870 MiB/s: it spends 77% of the wall clock
 * waiting for a matmul. Move the expert FFN to the GPU and the same drive is busy 83% of
 * the time without changing a single thing about the I/O path.
 *
 * THE LAST 17%
 *
 * Even then the disk stops during the (short) compute, because layer L+1's experts are not
 * known until layer L has produced its output — the router depends on it. That dependency
 * cannot be broken. But the drive does not have to sit still: a background prefetcher fills
 * every idle moment by pulling the HOTTEST expert that is not yet cached. It never blocks
 * the forward pass, it never competes for a slot the pass needs, and everything it fetches
 * is a miss that will not happen later. Idle disk bandwidth is the one resource here that
 * is free, and spending it on the cache is strictly better than spending it on nothing.
 */
#include <cuda_runtime.h>

#include "glm_engine.h"
#include "estream.h"
#include "kquant_cuda.h"

typedef struct {
    int    d_model, d_ff, topk;
    /* device buffers, allocated once */
    signed char *xq, *hq;
    float *x, *xd, *hd, *g, *u, *acc, *out;
    uint8_t *W;              /* staging for one expert's g|u|d */
    int64_t  woff[3], wcap;
    cudaStream_t s;
} MoeGpu;

static int moe_gpu_init(MoeGpu *M, int d_model, int d_ff, int topk, int64_t wcap,
                        const int64_t *woff){
    memset(M, 0, sizeof(*M));
    M->d_model = d_model; M->d_ff = d_ff; M->topk = topk; M->wcap = wcap;
    for(int i=0;i<3;i++) M->woff[i] = woff[i];
    const int n = d_model > d_ff ? d_model : d_ff;

    if(cudaStreamCreate(&M->s) != cudaSuccess) return 0;
    #define A(p, bytes) if(cudaMalloc((void**)&(p), (bytes)) != cudaSuccess) return 0
    A(M->x,   (size_t)n*sizeof(float));
    A(M->xq,  (size_t)n);
    A(M->xd,  (size_t)(n/32)*sizeof(float));
    A(M->hq,  (size_t)n);
    A(M->hd,  (size_t)(n/32)*sizeof(float));
    A(M->g,   (size_t)n*sizeof(float));
    A(M->u,   (size_t)n*sizeof(float));
    A(M->acc, (size_t)d_model*sizeof(float));
    A(M->out, (size_t)d_model*sizeof(float));
    A(M->W,   (size_t)wcap);
    #undef A
    return 1;
}

/* fp32 fallback for the few layers whose experts are not i-quants. `M->x` still holds the
 * layer's activation; `M->g` holds the SwiGLU output for the down projection. */
/* kq_cu_gemm returns 0 for two very different reasons — it declined to launch (a row length
 * it cannot block, a type it cannot decode), or the launch failed. Collapsing them into one
 * message sent me hunting for a missing Q5_K kernel that was there all along. Say which. */
static void moe_gpu_fp32(MoeGpu *M, float *y, const uint8_t *W, int ty, int I, int O){
    if(!kq_cu_gemm(y, M->x, W, ty, 1, I, O, kq_typesize(ty), M->s)){
        fprintf(stderr, "moe_gpu: gemm(%s, I=%d, O=%d) failed: %s\n",
                kq_name(ty), I, O, kq_cu_last_error()); exit(1); }
}
static void moe_gpu_fp32_h(MoeGpu *M, float *y, const uint8_t *W, int ty, int I, int O){
    if(!kq_cu_gemm(y, M->g, W, ty, 1, I, O, kq_typesize(ty), M->s)){
        fprintf(stderr, "moe_gpu: gemm(%s, I=%d, O=%d) failed: %s\n",
                kq_name(ty), I, O, kq_cu_last_error()); exit(1); }
}

/* One MoE layer's routed experts, entirely on the device.
 *
 * `eg/eu/ed` are HOST pointers into the streamer's buffers (staging or cache). They are
 * uploaded as-is: quantized bytes, never dequantized on the host, never expanded. The
 * activation is quantized ONCE for the layer — all 8 experts contract against the same x,
 * so re-quantizing per expert would repeat the work 8 times for nothing. */
/* Split into launch + wait so the host can run the CPU's share of the experts CONCURRENTLY.
 *
 * moe_gpu_launch issues every upload and kernel on the stream and RETURNS immediately — the
 * GPU then runs on its own while the caller does other work (the CPU experts). The acc copy
 * back is queued here too, async, so wait only has to synchronize. Nothing blocks between the
 * two, which is the entire point: max(GPU, CPU), not GPU then CPU. */
static void moe_gpu_launch(MoeGpu *M, float *acc_host, const float *xn,
                           const void **eg, const void **eu, const void **ed,
                           const int *tg, const int *tu, const int *td,
                           const int64_t *ng, const int64_t *nu, const int64_t *nd,
                           const float *w, int k){
    const int D = M->d_model, F = M->d_ff;

    cudaMemcpyAsync(M->x, xn, (size_t)D*sizeof(float), cudaMemcpyHostToDevice, M->s);
    kq_cu_quant_act(M->xq, M->xd, M->x, 1, D, M->s);      /* once per layer, not per expert */
    kq_cu_zero(M->acc, D, M->s);

    for(int i=0;i<k;i++){
        /* an error here is recorded, not raised — and the next cudaGetLastError() caller (the
         * gemm) reports it as ITS failure. Check the upload where the upload happens. */
        cudaError_t ce = cudaSuccess;
        cudaMemcpyAsync(M->W + M->woff[0], eg[i], ng[i], cudaMemcpyHostToDevice, M->s);
        if((ce=cudaGetLastError()) != cudaSuccess) goto upload_bad;
        cudaMemcpyAsync(M->W + M->woff[1], eu[i], nu[i], cudaMemcpyHostToDevice, M->s);
        if((ce=cudaGetLastError()) != cudaSuccess) goto upload_bad;
        cudaMemcpyAsync(M->W + M->woff[2], ed[i], nd[i], cudaMemcpyHostToDevice, M->s);
        if((ce=cudaGetLastError()) != cudaSuccess) goto upload_bad;
        if(0){ upload_bad:
            fprintf(stderr, "moe_gpu: expert %d/%d upload failed: %s\n"
                    "  region caps  g=%lld u=%lld d=%lld  (wcap %lld)\n"
                    "  this expert  g=%lld u=%lld d=%lld\n",
                    i, k, cudaGetErrorString(ce),
                    (long long)(M->woff[1]-M->woff[0]), (long long)(M->woff[2]-M->woff[1]),
                    (long long)(M->wcap-M->woff[2]), (long long)M->wcap,
                    (long long)ng[i], (long long)nu[i], (long long)nd[i]);
            exit(1);
        }

        /* The int8/dp4a path only knows IQ2_XS and IQ3_XXS. That covers 71 of GLM's 76 MoE
         * layers — "Dynamic" bumps a handful to Q2_K / Q3_K / IQ4_XS, and for those the int8
         * kernel DECLINES. It returns 0 and writes nothing, which is not a crash: the output
         * buffer keeps whatever was there, the residual goes to NaN, the router picks expert
         * -1, and the failure finally surfaces as a bad pread in a different file. So the
         * fallback is mandatory, and a refusal that is not handled must be fatal. */
        #define MG(dst, xq, xd, w, ty, I_, O_) do{                                      \
            if(!kq_cu_gemm_i8((dst),(xq),(xd),(w),(ty),1,(I_),(O_),M->s) &&             \
               !kq_cu_gemm((dst), NULL, (w), (ty), 1, (I_), (O_), kq_typesize(ty), M->s)) { \
                fprintf(stderr, "moe_gpu: no kernel for %s\n", kq_name(ty)); exit(1); } \
        }while(0)

        if(!kq_cu_gemm_i8(M->g, M->xq, M->xd, M->W + M->woff[0], tg[i], 1, D, F, M->s))
            moe_gpu_fp32(M, M->g, M->W + M->woff[0], tg[i], D, F);
        if(!kq_cu_gemm_i8(M->u, M->xq, M->xd, M->W + M->woff[1], tu[i], 1, D, F, M->s))
            moe_gpu_fp32(M, M->u, M->W + M->woff[1], tu[i], D, F);
        kq_cu_swiglu(M->g, M->u, F, M->s);

        /* down contracts the SwiGLU output, which belongs to this expert alone */
        kq_cu_quant_act(M->hq, M->hd, M->g, 1, F, M->s);
        if(!kq_cu_gemm_i8(M->out, M->hq, M->hd, M->W + M->woff[2], td[i], 1, F, D, M->s))
            moe_gpu_fp32_h(M, M->out, M->W + M->woff[2], td[i], F, D);

        kq_cu_axpy(M->acc, M->out, w[i], D, M->s);
    }
    cudaMemcpyAsync(acc_host, M->acc, (size_t)D*sizeof(float), cudaMemcpyDeviceToHost, M->s);
}

/* Block until the launched work (and its acc copy-back) has landed. */
static void moe_gpu_wait(MoeGpu *M){ cudaStreamSynchronize(M->s); }

/* Backward-compatible synchronous form: launch then immediately wait. */
static void moe_gpu_layer(MoeGpu *M, float *acc_host, const float *xn,
                          const void **eg, const void **eu, const void **ed,
                          const int *tg, const int *tu, const int *td,
                          const int64_t *ng, const int64_t *nu, const int64_t *nd,
                          const float *w, int k){
    moe_gpu_launch(M, acc_host, xn, eg, eu, ed, tg, tu, td, ng, nu, nd, w, k);
    moe_gpu_wait(M);
}

/* ---------------- the idle-disk prefetcher ----------------
 *
 * Runs on its own thread and keeps the NVMe busy whenever the forward pass is not asking it
 * for anything. It walks the heat histogram, finds the hottest expert that is NOT cached,
 * and pulls it in. Everything it manages to fetch is a miss the forward pass will never
 * pay for.
 *
 * It must never get in the way, so it yields on two conditions: it does no work at all
 * while a fetch job is in flight, and it takes the cache lock only to look and to publish,
 * never to copy. */
typedef struct {
    EStream *S;
    volatile int stop;
    pthread_t th;
    uint8_t *buf;
    uint64_t fetched;
} Prefetch;

static void *prefetch_worker(void *ud){
    Prefetch *P = (Prefetch*)ud;
    EStream *S = P->S;

    while(!P->stop){
        /* stay out of the way: the forward pass owns the drive when it wants it */
        pthread_mutex_lock(&S->mx);
        int busy = (S->job_next < S->job_k);
        pthread_mutex_unlock(&S->mx);
        if(busy){ usleep(200); continue; }

        /* hottest expert that is not resident */
        int best = -1; uint32_t bh = 1;              /* heat >= 2 to be worth a slot */
        pthread_mutex_lock(&S->mx);
        for(int l = 0; l < S->n_layer; l++){
            if(!S->L[l].g || S->lsz[l] > S->slot_bytes) continue;
            for(int e = 0; e < S->n_exp; e++){
                int key = l*S->n_exp + e;
                if(S->heat[key] <= bh) continue;
                if(es_find(S, key) >= 0) continue;   /* already cached */
                bh = S->heat[key]; best = key;
            }
        }
        int v = -1;
        if(best >= 0){
            v = es_victim(S);
            /* only displace something COLDER than what we are bringing in */
            if(v >= 0 && S->e[v].key >= 0 && S->heat[S->e[v].key] >= bh) v = -1;
            if(v >= 0){
                if(S->e[v].key >= 0){ S->e[v].key = -1; es_reindex(S); }
                S->e[v].key = -2;                    /* reserved */
            }
        }
        pthread_mutex_unlock(&S->mx);

        if(v < 0){ usleep(2000); continue; }

        const int l = best / S->n_exp, eid = best % S->n_exp;
        const gguf_tensor *T[3] = { S->L[l].g, S->L[l].u, S->L[l].d };
        int64_t nb[3], dl[3];
        int ok = 1;
        for(int q=0;q<3 && ok;q++){
            nb[q] = es_read(S, T[q], eid, P->buf + S->roff[q], &dl[q]);
            if(nb[q] < 0) ok = 0;
        }

        pthread_mutex_lock(&S->mx);
        if(ok){
            for(int q=0;q<3;q++)
                memcpy(S->e[v].p + S->coff[l][q], P->buf + S->roff[q] + dl[q], nb[q]);
            S->e[v].key = best;
            uint32_t h = es_kh(best) & S->mask;
            while(S->map[h] != -1) h = (h+1) & S->mask;
            S->map[h] = v;
            P->fetched++;
        } else {
            S->e[v].key = -1;
        }
        pthread_mutex_unlock(&S->mx);
    }
    return NULL;
}

static int prefetch_start(Prefetch *P, EStream *S){
    memset(P, 0, sizeof(*P));
    P->S = S;
    if(posix_memalign((void**)&P->buf, ES_ALIGN, S->ring_bytes)) return 0;
    return pthread_create(&P->th, NULL, prefetch_worker, P) == 0;
}
static void prefetch_stop(Prefetch *P){
    P->stop = 1;
    pthread_join(P->th, NULL);
}

#endif /* COLIBRI_MOE_GPU_H */
