/* pipe_test.c — saturate the disk, hide everything else behind it.
 *
 * Three measurements shaped this, all on the real 254 GB GLM-5.2-UD-Q2_K_XL:
 *
 *   1. QUEUE DEPTH. Blocking preads from one thread give 2.36 GB/s. The drive needs
 *      several requests in flight. With O_DIRECT it saturates at ~5.35 GB/s from QD4 up.
 *
 *   2. O_DIRECT. Buffered reads looked faster (up to 8 GB/s) but that was the page cache
 *      replaying offsets a benchmark had already touched. Against a 254 GB model with
 *      routing-driven access, the cache can never help — it only costs a copy and evicts
 *      whatever else the box needs. O_DIRECT is both honest and stable here.
 *      Cost: offsets and lengths must be 4 KB aligned, so we read the aligned superset
 *      of each expert slice and hand the GPU the interior.
 *
 *   3. THE HOST SLOT IS FREE ONCE PCIe HAS THE BYTES, not once the kernels finish. Two
 *      streams: copies on one, compute on the other, joined by events. A reader can refill
 *      a pinned slot as soon as its H2D completes (~0.5 ms), instead of waiting out the
 *      kernels (~1.1 ms). That is what keeps the drive from ever going idle.
 *
 * Result should be: wall == read time, everything else in its shadow.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include "../gguf.h"
#include "../kquant_cuda.h"

#define NSLOT   24      /* pinned host slots */
#define NREADER 8
#define NDEV     4      /* device weight buffers in flight */
#define ALIGN 4096

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
#define CU(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#c,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

typedef struct { const gguf_tensor *g,*u,*d; } EL;
typedef struct {
    int layer, eid;
    int64_t nb[3];        /* useful bytes of gate/up/down */
    int32_t delta[3];     /* where the useful bytes start inside the aligned read */
} Req;

typedef struct {
    gguf_model *m; EL *L; Req *q; int n;
    int fd[GGUF_MAX_SHARD];       /* O_DIRECT dups, one per shard */
    uint8_t *slot[NSLOT];
    int64_t region;               /* bytes reserved per tensor inside a slot */
    volatile int next, consumed;
    volatile int ready[NSLOT];
    pthread_mutex_t mx; pthread_cond_t cv;
    double t_io[NREADER], bytes[NREADER];
    int idgen;
} Pipe;

/* One expert slice -> one 4K-aligned pread. Returns useful bytes; *delta is the offset
 * of the real data inside the buffer. */
static int64_t read_aligned(Pipe *P, const gguf_tensor *T, int eid, uint8_t *dst,
                            int32_t *delta){
    int fd_unused; uint64_t off; int64_t nb, rows, cols;
    if(!gguf_expert_slice(P->m, T, eid, &fd_unused, &off, &nb, &rows, &cols)) return -1;
    uint64_t base = off & ~(uint64_t)(ALIGN-1);
    int64_t  pad  = (int64_t)(off - base);
    int64_t  len  = (pad + nb + ALIGN - 1) & ~(int64_t)(ALIGN-1);
    ssize_t r = pread(P->fd[T->shard], dst, len, base);
    if(r < pad + nb){ fprintf(stderr,"short read: %zd < %lld\n", r, (long long)(pad+nb)); return -1; }
    *delta = (int32_t)pad;
    return nb;
}

static void *reader(void *arg){
    Pipe *P = (Pipe*)arg;
    pthread_mutex_lock(&P->mx);
    int id = P->idgen++;
    pthread_mutex_unlock(&P->mx);

    for(;;){
        int i;
        pthread_mutex_lock(&P->mx);
        while(P->next < P->n && P->next - P->consumed >= NSLOT)
            pthread_cond_wait(&P->cv, &P->mx);
        if(P->next >= P->n){ pthread_mutex_unlock(&P->mx); break; }
        i = P->next++;
        pthread_mutex_unlock(&P->mx);

        Req *r = &P->q[i];
        uint8_t *b = P->slot[i % NSLOT];
        const EL *e = &P->L[r->layer];
        const gguf_tensor *T[3] = { e->g, e->u, e->d };

        double a = now();
        for(int t = 0; t < 3; t++){
            r->nb[t] = read_aligned(P, T[t], r->eid, b + t*P->region, &r->delta[t]);
            if(r->nb[t] < 0) exit(1);
            P->bytes[id] += r->nb[t];
        }
        P->t_io[id] += now() - a;

        pthread_mutex_lock(&P->mx);
        P->ready[i % NSLOT] = i;
        pthread_cond_broadcast(&P->cv);
        pthread_mutex_unlock(&P->mx);
    }
    return NULL;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <first-shard.gguf> [n_tokens]\n", argv[0]); return 2; }
    const int n_tok = argc > 2 ? atoi(argv[2]) : 2;

    struct cudaDeviceProp pr; CU(cudaGetDeviceProperties(&pr, 0));
    printf("GPU: %s | %d readers, %d slots, O_DIRECT\n", pr.name, NREADER, NSLOT);
    if(!kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl)){
        printf("table upload failed\n"); return 1; }

    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    const char *arch = gguf_str(&m, "general.architecture", "");
    char k[128];
    #define KV(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    int n_layer=KV("block_count"), topk=KV("expert_used_count");
    int n_exp=KV("expert_count"), d_model=KV("embedding_length");
    #undef KV

    EL *L = calloc(n_layer, sizeof(EL));
    int n_moe = 0;
    for(int i=0;i<n_layer;i++){
        char a[128],b[128],c[128];
        snprintf(a,sizeof(a),"blk.%d.ffn_gate_exps.weight",i);
        snprintf(b,sizeof(b),"blk.%d.ffn_up_exps.weight",i);
        snprintf(c,sizeof(c),"blk.%d.ffn_down_exps.weight",i);
        const gguf_tensor *g=gguf_find(&m,a),*u=gguf_find(&m,b),*d=gguf_find(&m,c);
        if(g&&u&&d){ L[n_moe].g=g; L[n_moe].u=u; L[n_moe].d=d; n_moe++; }
    }
    const int64_t d_ff = L[0].g->ne[1];
    const int64_t n_out = d_ff > d_model ? d_ff : d_model;

    int64_t big = 0;
    for(int i=0;i<n_moe;i++){
        int64_t s[3] = {
            kq_row_bytes(L[i].g->type, L[i].g->ne[0]*L[i].g->ne[1]),
            kq_row_bytes(L[i].u->type, L[i].u->ne[0]*L[i].u->ne[1]),
            kq_row_bytes(L[i].d->type, L[i].d->ne[0]*L[i].d->ne[1]) };
        for(int t=0;t<3;t++) if(s[t] > big) big = s[t];
    }
    const int64_t region = (big + 2*ALIGN + ALIGN-1) & ~(int64_t)(ALIGN-1);
    printf("%s: %d MoE layers, top-%d of %d | slot %.1f MB\n\n",
           arch, n_moe, topk, n_exp, 3.0*region/1e6);

    int n_req = n_tok * n_moe * topk;
    Req *q = calloc(n_req, sizeof(Req));
    for(int t=0,i=0;t<n_tok;t++) for(int l=0;l<n_moe;l++) for(int e=0;e<topk;e++,i++){
        q[i].layer=l;
        q[i].eid=(int)(((int64_t)t*7919 + (int64_t)l*104729 + e*31) % n_exp);
    }

    Pipe P; memset(&P,0,sizeof(P));
    P.m=&m; P.L=L; P.q=q; P.n=n_req; P.region=region;
    for(int i=0;i<NSLOT;i++) P.ready[i] = -1;
    for(int s=0;s<m.n_shards;s++){
        P.fd[s] = open(m.shard[s].path, O_RDONLY | O_DIRECT);
        if(P.fd[s] < 0){ perror("open O_DIRECT"); return 1; }
    }
    for(int i=0;i<NSLOT;i++) CU(cudaMallocHost((void**)&P.slot[i], (size_t)3*region));
    pthread_mutex_init(&P.mx,NULL); pthread_cond_init(&P.cv,NULL);

    /* copies and compute on separate streams, joined by events: a host slot is reusable
     * the moment PCIe has the bytes, without waiting for the kernels. */
    cudaStream_t s_cpy, s_cmp;
    CU(cudaStreamCreate(&s_cpy)); CU(cudaStreamCreate(&s_cmp));

    uint8_t *d_W[NDEV];
    cudaEvent_t ev_cpy[NSLOT], ev_cmp[NDEV];
    for(int i=0;i<NDEV;i++){
        CU(cudaMalloc((void**)&d_W[i], (size_t)3*region));
        CU(cudaEventCreateWithFlags(&ev_cmp[i], cudaEventDisableTiming));
    }
    for(int i=0;i<NSLOT;i++) CU(cudaEventCreateWithFlags(&ev_cpy[i], cudaEventDisableTiming));

    float *d_x,*d_g,*d_u;
    CU(cudaMalloc((void**)&d_x,(size_t)n_out*sizeof(float)));
    CU(cudaMalloc((void**)&d_g,(size_t)n_out*sizeof(float)));
    CU(cudaMalloc((void**)&d_u,(size_t)n_out*sizeof(float)));
    float *x = malloc((size_t)n_out*sizeof(float));
    for(int i=0;i<d_model;i++) x[i]=(float)(0.05*sin(i*0.013));
    CU(cudaMemcpy(d_x,x,(size_t)d_model*sizeof(float),cudaMemcpyHostToDevice));

    double bytes = 0, t_stall = 0;
    double t0 = now();
    pthread_t th[NREADER];
    for(int i=0;i<NREADER;i++) pthread_create(&th[i], NULL, reader, &P);

    for(int i=0;i<n_req;i++){
        double a = now();
        pthread_mutex_lock(&P.mx);
        while(P.ready[i % NSLOT] != i) pthread_cond_wait(&P.cv, &P.mx);
        pthread_mutex_unlock(&P.mx);
        t_stall += now() - a;          /* time the GPU spent waiting on the disk */

        Req *r = &q[i];
        uint8_t *hb = P.slot[i % NSLOT];
        uint8_t *db = d_W[i % NDEV];
        const EL *e = &L[r->layer];

        /* the device buffer we are about to overwrite must have finished its kernels */
        if(i >= NDEV) CU(cudaEventSynchronize(ev_cmp[i % NDEV]));

        for(int t=0;t<3;t++)
            CU(cudaMemcpyAsync(db + t*region, hb + t*region + r->delta[t],
                               r->nb[t], cudaMemcpyHostToDevice, s_cpy));
        CU(cudaEventRecord(ev_cpy[i % NSLOT], s_cpy));
        bytes += r->nb[0] + r->nb[1] + r->nb[2];

        CU(cudaStreamWaitEvent(s_cmp, ev_cpy[i % NSLOT], 0));
        kq_cu_gemv(d_g, d_x, db,            e->g->type, d_model, (int)d_ff, kq_typesize(e->g->type), s_cmp);
        kq_cu_gemv(d_u, d_x, db+region,     e->u->type, d_model, (int)d_ff, kq_typesize(e->u->type), s_cmp);
        kq_cu_gemv(d_x, d_g, db+2*region,   e->d->type, (int)d_ff, d_model, kq_typesize(e->d->type), s_cmp);
        CU(cudaEventRecord(ev_cmp[i % NDEV], s_cmp));

        /* host slot is free as soon as PCIe has it — NOT after the kernels */
        CU(cudaEventSynchronize(ev_cpy[i % NSLOT]));
        pthread_mutex_lock(&P.mx);
        P.consumed = i + 1;
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.mx);
    }
    CU(cudaDeviceSynchronize());
    for(int i=0;i<NREADER;i++) pthread_join(th[i], NULL);
    double wall = now() - t0;

    double io_sum=0, io_bytes=0;
    for(int i=0;i<NREADER;i++){ io_sum += P.t_io[i]; io_bytes += P.bytes[i]; }
    double agg = io_bytes/1e9 / (io_sum/NREADER);

    printf("results (%d tokens, %d expert fetches)\n", n_tok, n_req);
    printf("  bytes            : %.2f GB   (%.2f GB/token)\n", bytes/1e9, bytes/1e9/n_tok);
    printf("  NVMe aggregate   : %.2f GB/s (drive tops out at ~5.35 with O_DIRECT)\n", agg);
    printf("  GPU waiting on it: %.2f s of %.2f s wall  (%.0f %%)\n",
           t_stall, wall, 100.0*t_stall/wall);
    printf("  wall             : %.2f s  ->  %.3f tok/s\n\n", wall, n_tok/wall);
    printf("  disk-bound floor at %.2f GB/s : %.3f tok/s\n", agg, agg/(bytes/1e9/n_tok));
    printf("\n  history: 0.127 (CPU) -> 0.316 (GPU) -> 0.569 (QD8 buffered) -> %.3f\n",
           n_tok/wall);

    for(int i=0;i<NSLOT;i++) cudaFreeHost(P.slot[i]);
    for(int i=0;i<NDEV;i++) cudaFree(d_W[i]);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_u);
    free(x); free(q); free(L); gguf_close(&m);
    return 0;
}
