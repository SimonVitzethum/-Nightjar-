/* cache_test.c — the tiered expert cache ON the saturated-disk pipeline.
 *
 * The disk is maxed at 5.39 GB/s (O_DIRECT, QD8) and the pipeline already reaches 99.7%
 * of it. So the only remaining lever is reading fewer bytes: don't re-fetch an expert we
 * still have. Three tiers, all costs measured on this box:
 *
 *      VRAM hit : free            — kernels run on the resident copy
 *      RAM  hit : 11.8 GB/s PCIe  — 2.2x cheaper than the disk
 *      miss     : 5.39 GB/s NVMe  — plus the PCIe hop
 *
 * (A pinned staging buffer between RAM and VRAM measured 5.36 GB/s, SLOWER than a plain
 *  pageable copy: the extra host memcpy costs more than the faster DMA saves. So the RAM
 *  tier is plain malloc, copied straight to the device.)
 *
 * The reader pool checks the cache before touching the disk, so a hit costs no queue slot
 * and no I/O at all. Entries in flight are refcounted and cannot be evicted underneath the
 * GPU.
 *
 * RAM_RESERVE is honoured strictly — the cache never eats the last 2 GB, so the box stays
 * responsive no matter how big the model is.
 *
 * ROUTING SKEW IS A PARAMETER, NOT A MEASUREMENT. Whether a cache helps depends entirely
 * on how concentrated MoE routing is, and that needs a real forward pass (real hidden
 * states -> real router logits), which this port does not have yet. So we sweep it:
 * s=0 is uniform (the honest worst case, where no cache can help), higher s is more
 * skewed. Real routing is known to be skewed; by how much on THIS model, this test
 * cannot tell you.
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

#define NSLOT   24
#define NREADER 8
#define NDEV     4
#define ALIGN   4096
#define RAM_RESERVE  (2ll*1024*1024*1024)
#define VRAM_RESERVE (1200ll*1024*1024)

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
#define CU(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#c,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

static long long mem_available(void){
    FILE *f = fopen("/proc/meminfo","r");
    char line[256]; long long kb = 0;
    while(f && fgets(line,sizeof(line),f))
        if(sscanf(line,"MemAvailable: %lld kB",&kb)==1) break;
    if(f) fclose(f);
    return kb*1024;
}

typedef struct { const gguf_tensor *g,*u,*d; } EL;

enum { TIER_VRAM=2, TIER_RAM=1 };
typedef struct {
    int      key;
    int      tier;
    uint8_t *p;
    uint32_t heat;
    int      inflight;     /* the GPU is reading this: never evict it */
} Entry;

typedef struct {
    Entry *e; int n_vram, n_ram, n;
    int *map, mask;
    int64_t slot_bytes;
    uint64_t hit_vram, hit_ram, miss;

    /* PERSISTENT heat, one counter per (layer, expert), independent of residency.
     *
     * Tracking heat only on cached entries does not work: an entry evicted from RAM loses
     * its history, so a moderately-hot expert can never accumulate enough to earn a VRAM
     * slot, and VRAM ends up holding whichever experts happened to be seen FIRST. Measured:
     * VRAM served 50% of hits while the RAM tier served 0.2% — it was pure churn.
     *
     * 76*256 counters is 76 KB. Colibri already collects exactly this (.coli_usage), which
     * is what makes the whole idea practical: the router tells you which experts are hot
     * long before the cache does. */
    uint32_t *heat;      /* [n_layer * n_exp] */
    int n_exp;
} Cache;
static inline uint32_t *heat_of(Cache *C, int layer, int eid){
    return &C->heat[(size_t)layer*C->n_exp + eid];
}

static inline uint32_t kh(int key){ return (uint32_t)key * 2654435761u; }
static int cache_find(Cache *C, int key){
    uint32_t i = kh(key) & C->mask;
    while(C->map[i] != -1){
        if(C->e[C->map[i]].key == key) return C->map[i];
        i = (i+1) & C->mask;
    }
    return -1;
}
static void map_rebuild(Cache *C){
    for(int i=0;i<=C->mask;i++) C->map[i] = -1;
    for(int s=0;s<C->n;s++) if(C->e[s].key >= 0){
        uint32_t i = kh(C->e[s].key) & C->mask;
        while(C->map[i] != -1) i = (i+1) & C->mask;
        C->map[i] = s;
    }
}
/* coldest slot in [lo,hi) that the GPU is not currently reading */
static int victim(Cache *C, int lo, int hi){
    int best = -1; uint32_t bh = 0xFFFFFFFFu;
    for(int i=lo;i<hi;i++){
        if(C->e[i].inflight) continue;
        if(C->e[i].key < 0) return i;
        uint32_t h = C->heat[C->e[i].key];             /* persistent, not per-entry */
        if(h < bh){ bh = h; best = i; }
    }
    return best;
}
/* The coldest resident VRAM expert, and how cold it is. */
static uint32_t vram_floor(Cache *C, int *slot){
    uint32_t bh = 0xFFFFFFFFu; int best = -1;
    for(int i=0;i<C->n_vram;i++){
        if(C->e[i].inflight) continue;
        if(C->e[i].key < 0){ *slot = i; return 0; }
        uint32_t h = C->heat[C->e[i].key];
        if(h < bh){ bh = h; best = i; }
    }
    *slot = best;
    return best < 0 ? 0xFFFFFFFFu : bh;
}

typedef struct {
    int layer, eid;
    int slot;          /* cache slot if it was a hit, else -1 */
    int64_t nb[3];
    int64_t used;      /* nb[0]+nb[1]+nb[2]: copy this, not the whole padded slot */
} Req;

typedef struct {
    gguf_model *m; EL *L; Req *q; int n;
    Cache *C; int n_exp;
    int fd[GGUF_MAX_SHARD];
    uint8_t *ring[NSLOT];
    int64_t off_r[3], slot_bytes;
    int64_t (*coff)[3]; int64_t *lsz;
    volatile int next, consumed;
    volatile int ready[NSLOT];
    pthread_mutex_t mx;       /* ring + schedule */
    pthread_mutex_t cmx;      /* cache */
    pthread_cond_t  cv;
    double t_io[NREADER], bytes[NREADER];
    int idgen;
} Pipe;

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
        int key = r->layer * P->n_exp + r->eid;

        /* a cached expert costs no I/O at all — do not even take a ring slot for it */
        pthread_mutex_lock(&P->cmx);
        P->C->heat[key]++;                              /* count EVERY access, cached or not */
        int s = cache_find(P->C, key);
        if(s >= 0) P->C->e[s].inflight++;
        pthread_mutex_unlock(&P->cmx);
        r->slot = s;

        if(s < 0){
            const EL *e = &P->L[r->layer];
            const gguf_tensor *T[3] = { e->g, e->u, e->d };
            uint8_t *b = P->ring[i % NSLOT];
            double a = now();
            for(int q=0;q<3;q++){
                int fdx; uint64_t off; int64_t nb, rows, cols;
                gguf_expert_slice(P->m, T[q], r->eid, &fdx, &off, &nb, &rows, &cols);
                uint64_t base = off & ~(uint64_t)(ALIGN-1);
                int64_t pad = (int64_t)(off - base);
                int64_t len = (pad + nb + ALIGN-1) & ~(int64_t)(ALIGN-1);
                if(pread(P->fd[T[q]->shard], b + P->off_r[q], len, base) < pad + nb){
                    perror("pread"); exit(1); }
                /* slide the useful bytes to the front of the region so a cached copy needs
                 * no per-entry deltas later */
                if(pad) memmove(b + P->off_r[q], b + P->off_r[q] + pad, nb);
                r->nb[q] = nb;
                P->bytes[id] += nb;
            }
            P->t_io[id] += now() - a;
            r->used = P->off_r[2] + r->nb[2];

            /* Admit to the RAM tier HERE, in the reader, not on the main thread. The bytes
             * are already in this thread's hands and this thread is I/O-bound with CPU to
             * spare; doing the 12 MB memcpy on the main thread instead put ~1.6 ms of pure
             * copy on the critical path of every single miss and cost more than the cache
             * saved. VRAM admission stays on the main thread — it is an async D2D from a
             * buffer only that thread owns. */
            r->used = P->lsz[r->layer];                     /* compact size */
            if(r->used > P->C->slot_bytes) goto publish;   /* fat layer: stream it, never cache */

            /* ADMISSION FILTER. Caching every miss means the cache fills with one-shot
             * experts, which evict the ones that would have come back: measured, the RAM
             * tier served 0.2% of hits — it was churn, not cache. An expert must EARN a
             * slot by being routed to at least twice (the persistent histogram remembers,
             * even across evictions). This is the same idea as TinyLFU's admission window. */
            if(P->C->heat[key] < 2) goto publish;
            pthread_mutex_lock(&P->cmx);
            int v = victim(P->C, P->C->n_vram, P->C->n);
            if(v >= 0 && P->C->e[v].key != key){
                int evicted = (P->C->e[v].key >= 0);
                P->C->e[v].key = key; P->C->e[v].inflight = 1;
                pthread_mutex_unlock(&P->cmx);

                for(int q=0;q<3;q++)                        /* strip the ring's padding */
                    memcpy(P->C->e[v].p + P->coff[r->layer][q], b + P->off_r[q], r->nb[q]);

                pthread_mutex_lock(&P->cmx);
                P->C->e[v].inflight = 0;
                if(evicted) map_rebuild(P->C);
                else {
                    uint32_t h = kh(key) & P->C->mask;
                    while(P->C->map[h] != -1) h = (h+1) & P->C->mask;
                    P->C->map[h] = v;
                }
            }
            pthread_mutex_unlock(&P->cmx);
        }
publish:
        pthread_mutex_lock(&P->mx);
        P->ready[i % NSLOT] = i;
        pthread_cond_broadcast(&P->cv);
        pthread_mutex_unlock(&P->mx);
    }
    return NULL;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf> [n_tokens] [zipf_s]\n", argv[0]); return 2; }
    const int n_tok = argc > 2 ? atoi(argv[2]) : 8;
    const double zipf = argc > 3 ? atof(argv[3]) : 1.0;

    struct cudaDeviceProp pr; CU(cudaGetDeviceProperties(&pr, 0));
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

    /* The ring buffers must fit the FATTEST layer, so they use the max. The CACHE must not:
     * "Dynamic" means 71 of GLM's 76 MoE layers need 12.12 MB per expert while a single
     * layer needs 16.34, and sizing every cache slot for that one outlier throws away 25%
     * of the cache — more than every dense-reduction trick put together. So the cache slot
     * is the COMMON size, and the handful of experts that do not fit simply stay uncached
     * (they are 6.6% of all experts, and we get 1.33x more slots in exchange). */
    int64_t region[3] = {0,0,0};
    for(int i=0;i<n_moe;i++){
        const gguf_tensor *T[3]={L[i].g,L[i].u,L[i].d};
        for(int t=0;t<3;t++){
            int64_t b = kq_row_bytes(T[t]->type, T[t]->ne[0]*T[t]->ne[1]);
            int64_t a = (b + 2*ALIGN + ALIGN-1) & ~(int64_t)(ALIGN-1);
            if(a > region[t]) region[t] = a;
        }
    }
    const int64_t off_r[3] = { 0, region[0], region[0]+region[1] };
    const int64_t ring_bytes = region[0]+region[1]+region[2];      /* worst case, for I/O */

    /* per-layer footprint, and the most common one */
    int64_t *lsz = calloc(n_moe, sizeof(int64_t));
    for(int i=0;i<n_moe;i++){
        const gguf_tensor *T[3]={L[i].g,L[i].u,L[i].d};
        lsz[i] = off_r[2] + kq_row_bytes(T[2]->type, T[2]->ne[0]*T[2]->ne[1]);
        (void)T;
    }
    /* The cache stores the three tensors BACK TO BACK, with none of the 4K padding the
     * O_DIRECT ring needs. Copying the ring's layout verbatim would drag those gaps into
     * every cached expert. Compact offsets are per layer, and derived from the tensor sizes. */
    int64_t (*coff)[3] = calloc(n_moe, sizeof(*coff));
    for(int i=0;i<n_moe;i++){
        const gguf_tensor *T[3]={L[i].g,L[i].u,L[i].d};
        int64_t o = 0;
        for(int t=0;t<3;t++){
            coff[i][t] = o;
            o += kq_row_bytes(T[t]->type, T[t]->ne[0]*T[t]->ne[1]);
        }
        lsz[i] = o;                                   /* compact footprint of this layer */
    }
    int64_t slot_bytes = 0; int best_n = 0;
    for(int i=0;i<n_moe;i++){
        int n = 0;
        for(int j=0;j<n_moe;j++) if(lsz[j] == lsz[i]) n++;
        if(n > best_n){ best_n = n; slot_bytes = lsz[i]; }
    }
    int n_fit = 0;
    for(int i=0;i<n_moe;i++) if(lsz[i] <= slot_bytes) n_fit++;

    size_t vfree=0, vtot=0; CU(cudaMemGetInfo(&vfree,&vtot));
    long long ram_avail = mem_available();
    long long pinned = (long long)NSLOT * ring_bytes;

    /* The dense part is NOT optional and NOT streamable: attention, shared experts, the
     * dense FFN layers, embeddings and the indexer are touched on EVERY token. For
     * GLM-5.2-UD-Q2_K_XL that is 15.44 GB, of which attention alone is 10.38 GB. It has to
     * live somewhere, and wherever it lives it is not expert cache. Ignoring it (as the
     * first version of this benchmark did) inflates the cache by more than 2x and every
     * tok/s number with it. DENSE_GB models that reservation. */
    const char *dg = getenv("DENSE_GB");
    long long dense = (long long)((dg ? atof(dg) : 15.44) * 1e9);

    /* WHERE the dense part lives decides the whole architecture, so it is a knob:
     *
     *   DENSE=vram — fill VRAM with dense first. Sounds right (the GPU computes it) but is
     *     a trap: 15.44 GB does not fit in 8, so the remainder sits in RAM and has to cross
     *     PCIe on EVERY token — 8.7 GB at 11.8 GB/s = 0.74 s/token, which dominates
     *     everything. And it leaves zero VRAM for the expert cache, so every expert hit
     *     costs a PCIe transfer too.
     *
     *   DENSE=ram — keep it all in RAM and contract it ON THE CPU, which already has it and
     *     is otherwise idle. Measured: 62.5 GB/s read bandwidth over 16 threads, so 15.44 GB
     *     costs ~0.25 s/token — and that overlaps with the disk and the GPU instead of
     *     serializing behind them. VRAM then belongs entirely to the expert cache, whose
     *     hits become free. */
    const char *dp = getenv("DENSE");
    const int dense_in_ram = !dp || strcmp(dp, "vram") != 0;

    /* The KV cache lives in VRAM, because attention has to run on the GPU (on the CPU it
     * would be ~1.6 s/token at 32K) and shipping the KV over PCIe every token is worse
     * still. So it competes directly with the expert cache: 91 KB/token/layer at fp16
     * across 79 layers, i.e. every 12.09 MB of context evicts one expert. */
    const char *kg = getenv("KV_GB");
    long long kv = (long long)((kg ? atof(kg) : 0.0) * 1e9);

    long long vram_budget = (long long)vfree - VRAM_RESERVE - kv;
    if(vram_budget < 0) vram_budget = 0;
    long long dense_v = dense_in_ram ? 0 : (dense < vram_budget ? dense : vram_budget);
    vram_budget -= dense_v;
    long long ram_budget  = ram_avail - RAM_RESERVE - pinned - (512ll<<20) - (dense - dense_v);
    if(ram_budget<0) ram_budget=0;
    if(vram_budget<0) vram_budget=0;

    Cache C; memset(&C,0,sizeof(C));
    C.slot_bytes = slot_bytes;
    C.n_vram = (int)(vram_budget/slot_bytes);
    C.n_ram  = (int)(ram_budget /slot_bytes);
    C.n = C.n_vram + C.n_ram;
    C.n_exp = n_exp;
    C.heat  = calloc((size_t)n_moe*n_exp, sizeof(uint32_t));
    C.e = calloc(C.n,sizeof(Entry));
    for(int i=0;i<C.n;i++){ C.e[i].key=-1; C.e[i].tier = (i<C.n_vram)?TIER_VRAM:TIER_RAM; }
    int cap=16; while(cap < C.n*2) cap<<=1;
    C.mask=cap-1; C.map=malloc((size_t)cap*sizeof(int));
    for(int i=0;i<cap;i++) C.map[i]=-1;
    for(int i=0;i<C.n_vram;i++) CU(cudaMalloc((void**)&C.e[i].p, slot_bytes));
    for(int i=C.n_vram;i<C.n;i++){
        C.e[i].p = malloc(slot_bytes);
        if(!C.e[i].p){ C.n = i; break; }
    }

    printf("GPU: %s | %d readers, O_DIRECT, %d ring slots\n", pr.name, NREADER, NSLOT);
    printf("cache (RAM reserve %.1f GB kept free, dense weights %.2f GB reserved)\n",
           RAM_RESERVE/1e9, dense/1e9);
    printf("  slot %.2f MB (ring %.2f MB); %d of %d layers fit the cache\n",
           slot_bytes/1e6, ring_bytes/1e6, n_fit, n_moe);
    printf("  %d experts in VRAM (%.1f GB) + %d in RAM (%.1f GB) = %d of %d  (%.1f %%)\n\n",
           C.n_vram, C.n_vram*slot_bytes/1e9, C.n-C.n_vram, (C.n-C.n_vram)*slot_bytes/1e9,
           C.n, n_moe*n_exp, 100.0*C.n/(n_moe*(double)n_exp));

    /* routing model — a parameter, not a measurement (see header) */
    int *perm = malloc((size_t)n_moe*n_exp*sizeof(int));
    double *cdf = malloc((size_t)n_exp*sizeof(double));
    { double s=0;
      for(int r=0;r<n_exp;r++){ s += 1.0/pow(r+1,zipf); cdf[r]=s; }
      for(int r=0;r<n_exp;r++) cdf[r]/=s;
      unsigned sd=12345;
      for(int l=0;l<n_moe;l++){
          int *p=perm+(size_t)l*n_exp;
          for(int i=0;i<n_exp;i++) p[i]=i;
          for(int i=n_exp-1;i>0;i--){ int j=rand_r(&sd)%(i+1); int t=p[i];p[i]=p[j];p[j]=t; }
      }
    }

    int n_req = n_tok*n_moe*topk;
    Req *q = calloc(n_req,sizeof(Req));
    { unsigned sd=777;
      for(int t=0,i=0;t<n_tok;t++) for(int l=0;l<n_moe;l++) for(int e=0;e<topk;e++,i++){
          double u=(double)rand_r(&sd)/RAND_MAX;
          int lo=0,hi=n_exp-1;
          while(lo<hi){ int mid=(lo+hi)/2; if(cdf[mid]<u) lo=mid+1; else hi=mid; }
          q[i].layer=l; q[i].eid=perm[(size_t)l*n_exp+lo]; q[i].slot=-1;
      }
    }

    Pipe P; memset(&P,0,sizeof(P));
    P.m=&m; P.L=L; P.q=q; P.n=n_req; P.C=&C; P.n_exp=n_exp;
    P.slot_bytes=slot_bytes; P.coff=coff; P.lsz=lsz;
    for(int t=0;t<3;t++) P.off_r[t]=off_r[t];
    for(int i=0;i<NSLOT;i++) P.ready[i]=-1;
    for(int s=0;s<m.n_shards;s++){
        P.fd[s]=open(m.shard[s].path, O_RDONLY|O_DIRECT);
        if(P.fd[s]<0){ perror("open O_DIRECT"); return 1; }
    }
    for(int i=0;i<NSLOT;i++) CU(cudaMallocHost((void**)&P.ring[i], ring_bytes));
    pthread_mutex_init(&P.mx,NULL); pthread_mutex_init(&P.cmx,NULL);
    pthread_cond_init(&P.cv,NULL);

    uint8_t *d_W[NDEV];
    for(int i=0;i<NDEV;i++) CU(cudaMalloc((void**)&d_W[i], ring_bytes));
    /* int8 activation buffers. Quantized ONCE per layer for gate/up (every routed expert
     * of a layer contracts against the same x), and once per expert for down (whose input
     * is that expert's own SwiGLU output). */
    signed char *d_xq, *d_gq;
    float *d_xd, *d_gd;
    CU(cudaMalloc((void**)&d_xq, (size_t)n_out));
    CU(cudaMalloc((void**)&d_gq, (size_t)n_out));
    CU(cudaMalloc((void**)&d_xd, (size_t)(n_out/32)*sizeof(float)));
    CU(cudaMalloc((void**)&d_gd, (size_t)(n_out/32)*sizeof(float)));

    float *d_x,*d_g,*d_u;
    CU(cudaMalloc((void**)&d_x,(size_t)n_out*sizeof(float)));
    CU(cudaMalloc((void**)&d_g,(size_t)n_out*sizeof(float)));
    CU(cudaMalloc((void**)&d_u,(size_t)n_out*sizeof(float)));
    float *x = malloc((size_t)n_out*sizeof(float));
    for(int i=0;i<d_model;i++) x[i]=(float)(0.05*sin(i*0.013));
    CU(cudaMemcpy(d_x,x,(size_t)d_model*sizeof(float),cudaMemcpyHostToDevice));

    double bytes_disk=0;
    double t0=now();
    pthread_t th[NREADER];
    for(int i=0;i<NREADER;i++) pthread_create(&th[i],NULL,reader,&P);

    int inflight[NDEV]; for(int i=0;i<NDEV;i++) inflight[i]=-1;
    int last_layer = -1;

    for(int i=0;i<n_req;i++){
        pthread_mutex_lock(&P.mx);
        while(P.ready[i % NSLOT] != i) pthread_cond_wait(&P.cv,&P.mx);
        pthread_mutex_unlock(&P.mx);

        Req *r = &q[i];
        const EL *e = &L[r->layer];
        uint8_t *dev;
        int64_t dof[3];        /* where g/u/d sit in whatever buffer we end up using */

        /* retire the request that used this device buffer NDEV ago */
        if(i >= NDEV){
            CU(cudaStreamSynchronize(0));
            int old = inflight[i % NDEV];
            if(old >= 0){
                pthread_mutex_lock(&P.cmx);
                if(C.e[old].inflight > 0) C.e[old].inflight--;
                pthread_mutex_unlock(&P.cmx);
            }
        }
        inflight[i % NDEV] = -1;

        if(r->slot >= 0){
            Entry *en = &C.e[r->slot];
            if(en->tier == TIER_VRAM){
                C.hit_vram++;
                dev = en->p;                                   /* free */
                memcpy(dof, coff[r->layer], sizeof(dof));
                inflight[i % NDEV] = r->slot;                  /* hold it until the GPU is done */
            } else {
                C.hit_ram++;
                CU(cudaMemcpy(d_W[i % NDEV], en->p, lsz[r->layer], cudaMemcpyHostToDevice));
                dev = d_W[i % NDEV];
                memcpy(dof, coff[r->layer], sizeof(dof));

                /* PROMOTION. The bytes are on the device anyway, so moving a hot RAM entry
                 * into VRAM costs one device-to-device copy and buys a free hit forever
                 * after. Only displace a VRAM entry that is genuinely colder. */
                pthread_mutex_lock(&P.cmx);
                int v; uint32_t floor = vram_floor(&C, &v);
                if(v >= 0 && floor < C.heat[en->key]){
                    C.e[v].key  = en->key;
                    CU(cudaMemcpyAsync(C.e[v].p, d_W[i % NDEV], r->used,
                                       cudaMemcpyDeviceToDevice, 0));
                    en->key = -1;                       /* the RAM slot is now free */
                    map_rebuild(&C);
                }
                if(en->inflight > 0) en->inflight--;           /* the bytes are on the GPU now */
                pthread_mutex_unlock(&P.cmx);
            }
        } else {
            C.miss++;
            uint8_t *hb = P.ring[i % NSLOT];
            bytes_disk += r->nb[0] + r->nb[1] + r->nb[2];   /* real bytes, not the padded slot */
            for(int q=0;q<3;q++)
                CU(cudaMemcpyAsync(d_W[i % NDEV] + off_r[q], hb + off_r[q], r->nb[q],
                                   cudaMemcpyHostToDevice, 0));
            dev = d_W[i % NDEV];
            memcpy(dof, off_r, sizeof(off_r));             /* ring layout */
            /* The RAM tier was already filled by the reader thread. VRAM is only seeded
             * while it still has free slots; after that, experts must earn their way up
             * from RAM by being hit again (the promotion path above). Admitting a stranger
             * straight into VRAM would evict a proven-hot resident. */
            int key = r->layer*n_exp + r->eid;
            pthread_mutex_lock(&P.cmx);
            int v = -1;
            if(lsz[r->layer] <= slot_bytes){
                uint32_t floor = vram_floor(&C, &v);
                /* free slot, or this expert is genuinely hotter than the coldest resident.
                 * Without the second condition VRAM is first-come-first-served forever. */
                if(v >= 0 && !(C.e[v].key < 0 || C.heat[key] > floor)) v = -1;
            }
            if(v >= 0){
                int evicted = (C.e[v].key >= 0);
                C.e[v].key = key;
                for(int q=0;q<3;q++)                       /* compact it on the way in */
                    CU(cudaMemcpyAsync(C.e[v].p + coff[r->layer][q],
                                       d_W[i % NDEV] + off_r[q], r->nb[q],
                                       cudaMemcpyDeviceToDevice, 0));
                if(evicted) map_rebuild(&C);
                else {
                    uint32_t h = kh(key) & C.mask;
                    while(C.map[h] != -1) h = (h+1) & C.mask;
                    C.map[h] = v;
                }
            }
            pthread_mutex_unlock(&P.cmx);
        }

        /* gate/up share this layer's x, so quantize it once and reuse it for all 8 experts */
        if(r->layer != last_layer){
            CU(kq_cu_quant_act(d_xq, d_xd, d_x, 1, d_model, 0) ? cudaSuccess : cudaErrorUnknown);
            last_layer = r->layer;
        }
        if(!kq_cu_gemm_i8(d_g, d_xq, d_xd, dev+dof[0], e->g->type, 1, d_model, (int)d_ff, 0))
            kq_cu_gemv(d_g, d_x, dev+dof[0], e->g->type, d_model, (int)d_ff, kq_typesize(e->g->type), 0);
        if(!kq_cu_gemm_i8(d_u, d_xq, d_xd, dev+dof[1], e->u->type, 1, d_model, (int)d_ff, 0))
            kq_cu_gemv(d_u, d_x, dev+dof[1], e->u->type, d_model, (int)d_ff, kq_typesize(e->u->type), 0);

        /* down contracts the SwiGLU output, which is this expert's alone */
        kq_cu_quant_act(d_gq, d_gd, d_g, 1, (int)d_ff, 0);
        if(!kq_cu_gemm_i8(d_x, d_gq, d_gd, dev+dof[2], e->d->type, 1, (int)d_ff, d_model, 0))
            kq_cu_gemv(d_x, d_g, dev+dof[2], e->d->type, (int)d_ff, d_model, kq_typesize(e->d->type), 0);

        pthread_mutex_lock(&P.mx);
        P.consumed = i+1;
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.mx);
    }
    CU(cudaDeviceSynchronize());
    for(int i=0;i<NREADER;i++) pthread_join(th[i],NULL);
    double wall = now()-t0;

    uint64_t tot = C.hit_vram + C.hit_ram + C.miss;
    double io_sum=0, io_b=0;
    for(int i=0;i<NREADER;i++){ io_sum += P.t_io[i]; io_b += P.bytes[i]; }

    printf("results (%d tokens, zipf s=%.2f)\n", n_tok, zipf);
    printf("  VRAM hits : %5.1f %%  (free)\n",      100.0*C.hit_vram/tot);
    printf("  RAM  hits : %5.1f %%  (PCIe only)\n", 100.0*C.hit_ram/tot);
    printf("  misses    : %5.1f %%  (NVMe)\n",      100.0*C.miss/tot);
    printf("  disk read : %.2f GB/token   (uncached: 7.44)\n", bytes_disk/1e9/n_tok);
    if(io_sum > 0)
        printf("  NVMe agg  : %.2f GB/s\n", io_b/1e9/(io_sum/NREADER));
    printf("  wall      : %.2f s  ->  %.3f tok/s\n", wall, n_tok/wall);
    printf("\n  pipeline without cache: 0.722 tok/s (disk-bound at 5.39 GB/s)\n");
    return 0;
}
