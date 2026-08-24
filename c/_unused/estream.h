#ifndef COLIBRI_ESTREAM_H
#define COLIBRI_ESTREAM_H
/* estream.h — the expert streamer: a layer's worth of experts, in flight at once.
 *
 * The naive fetch (one pread, then the matmul that consumes it, then the next pread) was
 * measured at under 1 GB/s on a drive that does 5.39. Two reasons, and they compound:
 *
 *   QUEUE DEPTH. One outstanding read means the NVMe is latency-bound. Measured on this
 *   drive with expert-sized random reads: 2.36 GB/s at QD1, 5.39 GB/s at QD8 (O_DIRECT).
 *   A 4x difference, entirely from how the requests are issued.
 *
 *   IDLE DISK. Interleaving each read with its matmul means the drive does nothing during
 *   the compute and the CPU does nothing during the read. Both sit at ~50% and the wall
 *   clock is their sum instead of their max.
 *
 * The router names all 8 experts of a layer at once, so both problems have the same fix:
 * issue all 8 fetches concurrently, then compute. That is this file.
 *
 * O_DIRECT, because a 254 GB model against 31 GB of RAM can never hit the page cache — it
 * only costs a copy and evicts whatever else the box needs. It forces 4K-aligned offsets,
 * so each slice is read as its aligned superset and the caller gets a pointer to the
 * interior.
 *
 * The heat histogram is persistent per (layer, expert) and independent of residency. Heat
 * tracked only on cached entries cannot work: an evicted entry forgets its history, so a
 * warm expert never accumulates enough to earn its slot back, and the cache ends up holding
 * whatever it happened to see first.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include "gguf.h"

#define ES_ALIGN   4096
#define ES_MAXK    16
#define ES_THREADS 8

typedef struct {
    const gguf_tensor *g, *u, *d;
} ESLayer;

typedef struct {
    int key;                 /* layer*n_exp + eid, or -1 */
    uint8_t *p;              /* compact g|u|d, no padding */
    int64_t off[3];
} ESEntry;

typedef struct {
    gguf_model *m;
    ESLayer   *L;
    int        n_layer, n_exp, topk;
    int        fd[GGUF_MAX_SHARD];

    /* per-layer geometry */
    int64_t  (*coff)[3];     /* compact offsets within a cached entry */
    int64_t   *lsz;          /* compact bytes of one expert of that layer */
    int64_t    ring_bytes;   /* padded, for the O_DIRECT staging buffers */
    int64_t    roff[3];      /* padded offsets in a staging buffer */

    /* cache */
    ESEntry   *e;
    int        n_slots;
    int64_t    slot_bytes;
    int       *map, mask;
    uint32_t  *heat;         /* [n_layer * n_exp], persistent */
    uint64_t   hits, misses;

    /* staging: one buffer per in-flight expert */
    uint8_t   *stage[ES_MAXK];

    /* the pool */
    pthread_t  th[ES_THREADS];
    pthread_mutex_t mx;
    pthread_cond_t  cv_work, cv_done;
    int        job_layer, job_k, job_next, job_left, job_quit;
    const int *job_eid;
    const void **job_g, **job_u, **job_d;
} EStream;

static inline uint32_t es_kh(int k){ return (uint32_t)k * 2654435761u; }

static int es_find(EStream *S, int key){
    uint32_t i = es_kh(key) & S->mask;
    while(S->map[i] != -1){
        if(S->e[S->map[i]].key == key) return S->map[i];
        i = (i+1) & S->mask;
    }
    return -1;
}
static void es_reindex(EStream *S){
    for(int i=0;i<=S->mask;i++) S->map[i] = -1;
    for(int s=0;s<S->n_slots;s++) if(S->e[s].key >= 0){
        uint32_t i = es_kh(S->e[s].key) & S->mask;
        while(S->map[i] != -1) i = (i+1) & S->mask;
        S->map[i] = s;
    }
}
/* coldest slot by PERSISTENT heat */
static int es_victim(EStream *S){
    int best = -1; uint32_t bh = 0xFFFFFFFFu;
    for(int i=0;i<S->n_slots;i++){
        if(S->e[i].key == -2) continue;          /* another reader is filling it */
        if(S->e[i].key == -1) return i;          /* free */
        uint32_t h = S->heat[S->e[i].key];
        if(h < bh){ bh = h; best = i; }
    }
    return best;
}

/* One expert slice -> one 4K-aligned pread. Returns the useful bytes and where they start. */
static int64_t es_read(EStream *S, const gguf_tensor *T, int eid, uint8_t *dst, int64_t *delta){
    int fd_unused; uint64_t off; int64_t nb, rows, cols;
    if(!gguf_expert_slice(S->m, T, eid, &fd_unused, &off, &nb, &rows, &cols)) return -1;
    uint64_t base = off & ~(uint64_t)(ES_ALIGN-1);
    int64_t  pad  = (int64_t)(off - base);
    int64_t  len  = (pad + nb + ES_ALIGN - 1) & ~(int64_t)(ES_ALIGN-1);
    ssize_t r = pread(S->fd[T->shard], dst, len, base);
    if(r < pad + nb){
        fprintf(stderr, "estream: pread len=%lld off=%llu dst=%p -> %zd : %s\n",
                (long long)len, (unsigned long long)base, (void*)dst, r,
                r < 0 ? strerror(errno) : "short read");
        return -1;
    }
    *delta = pad;
    return nb;
}

static void *es_worker(void *arg){
    EStream *S = (EStream*)arg;
    for(;;){
        int j;
        pthread_mutex_lock(&S->mx);
        while(!S->job_quit && S->job_next >= S->job_k) pthread_cond_wait(&S->cv_work, &S->mx);
        if(S->job_quit){ pthread_mutex_unlock(&S->mx); break; }
        j = S->job_next++;
        pthread_mutex_unlock(&S->mx);

        const int l   = S->job_layer;
        const int eid = S->job_eid[j];
        const int key = l * S->n_exp + eid;
        const ESLayer *EL = &S->L[l];
        const gguf_tensor *T[3] = { EL->g, EL->u, EL->d };

        pthread_mutex_lock(&S->mx);
        S->heat[key]++;                                 /* count every access, cached or not */
        int slot = es_find(S, key);
        pthread_mutex_unlock(&S->mx);

        if(slot >= 0){
            uint8_t *p = S->e[slot].p;
            pthread_mutex_lock(&S->mx);
            S->hits++;
            pthread_mutex_unlock(&S->mx);
            S->job_g[j] = p + S->coff[l][0];
            S->job_u[j] = p + S->coff[l][1];
            S->job_d[j] = p + S->coff[l][2];
        } else {
            uint8_t *b = S->stage[j];
            int64_t nb[3], dl[3];
            for(int q=0;q<3;q++){
                nb[q] = es_read(S, T[q], eid, b + S->roff[q], &dl[q]);
                if(nb[q] < 0){
                    fprintf(stderr, "estream: read failed  layer=%d eid=%d q=%d "
                                    "tensor=%s ne=[%lld,%lld,%lld] n_dims=%d\n",
                            l, eid, q, T[q] ? T[q]->name : "(NULL)",
                            T[q] ? (long long)T[q]->ne[0] : -1,
                            T[q] ? (long long)T[q]->ne[1] : -1,
                            T[q] ? (long long)T[q]->ne[2] : -1,
                            T[q] ? T[q]->n_dims : -1);
                    exit(1);
                }
            }
            S->job_g[j] = b + S->roff[0] + dl[0];
            S->job_u[j] = b + S->roff[1] + dl[1];
            S->job_d[j] = b + S->roff[2] + dl[2];

            /* Admission. The 12 MB copy into the cache must NOT happen under the lock:
             * doing so serializes all 8 readers on a 1.2 ms memcpy and throws away exactly
             * the queue depth this whole file exists to create. Reserve the slot under the
             * lock, copy outside it, publish under the lock.
             *
             * An expert must also EARN its slot (heat >= 2). Admitting every miss fills the
             * cache with one-shot experts and evicts the ones that would have come back. */
            int v = -1;
            pthread_mutex_lock(&S->mx);
            S->misses++;
            if(S->n_slots && S->lsz[l] <= S->slot_bytes && S->heat[key] >= 2){
                v = es_victim(S);
                if(v >= 0 && S->e[v].key != key){
                    if(S->e[v].key >= 0){ S->e[v].key = -1; es_reindex(S); }
                    S->e[v].key = -2;                       /* reserved: not findable, not a victim */
                } else v = -1;
            }
            pthread_mutex_unlock(&S->mx);

            if(v >= 0){
                for(int q=0;q<3;q++)                        /* outside the lock */
                    memcpy(S->e[v].p + S->coff[l][q], b + S->roff[q] + dl[q], nb[q]);
                pthread_mutex_lock(&S->mx);
                S->e[v].key = key;
                uint32_t h = es_kh(key) & S->mask;
                while(S->map[h] != -1) h = (h+1) & S->mask;
                S->map[h] = v;
                pthread_mutex_unlock(&S->mx);
            }

            pthread_mutex_lock(&S->mx);
            S->job_left--;
            if(S->job_left == 0) pthread_cond_broadcast(&S->cv_done);
            pthread_mutex_unlock(&S->mx);
            continue;
        }
        pthread_mutex_lock(&S->mx);
        S->job_left--;
        if(S->job_left == 0) pthread_cond_broadcast(&S->cv_done);
        pthread_mutex_unlock(&S->mx);
    }
    return NULL;
}

/* the glm_fetch callback: all k experts of a layer, concurrently */
/* Every byte an expert occupies here is on its way to the GPU, and cudaMemcpyAsync cannot
 * DMA out of pageable memory — it silently falls back to a blocking staged copy (measured:
 * 4.4 GB/s and a 463 ms "async" issue time, against 27.3 GB/s and 0 ms when pinned). So the
 * staging ring and the expert cache are allocated PINNED when there is a GPU.
 *
 * This costs no extra RAM: these are anonymous allocations already, so pinning only makes
 * pages that were evictable unevictable. It is not a second copy of anything.
 *
 * ES_CUDA is defined by the callers that actually have a device; the CPU-only tests include
 * this header too and must keep building without CUDA. cudaHostAlloc returns page-aligned
 * memory, which satisfies O_DIRECT's alignment requirement as posix_memalign did. */
/* HOW MUCH RAM IS REALLY LEFT.
 *
 * This function exists because I OOM-killed the machine. The cache is pinned (it has to be,
 * or its uploads to the GPU block), and I sized it from MemAvailable minus a 2 GB reserve —
 * about 17 GB. That number looked safe and was not, for a reason worth writing down:
 *
 *   MemAvailable COUNTS RECLAIMABLE PAGE CACHE. But we need that page cache ourselves — the
 *   model is a 254 GB mmap and every expert read goes through it. Pinning up to MemAvailable
 *   takes memory the kernel was counting on being able to hand back, and hands it to something
 *   that can never give it back. The kernel has nowhere to go but the OOM killer.
 *
 * And the deeper mistake: I computed a budget once, up front, and then TRUSTED it through a
 * long sequence of allocations. For memory you cannot page out, that is not allowed. Check
 * before every single one, and stop the moment the floor is in sight. */
static int64_t es_mem_available(void){
    FILE *f = fopen("/proc/meminfo", "r");
    if(!f) return 0;
    char ln[256]; long kb = 0;
    while(fgets(ln, sizeof(ln), f))
        if(sscanf(ln, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return (int64_t)kb * 1024;
}

/* Never let an unevictable allocation take us below this floor.
 *
 * Set by COLIBRI_RESERVE_GB (default 10 GB). The floor must leave room not just for the
 * system but for the page cache the expert reads THEMSELVES churn through — a cache sized
 * right up to a too-low floor still OOMs once reads start faulting pages in. 10 GB is the
 * server's standing requirement; on a tighter box, lower it deliberately via the env var. */
static int64_t es_reserve(void){
    const char *e = getenv("COLIBRI_RESERVE_GB");
    double gb = e ? atof(e) : 10.0;
    if(gb < 1.0) gb = 1.0;
    return (int64_t)(gb * (1LL<<30));
}
#define ES_RESERVE (es_reserve())

#ifdef ES_CUDA
static int es_alloc(void **p, int64_t n){
    if(es_mem_available() - n < ES_RESERVE) return 0;      /* refuse, do not OOM */
    return cudaHostAlloc(p, (size_t)n, cudaHostAllocDefault) == cudaSuccess;
}
#else
static int es_alloc(void **p, int64_t n){
    return posix_memalign(p, ES_ALIGN, (size_t)n) == 0;
}
#endif

static int es_fetch(void *ud, int layer, const int *eid, int k,
                    const void **g, const void **u, const void **d){
    EStream *S = (EStream*)ud;
    if(k > ES_MAXK) return 0;

    pthread_mutex_lock(&S->mx);
    S->job_layer = layer; S->job_eid = eid; S->job_k = k;
    S->job_next = 0; S->job_left = k;
    S->job_g = g; S->job_u = u; S->job_d = d;
    pthread_cond_broadcast(&S->cv_work);
    while(S->job_left > 0) pthread_cond_wait(&S->cv_done, &S->mx);
    pthread_mutex_unlock(&S->mx);
    return 1;
}

static int es_init(EStream *S, gguf_model *m, ESLayer *L, int n_layer, int n_exp, int topk,
                   int64_t cache_bytes){
    memset(S, 0, sizeof(*S));
    S->m = m; S->L = L; S->n_layer = n_layer; S->n_exp = n_exp; S->topk = topk;

    for(int s=0;s<m->n_shards;s++){
        S->fd[s] = open(m->shard[s].path, O_RDONLY | O_DIRECT);
        if(S->fd[s] < 0){ perror("estream: O_DIRECT open"); return 0; }
    }

    S->coff = calloc(n_layer, sizeof(*S->coff));
    S->lsz  = calloc(n_layer, sizeof(int64_t));
    int64_t reg[3] = {0,0,0};
    for(int i=0;i<n_layer;i++){
        if(!L[i].g) continue;
        const gguf_tensor *T[3] = { L[i].g, L[i].u, L[i].d };
        int64_t o = 0;
        for(int q=0;q<3;q++){
            int64_t nb = kq_row_bytes(T[q]->type, T[q]->ne[0]*T[q]->ne[1]);
            S->coff[i][q] = o; o += nb;
            int64_t pad = (nb + 2*ES_ALIGN + ES_ALIGN-1) & ~(int64_t)(ES_ALIGN-1);
            if(pad > reg[q]) reg[q] = pad;
        }
        S->lsz[i] = o;
    }
    S->roff[0]=0; S->roff[1]=reg[0]; S->roff[2]=reg[0]+reg[1];
    S->ring_bytes = reg[0]+reg[1]+reg[2];
    for(int j=0;j<ES_MAXK;j++)
        if(!es_alloc((void**)&S->stage[j], S->ring_bytes)) return 0;

    /* the cache slot is the COMMON expert size, not the largest: sizing every slot for the
     * one fat layer throws away 25% of the cache. Experts that do not fit stay uncached. */
    int64_t common = 0; int best = 0;
    for(int i=0;i<n_layer;i++){
        if(!S->lsz[i]) continue;
        int n = 0;
        for(int j=0;j<n_layer;j++) if(S->lsz[j] == S->lsz[i]) n++;
        if(n > best){ best = n; common = S->lsz[i]; }
    }
    S->slot_bytes = common;
    S->n_slots = common ? (int)(cache_bytes / common) : 0;

    S->heat = calloc((size_t)n_layer*n_exp, sizeof(uint32_t));
    if(S->n_slots){
        S->e = calloc(S->n_slots, sizeof(ESEntry));
        for(int i=0;i<S->n_slots;i++){
            S->e[i].key = -1;
            /* es_alloc refuses rather than OOMs, so the cache simply stops growing at the
             * safe size. Take what the machine can actually spare, not what a stale budget
             * said it could. */
            if(!es_alloc((void**)&S->e[i].p, common)){ S->n_slots = i; break; }
        }
        int cap = 16; while(cap < S->n_slots*2) cap <<= 1;
        S->mask = cap-1;
        S->map = malloc((size_t)cap*sizeof(int));
        for(int i=0;i<cap;i++) S->map[i] = -1;
    }

    pthread_mutex_init(&S->mx,NULL);
    pthread_cond_init(&S->cv_work,NULL);
    pthread_cond_init(&S->cv_done,NULL);
    S->job_k = 0; S->job_next = 0;
    for(int i=0;i<ES_THREADS;i++) pthread_create(&S->th[i], NULL, es_worker, S);

    fprintf(stderr, "[estream] %d readers, O_DIRECT, cache %d experts (%.1f GB, slot %.2f MB)\n",
            ES_THREADS, S->n_slots, S->n_slots*common/1e9, common/1e6);
    return 1;
}

static void es_free(EStream *S){
    pthread_mutex_lock(&S->mx);
    S->job_quit = 1;
    pthread_cond_broadcast(&S->cv_work);
    pthread_mutex_unlock(&S->mx);
    for(int i=0;i<ES_THREADS;i++) pthread_join(S->th[i], NULL);
}

#endif /* COLIBRI_ESTREAM_H */
