#ifndef QWEN_KV_TIER_H
#define QWEN_KV_TIER_H
/* kv_tier.h — the KV cache as a streamed, tiered store. estream.h's principle, applied to
 * the one thing in this model that GROWS.
 *
 * WHY THE KV IS THE RIGHT THING TO STREAM, AND THE WEIGHTS ARE NOT
 *
 * Qwen3.5's weights are 15.6 GiB and every token reads all of them — but that number never
 * changes, so it is a fixed cost you pay once by placing each tensor on the side of the bus
 * that will compute it. The KV is different: at 1k tokens it is 33 MiB, at 250k it is 8.1
 * GiB, and by then it is the largest single object in the system and the dominant per-token
 * read. Streaming it is not an optimization, it is the only way 250k fits at all.
 *
 * AND IT IS THE FRIENDLIEST POSSIBLE STREAM
 *
 * Expert streaming is hostile: the router names 8 of 256 experts, late, at random. Attention
 * is the opposite — layer l scans positions 0..T IN ORDER, every token, forever. Perfectly
 * sequential, perfectly predictable, known one whole layer in advance. So:
 *
 *   - chunks are read AHEAD, never on demand. The read for chunk i+1 is issued before the
 *     compute on chunk i starts, so the drive never idles waiting to be asked.
 *   - the read for layer l+1 is issued while layers l..l+1 (3 GDN layers and 4 FFNs, ~20 ms
 *     of CPU-side RAM traffic) are still computing. That window is free bandwidth.
 *   - O_DIRECT with several chunks in flight: measured on this project's drive, expert-sized
 *     reads go 2.36 GB/s at QD1 and 5.39 GB/s at QD8. Queue depth is worth 2.3x and costs
 *     nothing but issuing the reads earlier.
 *
 * TIERS, NEWEST-FIRST
 *
 * Residency is per CHUNK, and chunks are assigned newest-first to the closest tier:
 *
 *     VRAM   the newest tokens        no bus traffic at all
 *     RAM    the middle band          PCIe, DMA'd under the compute
 *     NVMe   the cold prefix          O_DIRECT, read ahead
 *
 * Newest-first is not arbitrary. The tail is what a growing context touches while it is
 * still being written, and it is what gets rewritten on a branch or a rollback; the cold
 * prefix is write-once and read-sequentially, which is exactly what a disk is good at.
 *
 * WHAT IS STORED
 *
 * q8_0, the same wire format ggml uses, so the numbers are comparable to llama.cpp's
 * -ctk/-ctv q8_0. 34 KiB/token across all 16 attention layers against 64 KiB at fp16.
 * kv_tier_test measures the attention-output error this actually causes rather than
 * assuming it is small — on this project q4_0 was measured at 10% on MLA latents and
 * rejected, and that only came out because it was measured.
 *
 * LAYOUT — K AND V ARE SEPARATE REGIONS, AND THAT IS THE WHOLE POINT
 *
 * The obvious layout interleaves them per token:
 *
 *     [pos 0: K | V][pos 1: K | V] ...
 *
 * and it wastes half the read bandwidth. Attention makes TWO passes over the history: the
 * score pass reads only K, the weighted-sum pass reads only V. With K and V adjacent, each
 * pass drags the other's cache lines in and never touches them — 2176 bytes fetched per
 * token to use 1088. At 250k context that is 4.05 GiB per token of cache lines fetched and
 * discarded, on both the RAM and the NVMe side.
 *
 * So a chunk is split into two contiguous regions:
 *
 *     [layer L, chunk c] = [ K: pos c*C .. c*C+C ] [ V: pos c*C .. c*C+C ]
 *
 * Now each pass streams exactly what it consumes, back to back, at full line utilization.
 * Nothing else changes: the chunk is the same size, the write still touches one place per
 * layer per token, and the tiering above is untouched.
 *
 * The separation buys a second thing for later: K and V can be quantized at DIFFERENT
 * precision. K feeds a softmax and is the more sensitive of the two; V is averaged. That
 * choice is impossible while they share a block layout.
 *
 * ON-DISK FORMAT IS VERSIONED. The spill file now carries a header with the geometry and a
 * format version, so a file written by an older layout fails loudly instead of decoding as
 * plausible garbage — which is exactly what an interleaved file read as split would do.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <sys/vfs.h>
#include <time.h>

#include "kv_tier_simd.h"

#include "qwen35.h"

#define KVT_ALIGN     4096
#define KVT_MAXQ      16          /* prefetch slots in flight */
#define KVT_MAGIC     0x3556544bu /* "KVT5" */
#define KVT_VERSION   2           /* 1 = K|V interleaved per token, 2 = split K/V regions */

/* Written at offset 0 of the spill file, one aligned block. Everything past it is chunks. */
typedef struct {
    uint32_t magic, version;
    int32_t  fmt, n_layer, n_ctx, chunk, n_kv;
    int64_t  ent, chunk_bytes;
} KvtHeader;

#define KVT_MAXW      16          /* write-behinds in flight */

enum { KVT_RAM = 0, KVT_DISK = 1 };

typedef struct {
    /* geometry */
    int      fmt;
    int      n_layer;            /* full-attention layers only */
    int      n_ctx;
    int      n_kv;               /* values per K (and per V) = n_head_kv * d_head */
    int64_t  ent;                /* bytes of one token's K+V in one layer */
    int      chunk;              /* tokens per residency/IO unit */
    int      n_chunk;
    int64_t  chunk_bytes;        /* ent*chunk, padded to KVT_ALIGN (O_DIRECT needs it) */

    /* residency, indexed layer*n_chunk + c */
    uint8_t  *where;
    uint8_t **ram;

    /* the RAM tier: one slab, handed out in FIFO order */
    uint8_t  *arena;
    int64_t   arena_bytes;
    int       ram_chunks;
    int      *slot_owner;        /* arena slot -> layer*n_chunk+c, or -1 */
    uint8_t  *slot_writing;      /* arena slot -> 1 while its write-behind is in flight */
    uint8_t  *slot_clean;        /* arena slot -> 1 once its chunk is fully on disk.
                                  * A chunk is append-only and immutable once complete, so a
                                  * completed write makes it permanently clean and eviction
                                  * needs no second write. Without this every chunk was
                                  * written TWICE — once on completion, once at eviction —
                                  * and the write side reported nearly double the traffic. */
    int      *fifo;
    int       fifo_head, fifo_n;

    /* the disk tier */
    int      fd, odirect;
    char     path[512];
    uint8_t *stage;              /* KVT_MAXQ landing buffers for reads */

    /* ---- two independent queues, on purpose ----
     * Reads and writes shared one job array and one condvar in the first version. A reader
     * would find a WRITE job for the same (layer, chunk), wait for it to reach the reader's
     * "done" state, and hang — writes retire to "free" because nobody is waiting on them.
     * Nine threads in futex_wait and no way to tell which side was wrong. Split, they have
     * one invariant each and neither can wait on the other's completion protocol. */
    pthread_mutex_t mu;

    pthread_cond_t  cv_rjob, cv_rdone;
    struct { int layer, c, slot, state; } rq[KVT_MAXQ];   /* 0 free 1 queued 2 running 3 done */
    pthread_t       rth[KVT_MAXQ];
    int             n_rth;

    pthread_cond_t  cv_wjob, cv_wdone;
    struct { int slot, layer, c; } wq[KVT_MAXW];          /* ring */
    pthread_t       wth[4];
    int             wq_head, wq_n, n_wth;

    int      stop;

    /* accounting. Read/write rates are reported over the WALL-CLOCK ENVELOPE of the IO, not
     * the sum of per-thread durations — with 8 readers in flight the latter under-reports
     * the drive by roughly the thread count, which is how a saturated drive can look slow. */
    int64_t  bytes_disk, bytes_wr, bytes_ram;
    double   t_disk, tr_first, tr_last, tw_first, tw_last;
    int64_t  n_reads, n_writes;
} KvTier;

/* ---------------- packing: q8_0 via the SIMD kernels, fp16 straight ---------------- */

static void kvt_pack_f16(void *dst, const float *src, int n){
    uint16_t *p = (uint16_t*)dst;
    for(int i = 0; i < n; i++) p[i] = kvt_f2h_bits(src[i]);
}
static float kvt_dot_f16(const float *q, const void *w, int n){
    const uint16_t *p = (const uint16_t*)w;
    float s = 0;
    for(int i = 0; i < n; i++) s += q[i]*kvt_h2f_bits(p[i]);
    return s;
}
static void kvt_axpy_f16(float *out, float g, const void *w, int n){
    const uint16_t *p = (const uint16_t*)w;
    for(int i = 0; i < n; i++) out[i] += g*kvt_h2f_bits(p[i]);
}

static void kvt_pack(const KvTier *S, void *dst, const float *src){
    if(S->fmt == Q35_KV_Q8_0) kvt_pack_q8(dst, src, S->n_kv);
    else                      kvt_pack_f16(dst, src, S->n_kv);
}
static inline float kvt_dot(const KvTier *S, const float *q, const void *w, int n){
    return S->fmt == Q35_KV_Q8_0 ? kvt_dot_q8(q, w, n) : kvt_dot_f16(q, w, n);
}
static inline void kvt_axpy(const KvTier *S, float *o, float g, const void *w, int n){
    if(S->fmt == Q35_KV_Q8_0) kvt_axpy_q8(o, g, w, n); else kvt_axpy_f16(o, g, w, n);
}

/* ---------------- the two pools ---------------- */

static double kvt_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

static int64_t kvt_chunk_off(const KvTier *S, int layer, int c){
    return (int64_t)KVT_ALIGN + ((int64_t)layer*S->n_chunk + c) * S->chunk_bytes;
}

/* Byte offset of one token's K (and of its V) inside a chunk. `half` is one token's K across
 * all kv-heads; the V region starts a whole chunk of K's later. */
static inline int64_t kvt_koff(const KvTier *S, int t_in_chunk){
    return (int64_t)t_in_chunk * (S->ent/2);
}
static inline int64_t kvt_voff(const KvTier *S, int t_in_chunk){
    return (int64_t)S->chunk * (S->ent/2) + (int64_t)t_in_chunk * (S->ent/2);
}

static void *kvt_reader(void *arg){
    KvTier *S = (KvTier*)arg;
    for(;;){
        pthread_mutex_lock(&S->mu);
        int j = -1;
        for(;;){
            if(S->stop){ pthread_mutex_unlock(&S->mu); return NULL; }
            for(int i = 0; i < KVT_MAXQ; i++) if(S->rq[i].state == 1){ j = i; break; }
            if(j >= 0) break;
            pthread_cond_wait(&S->cv_rjob, &S->mu);
        }
        S->rq[j].state = 2;
        const int layer = S->rq[j].layer, c = S->rq[j].c, slot = S->rq[j].slot;
        pthread_mutex_unlock(&S->mu);

        const double t0 = kvt_now();
        uint8_t *dst = S->stage + (int64_t)slot*S->chunk_bytes;
        const int64_t off = kvt_chunk_off(S, layer, c);
        int64_t got = 0;
        while(got < S->chunk_bytes){
            ssize_t r = pread(S->fd, dst + got, (size_t)(S->chunk_bytes - got), off + got);
            if(r <= 0) break;
            got += r;
        }
        const double t1 = kvt_now();

        pthread_mutex_lock(&S->mu);
        S->rq[j].state = 3;
        S->bytes_disk += got; S->t_disk += t1-t0; S->n_reads++;
        if(S->tr_first == 0.0 || t0 < S->tr_first) S->tr_first = t0;
        if(t1 > S->tr_last) S->tr_last = t1;
        pthread_cond_broadcast(&S->cv_rdone);
        pthread_mutex_unlock(&S->mu);
    }
}

/* Write-behind. Exists so a prefill does not stall on one 8.5 MiB pwrite at a time: doing
 * the flush synchronously at eviction measured 0.22 GB/s, because the drive saw a single
 * request with nothing queued behind it and the token that triggered it waited for all of it. */
static void *kvt_writer(void *arg){
    KvTier *S = (KvTier*)arg;
    for(;;){
        pthread_mutex_lock(&S->mu);
        while(!S->stop && S->wq_n == 0) pthread_cond_wait(&S->cv_wjob, &S->mu);
        if(S->wq_n == 0){ pthread_mutex_unlock(&S->mu); return NULL; }
        const int slot  = S->wq[S->wq_head].slot;
        const int layer = S->wq[S->wq_head].layer;
        const int c     = S->wq[S->wq_head].c;
        S->wq_head = (S->wq_head + 1) % KVT_MAXW;
        S->wq_n--;
        pthread_cond_broadcast(&S->cv_wdone);       /* a producer may be blocked on a full ring */
        pthread_mutex_unlock(&S->mu);

        const double t0 = kvt_now();
        const uint8_t *src = S->arena + (int64_t)slot*S->chunk_bytes;
        const int64_t off  = kvt_chunk_off(S, layer, c);
        int64_t put = 0;
        while(put < S->chunk_bytes){
            ssize_t w = pwrite(S->fd, src + put, (size_t)(S->chunk_bytes - put), off + put);
            if(w <= 0) break;
            put += w;
        }
        const double t1 = kvt_now();

        pthread_mutex_lock(&S->mu);
        S->slot_writing[slot] = 0;
        S->slot_clean[slot]   = 1;
        S->bytes_wr += put; S->n_writes++;
        if(S->tw_first == 0.0 || t0 < S->tw_first) S->tw_first = t0;
        if(t1 > S->tw_last) S->tw_last = t1;
        pthread_cond_broadcast(&S->cv_wdone);
        pthread_mutex_unlock(&S->mu);
    }
}

/* ---------------- open / close ---------------- */

/* n_layer is passed explicitly rather than derived, because speculative decoding adds the
 * MTP block's attention layer to the cache and the geometry alone cannot know whether it is
 * in use. */
static int kvt_open_n(KvTier *S, const Q35Cfg *c, int n_layer, int n_ctx, int fmt,
                      int64_t ram_budget, const char *dir, int chunk_tokens, int n_threads){
    memset(S, 0, sizeof *S);
    S->fmt     = fmt;
    S->n_layer = n_layer;
    S->n_ctx   = n_ctx;
    S->n_kv    = c->n_head_kv * c->d_head;
    S->ent     = 2 * q35_kv_entry_bytes(fmt, S->n_kv);
    S->chunk   = chunk_tokens > 0 ? chunk_tokens : 4096;
    if(S->chunk > n_ctx) S->chunk = n_ctx;
    if(S->chunk < 1)     S->chunk = 1;
    S->n_chunk = (n_ctx + S->chunk - 1) / S->chunk;
    S->chunk_bytes = ((S->ent*S->chunk) + KVT_ALIGN-1) & ~(int64_t)(KVT_ALIGN-1);
    S->fd = -1;

    const int n_slots = S->n_layer * S->n_chunk;
    S->where = (uint8_t*)calloc(n_slots, 1);
    S->ram   = (uint8_t**)calloc(n_slots, sizeof(uint8_t*));
    if(!S->where || !S->ram) return 0;

    S->ram_chunks = ram_budget > 0 ? (int)(ram_budget / S->chunk_bytes) : 0;
    if(S->ram_chunks > n_slots) S->ram_chunks = n_slots;
    /* The hot window must at least be able to hold one open chunk per layer, or every layer
     * evicts the layer before it and nothing is ever resident when it is read back. */
    if(S->ram_chunks < S->n_layer){
        fprintf(stderr, "kv_tier: RAM budget %.2f GiB holds %d chunks of %.1f MiB, need at "
                "least %d (one open chunk per attention layer)\n",
                ram_budget/(1024.0*1024*1024), S->ram_chunks,
                S->chunk_bytes/(1024.0*1024.0), S->n_layer);
        return 0;
    }
    S->arena_bytes = (int64_t)S->ram_chunks * S->chunk_bytes;
    if(posix_memalign((void**)&S->arena, KVT_ALIGN, (size_t)S->arena_bytes) != 0){
        fprintf(stderr, "kv_tier: cannot allocate a %.2f GiB RAM tier\n",
                S->arena_bytes/(1024.0*1024*1024));
        return 0;
    }
    memset(S->arena, 0, (size_t)S->arena_bytes);
    S->slot_owner   = (int*)malloc(sizeof(int)*S->ram_chunks);
    S->fifo         = (int*)malloc(sizeof(int)*S->ram_chunks);
    S->slot_writing = (uint8_t*)calloc(S->ram_chunks, 1);
    S->slot_clean   = (uint8_t*)calloc(S->ram_chunks, 1);
    if(!S->slot_owner || !S->fifo || !S->slot_writing || !S->slot_clean) return 0;
    for(int i = 0; i < S->ram_chunks; i++){ S->slot_owner[i] = -1; S->fifo[i] = 0; }

    /* The spill file is created even when the whole cache currently fits, because the
     * context is allowed to grow past what fits and there must be somewhere for it to go. */
    const int need_disk = 1;
    (void)n_slots;
    if(need_disk){
        const char *env = getenv("QWEN_KV_SPILL");
        if(env) dir = env;
        if(!dir) dir = "/tmp";
        /* tmpfs is the quiet disaster: the cold tier would sit in the very RAM the tiering
         * exists to free, the plan's numbers become fiction, and the box swaps instead of
         * reading. Loud, because it otherwise looks like it works. */
        struct statfs sf;
        if(statfs(dir, &sf) == 0 && (long)sf.f_type == 0x01021994L)
            fprintf(stderr,
                "kv_tier: WARNING: %s is tmpfs — the cold KV tier would live in RAM, not on\n"
                "         disk, competing with the weights it is meant to make room for.\n"
                "         Set QWEN_KV_SPILL to a path on real storage.\n", dir);
        snprintf(S->path, sizeof S->path, "%s/colibri-kv-%d.bin", dir, (int)getpid());
#ifdef O_DIRECT
        S->fd = open(S->path, O_RDWR|O_CREAT|O_TRUNC|O_DIRECT, 0600);
        S->odirect = (S->fd >= 0);
#endif
        if(S->fd < 0){
            S->fd = open(S->path, O_RDWR|O_CREAT|O_TRUNC, 0600);
            if(S->fd >= 0) fprintf(stderr, "kv_tier: O_DIRECT refused on %s — reads will go "
                "through the page cache and evict the weights they share RAM with\n", S->path);
        }
        if(S->fd < 0){ fprintf(stderr, "kv_tier: cannot open %s: %s\n", S->path, strerror(errno)); return 0; }
        if(ftruncate(S->fd, (off_t)(KVT_ALIGN + (int64_t)n_slots*S->chunk_bytes)) != 0){
            fprintf(stderr, "kv_tier: cannot size the spill file\n"); return 0;
        }
        {   /* O_DIRECT needs an aligned buffer even for the header */
            void *hb = NULL;
            if(posix_memalign(&hb, KVT_ALIGN, KVT_ALIGN) != 0) return 0;
            memset(hb, 0, KVT_ALIGN);
            KvtHeader *H = (KvtHeader*)hb;
            H->magic = KVT_MAGIC; H->version = KVT_VERSION;
            H->fmt = fmt; H->n_layer = S->n_layer; H->n_ctx = S->n_ctx;
            H->chunk = S->chunk; H->n_kv = S->n_kv;
            H->ent = S->ent; H->chunk_bytes = S->chunk_bytes;
            if(pwrite(S->fd, hb, KVT_ALIGN, 0) != KVT_ALIGN){
                fprintf(stderr, "kv_tier: cannot write the spill header\n");
                free(hb); return 0;
            }
            free(hb);
        }
        if(posix_memalign((void**)&S->stage, KVT_ALIGN,
                          (size_t)KVT_MAXQ*S->chunk_bytes) != 0) return 0;
    }

    pthread_mutex_init(&S->mu, NULL);
    pthread_cond_init(&S->cv_rjob, NULL);  pthread_cond_init(&S->cv_rdone, NULL);
    pthread_cond_init(&S->cv_wjob, NULL);  pthread_cond_init(&S->cv_wdone, NULL);
    S->n_rth = need_disk ? (n_threads > 0 ? n_threads : 8) : 0;
    if(S->n_rth > KVT_MAXQ) S->n_rth = KVT_MAXQ;
    /* Writers are separate from readers because their cost profile is: NVMe sustained write
     * is well under its read rate, so it needs its own depth to keep up during a prefill. */
    { const char *e = getenv("QWEN_KV_WTHREADS");
      int w = e ? atoi(e) : 4; if(w < 1) w = 1; if(w > 4) w = 4;
      S->n_wth = need_disk ? w : 0; }
    for(int i = 0; i < S->n_rth; i++) pthread_create(&S->rth[i], NULL, kvt_reader, S);
    for(int i = 0; i < S->n_wth; i++) pthread_create(&S->wth[i], NULL, kvt_writer, S);
    return 1;
}

static int kvt_open(KvTier *S, const Q35Cfg *c, int n_ctx, int fmt,
                    int64_t ram_budget, const char *dir, int chunk_tokens, int n_threads){
    return kvt_open_n(S, c, q35_n_attn_layers(c), n_ctx, fmt, ram_budget, dir, chunk_tokens, n_threads);
}

static void kvt_close(KvTier *S){
    if(S->n_rth || S->n_wth){
        pthread_mutex_lock(&S->mu);
        S->stop = 1;
        pthread_cond_broadcast(&S->cv_rjob); pthread_cond_broadcast(&S->cv_rdone);
        pthread_cond_broadcast(&S->cv_wjob); pthread_cond_broadcast(&S->cv_wdone);
        pthread_mutex_unlock(&S->mu);
        for(int i = 0; i < S->n_rth; i++) pthread_join(S->rth[i], NULL);
        for(int i = 0; i < S->n_wth; i++) pthread_join(S->wth[i], NULL);
    }
    if(S->fd >= 0){ close(S->fd); unlink(S->path); }
    free(S->arena); free(S->stage); free(S->where); free(S->ram);
    free(S->slot_owner); free(S->fifo); free(S->slot_writing); free(S->slot_clean);
    memset(S, 0, sizeof *S);
    S->fd = -1;
}

static int kvt_grow(KvTier *S, int need_ctx);      /* defined below; used by kvt_put */

/* ---------------- write path ---------------- */

/* Queue a resident chunk's write WITHOUT evicting it: it stays readable in RAM, and all this
 * does is make the disk copy current early, so the arena slot is already clean when the FIFO
 * comes round to reuse it. Blocks only if the ring is full, which is correct back-pressure —
 * the queue depth belongs at the drive, not in this process. */
static void kvt_flush_begin(KvTier *S, int slot){
    if(S->fd < 0) return;
    pthread_mutex_lock(&S->mu);
    const int idx = S->slot_owner[slot];
    if(idx < 0 || S->slot_writing[slot] || S->slot_clean[slot]){
        pthread_mutex_unlock(&S->mu); return;
    }
    while(S->wq_n == KVT_MAXW && !S->stop) pthread_cond_wait(&S->cv_wdone, &S->mu);
    const int t = (S->wq_head + S->wq_n) % KVT_MAXW;
    S->wq[t].slot = slot; S->wq[t].layer = idx / S->n_chunk; S->wq[t].c = idx % S->n_chunk;
    S->wq_n++;
    S->slot_writing[slot] = 1;
    pthread_cond_signal(&S->cv_wjob);
    pthread_mutex_unlock(&S->mu);
}

static void kvt_flush_wait(KvTier *S, int slot){
    pthread_mutex_lock(&S->mu);
    while(S->slot_writing[slot]) pthread_cond_wait(&S->cv_wdone, &S->mu);
    pthread_mutex_unlock(&S->mu);
}

/* Append one token's K and V for one attention-layer slot. */
static void kvt_put(KvTier *S, int layer, int pos, const float *k, const float *v){
    if(pos >= S->n_ctx && !kvt_grow(S, pos + 1)) return;   /* at the floor: stop growing */
    const int c = pos / S->chunk;
    const int idx = layer*S->n_chunk + c;

    if(!S->ram[idx]){
        int slot;
        if(S->fifo_n < S->ram_chunks){
            slot = S->fifo_n;
        } else {
            /* Evict the oldest-WRITTEN chunk, which is also the oldest POSITION because
             * writes only ever move forward. That is what makes "newest-first residency"
             * fall out of a plain FIFO with no per-access bookkeeping. */
            slot = S->fifo[S->fifo_head];
            S->fifo_head = (S->fifo_head + 1) % S->ram_chunks;
            S->fifo_n--;
            const int old = S->slot_owner[slot];
            if(old >= 0){
                kvt_flush_begin(S, slot);
                kvt_flush_wait(S, slot);     /* usually already done: it was queued when full */
                S->where[old] = KVT_DISK;
                S->ram[old]   = NULL;
            }
        }
        S->fifo[(S->fifo_head + S->fifo_n) % S->ram_chunks] = slot;
        S->fifo_n++;
        S->slot_owner[slot] = idx;
        S->slot_clean[slot] = 0;                 /* new tenant, disk copy is stale again */
        S->ram[idx]   = S->arena + (int64_t)slot*S->chunk_bytes;
        S->where[idx] = KVT_RAM;
    }
    const int ti = pos % S->chunk;
    uint8_t *base = S->ram[idx];
    kvt_pack(S, base + kvt_koff(S, ti), k);
    kvt_pack(S, base + kvt_voff(S, ti), v);
    S->bytes_ram += S->ent;

    /* Chunk just filled: start its write NOW, while later chunks are still being computed.
     * That is what keeps the drive's write queue non-empty instead of stop-and-go. */
    if(S->fd >= 0 && (pos % S->chunk) == S->chunk - 1){
        for(int sl = 0; sl < S->ram_chunks; sl++)
            if(S->slot_owner[sl] == idx){ kvt_flush_begin(S, sl); break; }
    }
}

/* ---------------- read path ----------------
 *     for c in chunks:  kvt_prefetch_window(layer, c, depth);  scan(kvt_chunk(layer, c))
 * so the drive is always `depth` chunks ahead of the compute and never waits to be asked. */

static int kvt_find_read(KvTier *S, int layer, int c){
    for(int i = 0; i < KVT_MAXQ; i++)
        if(S->rq[i].state && S->rq[i].layer == layer && S->rq[i].c == c) return i;
    return -1;
}

static void kvt_prefetch(KvTier *S, int layer, int c){
    if(c < 0 || c >= S->n_chunk || S->fd < 0) return;
    if(S->ram[layer*S->n_chunk + c]) return;
    pthread_mutex_lock(&S->mu);
    if(kvt_find_read(S, layer, c) < 0){
        int j = -1;
        for(int i = 0; i < KVT_MAXQ; i++) if(!S->rq[i].state){ j = i; break; }
        if(j >= 0){
            S->rq[j].layer = layer; S->rq[j].c = c; S->rq[j].slot = j;
            S->rq[j].state = 1;
            pthread_cond_signal(&S->cv_rjob);
        }
    }
    pthread_mutex_unlock(&S->mu);
}

/* One staging slot per job slot (slot == j), so two in-flight reads can never land in the
 * same buffer while one of them is being scanned. */
static const uint8_t *kvt_chunk(KvTier *S, int layer, int c){
    const int idx = layer*S->n_chunk + c;
    if(S->ram[idx]) return S->ram[idx];
    if(S->fd < 0) return NULL;

    pthread_mutex_lock(&S->mu);
    int j = kvt_find_read(S, layer, c);
    if(j < 0){
        for(int i = 0; i < KVT_MAXQ; i++) if(!S->rq[i].state){ j = i; break; }
        if(j < 0){
            /* every slot in flight: wait for one to retire rather than fail the read */
            while(j < 0 && !S->stop){
                pthread_cond_wait(&S->cv_rdone, &S->mu);
                for(int i = 0; i < KVT_MAXQ; i++) if(!S->rq[i].state){ j = i; break; }
            }
            if(j < 0){ pthread_mutex_unlock(&S->mu); return NULL; }
        }
        S->rq[j].layer = layer; S->rq[j].c = c; S->rq[j].slot = j;
        S->rq[j].state = 1;
        pthread_cond_signal(&S->cv_rjob);
    }
    while(S->rq[j].state != 3 && !S->stop) pthread_cond_wait(&S->cv_rdone, &S->mu);
    const int slot = S->rq[j].slot;
    S->rq[j].state = 0;
    pthread_cond_broadcast(&S->cv_rdone);          /* a slot just freed up */
    pthread_mutex_unlock(&S->mu);
    return S->stage + (int64_t)slot*S->chunk_bytes;
}

/* Issue the next `depth` chunks at once. One-ahead is queue depth 1, and this drive was
 * measured at 2.36 GB/s at QD1 against 5.39 at QD8 — 2.3x for nothing but asking earlier. */
static void kvt_prefetch_window(KvTier *S, int layer, int c, int depth){
    for(int i = 1; i <= depth; i++) kvt_prefetch(S, layer, c+i);
}

/* ---------------- growth ----------------
 * The context is not a fixed size. n_ctx is only the CURRENT capacity of the chunk index;
 * when a write runs past it, the index grows and the spill file is extended. What does NOT
 * grow is the RAM arena — that is the hot window, deliberately bounded, and everything past
 * it lives on the drive. So the ceiling on context length is disk space, not RAM.
 *
 * Growth allocates only the index tables (a few ints per layer per chunk, kilobytes), and it
 * refuses to do even that below the memory floor. If the allocation fails outright the
 * process ends rather than retrying — see q35_oom. */
static int kvt_grow(KvTier *S, int need_ctx){
    if(need_ctx <= S->n_ctx) return 1;

    int new_ctx = S->n_ctx;
    while(new_ctx < need_ctx) new_ctx += (new_ctx/2 > S->chunk) ? new_ctx/2 : S->chunk;
    const int new_chunks = (new_ctx + S->chunk - 1) / S->chunk;
    const int new_slots  = S->n_layer * new_chunks;
    const int64_t idx_bytes = (int64_t)new_slots*(sizeof(uint8_t) + sizeof(uint8_t*));

    if(!q35_mem_room(idx_bytes)){
        fprintf(stderr, "kv_tier: at the memory floor (%.2f GiB available) — context stays at "
                "%d tokens\n", q35_ram_avail_bytes()/(1024.0*1024*1024), S->n_ctx);
        return 0;
    }

    /* The chunk index is [layer][chunk], so growing the chunk count RESHAPES it: every
     * layer's row moves. Copying row by row, backwards, is the only order that does not
     * overwrite a row before it has been read. */
    uint8_t  *nw = (uint8_t*) q35_xalloc((size_t)new_slots, "kv chunk index");
    uint8_t **nr = (uint8_t**)q35_xalloc((size_t)new_slots*sizeof(uint8_t*), "kv chunk table");
    memset(nw, 0, (size_t)new_slots);
    memset(nr, 0, (size_t)new_slots*sizeof(uint8_t*));
    for(int l = S->n_layer - 1; l >= 0; l--){
        memcpy(nw + (size_t)l*new_chunks, S->where + (size_t)l*S->n_chunk, (size_t)S->n_chunk);
        memcpy(nr + (size_t)l*new_chunks, S->ram   + (size_t)l*S->n_chunk,
               (size_t)S->n_chunk*sizeof(uint8_t*));
    }
    /* arena slots point back into the index by flat position, so those move too */
    for(int sl = 0; sl < S->ram_chunks; sl++){
        const int old = S->slot_owner[sl];
        if(old >= 0) S->slot_owner[sl] = (old / S->n_chunk)*new_chunks + (old % S->n_chunk);
    }
    free(S->where); free(S->ram);
    S->where = nw; S->ram = nr;
    S->n_chunk = new_chunks;
    S->n_ctx   = new_ctx;

    if(S->fd >= 0 && ftruncate(S->fd, (off_t)(KVT_ALIGN + (int64_t)new_slots*S->chunk_bytes)) != 0){
        fprintf(stderr, "kv_tier: cannot extend the spill file to %.2f GiB: %s\n",
                (double)new_slots*S->chunk_bytes/(1024.0*1024*1024), strerror(errno));
        return 0;
    }
    return 1;
}

static void kvt_report(const KvTier *S, FILE *o){
    fprintf(o, "  kv_tier: %s, %d layers x %d ctx, chunk %d tok (%.1f MiB)\n",
            q35_kv_fmt_name(S->fmt), S->n_layer, S->n_ctx, S->chunk, S->chunk_bytes/(1024.0*1024.0));
    fprintf(o, "    RAM tier %.3f GiB (%d chunks)   disk %s\n",
            S->arena_bytes/(1024.0*1024*1024), S->ram_chunks,
            S->fd >= 0 ? S->path : "(not needed)");
    if(S->n_reads){
        const double w = S->tr_last - S->tr_first;
        fprintf(o, "    read : %.3f GiB in %ld x %.1f MiB, O_DIRECT %s, %.3f s -> %.2f GB/s\n",
                S->bytes_disk/(1024.0*1024*1024), (long)S->n_reads,
                S->chunk_bytes/(1024.0*1024.0), S->odirect ? "on" : "OFF",
                w, w > 0 ? S->bytes_disk/w/1e9 : 0.0);
    }
    if(S->n_writes){
        const double w = S->tw_last - S->tw_first;
        fprintf(o, "    write: %.3f GiB in %ld x %.1f MiB, %.3f s -> %.2f GB/s\n",
                S->bytes_wr/(1024.0*1024*1024), (long)S->n_writes,
                S->chunk_bytes/(1024.0*1024.0), w, w > 0 ? S->bytes_wr/w/1e9 : 0.0);
    }
}

#endif
