#ifndef COLIBRI_KVSTORE_H
#define COLIBRI_KVSTORE_H
/* kvstore.h — quantized KV cache + a persistent prefix cache on disk.
 *
 * WHY THIS IS THE MOST VALUABLE PIECE IN THE ENGINE
 *
 * Decoding streams ~8 experts per layer per token. PREFILL does not: over N tokens the
 * UNION of routed experts saturates, and it saturates fast. Measured against the real
 * routing geometry (256 experts, top-8, 76 layers):
 *
 *      100 tokens -> 121 of 256 experts per layer -> 111 GB -> 21 s
 *      500 tokens -> 210 of 256                   -> 193 GB -> 36 s
 *     2000 tokens -> 253 of 256                   -> 233 GB -> 43 s
 *
 * In other words: prefilling ANY prompt longer than a few hundred tokens reads essentially
 * the entire 235 GB expert model off NVMe. ~44 seconds, every time, no matter how fast
 * decoding is. For a coding assistant that resends the same repo context on every turn,
 * that is the whole latency budget, spent re-deriving something it already knew.
 *
 * A cached prefix costs 25 KB/token (q4_0) instead:
 *
 *     10 000 tokens of context:  244 MB from disk = 47 ms   (vs 44 s)
 *
 * ~900x. Nothing else in this engine comes close.
 *
 * WHAT IS ACTUALLY CACHEABLE
 *
 * Not "hot tokens" — a KV entry depends on the ENTIRE preceding context, so the same token
 * has a different KV in every position and conversation. What repeats is PREFIXES: the
 * system prompt, the repo context, the file being edited, the conversation so far. So the
 * cache is keyed on a chained hash of the token sequence, in blocks, exactly like vLLM's
 * prefix cache: block k's key hashes (block k-1's key, its own tokens). Any request that
 * shares a leading run of blocks reuses them and prefills only the tail.
 *
 * MLA makes this cheap in a way normal attention does not: the cache entry is the shared
 * 576-value latent, not K and V per head. 25 KB/token at q4_0 instead of 1.2 MB.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#include "mla.h"

/* ---- KV quantization ----
 * The latent is dense, compressed information — every value carries signal, unlike a
 * normal K/V where much is redundant. So the quantization has to be validated, not
 * assumed; kv_test measures the attention-output error each format actually causes. */
enum { KV_F32 = 0, KV_F16 = 1, KV_Q8_0 = 2, KV_Q4_0 = 3 };

static const char *kv_fmt_name(int f){
    switch(f){ case KV_F32:return "fp32"; case KV_F16:return "fp16";
               case KV_Q8_0:return "q8_0"; case KV_Q4_0:return "q4_0"; default:return "?"; }
}
/* bytes for one token's [kv_lora + qk_rope] entry, in one layer */
static int64_t kv_entry_bytes(int fmt, int n){
    switch(fmt){
        case KV_F32:  return (int64_t)n*4;
        case KV_F16:  return (int64_t)n*2;
        case KV_Q8_0: return (int64_t)(n/32)*(2 + 32);        /* fp16 scale + 32 int8 */
        case KV_Q4_0: return (int64_t)(n/32)*(2 + 16);        /* fp16 scale + 32 nibbles */
        default: return 0;
    }
}

static inline uint16_t kv_f32_to_f16(float f){
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000;
    int32_t  e = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = b & 0x7FFFFF;
    if(e <= 0)  return (uint16_t)s;
    if(e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

/* Pack `n` floats (n % 32 == 0) into `dst`. Blocks of 32 with one fp16 absmax scale —
 * the same shape ggml uses, so the numbers are comparable to llama.cpp's KV quantization. */
static void kv_pack(void *dst, const float *src, int n, int fmt){
    if(fmt == KV_F32){ memcpy(dst, src, (size_t)n*4); return; }
    if(fmt == KV_F16){
        uint16_t *d = (uint16_t*)dst;
        for(int i=0;i<n;i++) d[i] = kv_f32_to_f16(src[i]);
        return;
    }
    uint8_t *d = (uint8_t*)dst;
    const int qmax = (fmt == KV_Q8_0) ? 127 : 7;
    const int per  = (fmt == KV_Q8_0) ? 34 : 18;
    for(int b = 0; b < n/32; b++){
        const float *x = src + b*32;
        float amax = 0;
        for(int i=0;i<32;i++){ float a = fabsf(x[i]); if(a > amax) amax = a; }
        const float s = amax / (float)qmax;
        const float inv = (s > 0) ? 1.0f/s : 0.0f;
        uint8_t *o = d + (size_t)b*per;
        uint16_t h = kv_f32_to_f16(s); memcpy(o, &h, 2);
        if(fmt == KV_Q8_0){
            for(int i=0;i<32;i++) o[2+i] = (uint8_t)(int8_t)lrintf(x[i]*inv);
        } else {
            for(int i=0;i<16;i++){
                int lo = (int)lrintf(x[i   ]*inv); if(lo> 7) lo= 7; if(lo<-8) lo=-8;
                int hi = (int)lrintf(x[i+16]*inv); if(hi> 7) hi= 7; if(hi<-8) hi=-8;
                o[2+i] = (uint8_t)(((lo + 8) & 0xF) | (((hi + 8) & 0xF) << 4));
            }
        }
    }
}

static void kv_unpack(float *dst, const void *src, int n, int fmt){
    if(fmt == KV_F32){ memcpy(dst, src, (size_t)n*4); return; }
    if(fmt == KV_F16){
        const uint16_t *s = (const uint16_t*)src;
        for(int i=0;i<n;i++) dst[i] = kq_half(s[i]);
        return;
    }
    const uint8_t *s = (const uint8_t*)src;
    const int per = (fmt == KV_Q8_0) ? 34 : 18;
    for(int b = 0; b < n/32; b++){
        const uint8_t *o = s + (size_t)b*per;
        uint16_t h; memcpy(&h, o, 2);
        const float sc = kq_half(h);
        float *y = dst + b*32;
        if(fmt == KV_Q8_0){
            for(int i=0;i<32;i++) y[i] = sc * (float)(int8_t)o[2+i];
        } else {
            for(int i=0;i<16;i++){
                y[i   ] = sc * (float)(((int)(o[2+i] & 0xF)) - 8);
                y[i+16] = sc * (float)(((int)(o[2+i] >>  4)) - 8);
            }
        }
    }
}

/* ================= persistent prefix cache ================= */

#define KVS_BLOCK   256          /* tokens per cache block: the granularity of a prefix match */
#define KVS_MAGIC   0x5356524Bu  /* "KRVS" */

typedef struct {
    uint64_t key;        /* chained hash: H(parent_key, this block's tokens) */
    uint64_t parent;     /* the block before it, 0 for the first */
    uint32_t n_tok;      /* <= KVS_BLOCK; a short final block is allowed */
    uint32_t heat;
    uint64_t off;        /* byte offset into the data file */
    uint64_t bytes;      /* size of this block's KV for ALL layers */
    uint64_t last_used;
} kvs_ent;

typedef struct {
    int      fd_data;
    char     dir[512];
    kvs_ent *e;
    int      n, cap;
    uint64_t used, budget;      /* bytes on disk */
    uint64_t clock;
    int      entry_bytes;       /* one token, one layer */
    int      n_layer, n_vals, fmt;
} kvstore;

/* FNV-1a over the block's tokens, chained onto the parent's key. Chaining is what makes a
 * match mean "the same tokens in the same order from position zero" — a block hash alone
 * would collide across different contexts that happen to share a window. */
static uint64_t kvs_hash(uint64_t parent, const int *tok, int n){
    uint64_t h = parent ? parent : 1469598103934665603ULL;
    for(int i=0;i<n;i++){
        h ^= (uint64_t)(uint32_t)tok[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1;              /* 0 is reserved for "no parent" */
}

static int kvs_open(kvstore *S, const char *dir, uint64_t budget_bytes,
                    int n_layer, int n_vals, int fmt){
    memset(S, 0, sizeof(*S));
    snprintf(S->dir, sizeof(S->dir), "%s", dir);
    S->budget      = budget_bytes;
    S->n_layer     = n_layer;
    S->n_vals      = n_vals;
    S->fmt         = fmt;
    S->entry_bytes = (int)kv_entry_bytes(fmt, n_vals);

    char p[600];
    snprintf(p, sizeof(p), "%s/kv.dat", dir);
    S->fd_data = open(p, O_RDWR | O_CREAT, 0644);
    if(S->fd_data < 0) return 0;

    S->cap = 4096;
    S->e = (kvs_ent*)calloc(S->cap, sizeof(kvs_ent));

    /* index */
    snprintf(p, sizeof(p), "%s/kv.idx", dir);
    FILE *f = fopen(p, "rb");
    if(f){
        uint32_t magic = 0, n = 0;
        if(fread(&magic,4,1,f)==1 && magic==KVS_MAGIC && fread(&n,4,1,f)==1){
            while((int)n > S->cap){ S->cap *= 2; S->e = realloc(S->e, (size_t)S->cap*sizeof(kvs_ent)); }
            if(fread(S->e, sizeof(kvs_ent), n, f) == n){
                S->n = (int)n;
                for(int i=0;i<S->n;i++){
                    S->used += S->e[i].bytes;
                    if(S->e[i].last_used > S->clock) S->clock = S->e[i].last_used;
                }
            }
        }
        fclose(f);
    }
    return 1;
}

static void kvs_flush(kvstore *S){
    char p[600];
    snprintf(p, sizeof(p), "%s/kv.idx", S->dir);
    FILE *f = fopen(p, "wb");
    if(!f) return;
    uint32_t magic = KVS_MAGIC, n = (uint32_t)S->n;
    fwrite(&magic,4,1,f); fwrite(&n,4,1,f);
    fwrite(S->e, sizeof(kvs_ent), S->n, f);
    fclose(f);
}

static kvs_ent *kvs_find(kvstore *S, uint64_t key){
    for(int i=0;i<S->n;i++) if(S->e[i].key == key) return &S->e[i];
    return NULL;
}

/* How many leading blocks of this prompt do we already have?
 * Returns the count and fills keys[] with each block's chained hash. */
static int kvs_match(kvstore *S, const int *tok, int n_tok, uint64_t *keys, int max_blk){
    uint64_t parent = 0;
    int hit = 0;
    for(int b = 0; b*KVS_BLOCK < n_tok && b < max_blk; b++){
        int n = n_tok - b*KVS_BLOCK;
        if(n > KVS_BLOCK) n = KVS_BLOCK;
        if(n < KVS_BLOCK) break;                    /* only whole blocks are cacheable */
        uint64_t k = kvs_hash(parent, tok + b*KVS_BLOCK, n);
        keys[b] = k;
        kvs_ent *e = kvs_find(S, k);
        if(!e) break;                               /* the chain ends here */
        e->heat++;
        e->last_used = ++S->clock;
        parent = k;
        hit = b + 1;
    }
    return hit;
}

/* Evict the coldest blocks until `need` bytes fit. Heat is persistent across runs, so a
 * prefix that a coding session hits every turn survives; a one-off paste does not. */
static void kvs_evict(kvstore *S, uint64_t need){
    while(S->used + need > S->budget && S->n > 0){
        int worst = 0;
        for(int i=1;i<S->n;i++){
            if(S->e[i].heat < S->e[worst].heat ||
              (S->e[i].heat == S->e[worst].heat && S->e[i].last_used < S->e[worst].last_used))
                worst = i;
        }
        S->used -= S->e[worst].bytes;
        S->e[worst] = S->e[S->n - 1];
        S->n--;
    }
}

/* Store one block. `kv` is [n_layer][n_tok][n_vals] float, already computed by prefill. */
static int kvs_put(kvstore *S, uint64_t key, uint64_t parent, const int *tok, int n_tok,
                   const float *kv){
    if(kvs_find(S, key)) return 1;                  /* already there */
    (void)tok;
    uint64_t bytes = (uint64_t)S->n_layer * n_tok * S->entry_bytes;
    if(bytes > S->budget) return 0;
    kvs_evict(S, bytes);

    uint8_t *buf = (uint8_t*)malloc(bytes);
    if(!buf) return 0;
    for(int l = 0; l < S->n_layer; l++)
        for(int t = 0; t < n_tok; t++)
            kv_pack(buf + ((size_t)l*n_tok + t)*S->entry_bytes,
                    kv + ((size_t)l*n_tok + t)*S->n_vals, S->n_vals, S->fmt);

    uint64_t off = S->used;                          /* append-only; eviction leaves holes we
                                                      * reuse only after a compaction, which
                                                      * is fine: the budget is the cap. */
    if(pwrite(S->fd_data, buf, bytes, off) != (ssize_t)bytes){ free(buf); return 0; }
    free(buf);

    if(S->n == S->cap){ S->cap *= 2; S->e = realloc(S->e, (size_t)S->cap*sizeof(kvs_ent)); }
    kvs_ent *e = &S->e[S->n++];
    e->key = key; e->parent = parent; e->n_tok = (uint32_t)n_tok;
    e->heat = 1; e->off = off; e->bytes = bytes; e->last_used = ++S->clock;
    S->used += bytes;
    return 1;
}

/* Load a cached block back into `kv` [n_layer][n_tok][n_vals] float. */
static int kvs_get(kvstore *S, const kvs_ent *e, float *kv){
    uint8_t *buf = (uint8_t*)malloc(e->bytes);
    if(!buf) return 0;
    if(pread(S->fd_data, buf, e->bytes, e->off) != (ssize_t)e->bytes){ free(buf); return 0; }
    for(int l = 0; l < S->n_layer; l++)
        for(uint32_t t = 0; t < e->n_tok; t++)
            kv_unpack(kv + ((size_t)l*e->n_tok + t)*S->n_vals,
                      buf + ((size_t)l*e->n_tok + t)*S->entry_bytes, S->n_vals, S->fmt);
    free(buf);
    return 1;
}

#endif /* COLIBRI_KVSTORE_H */
