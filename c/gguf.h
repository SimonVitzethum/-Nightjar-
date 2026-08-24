#ifndef COLIBRI_GGUF_H
#define COLIBRI_GGUF_H
/* gguf.h — GGUF v3 reader for streamed MoE inference.
 *
 * Scoped to what colibri actually needs, which is not what a general GGUF library
 * does. Two decisions drive the whole design:
 *
 * 1) DENSE TENSORS ARE MMAPPED, NEVER COPIED. The dense part (attention, shared
 *    experts, norms, embeddings) is resident for the whole run, so it wants to be
 *    page-cache backed: mmap gives that for free, costs no RSS the kernel cannot
 *    reclaim, and lets a second process share it. Nothing is dequantized at load —
 *    the k/i-quant bytes ARE the weights, and kquant.h contracts against them in
 *    place. Load time is therefore O(header), not O(model).
 *
 * 2) EXPERTS ARE NEVER MAPPED. In GGUF the 256 routed experts of a layer live in ONE
 *    3-D tensor, e.g.
 *        IQ2_XS  ne = [4096, 2048, 256]   blk.N.ffn_gate_exps.weight
 *    with ne[0] fastest, so expert e is a CONTIGUOUS slice of ne[0]*ne[1] weights at
 *    a computable offset. That is exactly the shape colibri's streaming wants: one
 *    pread per matrix per expert, no seeking, no parsing, no mmap of 90 GB we would
 *    only thrash. gguf_expert_slice() hands back that (file, offset, bytes) triple.
 *
 * Split shards: Unsloth ships shard 1 as metadata-only (0 tensors, all the KV and the
 * tokenizer) and the tensors in shards 2..N. We index every shard into one flat table.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "kquant.h"

#define GGUF_MAGIC   0x46554747u   /* "GGUF" */
#define GGUF_MAX_SHARD 64
#define GGUF_MAX_DIMS  4

/* GGUF metadata value types */
enum { GT_U8=0, GT_I8=1, GT_U16=2, GT_I16=3, GT_U32=4, GT_I32=5, GT_F32=6,
       GT_BOOL=7, GT_STR=8, GT_ARR=9, GT_U64=10, GT_I64=11, GT_F64=12 };

typedef struct {
    char     *key;
    uint32_t  type;        /* GT_* */
    uint64_t  u;           /* scalar payload (ints/bools) */
    double    f;           /* scalar payload (floats) */
    char     *s;           /* GT_STR payload */
    uint32_t  arr_type;    /* GT_ARR: element type */
    uint64_t  arr_n;       /* GT_ARR: element count */
    const uint8_t *arr;    /* GT_ARR: raw elements, in the mapped shard (non-string only) */
    char    **strs;        /* GT_ARR of GT_STR: the tokenizer's tokens / merges */
} gguf_kv;

typedef struct {
    char    *name;
    int      n_dims;
    int64_t  ne[GGUF_MAX_DIMS];
    int      type;         /* ggml type id; see kquant.h */
    uint64_t off;          /* ABSOLUTE byte offset in its shard file */
    int      shard;
    int64_t  bytes;        /* total size of the tensor payload */
} gguf_tensor;

typedef struct {
    int      fd;
    uint8_t *map;          /* mmap of the whole shard, or NULL if mapping was declined */
    size_t   size;
    uint64_t data_off;     /* where the tensor payload region begins */
    char     path[1024];
} gguf_shard;

typedef struct {
    gguf_shard  shard[GGUF_MAX_SHARD];
    int         n_shards;

    gguf_kv    *kv;
    int         n_kv;

    gguf_tensor *t;
    int          n_tensors;
    int         *hash;          /* open-addressed index into t[]; -1 empty */
    int          hash_mask;

    uint32_t     alignment;
} gguf_model;

/* ---------------- byte reader over a mapped shard ---------------- */
typedef struct { const uint8_t *p, *end; int bad; } gguf_rd;

static inline uint64_t gg_u64(gguf_rd *r){
    if(r->p + 8 > r->end){ r->bad = 1; return 0; }
    uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v;
}
static inline uint32_t gg_u32(gguf_rd *r){
    if(r->p + 4 > r->end){ r->bad = 1; return 0; }
    uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v;
}
/* GGUF strings are length-prefixed and NOT nul-terminated; we copy so callers can
 * treat them as C strings and so the model outlives any remap. */
static char *gg_str(gguf_rd *r){
    uint64_t n = gg_u64(r);
    if(r->bad || n > (uint64_t)(r->end - r->p) || n > (1u<<20)){ r->bad = 1; return NULL; }
    char *s = (char*)malloc(n + 1);
    if(!s){ r->bad = 1; return NULL; }
    memcpy(s, r->p, n); s[n] = 0; r->p += n;
    return s;
}
static inline int gg_scalar_size(uint32_t t){
    switch(t){ case GT_U8: case GT_I8: case GT_BOOL: return 1;
               case GT_U16: case GT_I16: return 2;
               case GT_U32: case GT_I32: case GT_F32: return 4;
               case GT_U64: case GT_I64: case GT_F64: return 8;
               default: return 0; }
}

/* ---------------- name -> tensor index ---------------- */
static uint64_t gg_hash(const char *s){
    uint64_t h = 1469598103934665603ULL;               /* FNV-1a */
    while(*s){ h ^= (uint8_t)*s++; h *= 1099511628211ULL; }
    return h;
}
static void gg_index(gguf_model *m){
    int cap = 16; while(cap < m->n_tensors * 2) cap <<= 1;
    m->hash_mask = cap - 1;
    m->hash = (int*)malloc((size_t)cap * sizeof(int));
    for(int i = 0; i < cap; i++) m->hash[i] = -1;
    for(int i = 0; i < m->n_tensors; i++){
        uint64_t h = gg_hash(m->t[i].name) & (uint64_t)m->hash_mask;
        while(m->hash[h] != -1) h = (h + 1) & (uint64_t)m->hash_mask;
        m->hash[h] = i;
    }
}
static const gguf_tensor *gguf_find(const gguf_model *m, const char *name){
    if(!m->hash) return NULL;
    uint64_t h = gg_hash(name) & (uint64_t)m->hash_mask;
    while(m->hash[h] != -1){
        const gguf_tensor *t = &m->t[m->hash[h]];
        if(strcmp(t->name, name) == 0) return t;
        h = (h + 1) & (uint64_t)m->hash_mask;
    }
    return NULL;
}

/* ---------------- metadata accessors ---------------- */
static const gguf_kv *gguf_kv_get(const gguf_model *m, const char *key){
    for(int i = 0; i < m->n_kv; i++) if(strcmp(m->kv[i].key, key) == 0) return &m->kv[i];
    return NULL;
}
static uint64_t gguf_u64(const gguf_model *m, const char *key, uint64_t dflt){
    const gguf_kv *k = gguf_kv_get(m, key); return k ? k->u : dflt;
}
static double gguf_f64(const gguf_model *m, const char *key, double dflt){
    const gguf_kv *k = gguf_kv_get(m, key); return k ? k->f : dflt;
}
static const char *gguf_str(const gguf_model *m, const char *key, const char *dflt){
    const gguf_kv *k = gguf_kv_get(m, key);
    return (k && k->type == GT_STR && k->s) ? k->s : dflt;
}
/* A string array (the tokenizer's tokens or merges). Returns the count and, via `out`,
 * the array itself — owned by the model, valid until gguf_close. */
static uint64_t gguf_strs(const gguf_model *m, const char *key, char ***out){
    const gguf_kv *k = gguf_kv_get(m, key);
    if(!k || k->type != GT_ARR || k->arr_type != GT_STR || !k->strs){ *out = NULL; return 0; }
    *out = k->strs;
    return k->arr_n;
}
/* An int array (token_type). */
static uint64_t gguf_i32s(const gguf_model *m, const char *key, const int32_t **out){
    const gguf_kv *k = gguf_kv_get(m, key);
    if(!k || k->type != GT_ARR || !k->arr ||
       (k->arr_type != GT_I32 && k->arr_type != GT_U32)){ *out = NULL; return 0; }
    *out = (const int32_t*)k->arr;
    return k->arr_n;
}

/* ---------------- one shard ---------------- */
static int gguf_load_shard(gguf_model *m, const char *path, int want_kv){
    if(m->n_shards >= GGUF_MAX_SHARD){ fprintf(stderr, "gguf: too many shards\n"); return 0; }
    gguf_shard *sh = &m->shard[m->n_shards];

    sh->fd = open(path, O_RDONLY);
    if(sh->fd < 0){ fprintf(stderr, "gguf: cannot open %s\n", path); return 0; }
    struct stat st;
    if(fstat(sh->fd, &st) != 0){ close(sh->fd); return 0; }
    sh->size = (size_t)st.st_size;
    snprintf(sh->path, sizeof(sh->path), "%s", path);

    /* Map the shard: the header walk needs random access to it, and the dense weights
     * will be served straight out of this mapping. MAP_NORESERVE + no MAP_POPULATE:
     * we want the page cache, not 90 GB of resident pages. */
    sh->map = (uint8_t*)mmap(NULL, sh->size, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, sh->fd, 0);
    if(sh->map == MAP_FAILED){
        fprintf(stderr, "gguf: mmap failed for %s\n", path);
        close(sh->fd); return 0;
    }
    /* The header is walked once, front to back; the expert region is not touched here
     * and must NOT be read ahead — that is what the pread path is for. */
    madvise(sh->map, sh->size, MADV_RANDOM);

    gguf_rd r = { sh->map, sh->map + sh->size, 0 };
    if(gg_u32(&r) != GGUF_MAGIC){ fprintf(stderr, "gguf: bad magic in %s\n", path); return 0; }
    uint32_t ver = gg_u32(&r);
    if(ver != 3){ fprintf(stderr, "gguf: version %u unsupported (need 3)\n", ver); return 0; }
    uint64_t n_tens = gg_u64(&r);
    uint64_t n_kv   = gg_u64(&r);

    /* KV block. Only shard 1 carries the real metadata; the rest repeat a stub, so we
     * keep the first shard's and skip past the others'. */
    int keep = want_kv && m->n_kv == 0;
    if(keep){
        m->kv = (gguf_kv*)calloc(n_kv, sizeof(gguf_kv));
        if(!m->kv) return 0;
    }
    for(uint64_t i = 0; i < n_kv && !r.bad; i++){
        char *key = gg_str(&r);
        uint32_t ty = gg_u32(&r);
        gguf_kv tmp; memset(&tmp, 0, sizeof(tmp));
        tmp.key = key; tmp.type = ty;

        if(ty == GT_STR){
            tmp.s = gg_str(&r);
        } else if(ty == GT_ARR){
            tmp.arr_type = gg_u32(&r);
            tmp.arr_n    = gg_u64(&r);
            if(tmp.arr_type == GT_STR){
                /* String arrays are the tokenizer: tokens (154880) and merges (321649).
                 * They are length-prefixed and NOT nul-terminated, so they cannot be handed
                 * out as C strings from the mapping — we materialize them once, here, and
                 * the vocab is built straight off this. Everything else is a pointer into
                 * the mmap; only this is copied, and only because the format forces it. */
                tmp.strs = (char**)calloc(tmp.arr_n ? tmp.arr_n : 1, sizeof(char*));
                if(!tmp.strs){ r.bad = 1; }
                else for(uint64_t j = 0; j < tmp.arr_n && !r.bad; j++)
                    tmp.strs[j] = gg_str(&r);
                tmp.arr = NULL;
            } else {
                int es = gg_scalar_size(tmp.arr_type);
                if(!es){ r.bad = 1; }
                else {
                    tmp.arr = r.p;                       /* points into the mapping; stays valid */
                    if((uint64_t)(r.end - r.p) < tmp.arr_n * (uint64_t)es) r.bad = 1;
                    else r.p += tmp.arr_n * (size_t)es;
                }
            }
        } else {
            int es = gg_scalar_size(ty);
            if(!es || r.p + es > r.end){ r.bad = 1; }
            else {
                uint64_t u = 0; double f = 0;
                switch(ty){
                    case GT_U8:  { uint8_t v;  memcpy(&v,r.p,1); u=v; break; }
                    case GT_I8:  { int8_t v;   memcpy(&v,r.p,1); u=(uint64_t)(int64_t)v; break; }
                    case GT_BOOL:{ uint8_t v;  memcpy(&v,r.p,1); u=v; break; }
                    case GT_U16: { uint16_t v; memcpy(&v,r.p,2); u=v; break; }
                    case GT_I16: { int16_t v;  memcpy(&v,r.p,2); u=(uint64_t)(int64_t)v; break; }
                    case GT_U32: { uint32_t v; memcpy(&v,r.p,4); u=v; break; }
                    case GT_I32: { int32_t v;  memcpy(&v,r.p,4); u=(uint64_t)(int64_t)v; break; }
                    case GT_F32: { float v;    memcpy(&v,r.p,4); f=v; u=(uint64_t)v; break; }
                    case GT_U64: { memcpy(&u,r.p,8); break; }
                    case GT_I64: { int64_t v;  memcpy(&v,r.p,8); u=(uint64_t)v; break; }
                    case GT_F64: { memcpy(&f,r.p,8); u=(uint64_t)f; break; }
                }
                tmp.u = u; tmp.f = f;
                r.p += es;
            }
        }
        if(keep) m->kv[m->n_kv++] = tmp;
        else { free(tmp.key); free(tmp.s); }
    }
    if(r.bad){ fprintf(stderr, "gguf: truncated metadata in %s\n", path); return 0; }

    /* Tensor directory. */
    if(n_tens){
        gguf_tensor *nt = (gguf_tensor*)realloc(m->t, (m->n_tensors + n_tens) * sizeof(gguf_tensor));
        if(!nt) return 0;
        m->t = nt;
    }
    int first = m->n_tensors;
    for(uint64_t i = 0; i < n_tens && !r.bad; i++){
        gguf_tensor t; memset(&t, 0, sizeof(t));
        t.name   = gg_str(&r);
        t.n_dims = (int)gg_u32(&r);
        if(t.n_dims < 1 || t.n_dims > GGUF_MAX_DIMS){ r.bad = 1; free(t.name); break; }
        int64_t n = 1;
        for(int d = 0; d < t.n_dims; d++){ t.ne[d] = (int64_t)gg_u64(&r); n *= t.ne[d]; }
        for(int d = t.n_dims; d < GGUF_MAX_DIMS; d++) t.ne[d] = 1;
        t.type  = (int)gg_u32(&r);
        t.off   = gg_u64(&r);                 /* relative to data_off, fixed up below */
        t.shard = m->n_shards;

        if(!kq_supported(t.type)){
            fprintf(stderr, "gguf: %s uses type %d, which colibri cannot decode\n",
                    t.name ? t.name : "?", t.type);
            r.bad = 1; free(t.name); break;
        }
        int bs = kq_blocksize(t.type);
        if(t.ne[0] % bs){
            fprintf(stderr, "gguf: %s row length %lld not a multiple of block %d\n",
                    t.name, (long long)t.ne[0], bs);
            r.bad = 1; free(t.name); break;
        }
        t.bytes = kq_row_bytes(t.type, n);
        m->t[m->n_tensors++] = t;
    }
    if(r.bad){ fprintf(stderr, "gguf: truncated tensor directory in %s\n", path); return 0; }

    /* Payload starts at the next `alignment` boundary after the directory. */
    uint32_t al = m->alignment ? m->alignment : 32;
    uint64_t pos = (uint64_t)(r.p - sh->map);
    sh->data_off = (pos + al - 1) & ~(uint64_t)(al - 1);

    for(int i = first; i < m->n_tensors; i++){
        m->t[i].off += sh->data_off;           /* now an absolute file offset */
        if(m->t[i].off + (uint64_t)m->t[i].bytes > sh->size){
            fprintf(stderr, "gguf: %s runs past the end of %s\n", m->t[i].name, path);
            return 0;
        }
    }
    m->n_shards++;
    return 1;
}

/* Open a split GGUF given ANY of its shards; discovers the rest by name.
 * Unsloth's layout is "<base>-00001-of-000NN.gguf". */
static int gguf_open(gguf_model *m, const char *path){
    memset(m, 0, sizeof(*m));
    m->alignment = 32;

    /* find the "-000i-of-000n" tail, if present */
    const char *base = path;
    char stem[1024]; int total = 1;
    const char *of = strstr(path, "-of-");
    if(of && of - path > 6){
        int idx = 0, n = 0;
        if(sscanf(of - 5, "%5d-of-%5d", &idx, &n) == 2 && n >= 1 && n <= GGUF_MAX_SHARD){
            total = n;
            size_t keep = (size_t)(of - 5 - path);
            if(keep >= sizeof(stem)) return 0;
            memcpy(stem, path, keep); stem[keep] = 0;
        } else { total = 1; }
    }
    if(total == 1){
        if(!gguf_load_shard(m, base, 1)) return 0;
    } else {
        for(int i = 1; i <= total; i++){
            char p[1200];
            snprintf(p, sizeof(p), "%s%05d-of-%05d.gguf", stem, i, total);
            if(!gguf_load_shard(m, p, 1)) return 0;
        }
    }
    gg_index(m);
    return 1;
}

static void gguf_close(gguf_model *m){
    for(int i = 0; i < m->n_shards; i++){
        if(m->shard[i].map) munmap(m->shard[i].map, m->shard[i].size);
        if(m->shard[i].fd >= 0) close(m->shard[i].fd);
    }
    for(int i = 0; i < m->n_kv; i++){
        free(m->kv[i].key); free(m->kv[i].s);
        if(m->kv[i].strs){
            for(uint64_t j = 0; j < m->kv[i].arr_n; j++) free(m->kv[i].strs[j]);
            free(m->kv[i].strs);
        }
    }
    for(int i = 0; i < m->n_tensors; i++) free(m->t[i].name);
    free(m->kv); free(m->t); free(m->hash);
    memset(m, 0, sizeof(*m));
}

/* Dense weights: a pointer straight into the mapping. No copy, no dequant. */
static const void *gguf_data(const gguf_model *m, const gguf_tensor *t){
    return m->shard[t->shard].map + t->off;
}

/* --------- the streaming entry point ---------
 * For a 3-D expert tensor [ne0, ne1, n_expert], hand back where expert `e` lives.
 * Contiguous by construction, so one pread lands it. Returns 0 if `t` is not an
 * expert tensor or `e` is out of range — callers must not silently read garbage. */
static int gguf_expert_slice(const gguf_model *m, const gguf_tensor *t, int e,
                             int *fd, uint64_t *off, int64_t *bytes,
                             int64_t *rows, int64_t *cols){
    if(t->n_dims != 3 || e < 0 || e >= t->ne[2]) return 0;
    int64_t per = t->ne[0] * t->ne[1];                 /* weights in one expert */
    int64_t sz  = kq_row_bytes(t->type, per);
    *fd    = m->shard[t->shard].fd;
    *off   = t->off + (uint64_t)e * (uint64_t)sz;
    *bytes = sz;
    if(rows) *rows = t->ne[1];                          /* output rows of this matrix */
    if(cols) *cols = t->ne[0];                          /* contraction length */
    return 1;
}

#endif /* COLIBRI_GGUF_H */
