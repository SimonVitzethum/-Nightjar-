#ifndef COLIBRI_ZEXP_H
#define COLIBRI_ZEXP_H
/* zexp.h — zstd container for streamed experts.
 *
 * Why this is worth doing at all: colibri is SSD-bound, not compute-bound. One token
 * pulls every routed expert off NVMe, so tok/s is (disk bandwidth)/(bytes per token) —
 * and the CPU sits idle ~95% of that time waiting. Bytes saved on disk convert straight
 * into tok/s, and the decode rides for free in the I/O shadow.
 *
 * Even on Q2_K (already 2.625 bpw) the k-quant symbol stream is not maximum-entropy:
 * the 4-bit scale/min nibbles and the 2-bit quants are both skewed. See zexp_test for
 * the ratio actually measured on ggml-quantized data.
 *
 * TILED on purpose. A single zstd frame over a 19 MB expert would force a serial decode
 * and give no random access. Fixed tiles instead:
 *   - each tile is an independent frame  -> OpenMP decodes them in parallel
 *   - decode target is the caller's slab -> the g/u/d QT views into it still hold,
 *     so nothing downstream in glm.c needs to know the expert was ever compressed
 *   - a tile is addressable              -> a future partial-expert read is possible
 *
 * Layout:  [zexp_hdr][zexp_tile x n_tiles][tile payloads, back to back]
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ZEXP_MAGIC   0x50584550u    /* "ZEXP" */
#define ZEXP_VERSION 1
#define ZEXP_TILE    (256*1024)     /* 256 KB: big enough for zstd to find matches,
                                     * small enough to keep ~76 tiles/expert busy on 20 threads */

typedef struct {
    uint32_t magic, version;
    uint64_t raw_bytes;             /* uncompressed slab size */
    uint32_t tile_size;             /* uncompressed bytes per tile (last may be shorter) */
    uint32_t n_tiles;
} zexp_hdr;

typedef struct {
    uint64_t off;                   /* payload offset, relative to end of the tile table */
    uint32_t csize;                 /* compressed bytes; == raw size means stored uncompressed */
    uint32_t rsize;                 /* uncompressed bytes of this tile */
} zexp_tile;

static inline uint64_t zexp_ntiles(uint64_t raw){ return (raw + ZEXP_TILE - 1) / ZEXP_TILE; }

/* Upper bound on the container size, for allocating the output buffer. */
static inline size_t zexp_bound(uint64_t raw){
    uint64_t nt = zexp_ntiles(raw);
    return sizeof(zexp_hdr) + nt * sizeof(zexp_tile) + ZSTD_compressBound(raw) + 64;
}

/* Pack `raw` into `out` (>= zexp_bound). Returns container bytes, or 0 on failure.
 * Offline path (the converter), so it is fine to be slow; level is the caller's call. */
static size_t zexp_pack(const void *raw, uint64_t raw_bytes, void *out, int level){
    uint32_t nt = (uint32_t)zexp_ntiles(raw_bytes);
    if(!nt) return 0;
    zexp_hdr *h = (zexp_hdr*)out;
    zexp_tile *tt = (zexp_tile*)((uint8_t*)out + sizeof(zexp_hdr));
    uint8_t *pay = (uint8_t*)(tt + nt);

    h->magic = ZEXP_MAGIC; h->version = ZEXP_VERSION;
    h->raw_bytes = raw_bytes; h->tile_size = ZEXP_TILE; h->n_tiles = nt;

    uint64_t off = 0;
    for(uint32_t i = 0; i < nt; i++){
        uint64_t o = (uint64_t)i * ZEXP_TILE;
        uint32_t rs = (uint32_t)((raw_bytes - o < ZEXP_TILE) ? (raw_bytes - o) : ZEXP_TILE);
        size_t cap = ZSTD_compressBound(rs);
        size_t cs = ZSTD_compress(pay + off, cap, (const uint8_t*)raw + o, rs, level);
        if(ZSTD_isError(cs)) return 0;
        /* incompressible tile: store it raw rather than pay zstd's framing to inflate it */
        if(cs >= rs){ memcpy(pay + off, (const uint8_t*)raw + o, rs); cs = rs; }
        tt[i].off = off; tt[i].csize = (uint32_t)cs; tt[i].rsize = rs;
        off += cs;
    }
    return (size_t)(pay - (uint8_t*)out) + off;
}

/* Decode a container straight into `dst` (must hold hdr->raw_bytes).
 * This is the hot path: called once per expert fetched from NVMe. Parallel over tiles.
 * Returns 1 on success. Any tile failing is fatal — a silently half-decoded expert
 * would be indistinguishable from a corrupt one downstream. */
static int zexp_unpack(const void *cont, size_t cont_bytes, void *dst, uint64_t dst_cap){
    if(cont_bytes < sizeof(zexp_hdr)) return 0;
    const zexp_hdr *h = (const zexp_hdr*)cont;
    if(h->magic != ZEXP_MAGIC || h->version != ZEXP_VERSION) return 0;
    if(h->raw_bytes > dst_cap) return 0;

    const zexp_tile *tt = (const zexp_tile*)((const uint8_t*)cont + sizeof(zexp_hdr));
    const uint8_t *pay = (const uint8_t*)(tt + h->n_tiles);
    if((size_t)(pay - (const uint8_t*)cont) > cont_bytes) return 0;

    int ok = 1;
    #pragma omp parallel for schedule(static) reduction(&:ok)
    for(int32_t i = 0; i < (int32_t)h->n_tiles; i++){
        uint64_t o = (uint64_t)i * h->tile_size;
        if(o + tt[i].rsize > h->raw_bytes){ ok = 0; continue; }
        if(tt[i].csize == tt[i].rsize){                      /* stored raw */
            memcpy((uint8_t*)dst + o, pay + tt[i].off, tt[i].rsize);
        } else {
            size_t r = ZSTD_decompress((uint8_t*)dst + o, tt[i].rsize,
                                       pay + tt[i].off, tt[i].csize);
            if(ZSTD_isError(r) || r != tt[i].rsize) ok = 0;
        }
    }
    return ok;
}

static inline uint64_t zexp_raw_bytes(const void *cont){
    const zexp_hdr *h = (const zexp_hdr*)cont;
    return (h->magic == ZEXP_MAGIC) ? h->raw_bytes : 0;
}

#endif /* COLIBRI_ZEXP_H */
