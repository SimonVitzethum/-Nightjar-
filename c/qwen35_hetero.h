#ifndef QWEN_QWEN35_HETERO_H
#define QWEN_QWEN35_HETERO_H
/* qwen35_hetero.h — the per-layer scheduler: which half of the model runs where.
 *
 * THE SPLIT, AND THE ONE NUMBER THAT DECIDES IT
 *
 * Per generated token the model reads 15.6 GiB of weights. The FFN is 9.65 of them and it
 * stays in RAM: it does not fit in 8 GiB of VRAM, and streaming it over this machine's x8
 * link at ~25 GB/s would be slower than reading it in place at 36. Everything else — gdn
 * 3.222, attn 0.889, output 0.971 — does fit, and once resident it is read at VRAM speed
 * and costs the DRAM controller nothing.
 *
 * That is the entire mechanism. Not overlap: the layer chain is strictly sequential inside a
 * token (attn -> +res -> ffn -> +res -> attn), so there is nothing to overlap. The saving is
 * bytes that stop being read on the slow side. Measured here: GPU 5.08 GiB at ~254 GB/s is
 * ~20 ms against the CPU's ~288 ms, so serializing the two costs ~7% and buys 37%.
 *
 * WHAT THE LEFTOVER VRAM DOES
 *
 * After the compute tensors and the 146 MiB of recurrent state there is ~1.8 GiB left. Two
 * candidates want it and they are worth EXACTLY the same: a KV window and a slice of FFN
 * layers both move ~1.8 GiB from the 36 GB/s side to the 254 GB/s side. So the tie is broken
 * on shape, not on speed — KV grows with context and FFN does not, so KV is served first up
 * to what the context actually needs and the FFN takes the remainder.
 *
 * SPLIT CONTEXT
 *
 * When the context outgrows the window, tokens [0, win) are attended on the GPU and
 * [win, T) on the CPU where those bytes already are, and the two partial results are merged
 * by the standard flash-decoding combine. Two vectors per layer cross PCIe instead of the
 * cache. It is exact up to rounding, which is what keeps the CPU comparison a hard gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "qwen35_cpu.h"
#include "qwen35_batch.h"
#include "qwen35_cuda.h"

typedef struct {
    void  *wqkv, *wgate, *w_alpha, *w_beta, *w_out;    /* gdn, quantized, device */
    float *conv1d, *ssm_a, *dt_bias, *ssm_norm;
    void  *wq, *wk, *wv, *wo;                          /* attn, quantized, device */
    float *q_norm, *k_norm;
    float *attn_norm, *post_attn_norm;
    void  *ffn_gate, *ffn_up, *ffn_down;               /* NULL => this layer's FFN is CPU-side */
    int    ffn_gpu;
} Q35CuLayer;

/* Per-stage timing costs a stream sync per stage: 192 a token instead of the 54 the schedule
 * actually needs, and each one is ~15 us of the CPU spinning instead of running the FFN. It
 * is how you find out where the time goes, so it stays -- behind a switch. */
static int g_q35_stage_timing = 0;
#define MOE_MAXG 16   /* n_expert_used + shared, with headroom */

typedef struct {
    const Q35Model *M;
    Q35CuLayer  L[Q35_MAX_LAYER];
    void       *output;
    float      *output_norm;

    /* work, device */
    float *x, *xn, *t, *qkv, *z, *ab, *o, *qg, *k, *v;
    float *lse, *o2, *lse2, *scratch, *logits, *ffn_g, *ffn_u;
    float *S, *conv;

    /* the device KV window: tokens [0, win) of every attention layer */
    uint8_t *Kc, *Vc;
    int      win, kv_fmt, kv_layers;
    int64_t  kv_half;          /* bytes of one token's K (or V) in one layer */

    /* pinned staging for the FFN hand-off */
    float *hxn, *ht, *hqg, *ho, *hlse;

    int64_t vram_weights, vram_state, vram_kv, vram_ffn, vram_work;
    int     n_ffn_gpu;
    int     ok;

    void   *upstage;        /* pinned staging for the weight upload; startup scratch only */
    size_t  upstage_n;

    /* ---- the streamed FFN ----
     * For layers whose weights did not fit in VRAM, the GPU still does a share of the work
     * by pulling its slice over PCIe WHILE the CPU computes the rest in place. The two read
     * the same DRAM, so the win is not free -- but the CPU reaches 38 GB/s and the DMA
     * engine adds 27.7 against a 64.8 GB/s ceiling, and the sides run at the same time.
     * Measured split point is therefore near 42% to the GPU; `stream_f` tracks it live,
     * because the balance moves with context length and with how busy the copy engine is. */
    uint8_t *stage;              /* device landing zone for one layer's slice */
    int64_t  stage_bytes;
    float   *sg, *su, *st;       /* device: gate/up partials and the output */
    float   *hg, *hu, *ht2;      /* pinned host: the CPU's partials */
    /* TWO split fractions, because decode and prefill are bound by different things.
     *
     * Decode reads each weight once and uses it once: both sides move bytes, the CPU at
     * ~38 GB/s and the DMA at 27.7, and the balance lands near 0.60.
     *
     * Prefill reads each weight once and uses it S times. Measured, the weights stream at
     * only 12.5 GB/s during prefill against 59 in decode — the CPU has stopped being
     * bandwidth-bound and is doing 8x the arithmetic per byte, while the GPU side is still
     * only a DMA and costs what it always did. At 0.60 that leaves the CPU the straggler by
     * 3.7x. Balancing PCIe time against CPU compute time puts prefill near 0.85. */
    double   stream_f, stream_f_b;
    int      stream_batch;       /* which of the two is in force right now */
    struct { int64_t gb, db; int r, s_unused; } slice[2];
    int      stage_half;
    double   t_cpu_side, t_gpu_side;
    int64_t  stream_bytes;
    int      stream_on, host_pinned;

    /* ---- the hetero MoE expert cache (qwen35moe / Ornith) ----
     * The routed experts are ~18 GiB and do not fit in VRAM. The hottest are cached on the card
     * and run there at VRAM speed; a miss is streamed over PCIe into a slot (admit-on-use, LRU
     * eviction) and also run on the GPU; once a per-layer PCIe budget is spent the remaining
     * misses are computed on the CPU IN PARALLEL with the GPU queue, so RAM and VRAM bandwidth
     * are both busy. moe_map/moe_slot are the cache directory; the three device arrays hold the
     * slots' gate/up/down; the shared expert of every layer is pinned resident. */
    int      moe;                          /* 1 for a MoE model */
    int      moe_nslot;                    /* VRAM cache capacity, in experts */
    int      moe_tg, moe_tu, moe_td;       /* expert gate/up/down quant types */
    int64_t  moe_gb, moe_ub, moe_db;       /* bytes of one expert's gate/up/down slice */
    uint8_t *moe_gpool, *moe_upool, *moe_dpool;   /* device: nslot * gb/ub/db */
    int     *moe_slot_gid;                 /* [nslot] global expert id in each slot, -1 empty */
    uint32_t*moe_slot_clk;                 /* [nslot] last-use tick (LRU) */
    int     *moe_map;                      /* [n_layer*n_expert] -> slot, or -1 */
    uint32_t moe_clk;                      /* global LRU tick */
    int      moe_budget;                   /* max experts streamed to the GPU per layer */
    float   *moe_wg, *moe_wu, *moe_wd;     /* device work: gate/up hidden [F], down out [D] */
    float   *moe_acc_dev;                  /* device: the CPU tier's partial, after h2d */
    void    *moe_shg[Q35_MAX_LAYER], *moe_shu[Q35_MAX_LAYER], *moe_shd[Q35_MAX_LAYER]; /* shared expert, resident */
    uint64_t moe_hit, moe_miss, moe_cpu, moe_res;   /* counters, for the report */
    /* Whole expert tensors kept permanently in VRAM for some layers, so those layers never
     * stream and — crucially — are EXCLUDED from RAM residency, which is what lets the ~13 GiB
     * of remaining experts fit in RAM and be pinned. NULL => this layer streams/caches. */
    void    *moe_rexp_g[Q35_MAX_LAYER], *moe_rexp_u[Q35_MAX_LAYER], *moe_rexp_d[Q35_MAX_LAYER];
    int      moe_nresident;
    /* grouped decode: one kernel for all experts. gg/uu are [MAXG*F] hidden, odg is [MAXG*D]
     * output; pg/pu/pd are DEVICE arrays of the experts' weight pointers, wdev their weights;
     * hpX and hwt are the pinned host staging that feeds those (persistent, so the async copy is
     * safe across the per-layer sync). */
    float   *moe_gg, *moe_uu, *moe_odg, *moe_wdev;
    void    *moe_pg, *moe_pu, *moe_pd;
    void   **moe_hpg, **moe_hpu, **moe_hpd; float *moe_hwt;
    /* router on the GPU: per-layer weight (E experts + the shared gate as row E) uploaded to
     * VRAM, so decode does the 524K-MAC route on the card instead of on one idle CPU thread. */
    void    *moe_router_dev[Q35_MAX_LAYER];
    float   *moe_logits_dev, *moe_logits_host;
} Q35Cu;

/* ---------------- upload ---------------- */

/* THE UPLOAD READS THE FILE ITSELF INSTEAD OF COPYING OUT OF THE MMAP.
 *
 * The first version was cudaMemcpy(dev, mmapped_source, bytes) once per tensor. Every 4 KiB of
 * that source is a page fault taken inside the copy, so the disk read, the fault and the DMA
 * are serialized on one thread with no overlap and a request size the disk considers small.
 * Measured: 5.085 GiB in 20.2 s = 0.25 GB/s, against 6.27 GB/s of disk and 6.8 GB/s of
 * pageable PCIe.
 *
 * Prefaulting the pages first — in parallel, with fadvise and a touch pass — got it to
 * 6.3 s + 3.7 s. Better, and still wrong, because touching pages is not reading a file: it
 * asks for 128 KiB at a time and waits.
 *
 * Big requests are half of it. The other half is threads, and I got that wrong once already:
 * the O_DIRECT benchmark says 19 MiB preads give 6.27 GB/s on one thread and the same at 16,
 * so I concluded the disk needs size and not concurrency. That holds for O_DIRECT. It does not
 * hold for BUFFERED reads, which is what this is — every read also fills the page cache, and
 * that path blocks per thread. The evidence was in this engine's own startup log: the parallel
 * residency pass gets 3.9 GB/s and this serial upload got 0.79, same disk, same second.
 *
 * So: fill a large pinned buffer with several 19 MiB preads IN PARALLEL, then hand the whole
 * filled span to one DMA. Size for the requests, threads for the buffering, and the DMA (27.7
 * GB/s pinned) is fast enough to disappear between fills.
 *
 * A tensor already living in anonymous RAM (the FFN layers the GPU shares) is copied from
 * there, not re-read: q35_wdata knows the difference. */
static int q35cu_stage_upload(Q35Cu *G, const Q35Model *M, const gguf_tensor *T, void *d){
    const void *host = q35_wdata(M, T);
    const gguf_shard *sh = &M->m.shard[T->shard];
    const int from_file = (host == (const void*)(sh->map ? sh->map + T->off : NULL))
                          && sh->fd >= 0 && G->upstage && G->upstage_n > 0;
    if(!from_file) return q35cu_h2d(d, host, (size_t)T->bytes);

    const int64_t REQ  = 19LL<<20;                  /* what the disk was measured to like */
    const int64_t SPAN = (int64_t)G->upstage_n;     /* what one DMA carries */
    for(int64_t base = 0; base < T->bytes; base += SPAN){
        const int64_t span = (T->bytes - base < SPAN) ? T->bytes - base : SPAN;
        const int nreq = (int)((span + REQ - 1)/REQ);
        int bad = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(|:bad)
        for(int k = 0; k < nreq; k++){
            const int64_t o   = (int64_t)k*REQ;
            const int64_t len = (span - o < REQ) ? span - o : REQ;
            int64_t done = 0;
            while(done < len){
                const ssize_t r = pread(sh->fd, (char*)G->upstage + o + done,
                                        (size_t)(len - done), (off_t)(T->off + base + o + done));
                if(r <= 0) break;
                done += r;
            }
            if(done != len) bad |= 1;
        }
        if(bad) return q35cu_h2d(d, host, (size_t)T->bytes);       /* fall back whole */
        if(!q35cu_h2d((char*)d + base, G->upstage, (size_t)span)) return 0;
        /* AND THEN DROP IT FROM THE PAGE CACHE.
         *
         * These bytes now live in VRAM. A copy in the page cache is not a cache of anything —
         * nothing will read the file again this run — and it is 5 GiB of pressure that has to
         * come from somewhere. Where it came from, measured, was the FFN: the residency pass
         * runs next, reads 9.65 GiB, and evicts exactly the tensors the upload just pulled in.
         * Next start those are cold again, which is why the upload sat at 6.2 s no matter how
         * the reads were shaped, while residency read the FFN warm at 4 GB/s. The two halves
         * of startup were evicting each other in a loop. */
        posix_fadvise(sh->fd, (off_t)(T->off + base), (off_t)span, POSIX_FADV_DONTNEED);
    }
    return 1;
}

static void *q35cu_up_t(Q35Cu *G, const Q35Model *M, const gguf_tensor *T, int64_t *acc){
    if(!T) return NULL;
    if(!q35cu_type_ok(T->type)){
        fprintf(stderr, "qwen35_cuda: %s has type %s, which has no device kernel\n",
                T->name, kq_name(T->type));
        G->ok = 0; return NULL;
    }
    void *d = q35cu_alloc((size_t)T->bytes);
    if(!d){
        fprintf(stderr, "qwen35_cuda: out of VRAM uploading %s (%.1f MiB): %s\n",
                T->name, T->bytes/1048576.0, q35cu_error());
        G->ok = 0; return NULL;
    }
    if(!q35cu_stage_upload(G, M, T, d)){
        fprintf(stderr, "qwen35_cuda: upload of %s failed: %s\n", T->name, q35cu_error());
        G->ok = 0; return NULL;
    }
    *acc += T->bytes;
    return d;
}

static float *q35cu_up_f(Q35Cu *G, const float *src, int n, int64_t *acc){
    if(!src) return NULL;
    float *d = (float*)q35cu_alloc((size_t)n*4);
    if(!d){ G->ok = 0; return NULL; }
    if(!q35cu_h2d(d, src, (size_t)n*4)){ G->ok = 0; return NULL; }
    *acc += (int64_t)n*4;
    return d;
}

static float *q35cu_work(Q35Cu *G, int64_t n){
    float *d = (float*)q35cu_alloc((size_t)n*4);
    if(!d){ fprintf(stderr, "qwen35_cuda: work buffer (%lld floats): %s\n",
                    (long long)n, q35cu_error()); G->ok = 0; return NULL; }
    G->vram_work += n*4;
    return d;
}

/* How many tokens of KV window fit in `budget` bytes. */
static int q35cu_win_tokens(const Q35Cu *G, int64_t budget){
    const int64_t per = (int64_t)G->kv_layers * G->kv_half * 2;   /* K and V */
    if(per <= 0) return 0;
    int64_t n = budget/per;
    if(n < 0) n = 0;
    return (int)(n > (int64_t)1<<30 ? (int64_t)1<<30 : n);
}

/* ---------------- init ----------------
 *
 * Order matters: the compute tensors are mandatory and go first, so a capacity failure
 * happens here and loudly rather than as a mystery slowdown three layers into a token. */
static int q35cu_moe_setup(Q35Cu *G);   /* hetero MoE expert cache; defined below */
static int q35cu_model_init(Q35Cu *G, const Q35Model *M, int n_ctx, int kv_fmt){
    memset(G, 0, sizeof *G);
    G->M = M; G->ok = 1; G->kv_fmt = kv_fmt;
    const Q35Cfg *c = &M->c;

    if(!q35cu_init(0)){
        fprintf(stderr, "qwen35_cuda: %s\n", q35cu_error());
        return 0;
    }
    size_t vfree = 0, vtot = 0;
    q35cu_mem(&vfree, &vtot);
    if(q35cu_log_on()) fprintf(stderr, "  [kernel] device        %s, %.2f GiB free\n",
                               q35cu_name(), vfree/1073741824.0);

    /* 19 MiB is not a round number, it is the request size this project measured the disk to
     * like. Pinned, so the DMA out of it runs at the bus rate rather than through the driver's
     * own staging copy. Freed as soon as the upload is done — it is startup scratch. */
    G->upstage_n = 152u<<20;   /* 8 x 19 MiB: eight requests in flight, one DMA */
    G->upstage   = q35cu_host_alloc(G->upstage_n);
    if(!G->upstage) G->upstage_n = 0;      /* not fatal: q35cu_stage_upload falls back */

    int64_t w = 0;
    for(int il = 0; il < c->n_layer && G->ok; il++){
        const Q35Layer *S = &M->L[il];
        Q35CuLayer *D = &G->L[il];
        D->attn_norm      = q35cu_up_f(G, S->attn_norm,      c->d_model, &w);
        D->post_attn_norm = q35cu_up_f(G, S->post_attn_norm, c->d_model, &w);
        if(S->kind == Q35_LAYER_ATTN){
            D->wq = q35cu_up_t(G, M, S->wq, &w);
            D->wk = q35cu_up_t(G, M, S->wk, &w);
            D->wv = q35cu_up_t(G, M, S->wv, &w);
            D->wo = q35cu_up_t(G, M, S->wo, &w);
            D->q_norm = q35cu_up_f(G, S->q_norm, c->d_head, &w);
            D->k_norm = q35cu_up_f(G, S->k_norm, c->d_head, &w);
        } else {
            D->wqkv    = q35cu_up_t(G, M, S->wqkv,    &w);
            D->wgate   = q35cu_up_t(G, M, S->wgate,   &w);
            D->w_alpha = q35cu_up_t(G, M, S->w_alpha, &w);
            D->w_beta  = q35cu_up_t(G, M, S->w_beta,  &w);
            D->w_out   = q35cu_up_t(G, M, S->w_out,   &w);
            D->conv1d   = q35cu_up_f(G, S->conv1d,  c->d_conv*c->conv_dim, &w);
            D->ssm_a    = q35cu_up_f(G, S->ssm_a,   c->n_v_heads, &w);
            D->dt_bias  = q35cu_up_f(G, S->dt_bias, c->n_v_heads, &w);
            D->ssm_norm = q35cu_up_f(G, S->ssm_norm, c->d_head_v, &w);
        }
    }
    G->output      = q35cu_up_t(G, M, M->output, &w);
    G->output_norm = q35cu_up_f(G, M->output_norm, c->d_model, &w);
    G->vram_weights = w;
    if(G->upstage){ q35cu_host_free(G->upstage); G->upstage = NULL; G->upstage_n = 0; }
    if(!G->ok) return 0;

    /* recurrent state: 48 layers x 48 heads x 64 KiB, constant in context length */
    int n_gdn = 0, n_attn = 0;
    for(int il = 0; il < c->n_layer; il++)
        (c->kind[il] == Q35_LAYER_ATTN) ? n_attn++ : n_gdn++;
    const int64_t s_n = (int64_t)n_gdn*c->n_v_heads*c->d_state*c->d_head_v;
    const int64_t c_n = (int64_t)n_gdn*(c->d_conv-1)*c->conv_dim;
    G->S    = (float*)q35cu_alloc((size_t)s_n*4);
    G->conv = (float*)q35cu_alloc((size_t)c_n*4);
    if(!G->S || !G->conv){ fprintf(stderr, "qwen35_cuda: gdn state: %s\n", q35cu_error()); return 0; }
    q35cu_zero(G->S, (size_t)s_n*4);
    q35cu_zero(G->conv, (size_t)c_n*4);
    G->vram_state = (s_n + c_n)*4;

    /* work buffers */
    const int nh = c->n_head, dh = c->d_head;
    G->x   = q35cu_work(G, c->d_model);
    G->xn  = q35cu_work(G, c->d_model);
    G->t   = q35cu_work(G, c->d_model);
    G->qkv = q35cu_work(G, c->conv_dim);
    G->z   = q35cu_work(G, c->d_inner);
    G->ab  = q35cu_work(G, 2*c->n_v_heads);
    G->o   = q35cu_work(G, nh*dh > c->d_inner ? nh*dh : c->d_inner);
    G->qg  = q35cu_work(G, 2*nh*dh);
    G->k   = q35cu_work(G, c->n_head_kv*dh);
    G->v   = q35cu_work(G, c->n_head_kv*dh);
    G->lse = q35cu_work(G, nh);
    G->o2  = q35cu_work(G, nh*dh);
    G->lse2= q35cu_work(G, nh);
    G->scratch = q35cu_work(G, (int64_t)q35cu_attn_scratch(nh, dh, n_ctx));
    G->logits  = q35cu_work(G, c->vocab);
    if(!G->ok) return 0;

    /* the KV window, then whatever is left goes to FFN layers */
    G->kv_layers = n_attn + (c->n_layer_mtp ? 1 : 0);
    G->kv_half   = q35_kv_entry_bytes(kv_fmt, c->n_head_kv*c->d_head);
    q35cu_mem(&vfree, &vtot);
    int64_t spare = (int64_t)vfree - (256<<20);        /* leave headroom for the allocator */
    if(spare < 0) spare = 0;

    /* The window is a CACHE, not a context limit: anything past it is attended on the CPU
     * out of the host tier and merged back with the flash combine (see §7.4). Capping it
     * explicitly is the knob that trades KV window against resident FFN layers — they are
     * worth the same per byte, so which one wins depends on how long the context actually
     * gets. */
    const int64_t kv_need = (int64_t)n_ctx * G->kv_layers * G->kv_half * 2;
    int64_t kv_budget = kv_need < spare ? kv_need : spare;
    G->win = q35cu_win_tokens(G, kv_budget);
    if(G->win > n_ctx) G->win = n_ctx;
    /* DEFAULT CAP, and the arithmetic behind it.
     *
     * A resident FFN layer costs 165 MiB and saves ~2.6 ms of every token, forever. The same
     * 165 MiB as KV window holds ~4978 tokens, and those only save anything once the context
     * has actually grown past the window — everything beyond it is attended on the CPU at
     * 36 GB/s, which is 2.6 ms per 2688 tokens. Solve the two against each other and a
     * 32768-token window does not overtake an 8192-token one until the context reaches
     * ~35k; at 8192 the breakeven is ~24k. So the window is capped low by default and the
     * VRAM goes to FFN layers, which pay from the first token. Raise it with
     * QWEN_CUDA_KV_TOK when the sessions really are that long. */
    { const char *we = getenv("QWEN_CUDA_KV_TOK");
      int cap = we ? atoi(we) : 8192;
      if(cap >= 0 && cap < G->win) G->win = cap; }
    if(G->win > 0){
        const size_t nb = (size_t)G->kv_layers*(size_t)G->win*(size_t)G->kv_half;
        G->Kc = (uint8_t*)q35cu_alloc(nb);
        G->Vc = (uint8_t*)q35cu_alloc(nb);
        if(!G->Kc || !G->Vc){ G->win = 0; q35cu_free(G->Kc); q35cu_free(G->Vc); G->Kc = G->Vc = NULL; }
        else G->vram_kv = 2*(int64_t)nb;
    }

    /* The streaming staging area, sized for the largest single-layer slice we would ever
     * ask for. Capped at half the remaining VRAM: a landing zone big enough to starve the
     * resident FFN of a layer is a net loss. */
    {
        const char *se = getenv("QWEN_CUDA_STREAM");
        G->stream_f = se ? atof(se) : 0.60;
        if(G->stream_f > 0.9) G->stream_f = 0.9;
        { const char *pe = getenv("QWEN_CUDA_STREAM_PREFILL");
          G->stream_f_b = pe ? atof(pe) : 0.85;
          if(G->stream_f_b > 0.92) G->stream_f_b = 0.92;
          if(G->stream_f_b < G->stream_f) G->stream_f_b = G->stream_f; }
        if(!c->is_moe && G->stream_f > 0){   /* MoE has no dense FFN to stream; experts stay in RAM */
            q35cu_mem(&vfree, &vtot);
            int64_t cap = (int64_t)vfree - (160<<20);
            int64_t need = 0;
            for(int il = 0; il < c->n_layer; il++){
                const Q35Layer *S = &M->L[il];
                const int64_t g = q35_tbytes(S->ffn_gate), u = q35_tbytes(S->ffn_up),
                              d = q35_tbytes(S->ffn_down);
                const double fmx = G->stream_f_b > G->stream_f ? G->stream_f_b : G->stream_f;
                const int64_t t = 2*(int64_t)(fmx*1.05*(double)(g + u + d));
                if(t > need) need = t;
            }
            if(need > cap) need = cap;
            /* Each half must be 256-byte aligned: k_gemv_q4k_r pulls d/dmin and the twelve
             * scale bytes as 32-bit loads, and an odd half-offset makes every one of them a
             * misaligned access. CUDA reports that as "misaligned address" from the NEXT
             * stream sync, several layers away from the cause, and the logits come back all
             * zero — which reads like a scheduling bug and is an alignment bug. */
            need &= ~(int64_t)511;
            if(need > (16<<20)){
                G->stage = (uint8_t*)q35cu_alloc((size_t)need);
                G->sg  = (float*)q35cu_alloc((size_t)c->d_ff*4);
                G->su  = (float*)q35cu_alloc((size_t)c->d_ff*4);
                G->st  = (float*)q35cu_alloc((size_t)c->d_model*4);
                G->hg  = (float*)q35cu_host_alloc((size_t)c->d_ff*4);
                G->hu  = (float*)q35cu_host_alloc((size_t)c->d_ff*4);
                G->ht2 = (float*)q35cu_host_alloc((size_t)c->d_model*4);
                if(G->stage && G->sg && G->su && G->st && G->hg && G->hu && G->ht2){
                    G->stage_bytes = need; G->stream_on = 1;
                } else {
                    q35cu_free(G->stage); G->stage = NULL;
                }
            }
        }
    }
    /* FFN layers into the remainder. Whole layers, not row slices: a split gate/up/down needs
     * the down projection cut along its CONTRACTED dim, which is a strided slice of every row
     * and buys nothing here, because the two sides cannot overlap anyway. */
    q35cu_mem(&vfree, &vtot);
    spare = (int64_t)vfree - (320<<20);
    const char *fe = getenv("QWEN_CUDA_FFN_GB");
    if(fe){ const double gb = atof(fe); const int64_t cap = (int64_t)(gb*1073741824.0);
            if(cap < spare) spare = cap; }
    if(!c->is_moe && spare > 0){   /* dense FFN layers into VRAM; a MoE model keeps every expert on the CPU (B1) */
        int64_t need_ffn = 0;
        G->ffn_g = q35cu_work(G, c->d_ff);
        G->ffn_u = q35cu_work(G, c->d_ff);
        spare -= 2*(int64_t)c->d_ff*4;
        for(int il = 0; il < c->n_layer; il++){
            const Q35Layer *S = &M->L[il];
            const int64_t need = q35_tbytes(S->ffn_gate) + q35_tbytes(S->ffn_up) + q35_tbytes(S->ffn_down);
            if(need > spare) break;
            int64_t got = 0;
            Q35CuLayer *D = &G->L[il];
            D->ffn_gate = q35cu_up_t(G, M, S->ffn_gate, &got);
            D->ffn_up   = q35cu_up_t(G, M, S->ffn_up,   &got);
            D->ffn_down = q35cu_up_t(G, M, S->ffn_down, &got);
            if(!G->ok || !D->ffn_gate || !D->ffn_up || !D->ffn_down){
                q35cu_free(D->ffn_gate); q35cu_free(D->ffn_up); q35cu_free(D->ffn_down);
                D->ffn_gate = D->ffn_up = D->ffn_down = NULL;
                G->ok = 1;                        /* not fatal: the CPU keeps this layer */
                break;
            }
            D->ffn_gpu = 1; G->n_ffn_gpu++; spare -= got; need_ffn += got;
        }
        G->vram_ffn = need_ffn;
    }

    /* pinned staging: the FFN hand-off and, at split context, the attention partials */
    G->hxn  = (float*)q35cu_host_alloc((size_t)c->d_model*4);
    G->ht   = (float*)q35cu_host_alloc((size_t)c->d_model*4);
    G->hqg  = (float*)q35cu_host_alloc((size_t)2*nh*dh*4);
    G->ho   = (float*)q35cu_host_alloc((size_t)nh*dh*4);
    G->hlse = (float*)q35cu_host_alloc((size_t)nh*4);
    if(!G->hxn || !G->ht || !G->hqg || !G->ho || !G->hlse){
        fprintf(stderr, "qwen35_cuda: pinned host staging failed\n"); return 0;
    }
    if(&M->c && M->c.is_moe && !q35cu_moe_setup(G)){
        fprintf(stderr, "qwen35moe: expert-cache setup failed\n"); return 0; }
    if(!q35cu_sync()){ fprintf(stderr, "qwen35_cuda: %s\n", q35cu_error()); return 0; }
    return G->ok;
}

/* The weights must be PINNED or the DMA runs at 6.8 GB/s instead of 27.7 and the whole
 * split is a loss. q35_reside_anon puts them all in one anonymous mapping, so this is one
 * call -- and it must happen after residency, not before. */
static void q35cu_pin_weights(Q35Cu *G, Q35Model *M){
    if((!G->stream_on && !G->moe) || !M->anon || M->anon_bytes <= 0) return;
    if(q35cu_host_register(M->anon, (size_t)M->anon_bytes)) G->host_pinned = 1;
    else {
        fprintf(stderr, "qwen35_cuda: cannot pin weights (%s) — streaming disabled,"
                        " pageable DMA would be slower than the CPU\n", q35cu_error());
        G->stream_on = 0;
    }
}

static void q35cu_model_free(Q35Cu *G){
    const Q35Cfg *c = &G->M->c;
    for(int il = 0; il < c->n_layer; il++){
        Q35CuLayer *D = &G->L[il];
        void *p[] = { D->wqkv,D->wgate,D->w_alpha,D->w_beta,D->w_out,D->conv1d,D->ssm_a,
                      D->dt_bias,D->ssm_norm,D->wq,D->wk,D->wv,D->wo,D->q_norm,D->k_norm,
                      D->attn_norm,D->post_attn_norm,D->ffn_gate,D->ffn_up,D->ffn_down };
        for(unsigned i = 0; i < sizeof p/sizeof *p; i++) q35cu_free(p[i]);
    }
    void *p[] = { G->output,G->output_norm,G->x,G->xn,G->t,G->qkv,G->z,G->ab,G->o,G->qg,
                  G->k,G->v,G->lse,G->o2,G->lse2,G->scratch,G->logits,G->ffn_g,G->ffn_u,
                  G->S,G->conv,G->Kc,G->Vc,G->stage,G->sg,G->su,G->st };
    for(unsigned i = 0; i < sizeof p/sizeof *p; i++) q35cu_free(p[i]);
    q35cu_host_free(G->hxn); q35cu_host_free(G->ht);
    q35cu_host_free(G->hqg); q35cu_host_free(G->ho); q35cu_host_free(G->hlse);
    q35cu_host_free(G->hg); q35cu_host_free(G->hu); q35cu_host_free(G->ht2);
    memset(G, 0, sizeof *G);
}

/* The device window ends on a host-tier chunk boundary. Otherwise one chunk is half on the
 * GPU and half in RAM, and the CPU's half-pass has to skip inside a chunk it just faulted in
 * whole — correct, but it reads bytes nobody uses. */
static void q35cu_align_win(Q35Cu *G, int chunk){
    if(chunk > 0 && G->win > chunk) G->win -= G->win % chunk;
}

/* Tell q35_reside_anon which FFN tensors are already in VRAM, so it does not pull a second
 * copy into RAM. Call between q35cu_model_init and q35_reside_anon. */
static void q35cu_mark_resident_elsewhere(Q35Cu *G){
    const Q35Model *M = G->M;
    static uint8_t *flag = NULL;
    free(flag);
    flag = (uint8_t*)calloc((size_t)M->m.n_tensors, 1);
    if(!flag) return;
    for(int il = 0; il < M->c.n_layer; il++){
        const gguf_tensor *g[3] = {0,0,0};
        if(G->L[il].ffn_gpu){ g[0]=M->L[il].ffn_gate; g[1]=M->L[il].ffn_up; g[2]=M->L[il].ffn_down; }
        else if(G->moe && G->moe_rexp_g[il]){                 /* whole expert tensors are in VRAM */
            g[0]=M->L[il].ffn_gate_exps; g[1]=M->L[il].ffn_up_exps; g[2]=M->L[il].ffn_down_exps; }
        for(int k = 0; k < 3; k++){
            if(!g[k]) continue;
            const int i = (int)(g[k] - M->m.t);
            if(i >= 0 && i < M->m.n_tensors) flag[i] = 1;
        }
    }
    g_q35_elsewhere = flag;
}

static void q35cu_state_reset(Q35Cu *G){
    const Q35Cfg *c = &G->M->c;
    int n_gdn = 0;
    for(int il = 0; il < c->n_layer; il++) if(c->kind[il] != Q35_LAYER_ATTN) n_gdn++;
    q35cu_zero(G->S,    (size_t)n_gdn*c->n_v_heads*c->d_state*c->d_head_v*4);
    q35cu_zero(G->conv, (size_t)n_gdn*(c->d_conv-1)*c->conv_dim*4);
    q35cu_sync();
}

typedef struct { int64_t gb, db; int r, s_unused; } Q35Slice;
static Q35Slice q35cu_slice(const Q35Cu *G, const Q35Layer *L, double f);

static void q35cu_report(const Q35Cu *G, FILE *o){
    size_t f = 0, t = 0; q35cu_mem(&f, &t);
    fprintf(o, "\n  device: %s\n", q35cu_name());
    fprintf(o, "    weights %6.3f GiB   state %6.3f   kv window %6.3f (%d tok)   ffn %6.3f (%d layers)   work %6.3f\n",
            G->vram_weights/1073741824.0, G->vram_state/1073741824.0,
            G->vram_kv/1073741824.0, G->win, G->vram_ffn/1073741824.0, G->n_ffn_gpu,
            G->vram_work/1073741824.0);
    fprintf(o, "    vram %.2f used / %.2f GiB, %.2f free\n",
            (t-f)/1073741824.0, t/1073741824.0, f/1073741824.0);
    if(G->stream_on){
        /* Print BOTH fractions and the slice they actually produce. The staging area caps the
         * slice, so a fraction that does not fit is silently reduced — and a split that did
         * not happen looks exactly like a split that did not help. */
        const Q35Layer *L0 = NULL;
        for(int il = 0; il < G->M->c.n_layer; il++) if(!G->L[il].ffn_gpu){ L0 = &G->M->L[il]; break; }
        int rd = 0, rp = 0;
        if(L0){ rd = q35cu_slice(G, L0, G->stream_f).r; rp = q35cu_slice(G, L0, G->stream_f_b).r; }
        fprintf(o, "    streamed ffn: decode %.0f%% (r=%d), prefill %.0f%% (r=%d),"
                   " %.0f MiB staging%s\n",
                100*G->stream_f, rd, 100*G->stream_f_b, rp, G->stage_bytes/1048576.0,
                G->host_pinned ? ", weights pinned" : ", NOT PINNED (slow)");
    }
}

/* ---------------- the CPU's half of a split context ----------------
 *
 * Computes attention over tokens [t_lo, t_hi) straight out of the host tier — where those
 * bytes already are — and returns the partial in flash-decoding form: the normalized output
 * and the log-sum-exp of its scores. The GPU's half arrives the same way and the two combine
 * exactly. This is what makes the KV window a CACHE rather than a context limit. */
static void q35_attn_host_partial(Q35State *R, const float *qg, int slot,
                                  int t_lo, int t_hi, float *o, float *lse){
    const Q35Cfg *c = &R->M->c;
    const int dh = c->d_head, nh = c->n_head, nkv = c->n_head_kv, grp = nh/nkv;
    const float scale = 1.0f/sqrtf((float)dh);
    KvTier *KV = &R->kv;
    const int64_t hb = q35_kv_entry_bytes(KV->fmt, dh);
    const int c0 = t_lo/KV->chunk, c1 = (t_hi + KV->chunk - 1)/KV->chunk;

    for(int h = 0; h < nh; h++) lse[h] = -INFINITY;
    memset(o, 0, sizeof(float)*(size_t)nh*dh);
    if(t_hi <= t_lo) return;

    for(int ci = c0; ci < c1; ci++){
        kvt_prefetch_window(KV, slot, ci, 1, c1);
        const uint8_t *blk = kvt_chunk(KV, slot, ci);
        if(!blk) continue;
        int a = ci*KV->chunk, b = a + KV->chunk;
        if(a < t_lo) a = t_lo;
        if(b > t_hi) b = t_hi;
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++){
            const float *q = qg + (size_t)h*2*dh;
            float *sc = R->att + (size_t)h*R->n_ctx;
            const int hk = h/grp;
            for(int t = a; t < b; t++){
                const uint8_t *kt = blk + kvt_koff(KV, t - ci*KV->chunk) + hk*hb;
                sc[t] = kvt_dot(KV, q, kt, dh)*scale;
            }
        }
    }
    /* one pass for the max, then an unnormalized weighted sum; the caller's merge divides */
    #pragma omp parallel for schedule(static)
    for(int h = 0; h < nh; h++){
        const float *sc = R->att + (size_t)h*R->n_ctx;
        float m = -INFINITY;
        for(int t = t_lo; t < t_hi; t++) if(sc[t] > m) m = sc[t];
        lse[h] = m;
    }
    for(int ci = c0; ci < c1; ci++){
        kvt_prefetch_window(KV, slot, ci, 1, c1);
        const uint8_t *blk = kvt_chunk(KV, slot, ci);
        if(!blk) continue;
        int a = ci*KV->chunk, b = a + KV->chunk;
        if(a < t_lo) a = t_lo;
        if(b > t_hi) b = t_hi;
        #pragma omp parallel for schedule(static)
        for(int h = 0; h < nh; h++){
            const float *sc = R->att + (size_t)h*R->n_ctx;
            const int hk = h/grp;
            float *oh = o + (size_t)h*dh;
            const float m = lse[h];
            for(int t = a; t < b; t++){
                const float p = expf(sc[t] - m);
                const uint8_t *vt = blk + kvt_voff(KV, t - ci*KV->chunk) + hk*hb;
                kvt_axpy(KV, oh, p, vt, dh);
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for(int h = 0; h < nh; h++){
        const float *sc = R->att + (size_t)h*R->n_ctx;
        const float m = lse[h];
        double s = 0;
        for(int t = t_lo; t < t_hi; t++) s += exp((double)(sc[t] - m));
        float *oh = o + (size_t)h*dh;
        const float inv = s > 0 ? (float)(1.0/s) : 0.f;
        for(int i = 0; i < dh; i++) oh[i] *= inv;
        lse[h] = m + (float)log(s > 0 ? s : 1.0);
    }
}

/* ---------------- layers ---------------- */

/* ---------------- the split FFN: CPU in place, GPU over PCIe, at the same time ----------
 *
 * The layer is cut so that NEITHER SIDE NEEDS THE OTHER'S RESULT until the very end.
 *
 * gate and up are [d_model, d_ff] with the output index slowest, so rows [0,r) are a
 * contiguous byte range -- the GPU takes those, the CPU takes [r, d_ff).
 *
 * down is [d_ff, d_model] and the obvious cut, by output row, does NOT work: every output
 * row needs the WHOLE swiglu vector, so each side would have to wait for the other's half
 * and the pipeline collapses into two barriers. Cutting it along the CONTRACTION instead --
 * the GPU sums i in [0,r), the CPU sums i in [r,d_ff) -- means each side uses exactly the
 * part of the swiglu vector it computed itself. The two partial sums are added at the end.
 * That costs one strided DMA (a column band of every row, cudaMemcpy2DAsync) and buys the
 * only true concurrency in the whole engine.
 *
 * r is a multiple of 256 because that is a k-quant super-block: cutting inside one would
 * mean decoding a block from the middle, which the format does not allow. */

static Q35Slice q35cu_slice(const Q35Cu *G, const Q35Layer *L, double f){
    const Q35Cfg *c = &G->M->c;
    Q35Slice S; memset(&S, 0, sizeof S);
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, c->d_model);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, c->d_ff);
    int r = (int)(f*(double)c->d_ff);
    r -= r % KQ_QK_K;                                 /* whole super-blocks only */
    for(;;){
        if(r < KQ_QK_K){ r = 0; break; }
        S.gb = (int64_t)r*grb;
        S.db = (int64_t)(r/KQ_QK_K)*kq_typesize(L->ffn_down->type)*(int64_t)c->d_model;
        if(2*S.gb + S.db <= G->stage_bytes/2) break;   /* double buffered */
        r -= KQ_QK_K;
    }
    S.r = r;
    if(r == 0){ S.gb = S.db = 0; }
    return S;
}

/* Issue one layer's slice onto the copy stream. Called a layer AHEAD of where it is used,
 * so the 2.6 ms of DMA sits underneath the previous layer's compute instead of in front of
 * this one's. */
static void q35cu_stream_issue(Q35Cu *G, int il, int half){
    uint8_t *dst = G->stage + (int64_t)half*(G->stage_bytes/2);
    q35cu_copy_join(half);        /* do not overwrite a half whose gemms are still running */
    Q35Slice *S = (Q35Slice*)&G->slice[half];
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35Layer *L = &M->L[il];
    *S = q35cu_slice(G, L, G->stream_batch ? G->stream_f_b : G->stream_f);
    if(!S->r){ q35cu_copy_mark(half); return; }
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, c->d_model);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, c->d_ff);
    const int64_t dw  = (int64_t)(S->r/KQ_QK_K)*kq_typesize(L->ffn_down->type);
    q35cu_h2d_async(dst,            q35_wdata(M, L->ffn_gate), (size_t)S->gb);
    q35cu_h2d_async(dst + S->gb,    q35_wdata(M, L->ffn_up),   (size_t)S->gb);
    q35cu_h2d_2d_async(dst + 2*S->gb, (size_t)dw,
                       q35_wdata(M, L->ffn_down), (size_t)drb,
                       (size_t)dw, (size_t)c->d_model);
    G->stream_bytes += 2*S->gb + S->db;
    q35cu_copy_mark(half);
}

/* The next layer whose FFN is not already resident in VRAM, or -1. */
static int q35cu_next_stream_layer(const Q35Cu *G, int from){
    for(int il = from; il < G->M->c.n_layer; il++) if(!G->L[il].ffn_gpu) return il;
    return -1;
}

static void q35cu_ffn_cpu_half(Q35Cu *G, const Q35Layer *L, const float *xn, float *out, int r);

/* One layer, both sides at once. Returns 0 if there was no slice and the caller should run
 * the plain CPU FFN. */
static int q35cu_ffn_split(Q35Cu *G, int il, const float *xn_dev, float *out_dev, int half){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35Layer *L = &M->L[il];
    const int r = G->slice[half].r;
    if(r <= 0) return 0;
    const int64_t gb = G->slice[half].gb;
    uint8_t *base = G->stage + (int64_t)half*(G->stage_bytes/2);
    const int Dm = c->d_model;

    q35cu_sync();
    q35cu_d2h(G->hxn, xn_dev, sizeof(float)*Dm);

    /* queue the GPU half behind its DMA -- this does not block the host */
    const double tg = q35_clk();
    q35cu_stream_join(half);
    q35cu_gemv(G->sg, xn_dev, base,            L->ffn_gate->type, Dm, r);
    q35cu_gemv(G->su, xn_dev, base + gb,       L->ffn_up->type,   Dm, r);
    q35cu_swiglu(G->sg, G->su, r);
    q35cu_gemv(out_dev, G->sg, base + 2*gb,    L->ffn_down->type, r, Dm);
    q35cu_compute_mark(half);     /* this half is free once these have run */

    /* the CPU half runs NOW, against the GPU and against the next layer's DMA */
    const double tc = q35_clk();
    q35cu_ffn_cpu_half(G, L, G->hxn, G->ht2, r);
    G->t_cpu_side += q35_clk() - tc;

    q35cu_h2d(G->st, G->ht2, sizeof(float)*Dm);
    q35cu_add(out_dev, G->st, Dm);
    G->t_gpu_side += q35_clk() - tg;
    return 1;
}

/* The CPU's half: rows [r, d_ff) of gate and up, then the tail of the contraction in down. */
static void q35cu_ffn_cpu_half(Q35Cu *G, const Q35Layer *L, const float *xn, float *out, int r){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const int Dff = c->d_ff, Dm = c->d_model;
    const int64_t grb = kq_row_bytes(L->ffn_gate->type, Dm);
    const int64_t drb = kq_row_bytes(L->ffn_down->type, Dff);
    const int nrow = Dff - r;
    kq_gemv_fast(G->hg + r, xn, (const uint8_t*)q35_wdata(M, L->ffn_gate) + (int64_t)r*grb,
                 L->ffn_gate->type, Dm, nrow);
    kq_gemv_fast(G->hu + r, xn, (const uint8_t*)q35_wdata(M, L->ffn_up) + (int64_t)r*grb,
                 L->ffn_up->type, Dm, nrow);
    for(int i = r; i < Dff; i++) G->hg[i] = q35_silu(G->hg[i])*G->hu[i];
    const uint8_t *Wd = (const uint8_t*)q35_wdata(M, L->ffn_down);
    const int64_t skip = (int64_t)(r/KQ_QK_K)*kq_typesize(L->ffn_down->type);
    /* kq_dot_fast, NOT kq_dot: the latter is the scalar reference at 0.71 GB/s per core
     * against AVX2's 2.66. Using it here made the CPU half of a 76% share cost 0.383 s where
     * the WHOLE FFN costs 0.246 — a 4x regression that looks exactly like "the split does
     * not work" and is not. */
    #pragma omp parallel for schedule(static)
    for(int o = 0; o < Dm; o++)
        out[o] = kq_dot_fast(L->ffn_down->type, Wd + (int64_t)o*drb + skip, G->hg + r, nrow);
}



static void q35cu_gdn_layer_x(Q35Cu *G, int il, int slot, const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *S = &M->L[il];
    const int nv = c->n_v_heads, nk = c->n_k_heads, dk = c->d_state, dv = c->d_head_v;
    const int kdim = dk*nk;

    q35cu_gemv(G->qkv, xn, D->wqkv,    S->wqkv->type,    c->d_model, c->conv_dim); q35cu_mark("wqkv");
    q35cu_gemv(G->z,   xn, D->wgate,   S->wgate->type,   c->d_model, c->d_inner); q35cu_mark("wgate");
    q35cu_gemv(G->ab,        xn, D->w_beta,  S->w_beta->type,  c->d_model, nv);
    q35cu_gemv(G->ab + nv,   xn, D->w_alpha, S->w_alpha->type, c->d_model, nv); q35cu_mark("w_alpha+beta");
    q35cu_gdn_gates(G->ab, G->ab + nv, D->dt_bias, D->ssm_a, nv); q35cu_mark("gates");

    q35cu_gdn_conv(G->qkv, G->conv + (size_t)slot*(c->d_conv-1)*c->conv_dim,
                   D->conv1d, c->conv_dim, c->d_conv); q35cu_mark("conv1d");
    q35cu_gdn_l2norm(G->qkv, 2*nk, dk, c->eps);          /* q and k, NOT rmsnorm */ q35cu_mark("l2norm");
    q35cu_gdn_delta(G->S + (size_t)slot*nv*dk*dv, G->o,
                    G->qkv, G->qkv + kdim, G->qkv + 2*kdim,
                    G->ab, G->ab + nv, nv, nk, dk, dv, g_q35_gdn_tile); q35cu_mark("delta");
    q35cu_gdn_norm_gate(G->o, G->z, D->ssm_norm, nv, dv, c->eps); q35cu_mark("norm_gate");
    q35cu_gemv(out, G->o, D->w_out, S->w_out->type, c->d_inner, c->d_model); q35cu_mark("w_out");
}

static void q35cu_attn_layer_x(Q35Cu *G, Q35State *R, int il, int slot, int pos,
                               const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *S = &M->L[il];
    const int nh = c->n_head, nkv = c->n_head_kv, dh = c->d_head;
    const int n_kv = nkv*dh, T = pos + 1;
    const float scale = 1.0f/sqrtf((float)dh);

    q35cu_gemv(G->qg, xn, D->wq, S->wq->type, c->d_model, 2*nh*dh); q35cu_mark("wq");
    q35cu_gemv(G->k,  xn, D->wk, S->wk->type, c->d_model, n_kv);
    q35cu_gemv(G->v,  xn, D->wv, S->wv->type, c->d_model, n_kv); q35cu_mark("wk+wv");
    q35cu_attn_qk(G->qg, G->k, D->q_norm, D->k_norm, nh, nkv, dh,
                  c->n_rot, c->rope_base, pos, c->eps); q35cu_mark("qknorm+rope");

    const int64_t lane = (int64_t)slot*G->win*G->kv_half;
    if(pos < G->win){
        q35cu_kv_store(G->Kc, G->Vc, G->k, G->v,
                       lane + (int64_t)pos*G->kv_half, lane + (int64_t)pos*G->kv_half,
                       n_kv, G->kv_fmt);
    } else {
        /* past the window: this token's KV belongs to the host tier */
        q35cu_sync();
        q35cu_d2h(R->k, G->k, sizeof(float)*n_kv);
        q35cu_d2h(R->v, G->v, sizeof(float)*n_kv);
        kvt_put(&R->kv, slot, pos, R->k, R->v);
    }

    const int T_gpu = T < G->win ? T : G->win;
    if(T_gpu > 0)
        q35cu_attn_flash(G->o, G->lse, G->qg, G->Kc + lane, G->Vc + lane,
                         nh, nkv, dh, T_gpu, G->kv_half, G->kv_half,
                         scale, G->kv_fmt, G->scratch); q35cu_mark("flash");

    if(T > G->win){
        q35cu_sync();
        q35cu_d2h(G->hqg, G->qg, sizeof(float)*2*nh*dh);
        q35_attn_host_partial(R, G->hqg, slot, G->win, T, G->ho, G->hlse);
        q35cu_h2d(G->o2,   G->ho,   sizeof(float)*nh*dh);
        q35cu_h2d(G->lse2, G->hlse, sizeof(float)*nh);
        if(T_gpu > 0) q35cu_lse_merge(G->o, G->lse, G->o2, G->lse2, nh, dh);
        else          q35cu_h2d(G->o, G->ho, sizeof(float)*nh*dh);
    }

    q35cu_attn_gate(G->o, G->qg, nh, dh); q35cu_mark("gate");
    q35cu_gemv(out, G->o, D->wo, S->wo->type, (int)S->wo->ne[0], c->d_model); q35cu_mark("wo");
}

/* ---- hetero MoE: build the VRAM expert cache ----
 * Called once, after the attention/GDN weights are on the card, for a MoE model. Uploads every
 * layer's shared expert (used each token, so always resident — 40 * ~1.9 MiB), then carves the
 * rest of free VRAM into as many expert-sized cache slots as fit. */
static int q35cu_moe_setup(Q35Cu *G){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    if(!c->is_moe) return 1;
    G->moe = 1;
    const Q35Layer *L0 = &M->L[0];
    const int D = c->d_model, F = c->d_ff_exp;
    /* Q4_K_M is not uniform: the down projection is Q6_K in some layers and Q4_K in others, so
     * the slot stride is the LARGEST any layer needs; each copy writes only that layer's bytes. */
    (void)L0;
    G->moe_gb = G->moe_ub = G->moe_db = 0;
    for(int il = 0; il < c->n_layer; il++){
        const Q35Layer *L = &M->L[il];
        int64_t gb = (int64_t)F * kq_row_bytes(L->ffn_gate_exps->type, D);
        int64_t ub = (int64_t)F * kq_row_bytes(L->ffn_up_exps->type,   D);
        int64_t db = (int64_t)D * kq_row_bytes(L->ffn_down_exps->type, F);
        if(gb > G->moe_gb) G->moe_gb = gb;
        if(ub > G->moe_ub) G->moe_ub = ub;
        if(db > G->moe_db) G->moe_db = db;
    }

    /* per-token work buffers */
    G->moe_wg = q35cu_work(G, F); G->moe_wu = q35cu_work(G, F); G->moe_wd = q35cu_work(G, D);
    G->moe_acc_dev = q35cu_work(G, D);
    G->moe_gg   = q35cu_work(G, MOE_MAXG*F);
    G->moe_uu   = q35cu_work(G, MOE_MAXG*F);
    G->moe_odg  = q35cu_work(G, MOE_MAXG*D);
    G->moe_wdev = q35cu_work(G, MOE_MAXG);
    G->moe_pg = q35cu_alloc(MOE_MAXG*sizeof(void*));
    G->moe_pu = q35cu_alloc(MOE_MAXG*sizeof(void*));
    G->moe_pd = q35cu_alloc(MOE_MAXG*sizeof(void*));
    G->moe_hpg = (void**)q35cu_host_alloc(MOE_MAXG*sizeof(void*));
    G->moe_hpu = (void**)q35cu_host_alloc(MOE_MAXG*sizeof(void*));
    G->moe_hpd = (void**)q35cu_host_alloc(MOE_MAXG*sizeof(void*));
    G->moe_hwt = (float*)q35cu_host_alloc(MOE_MAXG*sizeof(float));
    if(!G->moe_wg || !G->moe_wu || !G->moe_wd || !G->moe_acc_dev ||
       !G->moe_gg || !G->moe_uu || !G->moe_odg || !G->moe_wdev ||
       !G->moe_pg || !G->moe_pu || !G->moe_pd || !G->moe_hpg || !G->moe_hpu || !G->moe_hpd || !G->moe_hwt){
        G->ok = 0; return 0; }

    /* shared experts, one per layer, always resident */
    for(int il = 0; il < c->n_layer; il++){
        const Q35Layer *L = &M->L[il];
        int64_t acc = 0;
        G->moe_shg[il] = q35cu_up_t(G, M, L->ffn_gate_shexp, &acc);
        G->moe_shu[il] = q35cu_up_t(G, M, L->ffn_up_shexp,   &acc);
        G->moe_shd[il] = q35cu_up_t(G, M, L->ffn_down_shexp, &acc);
        if(!G->ok) return 0;
    }

    /* GPU router: [E experts ; shared gate] as (E+1) F32 rows, per layer */
    { const int Ec = c->n_expert;
      G->moe_logits_dev  = q35cu_work(G, Ec+1);
      G->moe_logits_host = (float*)q35cu_host_alloc((size_t)(Ec+1)*sizeof(float));
      if(!G->moe_logits_dev || !G->moe_logits_host){ G->ok = 0; return 0; }
      for(int il = 0; il < c->n_layer; il++){
          const Q35Layer *L = &M->L[il];
          G->moe_router_dev[il] = q35cu_work(G, (int64_t)(Ec+1)*D);
          if(!G->moe_router_dev[il]){ G->ok = 0; return 0; }
          q35cu_h2d((float*)G->moe_router_dev[il], (const float*)q35_wdata(M, L->ffn_gate_inp),
                    (size_t)Ec*D*sizeof(float));
          q35cu_h2d((float*)G->moe_router_dev[il] + (size_t)Ec*D, L->ffn_gate_inp_shexp,
                    (size_t)D*sizeof(float));
      } }

    /* RESIDENT EXPERT LAYERS: whole gate/up/down tensors kept permanently in VRAM for the
     * first few layers. Those layers never stream — but the real reason is that they are then
     * EXCLUDED from RAM residency (q35cu_mark_resident_elsewhere + q35_reside_anon). Dropping
     * ~5 GiB from the residency target is what lets the remaining ~13 GiB of experts fit in RAM
     * and be pinned, so the streamed tier for the OTHER layers runs at ~27 GB/s instead of the
     * ~10 GB/s pageable copy. QWEN_MOE_RESIDENT_GB overrides the ~5 GiB default (0 disables). */
    {
        size_t rvf=0, rvt=0; q35cu_mem(&rvf, &rvt);
        double rgb = 0.0;   /* opt-in: the big dynamic cache beats fixed layers when VRAM is the bottleneck */
        { const char *re = getenv("QWEN_MOE_RESIDENT_GB"); if(re) rgb = atof(re); }
        int64_t rbudget = (int64_t)(rgb*1073741824.0);
        const int64_t keep = 1200LL<<20;                 /* leave VRAM for the dynamic cache + work */
        if(rbudget > (int64_t)rvf - keep) rbudget = (int64_t)rvf - keep;
        int64_t used = 0;
        for(int il = 0; il < c->n_layer && G->ok && rbudget > 0; il++){
            const Q35Layer *L = &M->L[il];
            const int64_t need = L->ffn_gate_exps->bytes + L->ffn_up_exps->bytes + L->ffn_down_exps->bytes;
            if(used + need > rbudget) break;
            int64_t acc = 0;
            G->moe_rexp_g[il] = q35cu_up_t(G, M, L->ffn_gate_exps, &acc);
            G->moe_rexp_u[il] = q35cu_up_t(G, M, L->ffn_up_exps,   &acc);
            G->moe_rexp_d[il] = q35cu_up_t(G, M, L->ffn_down_exps, &acc);
            if(!G->ok || !G->moe_rexp_g[il] || !G->moe_rexp_u[il] || !G->moe_rexp_d[il]){
                q35cu_free(G->moe_rexp_g[il]); q35cu_free(G->moe_rexp_u[il]); q35cu_free(G->moe_rexp_d[il]);
                G->moe_rexp_g[il] = G->moe_rexp_u[il] = G->moe_rexp_d[il] = NULL; G->ok = 1; break;
            }
            used += acc; G->moe_nresident++;
        }
        if(G->moe_nresident)
            fprintf(stderr, "  moe resident: %d whole layers in VRAM (%.2f GiB), excluded from RAM\n",
                    G->moe_nresident, used/1073741824.0);
    }

    /* the cache: whatever VRAM is left, minus a margin, split into expert-sized slots */
    int64_t vfree = 0, vtot = 0; q35cu_mem(&vfree, &vtot);
    int64_t budget = vfree - (320<<20);
    { const char *ce = getenv("QWEN_MOE_CACHE_GB");
      if(ce){ int64_t cap = (int64_t)(atof(ce)*1073741824.0); if(cap < budget) budget = cap; } }
    const int64_t per = G->moe_gb + G->moe_ub + G->moe_db;
    int nslot = per > 0 ? (int)(budget / per) : 0;
    const int total = c->n_layer * c->n_expert;
    if(nslot > total) nslot = total;
    if(nslot < 1){ fprintf(stderr, "qwen35moe: no VRAM for an expert cache — experts on CPU\n"); G->moe_nslot = 0; }
    else {
        G->moe_gpool = (uint8_t*)q35cu_alloc((size_t)nslot*G->moe_gb);
        G->moe_upool = (uint8_t*)q35cu_alloc((size_t)nslot*G->moe_ub);
        G->moe_dpool = (uint8_t*)q35cu_alloc((size_t)nslot*G->moe_db);
        if(!G->moe_gpool || !G->moe_upool || !G->moe_dpool){
            fprintf(stderr, "qwen35moe: expert cache alloc failed — experts on CPU\n");
            q35cu_free(G->moe_gpool); q35cu_free(G->moe_upool); q35cu_free(G->moe_dpool);
            G->moe_gpool = G->moe_upool = G->moe_dpool = NULL; nslot = 0;
        }
    }
    G->moe_nslot = nslot;
    G->moe_slot_gid = (int*)malloc(sizeof(int)*(nslot>0?nslot:1));
    G->moe_slot_clk = (uint32_t*)malloc(sizeof(uint32_t)*(nslot>0?nslot:1));
    for(int i = 0; i < nslot; i++){ G->moe_slot_gid[i] = -1; G->moe_slot_clk[i] = 0; }
    G->moe_map = (int*)malloc(sizeof(int)*(size_t)total);
    for(int i = 0; i < total; i++) G->moe_map[i] = -1;
    G->moe_clk = 1;
    { const char *be = getenv("QWEN_MOE_STREAM"); G->moe_budget = be ? atoi(be) : c->n_expert_used; }
    fprintf(stderr, "  moe cache: %d slots x %.2f MiB = %.2f GiB (%.0f%% of %d experts), stream<=%d/layer\n",
            nslot, per/1048576.0, (double)nslot*per/1073741824.0,
            100.0*nslot/(double)total, total, G->moe_budget);
    return G->ok;
}

/* Compute one expert on the GPU from device weight pointers and accumulate w*expert(x) into
 * out (device). The three gemvs and the swiglu ride the compute stream, so successive calls
 * queue without a host sync — and the host is free to run CPU experts meanwhile. */
static void q35cu_moe_expert_gpu(Q35Cu *G, const float *xn, int D, int F,
                                 const void *g, int tg, const void *u, int tu,
                                 const void *d, int td, float w, float *out){
    q35cu_gemv(G->moe_wg, xn, g, tg, D, F);
    q35cu_gemv(G->moe_wu, xn, u, tu, D, F);
    q35cu_swiglu(G->moe_wg, G->moe_wu, F);
    q35cu_gemv(G->moe_wd, G->moe_wg, d, td, F, D);
    q35cu_scale_add(out, G->moe_wd, w, D);
}

/* Pick a cache slot for a miss: an empty one, else the least-recently-used. O(nslot), which is
 * dwarfed by the gemvs it precedes. */
static int q35cu_moe_evict(Q35Cu *G){
    int best = 0; uint32_t oldest = 0xffffffffu;
    for(int i = 0; i < G->moe_nslot; i++){
        if(G->moe_slot_gid[i] < 0) return i;
        if(G->moe_slot_clk[i] < oldest){ oldest = G->moe_slot_clk[i]; best = i; }
    }
    return best;
}

/* ---- hetero MoE decode: route on the host, then ONE grouped kernel per stage ----
 * The eight routed experts + the shared one used to be ~27 tiny gemv launches; now the routed
 * ones are gathered by pointer and computed in a single grouped gate/up/swiglu/down/reduce, so
 * the GPU runs them concurrently instead of one starved warp-block at a time. This is the fix
 * for the 94% the FFN was costing. Types are per layer (Q4_K_M mixes them). The shared expert
 * keeps its own single call because its down type can differ from the routed experts' by layer. */
static void q35cu_moe_layer(Q35Cu *G, Q35State *R, int il, const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35Layer *L = &M->L[il];
    const int D = c->d_model, F = c->d_ff_exp, E = c->n_expert, K = c->n_expert_used;
    if(K > MOE_MAXG - 1) return;

    /* route on the GPU: one gemv gives all E expert logits plus the shared gate (row E), so
     * the CPU only does the tiny softmax/top-K. xn goes to the host only when a CPU tier can
     * actually happen (budget below K). */
    q35cu_gemv(G->moe_logits_dev, xn, G->moe_router_dev[il], L->ffn_gate_inp->type, D, E+1);
    const int need_hxn = (G->moe_budget < K);
    q35cu_d2h_compute(G->moe_logits_host, G->moe_logits_dev, sizeof(float)*(E+1));
    if(need_hxn) q35cu_d2h_compute(G->hxn, xn, sizeof(float)*D);
    q35cu_sync();

    int idx[64]; float wt[64];
    const float sh_gate = 1.f/(1.f+expf(-G->moe_logits_host[E]));
    q35_moe_topk(G->moe_logits_host, E, K, idx, wt);

    const int tg = L->ffn_gate_exps->type, tu = L->ffn_up_exps->type, td = L->ffn_down_exps->type;
    const int64_t lgb = (int64_t)F * kq_row_bytes(tg, D);
    const int64_t lub = (int64_t)F * kq_row_bytes(tu, D);
    const int64_t ldb = (int64_t)D * kq_row_bytes(td, F);
    const uint8_t *bgate = (const uint8_t*)q35_wdata(M, L->ffn_gate_exps);
    const uint8_t *bup   = (const uint8_t*)q35_wdata(M, L->ffn_up_exps);
    const uint8_t *bdown = (const uint8_t*)q35_wdata(M, L->ffn_down_exps);
    const int resident = G->moe_rexp_g[il] != NULL;

    int ng = 0, ncpu = 0, streamed = 0;
    int cpu_e[64]; float cpu_w[64];
    for(int j = 0; j < K; j++){
        const int e = idx[j], gid = il*E + e;
        const uint8_t *pg, *pu, *pd;
        if(resident){
            pg = (const uint8_t*)G->moe_rexp_g[il] + (size_t)e*lgb;
            pu = (const uint8_t*)G->moe_rexp_u[il] + (size_t)e*lub;
            pd = (const uint8_t*)G->moe_rexp_d[il] + (size_t)e*ldb;
            G->moe_res++;
        } else {
            int slot = G->moe_nslot ? G->moe_map[gid] : -1;
            if(slot >= 0){ G->moe_hit++; G->moe_slot_clk[slot] = ++G->moe_clk; }
            else if(G->moe_nslot && streamed < G->moe_budget){
                G->moe_miss++; streamed++;
                slot = q35cu_moe_evict(G);
                if(G->moe_slot_gid[slot] >= 0) G->moe_map[G->moe_slot_gid[slot]] = -1;
                G->moe_slot_gid[slot] = gid; G->moe_map[gid] = slot; G->moe_slot_clk[slot] = ++G->moe_clk;
                q35cu_h2d_compute(G->moe_gpool + (size_t)slot*G->moe_gb, bgate + (size_t)e*lgb, (size_t)lgb);
                q35cu_h2d_compute(G->moe_upool + (size_t)slot*G->moe_ub, bup   + (size_t)e*lub, (size_t)lub);
                q35cu_h2d_compute(G->moe_dpool + (size_t)slot*G->moe_db, bdown + (size_t)e*ldb, (size_t)ldb);
            } else { cpu_e[ncpu] = e; cpu_w[ncpu] = wt[j]; ncpu++; continue; }
            pg = G->moe_gpool + (size_t)slot*G->moe_gb;
            pu = G->moe_upool + (size_t)slot*G->moe_ub;
            pd = G->moe_dpool + (size_t)slot*G->moe_db;
        }
        G->moe_hpg[ng] = (void*)pg; G->moe_hpu[ng] = (void*)pu; G->moe_hpd[ng] = (void*)pd;
        G->moe_hwt[ng] = wt[j]; ng++;
    }

    if(ng > 0){
        q35cu_h2d_compute(G->moe_pg,   G->moe_hpg, (size_t)ng*sizeof(void*));
        q35cu_h2d_compute(G->moe_pu,   G->moe_hpu, (size_t)ng*sizeof(void*));
        q35cu_h2d_compute(G->moe_pd,   G->moe_hpd, (size_t)ng*sizeof(void*));
        q35cu_h2d_compute(G->moe_wdev, G->moe_hwt, (size_t)ng*sizeof(float));
        q35cu_gemv_grp(G->moe_gg, xn, 0, (const void* const*)G->moe_pg, tg, D, F, ng);
        q35cu_gemv_grp(G->moe_uu, xn, 0, (const void* const*)G->moe_pu, tu, D, F, ng);
        q35cu_swiglu(G->moe_gg, G->moe_uu, ng*F);
        q35cu_gemv_grp(G->moe_odg, G->moe_gg, F, (const void* const*)G->moe_pd, td, F, D, ng);
        q35cu_moe_reduce(out, G->moe_odg, G->moe_wdev, D, ng);   /* out = sum_j w_j * expert_j */
    } else {
        q35cu_zero(out, sizeof(float)*D);
    }

    /* shared expert, its own per-layer type */
    q35cu_moe_expert_gpu(G, xn, D, F,
                         G->moe_shg[il], L->ffn_gate_shexp->type,
                         G->moe_shu[il], L->ffn_up_shexp->type,
                         G->moe_shd[il], L->ffn_down_shexp->type, sh_gate, out);

    /* CPU tier (PCIe budget spent), concurrent with the GPU queue above */
    if(ncpu){
        G->moe_cpu += (uint64_t)ncpu;
        for(int i = 0; i < D; i++) G->ht[i] = 0.f;
        for(int cc = 0; cc < ncpu; cc++){
            const int e = cpu_e[cc];
            q35_swiglu_row(M, G->hxn, D, F, L->ffn_gate_exps, L->ffn_up_exps, L->ffn_down_exps,
                           (size_t)e*lgb, (size_t)e*lub, (size_t)e*ldb, R->ffn_g, R->ffn_u, R->moe_od);
            for(int i = 0; i < D; i++) G->ht[i] += cpu_w[cc] * R->moe_od[i];
        }
        q35cu_h2d_compute(G->moe_acc_dev, G->ht, sizeof(float)*D);
        q35cu_add(out, G->moe_acc_dev, D);
    }
}

static void q35cu_ffn_layer_x(Q35Cu *G, Q35State *R, int il, const float *xn, float *out){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;
    const Q35CuLayer *D = &G->L[il]; const Q35Layer *S = &M->L[il];
    if(c->is_moe){ q35cu_moe_layer(G, R, il, xn, out); return; }
    if(D->ffn_gpu){
        q35cu_gemv(G->ffn_g, xn, D->ffn_gate, S->ffn_gate->type, c->d_model, c->d_ff);
        q35cu_gemv(G->ffn_u, xn, D->ffn_up,   S->ffn_up->type,   c->d_model, c->d_ff);
        q35cu_swiglu(G->ffn_g, G->ffn_u, c->d_ff);
        q35cu_gemv(out, G->ffn_g, D->ffn_down, S->ffn_down->type, c->d_ff, c->d_model);
    } else if(G->stream_on && q35cu_ffn_split(G, il, xn, out, G->stage_half)){
        /* handled: CPU and GPU each did a share, concurrently */
    } else {
        /* 20 KiB out, 20 KiB back, against 150 MiB of weights read in place. The hand-off is
         * not the cost; the alternative — moving those weights to VRAM over a x8 link — is. */
        q35cu_sync();
        q35cu_d2h(G->hxn, xn, sizeof(float)*c->d_model);
        q35_ffn(R, S, G->hxn, G->ht);
        q35cu_h2d(out, G->ht, sizeof(float)*c->d_model);
    }
}

static void q35cu_gdn_layer(Q35Cu *G, int il, int slot){
    q35cu_gdn_layer_x(G, il, slot, G->xn, G->t);
}
static void q35cu_attn_layer(Q35Cu *G, Q35State *R, int il, int slot, int pos){
    q35cu_attn_layer_x(G, R, il, slot, pos, G->xn, G->t);
}
static void q35cu_ffn_layer(Q35Cu *G, Q35State *R, int il){
    q35cu_ffn_layer_x(G, R, il, G->xn, G->t);
}

/* ---------------- one token ---------------- */

static void q35_forward_cu(Q35Cu *G, Q35State *R, int tok, int pos, float *logits){
    const Q35Model *M = G->M; const Q35Cfg *c = &M->c;

    {   const double t0 = g_q35_stage_timing ? q35_clk() : 0;
        q35_embed(R, tok, R->x);                       /* one row of a 0.67 GiB table */
        q35cu_h2d(G->x, R->x, sizeof(float)*c->d_model);
        if(g_q35_stage_timing) g_t_embd += q35_clk()-t0; }

    G->stream_batch = 0;      /* the single-token path is the decode regime */
    if(G->stream_on){
        const int f0 = q35cu_next_stream_layer(G, 0);
        G->stage_half = 0;
        if(f0 >= 0) q35cu_stream_issue(G, f0, 0);
    }
    for(int il = 0; il < c->n_layer; il++){
        const Q35CuLayer *D = &G->L[il];
        /* Prefetch the NEXT streamed layer into the other half before touching this one, so
         * its 2.6 ms of DMA sits underneath this layer's compute instead of in front of the
         * next one's. */
        if(G->stream_on && !D->ffn_gpu){
            const int nl = q35cu_next_stream_layer(G, il+1);
            if(nl >= 0) q35cu_stream_issue(G, nl, 1 - G->stage_half);
        }
        q35cu_rmsnorm(G->xn, G->x, D->attn_norm, c->d_model, c->eps);
        { const double t0 = g_q35_stage_timing ? q35_clk() : 0;
          if(c->kind[il] == Q35_LAYER_ATTN) q35cu_attn_layer(G, R, il, R->attn_slot[il], pos);
          else                              q35cu_gdn_layer(G, il, R->gdn_slot[il]);
          if(g_q35_stage_timing){ q35cu_sync();
              *(c->kind[il] == Q35_LAYER_ATTN ? &g_t_attn : &g_t_gdn) += q35_clk()-t0; } }
        q35cu_add(G->x, G->t, c->d_model);

        q35cu_rmsnorm(G->xn, G->x, D->post_attn_norm, c->d_model, c->eps);
        { const double t0 = g_q35_stage_timing ? q35_clk() : 0;
          q35cu_ffn_layer(G, R, il);
          if(G->stream_on && !D->ffn_gpu) G->stage_half = 1 - G->stage_half;
          if(g_q35_stage_timing){ q35cu_sync(); g_t_ffn += q35_clk()-t0; } }
        q35cu_add(G->x, G->t, c->d_model);
    }

    if(logits){
        const double t0 = g_q35_stage_timing ? q35_clk() : 0;
        q35cu_rmsnorm(G->xn, G->x, G->output_norm, c->d_model, c->eps);
        q35cu_gemv(G->logits, G->xn, G->output, M->output->type, c->d_model, c->vocab);
        q35cu_sync();
        q35cu_d2h(logits, G->logits, sizeof(float)*c->vocab);
        if(g_q35_stage_timing) g_t_out += q35_clk()-t0;
    }
    if(!q35cu_sync()) fprintf(stderr, "qwen35_cuda: %s\n", q35cu_error());
    /* Mirror the residual stream back to the host. 20 KiB against 15.6 GiB of weights, and
     * without it q35_hidden() feeds the MTP draft head the LAST TOKEN'S EMBEDDING instead of
     * the model's hidden state — which does not crash, does not warn, and quietly drops
     * draft acceptance from ~85% to ~8%, i.e. turns speculation from a 1.5x win into an
     * 18% loss. */
    q35cu_d2h(R->x, G->x, sizeof(float)*c->d_model);
    R->pos = pos + 1;
}

#endif
