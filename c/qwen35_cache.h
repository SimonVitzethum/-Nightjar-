#ifndef QWEN_QWEN35_CACHE_H
#define QWEN_QWEN35_CACHE_H
/* qwen35_cache.h — prefix caching across requests.
 *
 * WHY A PLAIN "DID THE LAST REQUEST EXTEND THIS ONE" CACHE IS NOT ENOUGH
 *
 * The OpenAI protocol is stateless: every turn resends the whole conversation. In a coding
 * session that conversation is mostly a fixed prefix — the agent's system prompt and tool
 * schemas run to thousands of tokens and never change — so the reachable hit rate is very
 * high. A single-sequence cache only collects it when consecutive requests extend each other
 * exactly. One short interleaved request (opencode sends a title-generation call at session
 * start) and the next agent turn pays the full prefill again: measured, 7148 tokens at 9.1
 * tok/s is 786 seconds.
 *
 * WHAT ACTUALLY HAS TO BE SAVED, AND WHAT DOES NOT
 *
 * Attention KV is append-only and **prefix-closed**: if the store holds the KV for sequence A
 * and a new sequence B shares A's first k tokens, then KV[0,k) is already correct for B. It
 * needs no copy, only a shorter logical length. Everything from k onward gets overwritten as
 * B is prefilled.
 *
 * The gated delta net has no such property. Its state is decayed and rank-1 updated per step
 * and that is not invertible, so "the state after k tokens" cannot be recovered from the state
 * after k+m. It is the ONLY thing that has to be snapshotted — 144 MiB of recurrent state plus
 * a 5.6 MiB conv ring, per checkpoint, and nothing else.
 *
 * So: keep N snapshots of the recurrent state at spaced positions along the live sequence. A
 * request that diverges at token p resumes from the largest checkpoint <= p, recomputing at
 * most `spacing` tokens instead of everything. Checkpoints past p are dropped, because the KV
 * they relied on is about to be overwritten.
 *
 * WHY THE CHECKPOINT ALSO CARRIES ITS OWN KV
 *
 * Prefix-closure only holds while nothing has overwritten the shared prefix. Measured: a long
 * agent prompt, then opencode's 16-token title call, then the agent prompt again — the title
 * call rewrote KV[0,16) and every checkpoint of the long sequence became unusable, so the
 * third request paid 140 s again. A checkpoint that carries the KV it needs is instead
 * SELF-CONTAINED: it survives any number of intervening requests and needs no invalidation
 * rule at all. That is what turns "consecutive turns are cheap" into "the system prompt is
 * cheap forever", which is the hit rate that actually matters in a coding session.
 *
 * Cost per checkpoint: 150 MiB of recurrent state plus 36 KiB per token of trunk attention KV
 * — at 8192 tokens, ~450 MiB, and ~20 ms to copy over PCIe from pinned memory. Against ~9
 * tokens of prefill per second, that pays for itself after 180 tokens of avoided recompute.
 *
 * The draft head's own KV is deliberately NOT checkpointed. It lives in the host tier and
 * would double the bookkeeping, and a stale draft KV costs acceptance, never correctness —
 * every speculated token is verified against the trunk regardless.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen35_cpu.h"
#ifndef QWEN_NO_CUDA
#include "qwen35_hetero.h"
#endif

typedef struct {
    int      pos;            /* how many tokens this checkpoint covers; -1 = empty */
    int     *toks;           /* the exact token prefix it corresponds to */
    float   *S, *conv;       /* host copies of the recurrent state */
    uint8_t *kv;             /* trunk attention KV for [0,pos), all layers, or NULL */
    uint64_t stamp;
    uint64_t chat;           /* which conversation this belongs to; 0 = unattributed */
} Q35Ckpt;

typedef struct {
    Q35State *R;
#ifndef QWEN_NO_CUDA
    Q35Cu    *G;
#endif
    int       gpu, pinned;
    size_t    s_bytes, c_bytes;

    Q35Ckpt  *ck;
    int       n_slots, n_used;
    uint64_t  clock;

    int      *ctx;           /* the token sequence the live KV and state correspond to */
    int       ctx_n, ctx_cap;

    int       spacing, next_mark;
    uint64_t  chat;          /* the conversation the CURRENT request belongs to */
    int       kv_win;        /* tokens of KV a checkpoint can carry (the device window) */
    size_t    kv_stride;     /* bytes of one layer's K (or V) for one token */
    int       kv_layers;     /* TRUNK attention layers only — the draft head's slot is skipped */
    size_t    kv_bytes;      /* per checkpoint, at kv_win tokens */
    int64_t   n_req, tok_total, tok_reused;
    double    t_copy;
} Q35Cache;

/* NOT pinned by default, and that is a deliberate reversal.
 *
 * Pinned pages copy at 27.7 GB/s instead of 6.8, so a 422 MiB checkpoint moves in 15 ms
 * rather than 62. Against a prefill it saves — 150 seconds and up — those 47 ms are noise.
 * What pinning does cost is real: the pages become unswappable, and this cache sits on top of
 * 9.3 GiB of already-pinned weights. Measured, 2.95 GiB of pinned cache on a 31 GiB machine
 * drove it into swap and the server was OOM-killed mid-request — silently, with no error
 * anywhere, because the kernel does not ask.
 *
 * A cache that occasionally kills the process is worse than a cache that copies four times
 * slower. QWEN_CACHE_PIN=1 restores the old behaviour for a machine with room. */
static void *q35_cache_alloc(Q35Cache *C, size_t n){
#ifndef QWEN_NO_CUDA
    static int want = -1;
    if(want < 0){ const char *e = getenv("QWEN_CACHE_PIN"); want = (e && *e && *e != '0'); }
    if(C->gpu && want){
        void *p = q35cu_host_alloc(n);
        if(p){ C->pinned = 1; return p; }
    }
#endif
    return malloc(n);
}
static void q35_cache_release(Q35Cache *C, void *p){
#ifndef QWEN_NO_CUDA
    if(C->pinned){ q35cu_host_free(p); return; }
#endif
    free(p);
}

#ifndef QWEN_NO_CUDA
static int q35_cache_init(Q35Cache *C, Q35State *R, Q35Cu *G, int slots, int spacing){
#else
static int q35_cache_init(Q35Cache *C, Q35State *R, void *G, int slots, int spacing){
#endif
    memset(C, 0, sizeof *C);
    const Q35Cfg *c = &R->M->c;
    C->R = R;
#ifndef QWEN_NO_CUDA
    C->G = G; C->gpu = (G != NULL);
#else
    (void)G;
#endif
    C->s_bytes = (size_t)R->n_gdn*c->n_v_heads*c->d_state*c->d_head_v*sizeof(float);
    C->c_bytes = (size_t)R->n_gdn*(c->d_conv-1)*c->conv_dim*sizeof(float);

#ifndef QWEN_NO_CUDA
    if(C->gpu){
        C->kv_win    = G->win;
        C->kv_stride = (size_t)G->kv_half;
        C->kv_layers = q35_n_attn_layers(c);      /* trunk only, not the MTP slot */
        C->kv_bytes  = (size_t)C->kv_layers*2u*(size_t)C->kv_win*C->kv_stride;
    }
#endif

    { const char *e = getenv("QWEN_CACHE_SPACING"); if(e) spacing = atoi(e); }
    if(spacing < 64) spacing = 64;

    /* Budget-driven rather than count-driven: a checkpoint's size depends on the KV window,
     * so "six slots" means very different things at 4k and at 32k. */
    double gb = 1.5;
    { const char *e = getenv("QWEN_CACHE_GB"); if(e) gb = atof(e); }
    { const char *e = getenv("QWEN_CACHE_SLOTS"); if(e) slots = atoi(e); else slots = 64; }
    const size_t per = C->s_bytes + C->c_bytes + C->kv_bytes;
    int by_budget = per ? (int)((int64_t)(gb*1073741824.0)/(int64_t)per) : 0;
    if(by_budget < 0) by_budget = 0;
    if(slots > by_budget) slots = by_budget;
    /* Never take the machine under its floor: an evicted FFN costs far more than any
     * re-prefill this saves. */
    /* Ask for TWICE what will be used. q35_mem_room measures MemAvailable, which counts
     * reclaimable page cache — and this allocation is the thing that stops it being
     * reclaimable, so the honest question is whether there is room to spare, not room to
     * fit exactly. */
    while(slots > 0 && !q35_mem_room(2*(int64_t)slots*(int64_t)per)) slots--;

    C->n_slots = slots; C->spacing = spacing;
    if(slots){
        C->ck = (Q35Ckpt*)calloc((size_t)slots, sizeof(Q35Ckpt));
        if(!C->ck) return 0;
        for(int i = 0; i < slots; i++){
            C->ck[i].S    = (float*)q35_cache_alloc(C, C->s_bytes);
            C->ck[i].conv = (float*)q35_cache_alloc(C, C->c_bytes);
            C->ck[i].toks = (int*)malloc(sizeof(int)*(size_t)(C->kv_win ? C->kv_win : 65536));
            C->ck[i].kv   = C->kv_bytes ? (uint8_t*)q35_cache_alloc(C, C->kv_bytes) : NULL;
            C->ck[i].pos  = -1;
            if(!C->ck[i].S || !C->ck[i].conv || !C->ck[i].toks ||
               (C->kv_bytes && !C->ck[i].kv)){ C->n_slots = i; break; }
        }
    }
    return 1;
}

static void q35_cache_free(Q35Cache *C){
    for(int i = 0; i < C->n_slots; i++){
        q35_cache_release(C, C->ck[i].S);
        q35_cache_release(C, C->ck[i].conv);
        q35_cache_release(C, C->ck[i].kv);
        free(C->ck[i].toks);
    }
    free(C->ck); free(C->ctx);
    memset(C, 0, sizeof *C);
}

static void q35_cache_push(Q35Cache *C, int tok){
    if(C->ctx_n == C->ctx_cap){
        C->ctx_cap = C->ctx_cap ? C->ctx_cap*2 : 8192;
        C->ctx = (int*)realloc(C->ctx, sizeof(int)*(size_t)C->ctx_cap);
    }
    C->ctx[C->ctx_n++] = tok;
}

/* ---- moving the recurrent state between device/host and a slot ---- */

/* The device window stores each attention layer's K (and V) contiguously over tokens, so one
 * layer's prefix is one contiguous range and a checkpoint is `kv_layers` pairs of copies. */
static void q35_cache_kv(Q35Cache *C, Q35Ckpt *k, int save){
#ifndef QWEN_NO_CUDA
    if(!C->gpu || !k->kv || !C->kv_bytes) return;
    const Q35Cu *G = C->G;
    const size_t n = q35_len(k->pos, "checkpoint pos")*C->kv_stride;
    for(int l = 0; l < C->kv_layers; l++){
        const size_t lane = (size_t)l*(size_t)C->kv_win*C->kv_stride;
        uint8_t *hk = k->kv + 2u*lane, *hv = hk + (size_t)C->kv_win*C->kv_stride;
        if(save){ q35cu_d2h(hk, G->Kc + lane, n); q35cu_d2h(hv, G->Vc + lane, n); }
        else     { q35cu_h2d(G->Kc + lane, hk, n); q35cu_h2d(G->Vc + lane, hv, n); }
    }
#else
    (void)C; (void)k; (void)save;
#endif
}

static void q35_cache_grab(Q35Cache *C, Q35Ckpt *k){
    if(k->pos <= 0 || (C->kv_win && k->pos > C->kv_win)) return;   /* caller must set pos first */
    const double t0 = q35_clk();
#ifndef QWEN_NO_CUDA
    if(C->gpu){
        q35cu_sync();
        q35cu_d2h(k->S,    C->G->S,    C->s_bytes);
        q35cu_d2h(k->conv, C->G->conv, C->c_bytes);
        q35_cache_kv(C, k, 1);
    } else
#endif
    { memcpy(k->S, C->R->S, C->s_bytes); memcpy(k->conv, C->R->conv, C->c_bytes); }
    memcpy(k->toks, C->ctx, sizeof(int)*q35_len(k->pos, "checkpoint pos"));
    C->t_copy += q35_clk() - t0;
}

static void q35_cache_put_back(Q35Cache *C, Q35Ckpt *k){
    const double t0 = q35_clk();
#ifndef QWEN_NO_CUDA
    if(C->gpu){
        q35cu_h2d(C->G->S,    k->S,    C->s_bytes);
        q35cu_h2d(C->G->conv, k->conv, C->c_bytes);
        q35_cache_kv(C, k, 0);
        q35cu_sync();
    } else
#endif
    { memcpy(C->R->S, k->S, C->s_bytes); memcpy(C->R->conv, k->conv, C->c_bytes); }
    /* the live sequence is now this checkpoint's prefix */
    if(C->ctx_cap < k->pos){
        C->ctx_cap = k->pos + 4096;
        C->ctx = (int*)realloc(C->ctx, sizeof(int)*(size_t)C->ctx_cap);
    }
    memcpy(C->ctx, k->toks, sizeof(int)*q35_len(k->pos, "checkpoint pos"));
    C->t_copy += q35_clk() - t0;
}

static void q35_cache_reset_state(Q35Cache *C){
    q35_state_reset(C->R);
#ifndef QWEN_NO_CUDA
    if(C->gpu) q35cu_state_reset(C->G);
#endif
}

/* IS THIS CHECKPOINT ON THE SAME LINE AS THE CURRENT REQUEST?
 *
 * Every site that compares two checkpoint positions has to ask this first, and the reason is
 * that a position is not a number — it is a number PLUS the conversation it counts within.
 * Positions from two chats share only the axis label; token 2048 of chat A and token 2048 of
 * chat B are different points in different spaces, and in C they look identical.
 *
 * There is exactly one place where comparing across chats is legitimate: q35_cache_begin, and
 * only because every candidate there has been memcmp'd against the SAME request prefix first.
 * That memcmp is what puts them on one line; without it, use this.
 *
 * (The structural answer is to put the chat into the position type so a cross-chat comparison
 * cannot be written at all. With two comparison sites a named predicate is the honest size of
 * the fix; if a third appears, the type is worth the churn.) */
static inline int q35_ck_mine(const Q35Cache *C, int i){
    return C->ck[i].pos >= 0 && C->ck[i].chat == C->chat;
}

/* Pick a slot for a checkpoint at `pos`.
 *
 * THE GAP HEURISTIC IS ONLY MEANINGFUL WITHIN ONE CONVERSATION.
 *
 * The original rule was "evict whichever checkpoint sits closest to a neighbour, since dropping
 * it widens the worst gap least". That is right for points along one sequence and wrong the
 * moment two conversations share the cache: a checkpoint of chat A at token 780 and one of
 * chat B at 800 look like neighbours, and one of them gets dropped to widen a gap that does not
 * exist — because they are not on the same line at all. The chat that lost its foothold then
 * pays a full prefill on its next turn, which is a hundred seconds and up.
 *
 * So: every conversation keeps at least one checkpoint while there are slots to go round, and
 * the gap rule applies inside a conversation. With N slots and more than N chats, the chat
 * whose turn was longest ago gives one up — LRU across chats, gap-based within one.
 *
 * Position 0 is never a checkpoint (resuming there is just a reset). */
static int q35_cache_slot_for(Q35Cache *C, int pos){
    for(int i = 0; i < C->n_slots; i++) if(C->ck[i].pos < 0) return i;

    /* how many slots each conversation currently holds */
    int mine = 0;
    for(int i = 0; i < C->n_slots; i++) if(q35_ck_mine(C, i)) mine++;

    if(mine >= 2){
        /* inside my own conversation the original argument holds exactly */
        int worst = -1, worst_gap = 1<<30;
        for(int i = 0; i < C->n_slots; i++){
            if(!q35_ck_mine(C, i)) continue;
            int lo = 0, hi = pos;
            for(int j = 0; j < C->n_slots; j++){
                if(j == i || !q35_ck_mine(C, j)) continue;
                if(C->ck[j].pos < C->ck[i].pos && C->ck[j].pos > lo) lo = C->ck[j].pos;
                if(C->ck[j].pos > C->ck[i].pos && C->ck[j].pos < hi) hi = C->ck[j].pos;
            }
            const int gap = hi - lo;
            if(gap < worst_gap){ worst_gap = gap; worst = i; }
        }
        if(worst >= 0) return worst;
    }

    /* otherwise take from whoever has the most, oldest first — never from a conversation that
     * is down to its last foothold unless everyone is */
    int best = -1, best_count = 0; uint64_t best_stamp = ~0ull;
    for(int i = 0; i < C->n_slots; i++){
        int count = 0;
        for(int j = 0; j < C->n_slots; j++) if(C->ck[j].chat == C->ck[i].chat) count++;
        if(count > best_count || (count == best_count && C->ck[i].stamp < best_stamp)){
            best_count = count; best_stamp = C->ck[i].stamp; best = i;
        }
    }
    return best < 0 ? 0 : best;
}

/* Called after the model has consumed token index `pos-1`, i.e. the state now covers `pos`.
 * The trigger is "at or past the next mark", not "an exact multiple": batched prefill advances
 * eight tokens at a time and the state simply does not exist in between. */
static void q35_cache_mark(Q35Cache *C, int pos){
    if(C->n_slots <= 0 || pos <= 0 || pos < C->next_mark) return;
    /* A checkpoint can only carry KV the device window actually holds. Past that the prefix
     * lives in the host tier and would need its own serialization; not worth it while the
     * window covers every prompt an agent sends. */
    if(C->kv_bytes && pos > C->kv_win) return;
    C->next_mark = pos + C->spacing;
    /* "is there already a checkpoint here" is a question about MY conversation. Without the
     * chat test this compared positions across chats, and the case is not hypothetical: two
     * conversations in one project share the system prompt and the tool schemas, so their marks
     * land on exactly the same positions — 512, 1024, 1536 — and whichever got there first
     * silently stopped the other from ever checkpointing at all. The 9.1x in Plan.md 19.2 was
     * measured with this bug still in place. */
    for(int i = 0; i < C->n_slots; i++) if(C->ck[i].pos == pos && q35_ck_mine(C, i)) return;
    const int s = q35_cache_slot_for(C, pos);
    /* pos FIRST. q35_cache_grab reads k->pos to size both the KV copy and the token memcpy,
     * and a fresh slot holds -1 — which as a size_t is 18 exabytes. It segfaulted inside libc
     * on whichever thread happened to be running, ~512 tokens into a prefill, with no message
     * of any kind. Prompts shorter than the checkpoint spacing never reached it, which is why
     * every short test passed. */
    C->ck[s].pos = pos;
    q35_cache_grab(C, &C->ck[s]);
    C->ck[s].stamp = ++C->clock;
    C->ck[s].chat  = C->chat;
}

/* ---- the lookup ----
 * Returns the token position to resume prefilling from, having put the engine into the
 * matching state. `*reused` reports how many prompt tokens were skipped. */
static int q35_cache_begin(Q35Cache *C, const int *req, int n, int *reused){
    /* The live state is the cheapest candidate: reusing it copies nothing. It only counts
     * when the request EXTENDS it exactly — a shorter or divergent one has to come from a
     * checkpoint, because there is no way to rewind the recurrent half. */
    int live = 0;
    while(live < n && live < C->ctx_n && req[live] == C->ctx[live]) live++;
    if(live != C->ctx_n) live = 0;

    /* Checkpoints are self-contained, so each is matched against its OWN token prefix rather
     * than against whatever ran last. That is what lets a long agent prompt survive the short
     * unrelated request opencode interleaves at session start. */
    /* THE ONE PLACE WHERE COMPARING ACROSS CHATS IS CORRECT. Every candidate is memcmp'd
     * against THIS request's token prefix, and that is what puts them all on one line — a
     * checkpoint that matches is a prefix of this request whichever conversation created it,
     * which is exactly what makes a shared system prompt free for everyone. The `<=` skip
     * before the memcmp only drops candidates that could not win anyway. */
    int best = -1;
    for(int i = 0; i < C->n_slots; i++){
        const Q35Ckpt *k = &C->ck[i];
        if(k->pos < 0 || k->pos > n) continue;
        if(best >= 0 && k->pos <= C->ck[best].pos) continue;
        if(memcmp(k->toks, req, sizeof(int)*(size_t)k->pos) == 0) best = i;
    }

    int start;
    if(best >= 0 && C->ck[best].pos > live){
        q35_cache_put_back(C, &C->ck[best]);
        C->ck[best].stamp = ++C->clock;
        /* A checkpoint that matched this request BELONGS to this request's conversation,
         * whatever it was tagged with before — two chats sharing a prefix are, for the purpose
         * of eviction, the same line. */
        C->ck[best].chat = C->chat;
        start = C->ck[best].pos;
    } else if(live > 0){
        start = live;                               /* state already correct, nothing to copy */
    } else {
        q35_cache_reset_state(C);
        start = 0;
    }
    C->ctx_n = start;
    C->next_mark = start + C->spacing;
    C->n_req++; C->tok_total += n; C->tok_reused += start;
    if(reused) *reused = start;
    return start;
}

/* Which conversation the next request belongs to. A 64-bit hash of the client's own id, so the
 * cache never has to hold or understand the string. Requests that carry no id share bucket 0 —
 * which is correct for opencode and curl, and costs them nothing they had before. */
static uint64_t q35_cache_chat_id(const char *s){
    if(!s || !*s) return 0;
    uint64_t h = 1469598103934665603ull;             /* FNV-1a */
    for(const unsigned char *p = (const unsigned char*)s; *p; p++){
        h ^= *p; h *= 1099511628211ull;
    }
    return h ? h : 1;
}
static void q35_cache_use_chat(Q35Cache *C, const char *id){ C->chat = q35_cache_chat_id(id); }

static void q35_cache_report(const Q35Cache *C, FILE *o){
    if(C->n_slots <= 0){ fprintf(o, "  prefix cache: off\n"); return; }
    int live = 0;
    for(int i = 0; i < C->n_slots; i++) if(C->ck[i].pos >= 0) live++;
    fprintf(o, "  prefix cache: %d slots x %.0f MiB%s, every %d tokens (%d live)"
               " — %lld of %lld prompt tokens reused (%.1f%%), %.2f s copying\n",
            C->n_slots, (C->s_bytes + C->c_bytes + C->kv_bytes)/1048576.0,
            C->pinned ? " pinned" : "", C->spacing, live,
            (long long)C->tok_reused, (long long)C->tok_total,
            C->tok_total ? 100.0*(double)C->tok_reused/(double)C->tok_total : 0.0,
            C->t_copy);
}

#endif
