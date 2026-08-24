/* chat_test.c — the whole stack, talking.
 *
 * Tokenizer -> chat template -> 78 layers of MLA + streamed MoE -> logits -> sampling ->
 * detokenizer. Every piece was validated against llama.cpp before it got here; this is the
 * first time they all run together and produce text.
 *
 * GLM-5.2's chat template starts with [gMASK]<sop> and then marks turns with <|user|> /
 * <|assistant|>. Those are CONTROL tokens: they must reach the model as single ids, which
 * is exactly what tok_gguf.h's special handling guarantees.
 *
 * It is slow — this runs on the CPU with no expert cache, so every token streams ~7.4 GB of
 * experts off NVMe. The GPU path and the tiered cache exist and are measured (1.6 tok/s);
 * they are simply not wired into this loop, because this loop is about whether the model
 * SPEAKS, not how fast.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../glm_engine.h"
#include "../tok_gguf.h"
#define ES_CUDA 1          /* the expert buffers feed the GPU: they must be pinned */
#include <cuda_runtime.h>
#include "../estream.h"
#include "../moe_gpu.h"
#include "../attn_gpu.h"
#include "../attn_requant.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* EVERY pinned allocation goes through this. No exceptions, no "this one is small", no "the
 * budget already accounted for it" — that reasoning is exactly what OOM-killed the machine
 * and took the user's editor with it. Pinned memory cannot be paged out, so the kernel has no
 * recourse; the only safe protocol is to look at what is left immediately before asking. */
static int pin_alloc(void **p, int64_t n, const char *what){
    const int64_t avail = es_mem_available();
    if(avail - n < ES_RESERVE){
        fprintf(stderr, "[mem] refusing to pin %.2f GB for %s: only %.2f GB left, "
                "floor is %.0f GB\n", n/1e9, what, avail/1e9, (double)(ES_RESERVE)/1e9);
        return 0;
    }
    if(cudaMallocHost(p, (size_t)n) != cudaSuccess){
        fprintf(stderr, "[mem] cudaMallocHost(%.2f GB) for %s failed\n", n/1e9, what);
        return 0;
    }
    return 1;
}

typedef struct { MoeGpu *G; GModel *M; EStream *S; AttnGpu *A;
                 int64_t sh_ng, sh_nu, sh_nd;      /* shared-expert byte counts */
                 uint8_t *out_w; int out_t;        /* output projection, resident in VRAM */
                 float *out_d; int vocab, d_model;
                 cudaStream_t out_s; } GpuCtx;

/* The output projection: 535 MB of Q4_K, one gemv per token. On the CPU's scalar decode path
 * that is 0.63 s; in VRAM it is resident, so it costs no PCIe at all — the weights go up once
 * at startup and stay. 0.54 GB of the 8, which we have. */
static void out_hook(void *ud, float *logits, const float *xn){
    GpuCtx *C = (GpuCtx*)ud;
    static float *x_d = NULL;
    if(!x_d) cudaMalloc((void**)&x_d, (size_t)C->d_model*sizeof(float));
    cudaMemcpyAsync(x_d, xn, (size_t)C->d_model*sizeof(float),
                    cudaMemcpyHostToDevice, C->out_s);
    if(!kq_cu_gemm(C->out_d, x_d, C->out_w, C->out_t, 1, C->d_model, C->vocab,
                   kq_typesize(C->out_t), C->out_s)){
        fprintf(stderr, "out_hook: no kernel for %s\n", kq_name(C->out_t)); exit(1); }
    cudaMemcpyAsync(logits, C->out_d, (size_t)C->vocab*sizeof(float),
                    cudaMemcpyDeviceToHost, C->out_s);
    cudaStreamSynchronize(C->out_s);
}

/* Where does a token actually go? 9.6 s/token with the drive at 1.5 GB/s does not add up:
 * the disk moves a layer's experts in ~20 ms, yet a layer takes ~126 ms. Something else owns
 * the other 106 ms, and guessing is what cost me the last two days. */
static double T_attn, T_moe, T_disk;
static long   N_attn, N_moe, N_disk;

/* the disk read sits INSIDE glm_forward, so wrap the fetch callback to time it */
static int es_fetch_timed(void *ud, int layer, const int *eid, int k,
                          const void **g, const void **u, const void **d){
    const double t0 = now();
    int r = es_fetch(ud, layer, eid, k, g, u, d);
    T_disk += now() - t0; N_disk++;
    return r;
}

/* The attention weights (133 MB/layer) are uploaded while the PREVIOUS layer computes, so
 * the PCIe transfer hides behind the disk read the forward pass is waiting on anyway. */
static void attn_hook(void *ud, int layer, float *out, const float *x, int pos,
                      const MLACfg *mc, const MLALayer *ml,
                      MLACache *kv, MLAScratch *s, float rope_base){
    GpuCtx *C = (GpuCtx*)ud;
    const double t_a0 = now();
    if(C->A->staged != layer){          /* not prefetched (first layer): fetch it now */
        attn_gpu_prefetch(C->A, mc, ml, layer);
        C->A->staged = layer;
    }
    mla_step_gpu(C->A, out, x, pos, mc, ml, layer, kv, s, rope_base);

    /* start the NEXT layer's weights moving right away */
    int nl = layer + 1;
    if(nl < C->M->c.n_layer){
        const GLayer *L = &C->M->L[nl];
        MLALayer next = {
            .pq_a=L->q_a, .pq_b=L->q_b, .pkv_a=L->kv_a, .pk_b=L->k_b, .pv_b=L->v_b, .po=L->o,
            .tq_a=L->t_q_a, .tq_b=L->t_q_b, .tkv_a=L->t_kv_a,
            .tk_b=L->t_k_b, .tv_b=L->t_v_b, .to=L->t_o,
            .attn_norm=L->attn_norm, .q_a_norm=L->q_a_norm, .kv_a_norm=L->kv_a_norm };
        attn_gpu_prefetch(C->A, mc, &next, nl);
        C->A->staged = nl;
    }
    T_attn += now() - t_a0; N_attn++;
}

/* glm_forward hands us host pointers to the k experts' quantized bytes; the GPU needs their
 * types and sizes too, and both live in the layer's tensor table. */
static void moe_hook(void *ud, int layer, float *acc, const float *xn,
                     const void **eg, const void **eu, const void **ed,
                     const float *w, int k){
    GpuCtx *C = (GpuCtx*)ud;
    const double t_m0 = now();
    const GLayer *L = &C->M->L[layer];
    int tg[16], tu[16], td[16];
    int64_t ng[16], nu[16], nd[16];
    const int64_t bg = kq_row_bytes(L->e_gate->type, L->e_gate->ne[0]*L->e_gate->ne[1]);
    const int64_t bu = kq_row_bytes(L->e_up->type,   L->e_up->ne[0]*L->e_up->ne[1]);
    const int64_t bd = kq_row_bytes(L->e_down->type, L->e_down->ne[0]*L->e_down->ne[1]);
    for(int i=0;i<k;i++){
        tg[i]=L->e_gate->type; tu[i]=L->e_up->type; td[i]=L->e_down->type;
        ng[i]=bg; nu[i]=bu; nd[i]=bd;
    }

    /* The shared expert has the SAME shape as a routed one — [6144x2048] gate/up, [2048x6144]
     * down — so it slots in as the (k+1)th expert with weight 1.0 and never touches the CPU.
     * Its TYPES differ (Q5_K/Q5_K/Q6_K against the routed IQ2_XS/IQ2_XS/IQ3_XXS) and so does
     * its SIZE (27.6 MB against 12.1), which is why moe_gpu takes types and byte counts per
     * expert, and why the caller must size the staging buffer for the fatter of the two. */
    const void *xg[17], *xu[17], *xd[17];
    float xw[17];
    for(int i=0;i<k;i++){ xg[i]=eg[i]; xu[i]=eu[i]; xd[i]=ed[i]; xw[i]=w[i]; }
    xg[k]=L->sh_gate; xu[k]=L->sh_up; xd[k]=L->sh_down;
    tg[k]=L->t_sh_gate; tu[k]=L->t_sh_up; td[k]=L->t_sh_down;
    /* PER-LAYER byte counts, not the max across layers. "Dynamic" gives some layers a Q6_K
     * shared expert (10.3 MB) where most get Q5_K (8.7). The max is what the staging BUFFER
     * must hold; telling cudaMemcpy to copy the max out of a Q5_K tensor reads 1.6 MB past
     * its end. Exactly the blk.8 bug, one buffer over. */
    const int Dm = C->d_model, Ff = C->M->c.d_ff_exp;
    ng[k] = kq_row_bytes(L->t_sh_gate, (int64_t)Dm*Ff);
    nu[k] = kq_row_bytes(L->t_sh_up,   (int64_t)Dm*Ff);
    nd[k] = kq_row_bytes(L->t_sh_down, (int64_t)Ff*Dm);
    xw[k]=1.0f;                                   /* no routing weight: it always fires */

    moe_gpu_layer(C->G, acc, xn, xg, xu, xd, tg, tu, td, ng, nu, nd, xw, k+1);
    T_moe += now() - t_m0; N_moe++;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf> [prompt] [n_tokens]\n", argv[0]); return 2; }
    const char *user = argc > 2 ? argv[2] : "Schreibe eine Python-Funktion, die zwei Zahlen addiert.";
    const int   n_gen = argc > 3 ? atoi(argv[3]) : 12;

    GModel M; memset(&M, 0, sizeof(M));
    if(!gguf_open(&M.m, argv[1])){ printf("open failed\n"); return 1; }

    Tok T; TokIds ids;
    if(!tok_from_gguf(&T, &M.m, &ids)){ printf("tokenizer failed\n"); return 1; }

    const char *arch = gguf_str(&M.m, "general.architecture", "");
    char k[160];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&M.m,k,0))
    #define KVF(s) (snprintf(k,sizeof(k),"%s." s,arch), (float)gguf_f64(&M.m,k,0))
    GCfg *c = &M.c;
    c->d_model=KVI("embedding_length");    c->n_layer=KVI("block_count");
    c->n_dense=KVI("leading_dense_block_count"); c->n_head=KVI("attention.head_count");
    c->vocab=KVI("vocab_size");            c->q_lora=KVI("attention.q_lora_rank");
    c->kv_lora=KVI("attention.kv_lora_rank"); c->qk_rope=KVI("rope.dimension_count");
    c->n_exp=KVI("expert_count");          c->topk=KVI("expert_used_count");
    c->n_shared=KVI("expert_shared_count"); c->d_ff=KVI("feed_forward_length");
    c->d_ff_exp=KVI("expert_feed_forward_length");
    c->gating=KVI("expert_gating_func");   c->w_norm=KVI("expert_weights_norm");
    c->eps=KVF("attention.layer_norm_rms_epsilon");
    c->rope_base=KVF("rope.freq_base");    c->w_scale=KVF("expert_weights_scale");
    #undef KVI
    #undef KVF
    const int n_run = c->n_layer - 1;      /* the MTP head is not part of the forward pass */

    M.L = calloc(c->n_layer, sizeof(GLayer));
    M.emb = gguf_find(&M.m, "token_embd.weight");
    const gguf_tensor *ow = gguf_find(&M.m, "output.weight");
    const gguf_tensor *on = gguf_find(&M.m, "output_norm.weight");
    M.out_w = gguf_data(&M.m, ow); M.t_out = ow->type;
    M.out_norm = (const float*)gguf_data(&M.m, on);

    for(int i=0;i<n_run;i++){
        GLayer *L = &M.L[i]; const gguf_tensor *t;
        #define AT(f,tf,nm) do{ snprintf(k,sizeof(k),nm,i); t=gguf_find(&M.m,k); \
            if(!t){printf("missing %s\n",k);return 1;} L->f=gguf_data(&M.m,t); L->tf=t->type; }while(0)
        #define NM(f,nm) do{ snprintf(k,sizeof(k),nm,i); t=gguf_find(&M.m,k); \
            if(!t){printf("missing %s\n",k);return 1;} L->f=(const float*)gguf_data(&M.m,t); }while(0)
        AT(q_a,t_q_a,"blk.%d.attn_q_a.weight");   AT(q_b,t_q_b,"blk.%d.attn_q_b.weight");
        AT(kv_a,t_kv_a,"blk.%d.attn_kv_a_mqa.weight");
        AT(k_b,t_k_b,"blk.%d.attn_k_b.weight");   AT(v_b,t_v_b,"blk.%d.attn_v_b.weight");
        AT(o,t_o,"blk.%d.attn_output.weight");
        NM(attn_norm,"blk.%d.attn_norm.weight");  NM(q_a_norm,"blk.%d.attn_q_a_norm.weight");
        NM(kv_a_norm,"blk.%d.attn_kv_a_norm.weight"); NM(ffn_norm,"blk.%d.ffn_norm.weight");
        if(i < c->n_dense){
            L->is_moe = 0;
            AT(gate,t_gate,"blk.%d.ffn_gate.weight"); AT(up,t_up,"blk.%d.ffn_up.weight");
            AT(down,t_down,"blk.%d.ffn_down.weight");
        } else {
            L->is_moe = 1;
            snprintf(k,sizeof(k),"blk.%d.ffn_gate_exps.weight",i); L->e_gate=gguf_find(&M.m,k);
            snprintf(k,sizeof(k),"blk.%d.ffn_up_exps.weight",i);   L->e_up  =gguf_find(&M.m,k);
            snprintf(k,sizeof(k),"blk.%d.ffn_down_exps.weight",i); L->e_down=gguf_find(&M.m,k);
            AT(sh_gate,t_sh_gate,"blk.%d.ffn_gate_shexp.weight");
            AT(sh_up,t_sh_up,"blk.%d.ffn_up_shexp.weight");
            AT(sh_down,t_sh_down,"blk.%d.ffn_down_shexp.weight");
            snprintf(k,sizeof(k),"blk.%d.ffn_gate_inp.weight",i);
            L->router=(const float*)gguf_data(&M.m,gguf_find(&M.m,k));
            snprintf(k,sizeof(k),"blk.%d.exp_probs_b.bias",i);
            t=gguf_find(&M.m,k); L->router_bias = t ? (const float*)gguf_data(&M.m,t) : NULL;
        }
        #undef AT
        #undef NM
    }
    const gguf_tensor *kb = gguf_find(&M.m,"blk.0.attn_k_b.weight");
    const gguf_tensor *vb = gguf_find(&M.m,"blk.0.attn_v_b.weight");
    c->qk_nope = (int)kb->ne[0];  c->v_head = (int)vb->ne[1];

    /* ---- the prompt, in GLM's chat format ---- */
    /* The OFFICIAL template, from tokenizer.chat_template in the GGUF. The ending is the
     * part I got wrong by hand and it is the part that matters: the assistant turn opens
     * with <think>. Leave it out and the model believes it is somewhere else in the format
     * and drifts — which is exactly what it did.
     *
     * <think></think> (an EMPTY reasoning block) tells GLM-5.2 to skip reasoning and answer
     * directly. With a bare <think> it would reason for hundreds of tokens first, which is
     * correct behaviour and useless for a smoke test at 20 s/token. */
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
             "[gMASK]<sop><|user|>%s<|assistant|><think></think>", user);
    int toks[2048];
    int n_prompt = tok_encode(&T, prompt, (int)strlen(prompt), toks, 2048);

    const int max_t = n_prompt + n_gen + 8;
    MLACfg mc = { c->n_head, c->qk_nope, c->qk_rope, c->v_head, c->kv_lora, c->q_lora,
                  c->d_model, c->eps, 1.0f/sqrtf((float)(c->qk_nope + c->qk_rope)) };
    M.kv = calloc(c->n_layer, sizeof(MLACache));
    for(int i=0;i<n_run;i++) mla_cache_init(&M.kv[i], &mc, max_t);
    mla_scratch_init(&M.s, &mc, max_t);

    ESLayer *EL = calloc(n_run, sizeof(ESLayer));
    for(int i=c->n_dense;i<n_run;i++){
        EL[i].g = M.L[i].e_gate; EL[i].u = M.L[i].e_up; EL[i].d = M.L[i].e_down;
    }
    /* Requantizing attention is not about making attention faster — at 2.8 ms/layer it is
     * already noise. It is about the RAM: the pinned attention weights are the expert cache's
     * only competitor, and the disk (10.3 ms/layer) is now the bottleneck. Every GB the
     * attention gives back is a GB of cache, which is a GB the drive does not have to read. */
    const char *aqn = getenv("ATTN_Q");
    const AQRecipe *RC = aq_find(aqn ? aqn : "free");
    if(!RC){ fprintf(stderr, "unknown ATTN_Q recipe '%s'; have:", aqn);
             for(int i=0;i<AQ_N;i++) fprintf(stderr, " %s", AQ_RECIPES[i].name);
             fprintf(stderr, "\n"); return 1; }

    /* Size the expert cache from what is ACTUALLY left, not from a constant.
     *
     * The attention weights get pinned later, and pinned memory is unevictable — so a cache
     * hard-coded to 6 GB either wastes RAM the attention did not need, or collides with it.
     * Both are decided by the same budget, so decide them together: attention first (it has
     * no choice about its size), the cache takes the rest, and 2 GB stays free for the system.
     *
     * This is why requantizing attention is worth anything at all. It does not make attention
     * faster; it hands its savings to the cache, and the cache is what the disk feels. */
    /* The attention goes to VRAM, so it costs NO host RAM — that is the entire point of
     * requantizing it. What still lives in pinned host memory is the shared expert (2.07 GB),
     * which streams per layer and does not fit in VRAM alongside the attention. Everything
     * else the machine has left belongs to the expert cache, because the disk is what we are
     * actually fighting. */
    int64_t host_pinned = 0;
    {
        const char *sn3[3] = { "ffn_gate_shexp","ffn_up_shexp","ffn_down_shexp" };
        char nm[128];
        for(int i=c->n_dense;i<n_run;i++) for(int q=0;q<3;q++){
            snprintf(nm,sizeof(nm),"blk.%d.%s.weight",i,sn3[q]);
            const gguf_tensor *tt = gguf_find(&M.m, nm);
            if(tt) host_pinned += (tt->bytes + 255) & ~255LL;
        }
    }
    int64_t avail = 0;
    { FILE *mi = fopen("/proc/meminfo","r");
      if(mi){ char ln[256];
          while(fgets(ln,sizeof(ln),mi))
              if(sscanf(ln,"MemAvailable: %ld kB",&avail)==1) break;
          fclose(mi); avail *= 1024; } }
    const int64_t RESERVE = ES_RESERVE;   /* the system floor, COLIBRI_RESERVE_GB (default 10) */

    EStream S;
    const char *cg = getenv("CACHE_GB");
    /* This is an UPPER BOUND, not a promise. es_alloc checks MemAvailable before every slot
     * and refuses once the 3 GB floor is in sight, so the cache lands wherever the machine can
     * actually afford — no matter what this number says. That is the whole point: an
     * unevictable allocation may never be sized from a budget computed earlier. */
    int64_t cache_bytes;
    if(cg) cache_bytes = (int64_t)(atof(cg) * 1e9);
    else {
        cache_bytes = avail - host_pinned - RESERVE;
        if(cache_bytes < 1LL<<30) cache_bytes = 1LL<<30;
    }
    fprintf(stderr, "[mem] %.1f GB available: shared expert %.1f pinned, cache %.1f, "
            "%.1f reserved (attention '%s' is in VRAM)\n",
            avail/1e9, host_pinned/1e9, cache_bytes/1e9, RESERVE/1e9, RC->name);

    /* ORDER MATTERS, and getting it wrong cost 19 ms/layer.
     *
     * The shared expert (2.08 GB) MUST be pinned — it uploads to the GPU on every layer, and
     * out of pageable memory that upload blocks. The expert cache, by contrast, is elastic:
     * it takes whatever is left and works fine with less.
     *
     * I sized the cache first. It grew to 15.5 GB, the shared expert then found only 4.9 GB
     * against a 3 GB floor, could not be pinned, and MoE went from 6.2 to 25.5 ms per layer.
     * Fixed allocations first; the thing that can shrink goes last. */
    {
        const char *sn3[3] = { "ffn_gate_shexp","ffn_up_shexp","ffn_down_shexp" };
        int64_t shb = 0; char nm3[128];
        for(int i=c->n_dense;i<n_run;i++) for(int q=0;q<3;q++){
            snprintf(nm3,sizeof(nm3),"blk.%d.%s.weight",i,sn3[q]);
            const gguf_tensor *tt = gguf_find(&M.m, nm3);
            if(tt) shb += (tt->bytes + 255) & ~255LL;
        }
        uint8_t *shp = NULL;
        if(!pin_alloc((void**)&shp, shb, "shared expert")){
            fprintf(stderr, "[gpu] shared expert unpinned — its uploads will BLOCK\n");
        } else {
            int64_t o = 0;
            for(int i=c->n_dense;i<n_run;i++){
                void **d3[3] = { (void**)&M.L[i].sh_gate, (void**)&M.L[i].sh_up,
                                 (void**)&M.L[i].sh_down };
                for(int q=0;q<3;q++){
                    snprintf(nm3,sizeof(nm3),"blk.%d.%s.weight",i,sn3[q]);
                    const gguf_tensor *tt = gguf_find(&M.m, nm3);
                    if(!tt) continue;
                    memcpy(shp + o, gguf_data(&M.m, tt), (size_t)tt->bytes);
                    *d3[q] = shp + o;
                    o += (tt->bytes + 255) & ~255LL;
                }
            }
            fprintf(stderr, "[mem] shared expert pinned (%.2f GB)\n", shb/1e9);
        }
    }


    /* ---- everything with a FIXED, MANDATORY size goes first ----
     *
     * Twice now I have put the expert cache before something it had to leave room for, and
     * twice the cache ate the budget and the thing it starved fell back to a 60x slower path
     * with no error — just a timer reading 0.0 ms. The cache is the ONLY elastic allocation
     * here: it works with whatever it gets. So it goes LAST, always, and everything with a
     * size it cannot negotiate goes before it. */
    MoeGpu G; AttnGpu AG; Prefetch PF; int gpu = 0;
    GpuCtx ctx = { &G, &M, &S, &AG };
    ctx.vocab = c->vocab; ctx.d_model = c->d_model;

    int cuda_ok = 0;
    if(!getenv("NOGPU")){
        cuda_ok = kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs,
                                      kmask_iq2xs, kvalues_iq4nl);
        if(!cuda_ok) fprintf(stderr, "[gpu] table upload failed, staying on CPU\n");
    }

    if(cuda_ok){
        /* the output projection: 535 MB in VRAM, once, no PCIe per token */
        const gguf_tensor *owt = gguf_find(&M.m, "output.weight");
        if(cudaMalloc((void**)&ctx.out_w, (size_t)owt->bytes) == cudaSuccess &&
           cudaMalloc((void**)&ctx.out_d, (size_t)c->vocab*sizeof(float)) == cudaSuccess &&
           cudaStreamCreate(&ctx.out_s) == cudaSuccess &&
           cudaMemcpy(ctx.out_w, gguf_data(&M.m, owt), (size_t)owt->bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess){
            ctx.out_t = owt->type;
            M.out = out_hook; M.out_ud = &ctx;
            fprintf(stderr, "[gpu] output projection resident in VRAM (%.0f MB, %s)\n",
                    owt->bytes/1e6, kq_name(owt->type));
        } else fprintf(stderr, "[gpu] output projection stays on the CPU\n");

            /* ---------------- attention: requantize, then keep it in VRAM ----------------
             *
             * The attention weights never change and are touched on every token, so streaming
             * them is pure waste — but at 10.25 GB they do not fit in 8 GB of VRAM, which is
             * the only reason they were ever streamed. Requantizing to ~4 bit/W gets them to
             * 6.35 GB, and then they can simply LIVE there.
             *
             * Two things fall out of that, and the second is the one that matters:
             *   - PCIe traffic for attention drops to zero (it was 2.8 ms/layer — small).
             *   - The ~12.5 GB of pinned, unevictable host RAM they were holding is FREED,
             *     and goes to the expert cache. The disk is the bottleneck now, and the cache
             *     is the only thing that makes it read less.
             *
             * Note the weights go mmap -> quantizer -> VRAM one layer at a time, through a
             * staging buffer of a single layer. There is never a 10 GB host copy at all. */
            int64_t mn[6] = {0,0,0,0,0,0};
            int rt[6];                              /* the type each tensor ENDS UP as */
            for(int q=0;q<6;q++){
                int64_t O,I;
                aq_shape(q, c->d_model, c->q_lora, c->kv_lora, c->n_head,
                         c->qk_nope, c->qk_rope, c->v_head, &O, &I);
                rt[q] = aq_can(RC->t[q], I) ? RC->t[q] : -1;
            }
            for(int i=0;i<n_run;i++){
                MLALayer Li = {
                    .tq_a =rt[0]>=0?rt[0]:M.L[i].t_q_a,  .tq_b =rt[1]>=0?rt[1]:M.L[i].t_q_b,
                    .tkv_a=rt[2]>=0?rt[2]:M.L[i].t_kv_a, .tk_b =rt[3]>=0?rt[3]:M.L[i].t_k_b,
                    .tv_b =rt[4]>=0?rt[4]:M.L[i].t_v_b,  .to   =rt[5]>=0?rt[5]:M.L[i].t_o };
                int64_t n6[6]; ag_sizes(&mc, &Li, n6);
                for(int q=0;q<6;q++) if(n6[q] > mn[q]) mn[q] = n6[q];
            }

            /* TWO ways to keep attention on the GPU, and BOTH must exist.
             *
             * Residency is better — zero PCIe, zero host RAM — but it needs the requantized
             * weights to fit in VRAM, which only 'hard' and below do. When they do not fit,
             * the answer is NOT to fall back to the CPU: CPU attention decodes k-quants
             * scalar at 0.85 GB/s and costs 156 ms/layer, which is 15.9 s/token. Deleting
             * this fallback when I built residency made the engine 7x SLOWER for every recipe
             * that does not fit, and the only symptom was a timer reading 0.0 ms.
             *
             * So: resident if it fits, streamed from pinned host RAM if it does not. */
            int attn_ok = 0;
            /* Same weights, same recipe, two different upload paths. If the text differs
             * between them, the bug is in MY residency code, not in the quantization — and I
             * would otherwise have blamed the quantizer for corrupting the model. */
            const int force_stream = getenv("ATTN_STREAM") != NULL;
            /* ATTN_CPU=1 keeps attention on the CPU, whose MLA is the one validated against
             * llama.cpp. It is 60x slower and it is the only ground truth I have. */
            const int want_cpu_attn = getenv("ATTN_CPU") != NULL;
            if(!want_cpu_attn && !force_stream &&
               attn_gpu_init(&AG, &mc, mn, max_t) && attn_gpu_reside(&AG, n_run)){
                uint8_t *stg = NULL;
                if(!pin_alloc((void**)&stg, AG.cap, "attention staging")){
                    fprintf(stderr, "[gpu] no staging buffer; attention stays on CPU\n");
                } else {
                    const char *tn6[6] = { "attn_q_a","attn_q_b","attn_kv_a_mqa",
                                           "attn_k_b","attn_v_b","attn_output" };
                    fprintf(stderr, "[gpu] requantizing attention to '%s' and uploading...\n",
                            RC->name);
                    const double tq0 = now();
                    char nm2[128];
                    for(int i=0;i<n_run;i++){
                        int *tdst[6] = { &M.L[i].t_q_a, &M.L[i].t_q_b, &M.L[i].t_kv_a,
                                         &M.L[i].t_k_b, &M.L[i].t_v_b, &M.L[i].t_o };
                        for(int q=0;q<6;q++){
                            snprintf(nm2,sizeof(nm2),"blk.%d.%s.weight",i,tn6[q]);
                            const gguf_tensor *tt = gguf_find(&M.m, nm2);
                            if(!tt){ fprintf(stderr,"missing %s\n",nm2); return 1; }
                            int64_t O,I;
                            aq_shape(q, c->d_model, c->q_lora, c->kv_lora, c->n_head,
                                     c->qk_nope, c->qk_rope, c->v_head, &O, &I);
                            if(rt[q] >= 0){
                                aq_tensor(stg + AG.off[q], gguf_data(&M.m, tt), tt->type,
                                          rt[q], O, I);
                                *tdst[q] = rt[q];   /* the kernel must decode the NEW type */
                            } else {
                                memcpy(stg + AG.off[q], gguf_data(&M.m, tt), (size_t)tt->bytes);
                            }
                        }
                        if(!attn_gpu_put(&AG, i, stg)){
                            fprintf(stderr, "[gpu] upload of layer %d failed\n", i); return 1; }
                    }
                    cudaFreeHost(stg);
                    fprintf(stderr, "[gpu] attention in VRAM after %.0f s — host RAM freed\n",
                            now() - tq0);
                    M.attn = attn_hook; M.attn_ud = &ctx; attn_ok = 1;
                }
            }

            if(!want_cpu_attn && !attn_ok && force_stream && !attn_gpu_init(&AG, &mc, mn, max_t))
                fprintf(stderr, "[gpu] attn init failed\n");
            if(!want_cpu_attn && !attn_ok){
                /* Does not fit in VRAM. Stream it from PINNED host RAM instead — still on the
                 * GPU, still a real DMA, just paying 2.8 ms/layer of PCIe and holding the RAM.
                 * Far better than the CPU, which costs 156 ms/layer. */
                int64_t need = 0; char nm2[128];
                const char *tn6[6] = { "attn_q_a","attn_q_b","attn_kv_a_mqa",
                                       "attn_k_b","attn_v_b","attn_output" };
                for(int i=0;i<n_run;i++) for(int q=0;q<6;q++){
                    snprintf(nm2,sizeof(nm2),"blk.%d.%s.weight",i,tn6[q]);
                    const gguf_tensor *tt = gguf_find(&M.m, nm2);
                    if(!tt) continue;
                    int64_t O,I;
                    aq_shape(q, c->d_model, c->q_lora, c->kv_lora, c->n_head,
                             c->qk_nope, c->qk_rope, c->v_head, &O, &I);
                    need += rt[q] >= 0 ? ((kq_row_bytes(rt[q], O*I) + 255) & ~255LL)
                                       : ((tt->bytes + 255) & ~255LL);
                }
                uint8_t *pin = NULL;
                if(!pin_alloc((void**)&pin, need, "attention (streamed)")){
                    fprintf(stderr, "[gpu] attention stays on the CPU — expect ~16 s/token\n");
                } else {
                    fprintf(stderr, "[gpu] attention streamed from %.2f GB pinned RAM "
                            "(does not fit in VRAM)\n", need/1e9);
                    int64_t o = 0;
                    for(int i=0;i<n_run;i++){
                        void **dst[6] = { (void**)&M.L[i].q_a, (void**)&M.L[i].q_b,
                                          (void**)&M.L[i].kv_a, (void**)&M.L[i].k_b,
                                          (void**)&M.L[i].v_b,  (void**)&M.L[i].o };
                        int *tds[6] = { &M.L[i].t_q_a, &M.L[i].t_q_b, &M.L[i].t_kv_a,
                                        &M.L[i].t_k_b, &M.L[i].t_v_b, &M.L[i].t_o };
                        for(int q=0;q<6;q++){
                            snprintf(nm2,sizeof(nm2),"blk.%d.%s.weight",i,tn6[q]);
                            const gguf_tensor *tt = gguf_find(&M.m, nm2);
                            if(!tt) continue;
                            int64_t O,I;
                            aq_shape(q, c->d_model, c->q_lora, c->kv_lora, c->n_head,
                                     c->qk_nope, c->qk_rope, c->v_head, &O, &I);
                            int64_t nb;
                            if(rt[q] >= 0){
                                aq_tensor(pin + o, gguf_data(&M.m, tt), tt->type, rt[q], O, I);
                                *tds[q] = rt[q];
                                nb = kq_row_bytes(rt[q], O*I);
                            } else {
                                memcpy(pin + o, gguf_data(&M.m, tt), (size_t)tt->bytes);
                                nb = tt->bytes;
                            }
                            *dst[q] = pin + o;
                            o += (nb + 255) & ~255LL;
                        }
                    }
                    M.attn = attn_hook; M.attn_ud = &ctx;
                }
            }

    }

    /* ---- and only now the cache, with what is genuinely left ---- */
    if(!es_init(&S, &M.m, EL, n_run, c->n_exp, c->topk, cache_bytes)) return 1;
    fprintf(stderr, "[mem] cache settled at %.1f GB (%d experts); %.1f GB still available\n",
            (double)S.n_slots*S.slot_bytes/1e9, S.n_slots, es_mem_available()/1e9);

    /* ---- the GPU expert path (needs the streamer's region layout, so it comes after) ---- */

    /* The GPU's expert staging is sized from the ROUTED experts, but the shared expert now
     * goes through the same buffer and it is FATTER: 27.6 MB (Q5_K/Q5_K/Q6_K) against 12.1
     * (IQ2_XS/IQ2_XS/IQ3_XXS). Sizing from the routed ones would overrun each region into the
     * next — the same silent corruption that blk.8's Q6_K caused in the attention upload. */
    int64_t wreg[3] = { S.roff[1]-S.roff[0], S.roff[2]-S.roff[1], S.ring_bytes-S.roff[2] };
    {
        const char *sn[3] = { "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
        int64_t shmax[3] = {0,0,0};
        for(int i=c->n_dense;i<n_run;i++) for(int q=0;q<3;q++){
            char nm[128]; snprintf(nm,sizeof(nm),"blk.%d.%s.weight",i,sn[q]);
            const gguf_tensor *t = gguf_find(&M.m, nm);
            if(t && t->bytes > shmax[q]) shmax[q] = t->bytes;
        }
        ctx.sh_ng = shmax[0]; ctx.sh_nu = shmax[1]; ctx.sh_nd = shmax[2];
        for(int q=0;q<3;q++){
            int64_t pad = (shmax[q] + 255) & ~255LL;
            if(pad > wreg[q]) wreg[q] = pad;
        }
    }
    int64_t woff[3] = { 0, wreg[0], wreg[0]+wreg[1] };
    const int64_t wcap = wreg[0]+wreg[1]+wreg[2];

    if(cuda_ok){
        if(!moe_gpu_init(&G, c->d_model, c->d_ff_exp, c->topk + 1, wcap, woff))
            fprintf(stderr, "[gpu] moe init failed, experts stay on the CPU\n");
        else {
            M.moe = moe_hook; M.moe_ud = &ctx; M.moe_shared = 1; gpu = 1;
            fprintf(stderr, "[gpu] routed experts + shared expert on the GPU (int8/dp4a)\n");
        }
    }
    /* Keep the drive busy: the forward pass cannot ask for layer L+1's experts until layer L
     * has run the router, so the disk would otherwise stop during every compute. The
     * prefetcher spends that idle time pulling the hottest expert that is not yet cached. */
    glm_route_log_init(c->n_layer, c->topk);
    if(!prefetch_start(&PF, &S)) return 1;

    printf("\nprompt (%d tokens): %s\n", n_prompt, user);
    printf("---------------------------------------------------------------\n");
    fflush(stdout);

    float *logits = malloc((size_t)c->vocab*sizeof(float));
    int keep = c->n_layer; c->n_layer = n_run;      /* skip the MTP head */

    /* Greedy text on a predictable prompt is a WEAK oracle: "Ein **Mutex** (kurz für *Mutual
     * Ex" is near-deterministic, so the argmax survives a logit perturbation that would wreck
     * a genuinely uncertain choice. Dump the logits and compare the DISTRIBUTION. */
    const char *ldump = getenv("LOGIT_DUMP");

    /* Greedy text on a predictable prompt is a WEAK oracle. "Ein **Mutex** (kurz für *Mutual
     * Ex" is a near-deterministic continuation, so the argmax survives a logit perturbation
     * that would wreck a genuinely uncertain choice — all five recipes produced byte-identical
     * text, including the one with a 3.8% per-layer error. That proves nothing. Dump the
     * logits and compare the DISTRIBUTION against the unmodified weights. */
    
    double t0 = now();
    int pos = 0;
    /* prefill: run the prompt through, keep only the last token's logits */
    /* Dump EVERY prefill position, not just the last.
     *
     * One position is a sample size of one — and the thing being sampled is discrete. The
     * attention output feeds the router, which picks 8 of 256 experts in each of 75 layers, so
     * a small weight perturbation does not shift the logits smoothly: it FLIPS routing
     * decisions, and each flip is a step change. That is why the mild recipe measured WORSE
     * than the aggressive one on a single prompt — not noise (the engine is bit-exact
     * reproducible), but a sample of one over a coin-flip. Average over the whole prompt. */
    FILE *lf = ldump ? fopen(ldump, "wb") : NULL;
    for(int i=0;i<n_prompt;i++){
        glm_forward(&M, toks[i], pos++, logits, es_fetch_timed, &S);
        if(lf) fwrite(logits, sizeof(float), (size_t)c->vocab, lf);
    }
    double t_pre = now() - t0;
    if(lf){ fclose(lf);
        fprintf(stderr, "[dbg] %d x %d logits -> %s\n", n_prompt, c->vocab, ldump); }
    if(ldump){
        FILE *f = fopen(ldump, "wb");
        fwrite(logits, sizeof(float), (size_t)c->vocab, f);
        fclose(f);
        fprintf(stderr, "[dbg] wrote %d prompt logits to %s\n", c->vocab, ldump);
    }

    char piece[64];
    double t1 = now();
    for(int g=0; g<n_gen; g++){
        int best = 0;                               /* greedy: no sampling noise to explain */
        for(int i=1;i<c->vocab;i++) if(logits[i] > logits[best]) best = i;
        if(best == ids.eos || best == ids.eot) break;

        int m = tok_decode(&T, &best, 1, piece, sizeof(piece));
        fwrite(piece, 1, m, stdout); fflush(stdout);

        glm_forward(&M, best, pos++, logits, es_fetch, &S);
    }
    double t_gen = now() - t1;
    c->n_layer = keep;

    printf("\n---------------------------------------------------------------\n");
    printf("prefill %d tokens in %.0f s (%.1f s/token)\n", n_prompt, t_pre, t_pre/n_prompt);
    printf("decode  %.1f s/token\n", t_gen/n_gen);
    printf("experts: %llu hits, %llu misses (%.0f%% cached)\n",
           (unsigned long long)S.hits, (unsigned long long)S.misses,
           100.0*S.hits/(S.hits + S.misses + 1));
    printf("prefetcher: %llu experts pulled in during idle disk time\n",
           (unsigned long long)PF.fetched);
    printf("moe on: %s\n", gpu ? "GPU (int8/dp4a)" : "CPU");

    extern double GLM_margin_sum; extern long GLM_margin_n, GLM_margin_hist[6];
    if(GLM_margin_n){
        printf("\nrouter margin (score of the 8th expert minus the 9th — the one that lost)\n");
        printf("  mean margin %.2e over %ld decisions\n", GLM_margin_sum/GLM_margin_n, GLM_margin_n);
        const char *lb[6] = { "< 1e-6  (a coin flip!)", "< 1e-5", "< 1e-4",
                              "< 1e-3", "< 1e-2", ">= 1e-2 (safe)" };
        for(int i=0;i<6;i++)
            printf("    %-22s %6.2f%%\n", lb[i], 100.0*GLM_margin_hist[i]/GLM_margin_n);
    }

    extern long GLM_route_hits, GLM_route_total, GLM_route_hist[9];
    if(GLM_route_total){
        printf("\nrouting stability (does a layer pick the same experts as last token?)\n");
        printf("  %ld of %ld picks repeated = %.1f%%\n", GLM_route_hits, GLM_route_total,
               100.0*GLM_route_hits/GLM_route_total);
        printf("  of 8 picks, how many repeated:\n");
        long tot = 0; for(int i=0;i<9;i++) tot += GLM_route_hist[i];
        for(int i=0;i<9;i++) if(GLM_route_hist[i])
            printf("    %d/8  %5.1f%%  %s\n", i, 100.0*GLM_route_hist[i]/tot,
                   "########################################" + (40 - (int)(40.0*GLM_route_hist[i]/tot)));
        printf("\n  A predictive prefetch is worth building only if this is high.\n");
    }

    /* Divide each timer by ITS OWN count. Dividing everything by N_attn printed 0.0 ms for
     * every line the moment attention fell back to the CPU (N_attn = 0) — so the profile said
     * "nothing takes any time" while the engine ran at 15.9 s/token. A profiler that goes
     * quiet exactly when something breaks is worse than none. */
    printf("\nwhere a layer's time goes:\n");
    printf("  attention (incl. PCIe)  %7.1f ms  %s\n",
           N_attn ? 1e3*T_attn/N_attn : 0.0,
           N_attn ? "(GPU)" : "(ON THE CPU — the GPU path did not engage!)");
    printf("  moe compute (GPU)       %7.1f ms\n", N_moe  ? 1e3*T_moe /N_moe  : 0.0);
    printf("  disk wait (es_fetch)    %7.1f ms\n", N_disk ? 1e3*T_disk/N_disk : 0.0);
    const double per = N_disk ? 1e3/N_disk : 0;
    printf("  ----------------------------------\n");
    printf("  dense ffn (CPU, 3 lyrs) %7.1f ms\n", GLM_T_dense*per);
    printf("  router (CPU)            %7.1f ms\n", GLM_T_route*per);
    printf("  ----------------------------------\n");
    printf("  accounted               %7.1f ms\n",
           (T_attn+T_moe+T_disk+GLM_T_dense+GLM_T_route)*per);
    prefetch_stop(&PF);
    es_free(&S);
    return 0;
}
