/* kv_test.c — (1) is q4_0 safe on the MLA latent?  (2) does the prefix cache work?
 *
 * (1) THE QUANTIZATION QUESTION IS NOT ACADEMIC. In normal attention the KV cache holds K
 *     and V per head, which are redundant and forgiving. MLA's cache holds the LATENT: 512
 *     numbers that every one of the 64 heads reconstructs its key and value from. It is
 *     dense, compressed information — there is no slack in it. So we do not assume q4_0 is
 *     fine because llama.cpp quantizes normal KV caches; we measure what it does to the
 *     ATTENTION SCORES, on real GLM-5.2 latents computed from real weights.
 *
 *     The score is what matters, not the latent's own RMSE: attention is a softmax, and a
 *     small score error in the wrong place reorders which tokens the model looks at.
 *
 * (2) The prefix cache is the highest-value piece in the engine (prefill reads ~235 GB and
 *     takes ~44 s; a cached 10k-token prefix is 244 MB and 47 ms). So it has to be right:
 *     a wrong prefix match silently answers with someone else's context.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../gguf.h"
#include "../mla.h"
#include "../kvstore.h"

#define NTOK 64          /* pseudo-tokens: enough latents to get a stable score distribution */

static int fails = 0;
static void ck(int ok, const char *msg){
    printf("  %s %s\n", ok ? "ok  " : "FAIL", msg);
    if(!ok) fails++;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }

    const char *arch = gguf_str(&m, "general.architecture", "");
    char k[128];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    #define KVF(s) (snprintf(k,sizeof(k),"%s." s,arch), (float)gguf_f64(&m,k,0))
    const int d_model = KVI("embedding_length");
    const int n_head  = KVI("attention.head_count");
    const int kv_lora = KVI("attention.kv_lora_rank");
    const int rope    = KVI("rope.dimension_count");
    const float eps   = KVF("attention.layer_norm_rms_epsilon");
    #undef KVI
    #undef KVF
    const int NV = kv_lora + rope;              /* 576: what one token costs, for ALL heads */

    #define T(n) gguf_find(&m, n)
    const gguf_tensor *emb  = T("token_embd.weight");
    const gguf_tensor *an   = T("blk.0.attn_norm.weight");
    const gguf_tensor *kv_a = T("blk.0.attn_kv_a_mqa.weight");
    const gguf_tensor *kvan = T("blk.0.attn_kv_a_norm.weight");
    const gguf_tensor *q_a  = T("blk.0.attn_q_a.weight");
    const gguf_tensor *qan  = T("blk.0.attn_q_a_norm.weight");
    const gguf_tensor *q_b  = T("blk.0.attn_q_b.weight");
    const gguf_tensor *k_b  = T("blk.0.attn_k_b.weight");
    #undef T
    if(!emb||!an||!kv_a||!kvan||!q_a||!qan||!q_b||!k_b){ printf("missing tensor\n"); return 1; }

    const int qk_nope = (int)k_b->ne[0];
    const int q_lora  = (int)q_a->ne[1];
    const int qk = qk_nope + rope;

    /* ---- real latents for NTOK real embedding rows ---- */
    float *x   = malloc((size_t)d_model*sizeof(float));
    float *xn  = malloc((size_t)d_model*sizeof(float));
    float *kva = malloc((size_t)NV*sizeof(float));
    float *lat = malloc((size_t)NTOK*NV*sizeof(float));
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);

    for(int t=0;t<NTOK;t++){
        int tok = 1000 + t*137;                        /* spread across the vocab */
        kq_dequant_row(emb->type, (const uint8_t*)gguf_data(&m,emb) + (size_t)tok*erb, x, d_model);
        mla_rms(xn, x, (const float*)gguf_data(&m,an), d_model, eps);
        kq_gemm(kva, xn, gguf_data(&m,kv_a), kv_a->type, 1, d_model, NV);
        float *e = lat + (size_t)t*NV;
        mla_rms(e, kva, (const float*)gguf_data(&m,kvan), kv_lora, eps);   /* the latent */
        memcpy(e + kv_lora, kva + kv_lora, (size_t)rope*sizeof(float));    /* the rope key */
    }

    /* ---- a real query, pushed into latent space (the absorbed form) ---- */
    float *qa = malloc((size_t)q_lora*sizeof(float));
    float *q  = malloc((size_t)n_head*qk*sizeof(float));
    kq_dequant_row(emb->type, (const uint8_t*)gguf_data(&m,emb) + (size_t)77451*erb, x, d_model);
    mla_rms(xn, x, (const float*)gguf_data(&m,an), d_model, eps);
    kq_gemm(qa, xn, gguf_data(&m,q_a), q_a->type, 1, d_model, q_lora);
    mla_rms(qa, qa, (const float*)gguf_data(&m,qan), q_lora, eps);
    kq_gemm(q, qa, gguf_data(&m,q_b), q_b->type, 1, q_lora, n_head*qk);

    float *qlat = malloc((size_t)kv_lora*sizeof(float));
    const int64_t kbrow = kq_row_bytes(k_b->type, (int64_t)qk_nope*kv_lora);
    kq_gemm(qlat, q, gguf_data(&m,k_b), k_b->type, 1, qk_nope, kv_lora);   /* head 0 */
    (void)kbrow;

    /* ---- what each format does to the ATTENTION SCORES ---- */
    printf("MLA latent quantization, measured on real GLM-5.2 layer-0 latents\n");
    printf("(the latent is 512 numbers that all 64 heads rebuild K and V from — it has no slack)\n\n");
    printf("  %-6s %9s %14s %16s %14s\n",
           "format", "KB/token", "latent RMSE", "score RMSE", "top-1 preserved");

    float *ref = malloc((size_t)NTOK*sizeof(float));
    for(int t=0;t<NTOK;t++){
        const float *e = lat + (size_t)t*NV;
        double s = 0;
        for(int i=0;i<kv_lora;i++) s += (double)qlat[i]*e[i];
        for(int i=0;i<rope;i++)    s += (double)q[qk_nope+i]*e[kv_lora+i];
        ref[t] = (float)s;
    }
    int ref_top = 0;
    for(int t=1;t<NTOK;t++) if(ref[t] > ref[ref_top]) ref_top = t;

    void  *packed = malloc((size_t)NV*4);
    float *back   = malloc((size_t)NV*sizeof(float));
    float *sc     = malloc((size_t)NTOK*sizeof(float));

    int fmts[4] = { KV_F32, KV_F16, KV_Q8_0, KV_Q4_0 };
    for(int f=0; f<4; f++){
        const int fmt = fmts[f];
        double se_lat = 0, sr_lat = 0, se_sc = 0, sr_sc = 0;
        for(int t=0;t<NTOK;t++){
            const float *e = lat + (size_t)t*NV;
            kv_pack(packed, e, NV, fmt);
            kv_unpack(back, packed, NV, fmt);
            for(int i=0;i<NV;i++){ double d = e[i]-back[i]; se_lat += d*d; sr_lat += (double)e[i]*e[i]; }
            double s = 0;
            for(int i=0;i<kv_lora;i++) s += (double)qlat[i]*back[i];
            for(int i=0;i<rope;i++)    s += (double)q[qk_nope+i]*back[kv_lora+i];
            sc[t] = (float)s;
            double d = sc[t]-ref[t]; se_sc += d*d; sr_sc += (double)ref[t]*ref[t];
        }
        int top = 0;
        for(int t=1;t<NTOK;t++) if(sc[t] > sc[top]) top = t;

        printf("  %-6s %8.1f  %12.3f%% %14.3f%% %14s\n",
               kv_fmt_name(fmt), kv_entry_bytes(fmt,NV)*79/1024.0,
               100.0*sqrt(se_lat/sr_lat), 100.0*sqrt(se_sc/sr_sc),
               top == ref_top ? "yes" : "NO");
    }

    /* ---- the prefix cache ---- */
    printf("\npersistent prefix cache\n");
    system("rm -rf /tmp/kvs_test && mkdir -p /tmp/kvs_test");
    kvstore S;
    ck(kvs_open(&S, "/tmp/kvs_test", 35ull<<30, 79, NV, KV_Q4_0), "opened (35 GB budget)");

    /* a "repo context" of 3 blocks, then two different questions after it */
    const int NB = 3;
    int *toks = malloc((size_t)NB*KVS_BLOCK*sizeof(int));
    for(int i=0;i<NB*KVS_BLOCK;i++) toks[i] = 5000 + (i*31 % 4000);

    float *blk = calloc((size_t)79*KVS_BLOCK*NV, sizeof(float));
    for(size_t i=0;i<(size_t)79*KVS_BLOCK*NV;i++) blk[i] = (float)sin(i*0.001);

    uint64_t keys[16];
    ck(kvs_match(&S, toks, NB*KVS_BLOCK, keys, 16) == 0, "cold cache: nothing matches");

    uint64_t parent = 0;
    for(int b=0;b<NB;b++){
        uint64_t key = kvs_hash(parent, toks + b*KVS_BLOCK, KVS_BLOCK);
        ck(kvs_put(&S, key, parent, toks + b*KVS_BLOCK, KVS_BLOCK, blk), "stored a block");
        parent = key;
    }
    ck(kvs_match(&S, toks, NB*KVS_BLOCK, keys, 16) == NB, "warm cache: all 3 blocks match");

    /* the same context with a DIFFERENT question appended must still reuse the prefix */
    int *toks2 = malloc((size_t)(NB*KVS_BLOCK + 50)*sizeof(int));
    memcpy(toks2, toks, (size_t)NB*KVS_BLOCK*sizeof(int));
    for(int i=0;i<50;i++) toks2[NB*KVS_BLOCK + i] = 900 + i;
    ck(kvs_match(&S, toks2, NB*KVS_BLOCK + 50, keys, 16) == NB,
       "same context, new question: prefix still hits");

    /* one token changed in the middle must invalidate everything from that block on */
    int save = toks[KVS_BLOCK + 7];
    toks[KVS_BLOCK + 7] = 424242;
    ck(kvs_match(&S, toks, NB*KVS_BLOCK, keys, 16) == 1,
       "one token changed in block 2: blocks 2 and 3 correctly miss");
    toks[KVS_BLOCK + 7] = save;

    /* round trip */
    kvs_ent *e0 = kvs_find(&S, kvs_hash(0, toks, KVS_BLOCK));
    float *rt = malloc((size_t)79*KVS_BLOCK*NV*sizeof(float));
    ck(e0 && kvs_get(&S, e0, rt), "read a block back from disk");
    if(e0){
        double se=0, sr=0;
        for(size_t i=0;i<(size_t)79*KVS_BLOCK*NV;i++){
            double d = blk[i]-rt[i]; se += d*d; sr += (double)blk[i]*blk[i];
        }
        printf("       round-trip error at q4_0: %.3f%%\n", 100.0*sqrt(se/sr));
    }

    printf("\n  one block (%d tokens, 79 layers, q4_0) = %.1f MB on disk\n",
           KVS_BLOCK, (double)e0->bytes/1e6);
    printf("  35 GB budget holds %.0f blocks = %.2f M tokens of context\n",
           (double)(35ull<<30)/e0->bytes, (double)(35ull<<30)/e0->bytes*KVS_BLOCK/1e6);

    kvs_flush(&S);
    printf("\n%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
