/* rope_test.c — the test that should have existed from the start.
 *
 * mla_test validated layer 0 against llama.cpp at position 0 and passed perfectly. It
 * proved nothing about RoPE: at pos = 0 the rotation angle is zero and RoPE is the
 * IDENTITY. Both possible pairings — NeoX halves (i, i+n/2) and NORM adjacent pairs
 * (2i, 2i+1) — give byte-identical results there.
 *
 * So the engine ran with NeoX (wrong for glm-dsa, which llama.cpp maps to
 * LLAMA_ROPE_TYPE_NORM), every position from 1 on was quietly wrong, and what came out was
 * a model that opens a sentence correctly and then loses the thread. Nothing crashed.
 * Nothing was out of range. It just stopped making sense, slowly.
 *
 * The fix is one line. The lesson is not: AN ORACLE FOR A POSITIONAL ENCODING MUST RUN AT
 * pos > 0, or it is not an oracle at all.
 *
 * Reference: llama-eval-callback, prompt "Hallo Welt wie geht es dir heute" (7 tokens),
 * layer 0, sums taken over ALL positions:
 *
 *      q_pe  before rope  -2491.340332      after rope  -2831.006104
 *      k_pe  before rope    -10.204261      after rope     -6.364964
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../gguf.h"
#include "../mla.h"

static int fails = 0;
static void ck(const char *what, double got, double want, double tol){
    double rel = fabs(got - want) / (fabs(want) + 1e-6);
    int ok = rel < tol;
    printf("  %-22s %14.4f   llama.cpp %14.4f   rel %.1e  %s\n",
           what, got, want, rel, ok ? "ok" : "MISMATCH");
    if(!ok) fails++;
}

/* the pairing we rejected, kept here so the test can show WHY it was rejected */
static void rope_neox(float *v, int n, int pos, float base){
    const int half = n/2;
    for(int i=0;i<half;i++){
        float ang = (float)pos * powf(base, -2.0f*(float)i/(float)n);
        float cs = cosf(ang), sn = sinf(ang);
        float a = v[i], b = v[i+half];
        v[i]      = a*cs - b*sn;
        v[i+half] = a*sn + b*cs;
    }
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }

    const char *arch = gguf_str(&m, "general.architecture", "");
    char k[128];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    #define KVF(s) (snprintf(k,sizeof(k),"%s." s,arch), (float)gguf_f64(&m,k,0))
    const int D = KVI("embedding_length"), H = KVI("attention.head_count");
    const int kv_lora = KVI("attention.kv_lora_rank");
    const int rope = KVI("rope.dimension_count");
    const int q_lora = KVI("attention.q_lora_rank");
    const float eps = KVF("attention.layer_norm_rms_epsilon");
    const float base = KVF("rope.freq_base");
    #undef KVI
    #undef KVF

    #define T(n) gguf_find(&m, n)
    const gguf_tensor *emb=T("token_embd.weight"), *an=T("blk.0.attn_norm.weight");
    const gguf_tensor *qa=T("blk.0.attn_q_a.weight"), *qan=T("blk.0.attn_q_a_norm.weight");
    const gguf_tensor *qb=T("blk.0.attn_q_b.weight"), *kva=T("blk.0.attn_kv_a_mqa.weight");
    const gguf_tensor *kb=T("blk.0.attn_k_b.weight");
    #undef T
    const int nope = (int)kb->ne[0], qk = nope + rope;

    printf("rope_base %.0f, rope dims %d, %d heads\n", base, rope, H);
    printf("prompt: \"Hallo Welt wie geht es dir heute\" (7 tokens, so pos 0..6)\n\n");

    const int toks[7] = { 77451, 44701, 13359, 39052, 1531, 5419, 48310 };

    float *x=malloc(D*4), *xn=malloc(D*4), *qav=malloc(q_lora*4);
    float *q=malloc((size_t)H*qk*4), *kv=malloc((size_t)(kv_lora+rope)*4);
    float *tmp=malloc((size_t)rope*4);
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);

    double q_pre=0, q_norm=0, q_neox=0, k_pre=0, k_norm=0, k_neox=0;

    for(int t=0;t<7;t++){
        kq_dequant_row(emb->type,(const uint8_t*)gguf_data(&m,emb)+(size_t)toks[t]*erb, x, D);
        mla_rms(xn, x, (const float*)gguf_data(&m,an), D, eps);

        kq_gemm(qav, xn, gguf_data(&m,qa), qa->type, 1, D, q_lora);
        mla_rms(qav, qav, (const float*)gguf_data(&m,qan), q_lora, eps);
        kq_gemm(q, qav, gguf_data(&m,qb), qb->type, 1, q_lora, H*qk);

        kq_gemm(kv, xn, gguf_data(&m,kva), kva->type, 1, D, kv_lora + rope);

        /* q_pe: the rope half of every head's query */
        for(int h=0; h<H; h++){
            const float *pe = q + (size_t)h*qk + nope;
            for(int i=0;i<rope;i++) q_pre += pe[i];

            memcpy(tmp, pe, (size_t)rope*4);
            mla_rope(tmp, rope, t, base);                      /* NORM: what we now do */
            for(int i=0;i<rope;i++) q_norm += tmp[i];

            memcpy(tmp, pe, (size_t)rope*4);
            rope_neox(tmp, rope, t, base);                     /* NEOX: what we did before */
            for(int i=0;i<rope;i++) q_neox += tmp[i];
        }

        /* k_pe: one shared rope key per token */
        const float *kpe = kv + kv_lora;
        for(int i=0;i<rope;i++) k_pre += kpe[i];

        memcpy(tmp, kpe, (size_t)rope*4);
        mla_rope(tmp, rope, t, base);
        for(int i=0;i<rope;i++) k_norm += tmp[i];

        memcpy(tmp, kpe, (size_t)rope*4);
        rope_neox(tmp, rope, t, base);
        for(int i=0;i<rope;i++) k_neox += tmp[i];
    }

    printf("before rotation (a pairing-independent sanity check)\n");
    ck("q_pe (pre-rope)", q_pre, -2491.340332, 5e-3);
    ck("k_pe (pre-rope)", k_pre,   -10.204261, 5e-3);

    printf("\nafter rotation — this is where the two pairings diverge\n");
    ck("q_pe  NORM  (2i,2i+1)", q_norm, -2831.006104, 5e-3);
    ck("k_pe  NORM  (2i,2i+1)", k_norm,    -6.364964, 5e-3);

    printf("\nand the pairing we had before, for the record:\n");
    printf("  %-22s %14.4f   llama.cpp %14.4f   -> off by %.0f%%\n",
           "q_pe  NEOX (i,i+n/2)", q_neox, -2831.006104,
           100.0*fabs(q_neox + 2831.006104)/2831.006104);
    printf("  %-22s %14.4f   llama.cpp %14.4f   -> off by %.0f%%\n",
           "k_pe  NEOX (i,i+n/2)", k_neox, -6.364964,
           100.0*fabs(k_neox + 6.364964)/6.364964);
    printf("\n  At pos 0 BOTH are exact. That is why the layer-0 test passed and the model\n");
    printf("  still lost the thread after the first token.\n");

    printf("\n%s\n", fails ? "FAILED" : "RoPE is NORM (adjacent pairs), and now matches llama.cpp");
    return fails ? 1 : 0;
}
