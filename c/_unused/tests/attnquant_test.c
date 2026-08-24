/* attnquant_test.c — how far can the ATTENTION weights be squeezed?
 *
 * WHY IT MATTERS
 *
 * The attention weights are 10.4 GB and never change, yet they cross PCIe on every single
 * token — 78 layers x 133 MB — purely because they do not fit in 8 GB of VRAM. That transfer
 * (0.43 s/token) is what would cap throughput at ~1.7 tok/s even with a 14 GB/s NVMe: the
 * disk would get faster and PCIe would become the wall.
 *
 * If they fit in VRAM, PCIe drops to zero for them and the disk becomes the sole bottleneck
 * again — which is the state where buying a faster drive actually pays.
 *
 * WHAT WE ARE MEASURING
 *
 * Unsloth's imatrix deliberately kept attention HIGH — q_b at Q8_0 (8.5 bpw) while the
 * routed experts sit at 2.3. Attention is ~5% of the parameters and gets 3x the precision.
 * That is a deliberate signal that it is sensitive, so we do not assume it can be squeezed;
 * we measure the error each format causes in the ATTENTION OUTPUT, on real GLM-5.2 weights.
 *
 * The output error is the thing that propagates. Weight RMSE is not.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../gguf.h"
#include "../mla.h"

/* ggml's quantizers (compiled in from llama.cpp) */
void ggml_abort(const char *f,int l,const char *fmt,...);
void ggml_abort(const char *f,int l,const char *fmt,...){ (void)fmt;
    fprintf(stderr,"ggml_abort %s:%d\n",f,l); exit(1); }
size_t ggml_row_size(int t,int64_t n); size_t ggml_row_size(int t,int64_t n){ return (size_t)kq_row_bytes(t,n); }
size_t ggml_type_size(int t);           size_t ggml_type_size(int t){ return (size_t)kq_typesize(t); }
const char *ggml_type_name(int t);      const char *ggml_type_name(int t){ return kq_name(t); }
void quantize_row_q4_K_ref(const float*,void*,int64_t);
void quantize_row_q5_K_ref(const float*,void*,int64_t);
void quantize_row_q2_K_ref(const float*,void*,int64_t);
void quantize_row_q3_K_ref(const float*,void*,int64_t);
size_t quantize_iq2_xs (const float*,void*,int64_t,int64_t,const float*);
size_t quantize_iq3_xxs(const float*,void*,int64_t,int64_t,const float*);
void iq2xs_init_impl(int); void iq3xs_init_impl(int);

/* requantize a whole tensor row by row, into `dst` of type `nt` */
static void requant(void *dst, const void *src, int st, int nt, int64_t O, int64_t I,
                    float *tmp, const float *imp){
    const int64_t srb = kq_row_bytes(st, I), drb = kq_row_bytes(nt, I);
    for(int64_t o=0;o<O;o++){
        kq_dequant_row(st, (const uint8_t*)src + o*srb, tmp, I);
        uint8_t *d = (uint8_t*)dst + o*drb;
        switch(nt){
            case KQ_Q4_K: quantize_row_q4_K_ref(tmp, d, I); break;
            case KQ_Q5_K: quantize_row_q5_K_ref(tmp, d, I); break;
            case KQ_Q3_K: quantize_row_q3_K_ref(tmp, d, I); break;
            case KQ_Q2_K: quantize_row_q2_K_ref(tmp, d, I); break;
            case KQ_IQ2_XS:  quantize_iq2_xs (tmp, d, 1, I, imp); break;
            case KQ_IQ3_XXS: quantize_iq3_xxs(tmp, d, 1, I, imp); break;
            default: exit(1);
        }
    }
}


/* THE TRAP: with a single token the softmax has one element, so it is 1.0 no matter what the
 * query says. Q only shapes the SCORES, and a score you never compare against is free. A
 * one-token oracle therefore reports 0.00% error for q_a/q_b at EVERY precision, down to 2
 * bits — and would have us shrink the second-largest tensor in the layer for nothing.
 *
 * Same shape of bug as RoPE-at-pos-0. Any oracle for something that only acts through a
 * comparison has to run long enough for the comparison to happen. We use 8 tokens and read
 * the error at the last one, where the softmax is actually discriminating over a history. */
#define NTOK 8
static const int TOKS[NTOK] = { 77451, 44701, 13359, 8930, 102, 55012, 31, 96604 };

static void run_seq(float *out, const gguf_model *m, const gguf_tensor *emb,
                    const MLACfg *c, const MLALayer *L, float base, int D){
    MLACache kv; MLAScratch s;
    mla_cache_init(&kv, c, NTOK); mla_scratch_init(&s, c, NTOK);
    float *x = malloc((size_t)D*sizeof(float));
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);
    for(int t=0;t<NTOK;t++){
        kq_dequant_row(emb->type,
            (const uint8_t*)gguf_data((gguf_model*)m,emb)+(size_t)TOKS[t]*erb, x, D);
        mla_step(out, x, t, c, L, &kv, &s, base);   /* last write wins = last position */
    }
    free(x);
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    iq2xs_init_impl(KQ_IQ2_XS); iq3xs_init_impl(256);

    const char *arch = gguf_str(&m,"general.architecture","");
    char k[128];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    #define KVF(s) (snprintf(k,sizeof(k),"%s." s,arch), (float)gguf_f64(&m,k,0))
    const int D=KVI("embedding_length"), H=KVI("attention.head_count");
    const int KVL=KVI("attention.kv_lora_rank"), ROPE=KVI("rope.dimension_count");
    const int QL=KVI("attention.q_lora_rank");
    const float eps=KVF("attention.layer_norm_rms_epsilon"), base=KVF("rope.freq_base");
    #undef KVI
    #undef KVF

    #define T(n) gguf_find(&m,n)
    const gguf_tensor *emb=T("token_embd.weight");
    const gguf_tensor *an=T("blk.0.attn_norm.weight"), *qan=T("blk.0.attn_q_a_norm.weight");
    const gguf_tensor *kvan=T("blk.0.attn_kv_a_norm.weight");
    const gguf_tensor *TT[6] = { T("blk.0.attn_q_a.weight"), T("blk.0.attn_q_b.weight"),
                                 T("blk.0.attn_kv_a_mqa.weight"), T("blk.0.attn_k_b.weight"),
                                 T("blk.0.attn_v_b.weight"), T("blk.0.attn_output.weight") };
    #undef T
    const char *NM[6] = { "q_a","q_b","kv_a","k_b","v_b","attn_output" };
    const int nope = (int)TT[3]->ne[0];

    MLACfg c = { H, nope, ROPE, (int)TT[4]->ne[1], KVL, QL, D, eps,
                 1.0f/sqrtf((float)(nope+ROPE)) };

    /* the shapes each tensor is contracted with (O rows of I) */
    const int64_t OO[6] = { QL, (int64_t)H*(nope+ROPE), KVL+ROPE,
                            (int64_t)KVL*H, (int64_t)c.v_head*H, D };
    const int64_t II[6] = { D, QL, D, nope, KVL, (int64_t)H*c.v_head };

    printf("Attention of one layer, as Unsloth ships it:\n");
    double total = 0, params = 0;
    for(int i=0;i<6;i++){
        double p = (double)OO[i]*II[i];
        printf("  %-12s %-6s %8.1f MB   %5.2f bit/W\n", NM[i], kq_name(TT[i]->type),
               TT[i]->bytes/1e6, 8.0*TT[i]->bytes/p);
        total += TT[i]->bytes; params += p;
    }
    printf("  %-12s        %8.1f MB   %5.2f bit/W   -> x78 layers = %.1f GB\n\n",
           "TOTAL", total/1e6, 8.0*total/params, total*78/1e9);

    /* one real token through the attention */
    float *x = malloc((size_t)D*sizeof(float));
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);
    kq_dequant_row(emb->type,(const uint8_t*)gguf_data(&m,emb)+(size_t)77451*erb, x, D);

    MLALayer L0 = {
        .pq_a=gguf_data(&m,TT[0]), .pq_b=gguf_data(&m,TT[1]), .pkv_a=gguf_data(&m,TT[2]),
        .pk_b=gguf_data(&m,TT[3]), .pv_b=gguf_data(&m,TT[4]), .po=gguf_data(&m,TT[5]),
        .tq_a=TT[0]->type, .tq_b=TT[1]->type, .tkv_a=TT[2]->type,
        .tk_b=TT[3]->type, .tv_b=TT[4]->type, .to=TT[5]->type,
        .attn_norm=(const float*)gguf_data(&m,an),
        .q_a_norm=(const float*)gguf_data(&m,qan),
        .kv_a_norm=(const float*)gguf_data(&m,kvan) };

    float *ref = malloc((size_t)D*sizeof(float));
    run_seq(ref, &m, emb, &c, &L0, base, D);

    /* requantize everything to one target type and see what the OUTPUT does */
    int64_t maxel = 0, maxb = 0;
    for(int i=0;i<6;i++){
        if(OO[i]*II[i] > maxel) maxel = OO[i]*II[i];
        if(TT[i]->bytes > maxb) maxb = TT[i]->bytes;
    }
    float *tmp = malloc((size_t)II[5]*sizeof(float));
    float *imp = malloc((size_t)II[5]*sizeof(float));
    for(int64_t i=0;i<II[5];i++) imp[i]=1.0f;
    void *buf[6];
    for(int i=0;i<6;i++) buf[i] = malloc((size_t)kq_row_bytes(KQ_Q5_K, OO[i]*II[i]) + 4096);

    /* k_b's rows are 192 long — NOT a multiple of 256, so it cannot be a k-quant at all.
     * Only Q8_0 (32-wide blocks) fits it. That is not Unsloth being generous, it is the only
     * legal choice: 6.7 of the 133 MB are simply not compressible by these formats. */
    #define KB 3

    struct { int t; const char *n; } TG[] = {
        {KQ_Q5_K,"Q5_K"}, {KQ_Q4_K,"Q4_K"}, {KQ_Q3_K,"Q3_K"},
        {KQ_IQ3_XXS,"IQ3_XXS"}, {KQ_Q2_K,"Q2_K"}, {KQ_IQ2_XS,"IQ2_XS"}
    };
    const int NG = (int)(sizeof(TG)/sizeof(TG[0]));

    /* run the attention with a per-tensor type vector, return the output error vs `ref` */
    double err_of(const int *ty, const void **pp);
    #define RUN(ty, pp, out_bytes) do{ } while(0)

    /* --- 1. ONE TENSOR AT A TIME: which of the six is actually sensitive? --- */
    printf("PER-TENSOR SENSITIVITY — one tensor lowered, the other five left as shipped\n");
    printf("(so the error column is purely that tensor's fault)\n\n");
    printf("  %-12s %7s", "tensor", "MB");
    for(int g=0; g<NG; g++) printf(" %9s", TG[g].n);
    printf("\n");

    double save[6][8];   /* MB saved by tensor i at format g */
    double derr[6][8];   /* output error caused */

    for(int i=0;i<6;i++){
        printf("  %-12s %7.1f", NM[i], TT[i]->bytes/1e6);
        for(int g=0; g<NG; g++){
            if(II[i] % 256){ printf(" %9s", "--"); save[i][g]=0; derr[i][g]=1e9; continue; }
            requant(buf[i], gguf_data(&m,TT[i]), TT[i]->type, TG[g].t, OO[i], II[i], tmp, imp);

            MLALayer Lq = L0;
            const void *pq[6] = { Lq.pq_a, Lq.pq_b, Lq.pkv_a, Lq.pk_b, Lq.pv_b, Lq.po };
            pq[i] = buf[i];
            Lq.pq_a=pq[0]; Lq.pq_b=pq[1]; Lq.pkv_a=pq[2];
            Lq.pk_b=pq[3]; Lq.pv_b=pq[4]; Lq.po=pq[5];
            int tq[6] = { Lq.tq_a, Lq.tq_b, Lq.tkv_a, Lq.tk_b, Lq.tv_b, Lq.to };
            tq[i] = TG[g].t;
            Lq.tq_a=tq[0]; Lq.tq_b=tq[1]; Lq.tkv_a=tq[2];
            Lq.tk_b=tq[3]; Lq.tv_b=tq[4]; Lq.to=tq[5];

            float *got = malloc((size_t)D*sizeof(float));
            run_seq(got, &m, emb, &c, &Lq, base, D);
            double se=0, sr=0;
            for(int q=0;q<D;q++){ double d=ref[q]-got[q]; se+=d*d; sr+=(double)ref[q]*ref[q]; }
            free(got);

            derr[i][g] = 100.0*sqrt(se/sr);
            save[i][g] = (TT[i]->bytes - kq_row_bytes(TG[g].t, OO[i]*II[i]))/1e6;
            printf(" %8.2f%%", derr[i][g]);
        }
        printf("\n");
    }

    /* --- 2. MIXED RECIPES: spend the error budget where it buys the most MB --- */
    printf("\nMIXED RECIPES — each tensor gets its own format\n\n");

    struct { const char *n; int t[6]; } RC[] = {
      /*             q_a       q_b       kv_a      k_b      v_b       o        */
      {"as shipped",{ -1,      -1,       -1,       -1,      -1,       -1       }},
      {"mild",      { KQ_Q4_K, KQ_Q5_K,  -1,       -1,      -1,       KQ_Q4_K  }},
      {"medium",    { KQ_Q4_K, KQ_Q4_K,  KQ_Q5_K,  -1,      KQ_Q5_K,  KQ_Q4_K  }},
      {"hard",      { KQ_Q3_K, KQ_Q4_K,  KQ_Q5_K,  -1,      KQ_Q5_K,  KQ_Q3_K  }},
      {"VRAM-fit",  { KQ_Q2_K, KQ_Q3_K,  KQ_Q4_K,  -1,      KQ_Q4_K,  KQ_Q2_K  }},
    };
    printf("  %-11s %8s %9s %8s %14s\n", "recipe", "MB/layer", "GB x78", "bit/W", "output error");
    for(unsigned r=0; r<sizeof(RC)/sizeof(RC[0]); r++){
        MLALayer Lq = L0;
        const void *pq[6] = { L0.pq_a, L0.pq_b, L0.pkv_a, L0.pk_b, L0.pv_b, L0.po };
        int tq[6] = { L0.tq_a, L0.tq_b, L0.tkv_a, L0.tk_b, L0.tv_b, L0.to };
        double bytes = 0;
        for(int i=0;i<6;i++){
            int nt = RC[r].t[i];
            if(nt < 0 || (II[i] % 256)){ bytes += TT[i]->bytes; continue; }
            requant(buf[i], gguf_data(&m,TT[i]), TT[i]->type, nt, OO[i], II[i], tmp, imp);
            pq[i] = buf[i]; tq[i] = nt;
            bytes += kq_row_bytes(nt, OO[i]*II[i]);
        }
        Lq.pq_a=pq[0]; Lq.pq_b=pq[1]; Lq.pkv_a=pq[2];
        Lq.pk_b=pq[3]; Lq.pv_b=pq[4]; Lq.po=pq[5];
        Lq.tq_a=tq[0]; Lq.tq_b=tq[1]; Lq.tkv_a=tq[2];
        Lq.tk_b=tq[3]; Lq.tv_b=tq[4]; Lq.to=tq[5];

        float *got = malloc((size_t)D*sizeof(float));
        run_seq(got, &m, emb, &c, &Lq, base, D);
        double se=0, sr=0;
        for(int q=0;q<D;q++){ double d=ref[q]-got[q]; se+=d*d; sr+=(double)ref[q]*ref[q]; }
        free(got);

        double gb = bytes*78/1e9;
        printf("  %-11s %8.1f %9.1f %8.2f %12.2f%%%s\n", RC[r].n, bytes/1e6, gb,
               8.0*bytes/params, 100.0*sqrt(se/sr),
               gb < 5.5 ? "   <- attention fits in VRAM" : "");
    }

    printf("\n  (Unsloth ships this layer at %.2f bit/W — 3x what it gives the experts.\n",
           8.0*total/params);
    printf("   That is the imatrix saying attention is sensitive. The numbers above say\n");
    printf("   whether it is right.)\n");
    return 0;
}
