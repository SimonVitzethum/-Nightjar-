/* attngpu_test.c — the GPU attention against the CPU attention, which is itself matched to
 * llama.cpp. Anything else is guessing: a wrong GPU projection does not crash, it produces
 * a NaN sixty lines later and surfaces as "estream: read failed" in a different file.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../gguf.h"
#include "../mla.h"
#include "../attn_gpu.h"

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }
    if(!kq_cu_upload_tables(iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, kvalues_iq4nl)){
        printf("tables failed\n"); return 1; }

    const char *arch = gguf_str(&m, "general.architecture", "");
    char k[128];
    #define KVI(s) (snprintf(k,sizeof(k),"%s." s,arch), (int)gguf_u64(&m,k,0))
    #define KVF(s) (snprintf(k,sizeof(k),"%s." s,arch), (float)gguf_f64(&m,k,0))
    const int D=KVI("embedding_length"), H=KVI("attention.head_count");
    const int KVL=KVI("attention.kv_lora_rank"), ROPE=KVI("rope.dimension_count");
    const int QL=KVI("attention.q_lora_rank");
    const float eps=KVF("attention.layer_norm_rms_epsilon"), base=KVF("rope.freq_base");
    #undef KVI
    #undef KVF

    #define T(n) gguf_find(&m, n)
    const gguf_tensor *emb=T("token_embd.weight");
    const gguf_tensor *an=T("blk.0.attn_norm.weight"), *qan=T("blk.0.attn_q_a_norm.weight");
    const gguf_tensor *kvan=T("blk.0.attn_kv_a_norm.weight");
    const gguf_tensor *qa=T("blk.0.attn_q_a.weight"), *qb=T("blk.0.attn_q_b.weight");
    const gguf_tensor *kva=T("blk.0.attn_kv_a_mqa.weight"), *kb=T("blk.0.attn_k_b.weight");
    const gguf_tensor *vb=T("blk.0.attn_v_b.weight"), *o=T("blk.0.attn_output.weight");
    #undef T

    MLACfg c = { H, (int)kb->ne[0], ROPE, (int)vb->ne[1], KVL, QL, D, eps,
                 1.0f/sqrtf((float)((int)kb->ne[0] + ROPE)) };
    MLALayer L = {
        .pq_a=gguf_data(&m,qa), .pq_b=gguf_data(&m,qb), .pkv_a=gguf_data(&m,kva),
        .pk_b=gguf_data(&m,kb), .pv_b=gguf_data(&m,vb), .po=gguf_data(&m,o),
        .tq_a=qa->type, .tq_b=qb->type, .tkv_a=kva->type,
        .tk_b=kb->type, .tv_b=vb->type, .to=o->type,
        .attn_norm=(const float*)gguf_data(&m,an),
        .q_a_norm=(const float*)gguf_data(&m,qan),
        .kv_a_norm=(const float*)gguf_data(&m,kvan) };

    printf("layer 0: q_a %s  q_b %s  kv_a %s  k_b %s  v_b %s  o %s\n",
           kq_name(qa->type), kq_name(qb->type), kq_name(kva->type),
           kq_name(kb->type), kq_name(vb->type), kq_name(o->type));
    printf("shapes: k_b contracts %d (not a multiple of 256!), v_b contracts %d\n\n",
           c.qk_nope, c.kv_lora);

    const int NT = 3;
    float *x = malloc((size_t)D*sizeof(float));
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);

    MLACache kv1, kv2; MLAScratch s1, s2;
    mla_cache_init(&kv1, &c, 8); mla_cache_init(&kv2, &c, 8);
    mla_scratch_init(&s1, &c, 8); mla_scratch_init(&s2, &c, 8);

    /* size the staging from the FATTEST layer, not layer 0 — blk.8 is Q6_K where the rest
     * are Q5_K, and that one difference is what corrupted the upload. */
    int64_t mn[6] = {0,0,0,0,0,0};
    for(int l=0;l<78;l++){
        char nm[128];
        #define G(f) (snprintf(nm,sizeof(nm),"blk.%d." f ".weight",l), gguf_find(&m,nm))
        const gguf_tensor *t[6] = { G("attn_q_a"), G("attn_q_b"), G("attn_kv_a_mqa"),
                                    G("attn_k_b"), G("attn_v_b"), G("attn_output") };
        #undef G
        if(!t[0]) continue;
        MLALayer Li = L;
        Li.tq_a=t[0]->type; Li.tq_b=t[1]->type; Li.tkv_a=t[2]->type;
        Li.tk_b=t[3]->type; Li.tv_b=t[4]->type; Li.to=t[5]->type;
        int64_t n6[6]; ag_sizes(&c, &Li, n6);
        for(int q=0;q<6;q++) if(n6[q] > mn[q]) mn[q] = n6[q];
    }
    AttnGpu A;
    if(!attn_gpu_init(&A, &c, mn, 8)){ printf("gpu init failed\n"); return 1; }

    float *cpu = malloc((size_t)D*sizeof(float));
    float *gpu = malloc((size_t)D*sizeof(float));
    int fails = 0;

    const int toks[3] = { 77451, 44701, 13359 };
    for(int t=0;t<NT;t++){
        kq_dequant_row(emb->type,(const uint8_t*)gguf_data(&m,emb)+(size_t)toks[t]*erb, x, D);

        mla_step(cpu, x, t, &c, &L, &kv1, &s1, base);

        attn_gpu_prefetch(&A, &c, &L, 0);
        mla_step_gpu(&A, gpu, x, t, &c, &L, 0, &kv2, &s2, base);

        double worst=0, den=0, sc=0, sg=0;
        int nan_c=0, nan_g=0;
        for(int i=0;i<D;i++){
            if(!isfinite(cpu[i])) nan_c++;
            if(!isfinite(gpu[i])) nan_g++;
            double d = fabs(cpu[i]-gpu[i]);
            if(d > worst) worst = d;
            if(fabs(cpu[i]) > den) den = fabs(cpu[i]);
            sc += cpu[i]; sg += gpu[i];
        }
        double rel = worst/(den+1e-9);
        int ok = rel < 1e-2 && !nan_c && !nan_g;
        printf("  pos %d: cpu sum %12.4f  gpu sum %12.4f  max rel %.2e  %s%s\n",
               t, sc, sg, rel, ok ? "ok" : "MISMATCH",
               nan_g ? "  (GPU produced NaN/Inf!)" : "");
        if(!ok) fails++;
    }

    printf("\n%s\n", fails ? "FAILED — the GPU attention does not match the CPU one"
                           : "GPU attention matches the CPU attention");
    return fails ? 1 : 0;
}
