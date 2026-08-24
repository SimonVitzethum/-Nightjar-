/* attnproj_test.c — WHICH projection does the GPU get wrong?
 *
 * The GPU attention writes "Der Hauptuntchied" where the CPU attention — the one validated
 * against llama.cpp — writes "Der Hauptunterschied". Same weights. So one of the six matrices
 * decodes wrong on the device, and attngpu_test never caught it because it accepts a relative
 * error under 1e-2, which is exactly the size of the defect.
 *
 * A tolerance you picked to make a test pass is not a tolerance. Two fp32 paths doing the same
 * dot product in a different ORDER differ by ~1e-6 relative; anything above ~1e-4 is a bug,
 * not rounding. So this checks every projection of every layer at 1e-4 and names the culprit.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../gguf.h"
#include "../mla.h"
#include "../attn_gpu.h"

static double rel_err(const float *a, const float *b, int n, double *biggest){
    double se=0, sr=0; *biggest = 0;
    for(int i=0;i<n;i++){
        double d = fabs((double)a[i]-b[i]);
        if(d > *biggest) *biggest = d;
        se += d*d; sr += (double)a[i]*a[i];
    }
    return sr > 0 ? sqrt(se/sr) : 0;
}

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
    const int QL=KVI("attention.q_lora_rank"), NL=KVI("block_count")-1;
    const float eps=KVF("attention.layer_norm_rms_epsilon"), base=KVF("rope.freq_base");
    #undef KVI
    #undef KVF

    const gguf_tensor *emb = gguf_find(&m, "token_embd.weight");
    const gguf_tensor *kb0 = gguf_find(&m, "blk.0.attn_k_b.weight");
    const gguf_tensor *vb0 = gguf_find(&m, "blk.0.attn_v_b.weight");
    MLACfg c = { H, (int)kb0->ne[0], ROPE, (int)vb0->ne[1], KVL, QL, D, eps,
                 1.0f/sqrtf((float)((int)kb0->ne[0] + ROPE)) };

    /* the fattest layer sizes the device regions — blk.8 is Q6_K where the rest are Q5_K */
    const char *TN[6] = { "attn_q_a","attn_q_b","attn_kv_a_mqa","attn_k_b","attn_v_b","attn_output" };
    int64_t mn[6] = {0,0,0,0,0,0};
    for(int l=0;l<NL;l++){
        MLALayer Li; memset(&Li,0,sizeof(Li));
        int *ty[6] = { &Li.tq_a,&Li.tq_b,&Li.tkv_a,&Li.tk_b,&Li.tv_b,&Li.to };
        for(int q=0;q<6;q++){
            snprintf(k,sizeof(k),"blk.%d.%s.weight",l,TN[q]);
            const gguf_tensor *t = gguf_find(&m,k);
            if(!t){ printf("missing %s\n",k); return 1; }
            *ty[q] = t->type;
        }
        int64_t n6[6]; ag_sizes(&c, &Li, n6);
        for(int q=0;q<6;q++) if(n6[q] > mn[q]) mn[q] = n6[q];
    }

    AttnGpu A;
    if(!attn_gpu_init(&A, &c, mn, 8)){ printf("gpu init failed\n"); return 1; }

    /* eight real tokens, so the softmax actually discriminates and Q matters */
    const int NT = 8;
    const int toks[8] = { 77451, 44701, 13359, 8930, 102, 55012, 31, 96604 };
    float *x = malloc((size_t)D*sizeof(float));
    const int64_t erb = kq_row_bytes(emb->type, emb->ne[0]);

    float *cpu = malloc((size_t)D*sizeof(float));
    float *gpu = malloc((size_t)D*sizeof(float));

    printf("GPU attention vs CPU attention, per layer, %d tokens.\n", NT);
    printf("Two fp32 paths differ by ~1e-6 from summation order alone. Anything above 1e-4\n");
    printf("is a defect, not rounding.\n\n");
    printf("  %-5s %-6s %-6s %-6s %-6s %-6s %-6s %10s %10s\n",
           "layer","q_a","q_b","kv_a","k_b","v_b","o","rel err","max |d|");

    int bad = 0;
    double worst_rel = 0; int worst_l = -1;
    for(int l=0; l<NL; l++){
        MLALayer L; memset(&L,0,sizeof(L));
        const void **pp[6] = { &L.pq_a,&L.pq_b,&L.pkv_a,&L.pk_b,&L.pv_b,&L.po };
        int *ty[6] = { &L.tq_a,&L.tq_b,&L.tkv_a,&L.tk_b,&L.tv_b,&L.to };
        for(int q=0;q<6;q++){
            snprintf(k,sizeof(k),"blk.%d.%s.weight",l,TN[q]);
            const gguf_tensor *t = gguf_find(&m,k);
            *pp[q] = gguf_data(&m,t); *ty[q] = t->type;
        }
        #define NRM(f,nm) do{ snprintf(k,sizeof(k),"blk.%d." nm ".weight",l); \
            L.f = (const float*)gguf_data(&m, gguf_find(&m,k)); }while(0)
        NRM(attn_norm,"attn_norm"); NRM(q_a_norm,"attn_q_a_norm"); NRM(kv_a_norm,"attn_kv_a_norm");
        #undef NRM

        MLACache kv1, kv2; MLAScratch s1, s2;
        mla_cache_init(&kv1,&c,NT); mla_cache_init(&kv2,&c,NT);
        mla_scratch_init(&s1,&c,NT); mla_scratch_init(&s2,&c,NT);

        double rel = 0, big = 0;
        for(int t=0;t<NT;t++){
            kq_dequant_row(emb->type,(const uint8_t*)gguf_data(&m,emb)+(size_t)toks[t]*erb,x,D);
            mla_step(cpu, x, t, &c, &L, &kv1, &s1, base);
            attn_gpu_prefetch(&A, &c, &L, 0);
            mla_step_gpu(&A, gpu, x, t, &c, &L, 0, &kv2, &s2, base);
            double b; double r = rel_err(cpu, gpu, D, &b);
            if(r > rel){ rel = r; big = b; }
        }
        if(rel > worst_rel){ worst_rel = rel; worst_l = l; }

        const int fail = rel > 1e-4;
        if(fail || l < 3 || rel > 1e-5){
            printf("  %-5d %-6s %-6s %-6s %-6s %-6s %-6s %10.2e %10.2e %s\n", l,
                   kq_name(L.tq_a), kq_name(L.tq_b), kq_name(L.tkv_a),
                   kq_name(L.tk_b), kq_name(L.tv_b), kq_name(L.to),
                   rel, big, fail ? "  <-- WRONG" : "");
        }
        if(fail) bad++;
    }

    printf("\n  %d of %d layers exceed 1e-4. Worst: layer %d at %.2e\n",
           bad, NL, worst_l, worst_rel);
    return bad ? 1 : 0;
}
