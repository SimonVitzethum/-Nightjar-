#ifndef COLIBRI_ATTN_REQUANT_H
#define COLIBRI_ATTN_REQUANT_H
/* attn_requant.h — shrink the attention weights at load time.
 *
 * WHY
 *
 * The attention weights are 10.4 GB as Unsloth ships them, and they are pinned, so they hold
 * 12.5 GB of RAM hostage for the whole run. That RAM is the expert cache's only competitor:
 * the cache gets 6 GB, hits 17% of the time, and the disk is now the bottleneck. Shrinking
 * attention does not speed up attention (it costs 2.8 ms/layer, nothing); it speeds up the
 * DISK, by paying for a bigger cache with the RAM it gives back.
 *
 * WHAT UNSLOTH ACTUALLY DID, AND WHY IT MATTERS
 *
 * Measured per-tensor, one layer, error in the ATTENTION OUTPUT over an 8-token sequence:
 *
 *      tensor        shipped    MB     Q5_K    Q4_K    Q3_K    Q2_K
 *      q_a           Q5_K       8.7      -     0.18%   0.65%   1.09%
 *      q_b           Q8_0      35.7    0.10%   0.15%   0.59%   0.82%
 *      kv_a          Q8_0       3.8    0.24%   0.44%   0.77%   1.43%
 *      k_b           Q8_0       6.7      --      --      --      --
 *      v_b           Q8_0       8.9    0.19%   0.45%   0.84%   2.06%
 *      attn_output   Q5_K      69.2      -     0.73%   1.63%   3.43%
 *
 * Two things fall out of that table.
 *
 * q_b sits at 8.5 bits and loses 0.10% at 5.5. That is 12.6 MB per layer for nothing, and
 * requantizing it is CLEAN — Q8_0 is near-lossless as a source, so there is no double
 * quantization damage. Same for kv_a and v_b. Those three are free money.
 *
 * k_b CANNOT be touched. Its rows are 192 long, and every k-quant needs a multiple of 256.
 * Q8_0's 32-wide blocks are the only format that fits it. Unsloth was not being generous
 * there; it had no choice, and neither do we.
 *
 * THE HONEST CAVEAT
 *
 * Requantizing q_a and attn_output means going Q5_K -> Q4_K or lower, i.e. quantizing data
 * that is ALREADY quantized. That is double damage, and the table above measures exactly it,
 * so the numbers are real. What the table does NOT measure is how a 1.8% or 3.8% per-layer
 * error compounds across 78 layers. Nobody should assume that is fine — it has to be run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kquant.h"

/* ggml-quants.c wants a few symbols from the rest of ggml that we do not link. It only ever
 * calls them on paths we do not take (the lattice search for iq2/iq3, which we never invoke
 * from here), so stubbing them is honest — but they must ABORT, not return, or a path we did
 * not anticipate would silently produce garbage weights. */
void ggml_abort(const char *f, int l, const char *fmt, ...);
void ggml_abort(const char *f, int l, const char *fmt, ...){
    (void)fmt; fprintf(stderr, "ggml_abort at %s:%d\n", f, l); exit(1);
}
size_t ggml_row_size(int t, int64_t n);
size_t ggml_row_size(int t, int64_t n){ return (size_t)kq_row_bytes(t, n); }
size_t ggml_type_size(int t);
size_t ggml_type_size(int t){ return (size_t)kq_typesize(t); }
const char *ggml_type_name(int t);
const char *ggml_type_name(int t){ return kq_name(t); }

/* ggml's reference quantizers, linked in from llama.cpp */
void quantize_row_q4_K_ref(const float*, void*, int64_t);
void quantize_row_q5_K_ref(const float*, void*, int64_t);
void quantize_row_q3_K_ref(const float*, void*, int64_t);
void quantize_row_q2_K_ref(const float*, void*, int64_t);

/* q_a, q_b, kv_a, k_b, v_b, attn_output. -1 = leave as shipped. */
typedef struct { const char *name; int t[6]; } AQRecipe;

static const AQRecipe AQ_RECIPES[] = {
  /* name          q_a       q_b       kv_a      k_b  v_b       attn_output */
  { "shipped",  { -1,       -1,       -1,       -1,  -1,       -1        } },
  { "free",     { -1,       KQ_Q5_K,  KQ_Q5_K,  -1,  KQ_Q5_K,  -1        } },
  { "medium",   { KQ_Q4_K,  KQ_Q4_K,  KQ_Q5_K,  -1,  KQ_Q5_K,  KQ_Q4_K   } },
  { "hard",     { KQ_Q3_K,  KQ_Q4_K,  KQ_Q5_K,  -1,  KQ_Q5_K,  KQ_Q3_K   } },
  { "vramfit",  { KQ_Q2_K,  KQ_Q3_K,  KQ_Q4_K,  -1,  KQ_Q4_K,  KQ_Q2_K   } },
};
#define AQ_N ((int)(sizeof(AQ_RECIPES)/sizeof(AQ_RECIPES[0])))

static const AQRecipe *aq_find(const char *name){
    for(int i=0;i<AQ_N;i++) if(!strcmp(AQ_RECIPES[i].name, name)) return &AQ_RECIPES[i];
    return NULL;
}

/* Requantize one tensor, row by row, from `src` (type `st`) into `dst` (type `nt`).
 * O rows of I values each. Rows are independent, so this parallelizes cleanly — without it,
 * requantizing 12.9B parameters at load takes 40 minutes and nobody would ever run it. */
static void aq_tensor(void *dst, const void *src, int st, int nt, int64_t O, int64_t I){
    const int64_t srb = kq_row_bytes(st, I), drb = kq_row_bytes(nt, I);
    #pragma omp parallel
    {
        float *tmp = (float*)malloc((size_t)I*sizeof(float));
        #pragma omp for schedule(static)
        for(int64_t o=0;o<O;o++){
            kq_dequant_row(st, (const uint8_t*)src + o*srb, tmp, I);
            uint8_t *d = (uint8_t*)dst + o*drb;
            switch(nt){
                case KQ_Q5_K: quantize_row_q5_K_ref(tmp, d, I); break;
                case KQ_Q4_K: quantize_row_q4_K_ref(tmp, d, I); break;
                case KQ_Q3_K: quantize_row_q3_K_ref(tmp, d, I); break;
                case KQ_Q2_K: quantize_row_q2_K_ref(tmp, d, I); break;
                default: fprintf(stderr,"aq: no quantizer for %s\n", kq_name(nt)); exit(1);
            }
        }
        free(tmp);
    }
}

/* The (O, I) each attention tensor is contracted with. NOT the GGUF ne[] — k_b and v_b are
 * per-head 3-D tensors, and what matters for row-wise quantization is the contraction length,
 * because that is what has to be a multiple of the block size. */
static void aq_shape(int i, int d_model, int q_lora, int kv_lora, int n_head,
                     int qk_nope, int qk_rope, int v_head, int64_t *O, int64_t *I){
    switch(i){
        case 0: *O=q_lora;                       *I=d_model;                  break; /* q_a  */
        case 1: *O=(int64_t)n_head*(qk_nope+qk_rope); *I=q_lora;              break; /* q_b  */
        case 2: *O=kv_lora+qk_rope;              *I=d_model;                  break; /* kv_a */
        case 3: *O=(int64_t)kv_lora*n_head;      *I=qk_nope;                  break; /* k_b  */
        case 4: *O=(int64_t)v_head*n_head;       *I=kv_lora;                  break; /* v_b  */
        default:*O=d_model;                      *I=(int64_t)n_head*v_head;   break; /* o    */
    }
}

/* A tensor is only requantizable if its contraction length is a multiple of the target's
 * block size. k_b contracts 192, so every k-quant is out and it keeps its Q8_0 — silently
 * skipping this check would corrupt it. */
static int aq_can(int nt, int64_t I){
    if(nt < 0) return 0;
    return (I % (nt == KQ_Q8_0 ? 32 : 256)) == 0;
}

#endif
