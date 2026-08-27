#ifndef QWEN_KQUANT_H
#define QWEN_KQUANT_H
/* kquant.h — GGUF quant decoders for the types Unsloth's Dynamic GGUFs actually use.
 *
 * The name "UD-Q2_K_XL" is misleading: dumping the tensor directory of
 * DeepSeek-V4-Flash-UD-Q2_K_XL shows the ROUTED EXPERTS — the tensors that dominate
 * streaming I/O and are the entire point — are not Q2_K at all:
 *
 *   IQ2_XS  [4096, 2048, 256]   blk.N.ffn_gate_exps.weight   (2.3125 bpw)
 *   IQ2_XS  [4096, 2048, 256]   blk.N.ffn_up_exps.weight
 *   IQ3_XXS [2048, 4096, 256]   blk.N.ffn_down_exps.weight   (3.0625 bpw)
 *
 * with Q5_K/Q6_K/Q8_0/BF16/F32 carrying the dense part. So a loader must speak the
 * i-quants (lattice codebooks, see iq_grids.h) as well as the k-quants; supporting
 * only the headline type would decode every expert to noise.
 *
 * Layouts mirror ggml-common.h / ggml-quants.c bit for bit; kquant_test.c asserts
 * that against ggml itself. Do not "clean up" the shift patterns — they are the
 * on-disk format, not a style choice.
 *
 * The dot-product entry points (kq_dot_*) consume the quantized row DIRECTLY.
 * A k-quant super-block is a linear code (x = d*q - m), so the dot folds into
 * per-sub-block scale*sum(q*x) terms; an i-quant is a signed lattice point, so the
 * dot folds into a codebook lookup plus sign flips. Neither ever materializes the
 * row — which is what keeps a streamed expert off the heap and out of RAM.
 */
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "iq_grids.h"

#define KQ_QK_K 256            /* super-block: 256 weights */
#define KQ_K_SCALE_SIZE 12     /* Q4_K/Q5_K packed 6-bit scales+mins */

/* ggml half -> float. Portable; no _Float16, no F16C assumed. */
static inline float kq_half(uint16_t h){
    uint32_t s=(uint32_t)(h>>15), e=(uint32_t)((h>>10)&0x1F), m=(uint32_t)(h&0x3FF), b;
    if(e==0){                                   /* subnormal / zero */
        if(m==0){ b=s<<31; }
        else{ e=127-15+1; while(!(m&0x400)){ m<<=1; e--; } m&=0x3FF; b=(s<<31)|(e<<23)|(m<<13); }
    } else if(e==31){                           /* inf / nan */
        b=(s<<31)|(0xFFu<<23)|(m<<13);
    } else {
        b=(s<<31)|((e-15+127)<<23)|(m<<13);
    }
    float f; memcpy(&f,&b,4); return f;
}

/* ---- block layouts (byte-identical to ggml-common.h) ---- */
typedef struct {                       /* 84 B / 256 w = 2.625 bpw */
    uint8_t  scales[KQ_QK_K/16];       /* 4-bit scale + 4-bit min, packed */
    uint8_t  qs[KQ_QK_K/4];            /* 2-bit quants */
    uint16_t d, dmin;                  /* fp16 */
} kq_q2_K;

typedef struct {                       /* 144 B / 256 w = 4.5 bpw */
    uint16_t d, dmin;                  /* fp16 (NOTE: d/dmin come FIRST here) */
    uint8_t  scales[KQ_K_SCALE_SIZE];  /* 6-bit scales+mins */
    uint8_t  qs[KQ_QK_K/2];            /* 4-bit quants */
} kq_q4_K;

typedef struct {                       /* 110 B / 256 w = 3.4375 bpw */
    uint8_t  hmask[KQ_QK_K/8];         /* 3rd bit */
    uint8_t  qs[KQ_QK_K/4];            /* low 2 bits */
    uint8_t  scales[12];               /* 6-bit scales, bit-sliced */
    uint16_t d;
} kq_q3_K;

typedef struct {                       /* 210 B / 256 w = 6.5625 bpw */
    uint8_t  ql[KQ_QK_K/2];
    uint8_t  qh[KQ_QK_K/4];
    int8_t   scales[KQ_QK_K/16];
    uint16_t d;
} kq_q6_K;

typedef struct {                       /* 18 B / 32 w = 4.5 bpw */
    uint16_t d;                        /* fp16 */
    uint8_t  qs[16];                   /* 32 4-bit indices into kvalues_iq4nl */
} kq_iq4_nl;

typedef struct {                       /* 136 B / 256 w = 4.25 bpw */
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[KQ_QK_K/64];
    uint8_t  qs[KQ_QK_K/2];            /* 4-bit index into the 16-entry kvalues_iq4nl */
} kq_iq4_xs;

typedef struct {                       /* 176 B / 256 w = 5.5 bpw */
    uint16_t d, dmin;
    uint8_t  scales[KQ_K_SCALE_SIZE];
    uint8_t  qh[KQ_QK_K/8];            /* 5th bit */
    uint8_t  qs[KQ_QK_K/2];
} kq_q5_K;

typedef struct { uint16_t d; int8_t qs[32]; } kq_q8_0;   /* 34 B / 32 w */

/* ---- i-quants: lattice codebook, not a scalar grid ---- */
typedef struct {                       /* 74 B / 256 w = 2.3125 bpw  <- the experts */
    uint16_t d;
    uint16_t qs[KQ_QK_K/8];            /* 9 bits lattice index + 7 bits sign index */
    uint8_t  scales[KQ_QK_K/32];       /* two 4-bit scales per byte */
} kq_iq2_xs;

typedef struct {                       /* 98 B / 256 w = 3.0625 bpw  <- down_exps */
    uint16_t d;
    uint8_t  qs[3*KQ_QK_K/8];          /* 64 lattice bytes, then 32 B of packed scale+signs */
} kq_iq3_xxs;

/* ggml type ids we support. Others are rejected at load time, loudly.
 *
 * This exact set is not a guess: it is every type present in Unsloth's
 * GLM-5.2-UD-Q2_K_XL (all 7 shards) and DeepSeek-V4-Flash-UD-Q2_K_XL. A "Q2_K_XL"
 * model contains almost no Q2_K — the routed experts are IQ2_XS/IQ3_XXS, with a
 * handful of layers bumped to IQ4_XS/Q3_K/Q2_K where the imatrix says it matters.
 * That per-layer mixing IS what "Dynamic" means. */
enum { KQ_F32=0, KQ_F16=1, KQ_Q8_0=8, KQ_Q2_K=10, KQ_Q3_K=11, KQ_Q4_K=12, KQ_Q5_K=13,
       KQ_Q6_K=14, KQ_IQ2_XS=17, KQ_IQ3_XXS=18, KQ_IQ4_NL=20, KQ_IQ4_XS=23, KQ_BF16=30 };

static inline int kq_blocksize(int t){
    switch(t){ case KQ_Q8_0: case KQ_IQ4_NL: return 32;
               case KQ_Q2_K: case KQ_Q3_K: case KQ_Q4_K: case KQ_Q5_K: case KQ_Q6_K:
               case KQ_IQ2_XS: case KQ_IQ3_XXS: case KQ_IQ4_XS: return KQ_QK_K;
               default: return 1; }
}
static inline int kq_typesize(int t){
    switch(t){ case KQ_F32: return 4; case KQ_F16: case KQ_BF16: return 2;
               case KQ_Q8_0:    return (int)sizeof(kq_q8_0);
               case KQ_Q2_K:    return (int)sizeof(kq_q2_K);
               case KQ_Q3_K:    return (int)sizeof(kq_q3_K);
               case KQ_Q4_K:    return (int)sizeof(kq_q4_K);
               case KQ_Q5_K:    return (int)sizeof(kq_q5_K);
               case KQ_Q6_K:    return (int)sizeof(kq_q6_K);
               case KQ_IQ2_XS:  return (int)sizeof(kq_iq2_xs);
               case KQ_IQ3_XXS: return (int)sizeof(kq_iq3_xxs);
               case KQ_IQ4_NL:  return (int)sizeof(kq_iq4_nl);
               case KQ_IQ4_XS:  return (int)sizeof(kq_iq4_xs);
               default: return 0; }
}
static inline int kq_supported(int t){ return kq_typesize(t)!=0; }
static const char *kq_name(int t){
    switch(t){ case KQ_F32:return "F32"; case KQ_F16:return "F16"; case KQ_BF16:return "BF16";
               case KQ_Q8_0:return "Q8_0"; case KQ_Q2_K:return "Q2_K"; case KQ_Q3_K:return "Q3_K";
               case KQ_Q4_K:return "Q4_K"; case KQ_Q5_K:return "Q5_K"; case KQ_Q6_K:return "Q6_K";
               case KQ_IQ2_XS:return "IQ2_XS"; case KQ_IQ3_XXS:return "IQ3_XXS";
               case KQ_IQ4_NL:return "IQ4_NL"; case KQ_IQ4_XS:return "IQ4_XS"; default:return "?"; }
}
/* bytes for n weights of type t (n must be a multiple of the block size) */
static inline int64_t kq_row_bytes(int t,int64_t n){
    int bs=kq_blocksize(t); return (n/bs)*kq_typesize(t);
}

/* ---- Q4_K/Q5_K 6-bit scale+min unpack (ggml get_scale_min_k4) ---- */
static inline void kq_sc_min_k4(int j,const uint8_t *q,uint8_t *d,uint8_t *m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; }
    else   { *d=(uint8_t)((q[j+4]&0xF)|((q[j-4]>>6)<<4));
             *m=(uint8_t)((q[j+4]>>4)  |((q[j-0]>>6)<<4)); }
}

/* ================= dequantization (reference path) ================= */

static void kq_deq_q2_K(const void *vx,float *y,int64_t k){
    const kq_q2_K *x=(const kq_q2_K*)vx;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const float d=kq_half(x[i].d), mn=kq_half(x[i].dmin);
        const uint8_t *q=x[i].qs; int is=0;
        for(int n=0;n<KQ_QK_K;n+=128){
            int shift=0;
            for(int j=0;j<4;j++){
                uint8_t sc=x[i].scales[is++];
                float dl=d*(sc&0xF), ml=mn*(sc>>4);
                for(int l=0;l<16;l++) *y++ = dl*(float)((int8_t)((q[l]>>shift)&3)) - ml;
                sc=x[i].scales[is++];
                dl=d*(sc&0xF); ml=mn*(sc>>4);
                for(int l=0;l<16;l++) *y++ = dl*(float)((int8_t)((q[l+16]>>shift)&3)) - ml;
                shift+=2;
            }
            q+=32;
        }
    }
}

static void kq_deq_q4_K(const void *vx,float *y,int64_t k){
    const kq_q4_K *x=(const kq_q4_K*)vx;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const uint8_t *q=x[i].qs;
        const float d=kq_half(x[i].d), mn=kq_half(x[i].dmin);
        int is=0; uint8_t sc,m;
        for(int j=0;j<KQ_QK_K;j+=64){
            kq_sc_min_k4(is+0,x[i].scales,&sc,&m); const float d1=d*sc, m1=mn*m;
            kq_sc_min_k4(is+1,x[i].scales,&sc,&m); const float d2=d*sc, m2=mn*m;
            for(int l=0;l<32;l++) *y++ = d1*(float)(q[l]&0xF) - m1;
            for(int l=0;l<32;l++) *y++ = d2*(float)(q[l]>>4)  - m2;
            q+=32; is+=2;
        }
    }
}

static void kq_deq_q6_K(const void *vx,float *y,int64_t k){
    const kq_q6_K *x=(const kq_q6_K*)vx;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const float d=kq_half(x[i].d);
        const uint8_t *ql=x[i].ql, *qh=x[i].qh; const int8_t *sc=x[i].scales;
        for(int n=0;n<KQ_QK_K;n+=128){
            for(int l=0;l<32;l++){
                int is=l/16;
                int8_t q1=(int8_t)((ql[l+ 0]&0xF)|(((qh[l]>>0)&3)<<4)) - 32;
                int8_t q2=(int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4)) - 32;
                int8_t q3=(int8_t)((ql[l+ 0]>>4) |(((qh[l]>>4)&3)<<4)) - 32;
                int8_t q4=(int8_t)((ql[l+32]>>4) |(((qh[l]>>6)&3)<<4)) - 32;
                y[l+ 0]=d*sc[is+0]*q1;  y[l+32]=d*sc[is+2]*q2;
                y[l+64]=d*sc[is+4]*q3;  y[l+96]=d*sc[is+6]*q4;
            }
            y+=128; ql+=64; qh+=32; sc+=8;
        }
    }
}

/* Q3_K: 2 low bits in qs + an INVERTED 3rd bit in hmask (set => don't subtract 4),
 * and 6-bit scales bit-sliced across 12 bytes. Both are easy to get subtly wrong. */
static void kq_deq_q3_K(const void *vx,float *y,int64_t k){
    const kq_q3_K *x=(const kq_q3_K*)vx;
    const uint32_t kmask1=0x03030303u, kmask2=0x0f0f0f0fu;
    uint32_t aux[4]; const int8_t *sc=(const int8_t*)aux;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const float d_all=kq_half(x[i].d);
        const uint8_t *q=x[i].qs, *hm=x[i].hmask;
        uint8_t m=1;
        memcpy(aux,x[i].scales,12);
        uint32_t tmp=aux[2];
        aux[2]=((aux[0]>>4)&kmask2)|(((tmp>>4)&kmask1)<<4);
        aux[3]=((aux[1]>>4)&kmask2)|(((tmp>>6)&kmask1)<<4);
        aux[0]=( aux[0]    &kmask2)|(((tmp>>0)&kmask1)<<4);
        aux[1]=( aux[1]    &kmask2)|(((tmp>>2)&kmask1)<<4);
        int is=0;
        for(int n=0;n<KQ_QK_K;n+=128){
            int shift=0;
            for(int j=0;j<4;j++){
                float dl=d_all*(float)(sc[is++]-32);
                for(int l=0;l<16;l++)
                    *y++ = dl*(float)((int8_t)((q[l+0]>>shift)&3) - ((hm[l+0]&m)?0:4));
                dl=d_all*(float)(sc[is++]-32);
                for(int l=0;l<16;l++)
                    *y++ = dl*(float)((int8_t)((q[l+16]>>shift)&3) - ((hm[l+16]&m)?0:4));
                shift+=2; m<<=1;
            }
            q+=32;
        }
    }
}

/* IQ4_XS: 4-bit index into a 16-entry NON-uniform value table (kvalues_iq4nl), with a
 * 6-bit scale per 32 weights split across scales_l (low 4) and scales_h (high 2). */
/* IQ4_NL is IQ4_XS without the superblock: 32 weights, one fp16 scale, no sub-scales, the
 * same non-uniform 16-entry table. Qwen4's per-layer embedding table ships in it, which is
 * why it is here -- the engine could read every other tensor in that model and stopped at
 * one. */
static void kq_deq_iq4_nl(const void *vx,float *y,int64_t k){
    const kq_iq4_nl *x=(const kq_iq4_nl*)vx;
    for(int64_t i=0;i<k/32;i++){
        const float d=kq_half(x[i].d);
        const uint8_t *qs=x[i].qs;
        for(int j=0;j<16;j++){
            y[j+ 0]=d*(float)kvalues_iq4nl[qs[j]&0xF];
            y[j+16]=d*(float)kvalues_iq4nl[qs[j]>>4];
        }
        y+=32;
    }
}

static void kq_deq_iq4_xs(const void *vx,float *y,int64_t k){
    const kq_iq4_xs *x=(const kq_iq4_xs*)vx;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const uint8_t *qs=x[i].qs;
        const float d=kq_half(x[i].d);
        for(int ib=0;ib<KQ_QK_K/32;ib++){
            const int ls=((x[i].scales_l[ib/2]>>(4*(ib%2)))&0xF) | (((x[i].scales_h>>(2*ib))&3)<<4);
            const float dl=d*(float)(ls-32);
            for(int j=0;j<16;j++){
                y[j+ 0]=dl*(float)kvalues_iq4nl[qs[j]&0xF];
                y[j+16]=dl*(float)kvalues_iq4nl[qs[j]>>4];
            }
            y+=32; qs+=16;
        }
    }
}

static void kq_deq_q5_K(const void *vx,float *y,int64_t k){
    const kq_q5_K *x=(const kq_q5_K*)vx;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const uint8_t *ql=x[i].qs, *qh=x[i].qh;
        const float d=kq_half(x[i].d), mn=kq_half(x[i].dmin);
        int is=0; uint8_t sc,m,u1=1,u2=2;
        for(int j=0;j<KQ_QK_K;j+=64){
            kq_sc_min_k4(is+0,x[i].scales,&sc,&m); const float d1=d*sc, m1=mn*m;
            kq_sc_min_k4(is+1,x[i].scales,&sc,&m); const float d2=d*sc, m2=mn*m;
            for(int l=0;l<32;l++) *y++ = d1*(float)((ql[l]&0xF)+((qh[l]&u1)?16:0)) - m1;
            for(int l=0;l<32;l++) *y++ = d2*(float)((ql[l]>>4) +((qh[l]&u2)?16:0)) - m2;
            ql+=32; is+=2; u1<<=2; u2<<=2;
        }
    }
}

/* IQ2_XS: each 16-bit qs entry is a 9-bit index into a 512-point lattice (8 int8 coords)
 * plus a 7-bit index into the 128 sign patterns. Scale is 4 bits per 32-weight group.
 * This IS a codebook format — the grid is the codebook, shared by the whole model. */
static void kq_deq_iq2_xs(const void *vx,float *y,int64_t k){
    const kq_iq2_xs *x=(const kq_iq2_xs*)vx;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const float d=kq_half(x[i].d); float db[2];
        for(int ib32=0;ib32<KQ_QK_K/32;ib32++){
            db[0]=d*(0.5f+(float)(x[i].scales[ib32]&0xF))*0.25f;
            db[1]=d*(0.5f+(float)(x[i].scales[ib32]>>4)) *0.25f;
            for(int l=0;l<4;l++){
                const uint8_t *grid=(const uint8_t*)(iq2xs_grid+(x[i].qs[4*ib32+l]&511));
                const uint8_t signs=ksigns_iq2xs[x[i].qs[4*ib32+l]>>9];
                for(int j=0;j<8;j++)
                    y[j]=db[l/2]*(float)grid[j]*((signs&kmask_iq2xs[j])?-1.f:1.f);
                y+=8;
            }
        }
    }
}

/* IQ3_XXS: two 8-bit lattice indices per 8 weights (4 coords each), with scale and
 * signs packed into a trailing uint32 per 32-weight group. */
static void kq_deq_iq3_xxs(const void *vx,float *y,int64_t k){
    const kq_iq3_xxs *x=(const kq_iq3_xxs*)vx;
    uint32_t aux32;
    for(int64_t i=0;i<k/KQ_QK_K;i++){
        const float d=kq_half(x[i].d);
        const uint8_t *qs=x[i].qs;
        const uint8_t *ss=qs+KQ_QK_K/4;
        for(int ib32=0;ib32<KQ_QK_K/32;ib32++){
            memcpy(&aux32,ss+4*ib32,4);
            const float db=d*(0.5f+(float)(aux32>>28))*0.5f;
            for(int l=0;l<4;l++){
                const uint8_t signs=ksigns_iq2xs[(aux32>>(7*l))&127];
                const uint8_t *g1=(const uint8_t*)(iq3xxs_grid+qs[2*l+0]);
                const uint8_t *g2=(const uint8_t*)(iq3xxs_grid+qs[2*l+1]);
                for(int j=0;j<4;j++){
                    y[j+0]=db*(float)g1[j]*((signs&kmask_iq2xs[j+0])?-1.f:1.f);
                    y[j+4]=db*(float)g2[j]*((signs&kmask_iq2xs[j+4])?-1.f:1.f);
                }
                y+=8;
            }
            qs+=8;
        }
    }
}

static void kq_deq_q8_0(const void *vx,float *y,int64_t k){
    const kq_q8_0 *x=(const kq_q8_0*)vx;
    for(int64_t i=0;i<k/32;i++){
        const float d=kq_half(x[i].d);
        for(int l=0;l<32;l++) *y++ = d*(float)x[i].qs[l];
    }
}

static void kq_deq_f16(const void *vx,float *y,int64_t k){
    const uint16_t *x=(const uint16_t*)vx;
    for(int64_t i=0;i<k;i++) y[i]=kq_half(x[i]);
}
static void kq_deq_bf16(const void *vx,float *y,int64_t k){
    const uint16_t *x=(const uint16_t*)vx;
    for(int64_t i=0;i<k;i++){ uint32_t b=(uint32_t)x[i]<<16; memcpy(&y[i],&b,4); }
}

/* Dequantize one row of n weights. Returns 0 on unsupported type. */
static int kq_dequant_row(int t,const void *src,float *dst,int64_t n){
    switch(t){
        case KQ_F32:  memcpy(dst,src,(size_t)n*4);     return 1;
        case KQ_F16:  kq_deq_f16(src,dst,n);           return 1;
        case KQ_BF16: kq_deq_bf16(src,dst,n);          return 1;
        case KQ_Q8_0: kq_deq_q8_0(src,dst,n);          return 1;
        case KQ_Q2_K: kq_deq_q2_K(src,dst,n);          return 1;
        case KQ_Q3_K: kq_deq_q3_K(src,dst,n);          return 1;
        case KQ_Q4_K: kq_deq_q4_K(src,dst,n);          return 1;
        case KQ_Q5_K: kq_deq_q5_K(src,dst,n);          return 1;
        case KQ_Q6_K: kq_deq_q6_K(src,dst,n);          return 1;
        case KQ_IQ2_XS:  kq_deq_iq2_xs(src,dst,n);     return 1;
        case KQ_IQ3_XXS: kq_deq_iq3_xxs(src,dst,n);    return 1;
        case KQ_IQ4_NL:  kq_deq_iq4_nl(src,dst,n);     return 1;
        case KQ_IQ4_XS:  kq_deq_iq4_xs(src,dst,n);     return 1;
        default: return 0;
    }
}

/* ============ dot product straight off the quantized row ============
 * y = sum_i w[i]*x[i] without materializing w.
 *
 * Q2_K is x = d*sc*q - dmin*mn, so the sub-block contributes
 *     d*sc * sum(q_l * x_l)  -  dmin*mn * sum(x_l)
 * i.e. two accumulators over 16 elements, both reusable. This is the
 * "compute on the compressed representation" path — no dequant buffer, no
 * extra pass over memory, which is what matters when the row just came off
 * an NVMe stream and is cold in cache. */
static float kq_dot_q2_K(const void *vw,const float *x,int64_t n){
    const kq_q2_K *w=(const kq_q2_K*)vw; float acc=0.0f;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const float d=kq_half(w[i].d), mn=kq_half(w[i].dmin);
        const uint8_t *q=w[i].qs; const float *xb=x+i*KQ_QK_K;
        int is=0;
        for(int nn=0;nn<KQ_QK_K;nn+=128){
            int shift=0;
            for(int j=0;j<4;j++){
                for(int half=0;half<2;half++){
                    uint8_t sc=w[i].scales[is++];
                    const uint8_t *qq=q+16*half;
                    const float *xx=xb+nn+j*32+16*half;
                    float sq=0.0f, sx=0.0f;
                    for(int l=0;l<16;l++){
                        float xv=xx[l];
                        sq += (float)((qq[l]>>shift)&3) * xv;
                        sx += xv;
                    }
                    acc += d*(float)(sc&0xF)*sq - mn*(float)(sc>>4)*sx;
                }
                shift+=2;
            }
            q+=32;
        }
    }
    return acc;
}

static float kq_dot_q4_K(const void *vw,const float *x,int64_t n){
    const kq_q4_K *w=(const kq_q4_K*)vw; float acc=0.0f;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const uint8_t *q=w[i].qs;
        const float d=kq_half(w[i].d), mn=kq_half(w[i].dmin);
        const float *xb=x+i*KQ_QK_K;
        int is=0; uint8_t sc,m;
        for(int j=0;j<KQ_QK_K;j+=64){
            kq_sc_min_k4(is+0,w[i].scales,&sc,&m); const float d1=d*sc, m1=mn*m;
            kq_sc_min_k4(is+1,w[i].scales,&sc,&m); const float d2=d*sc, m2=mn*m;
            float s1=0,x1=0,s2=0,x2=0;
            for(int l=0;l<32;l++){
                float xa=xb[j+l], xb2=xb[j+32+l];
                s1 += (float)(q[l]&0xF)*xa; x1 += xa;
                s2 += (float)(q[l]>>4) *xb2; x2 += xb2;
            }
            acc += d1*s1 - m1*x1 + d2*s2 - m2*x2;
            q+=32; is+=2;
        }
    }
    return acc;
}

static float kq_dot_q6_K(const void *vw,const float *x,int64_t n){
    const kq_q6_K *w=(const kq_q6_K*)vw; float acc=0.0f;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const float d=kq_half(w[i].d);
        const uint8_t *ql=w[i].ql,*qh=w[i].qh; const int8_t *sc=w[i].scales;
        const float *y=x+i*KQ_QK_K;
        for(int nn=0;nn<KQ_QK_K;nn+=128){
            for(int l=0;l<32;l++){
                int is=l/16;
                int8_t q1=(int8_t)((ql[l+ 0]&0xF)|(((qh[l]>>0)&3)<<4)) - 32;
                int8_t q2=(int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4)) - 32;
                int8_t q3=(int8_t)((ql[l+ 0]>>4) |(((qh[l]>>4)&3)<<4)) - 32;
                int8_t q4=(int8_t)((ql[l+32]>>4) |(((qh[l]>>6)&3)<<4)) - 32;
                acc += d*sc[is+0]*q1*y[l+ 0];  acc += d*sc[is+2]*q2*y[l+32];
                acc += d*sc[is+4]*q3*y[l+64];  acc += d*sc[is+6]*q4*y[l+96];
            }
            y+=128; ql+=64; qh+=32; sc+=8;
        }
    }
    return acc;
}

static float kq_dot_q5_K(const void *vw,const float *x,int64_t n){
    const kq_q5_K *w=(const kq_q5_K*)vw; float acc=0.0f;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const uint8_t *ql=w[i].qs,*qh=w[i].qh;
        const float d=kq_half(w[i].d), mn=kq_half(w[i].dmin);
        const float *xb=x+i*KQ_QK_K;
        int is=0; uint8_t sc,m,u1=1,u2=2;
        for(int j=0;j<KQ_QK_K;j+=64){
            kq_sc_min_k4(is+0,w[i].scales,&sc,&m); const float d1=d*sc, m1=mn*m;
            kq_sc_min_k4(is+1,w[i].scales,&sc,&m); const float d2=d*sc, m2=mn*m;
            float s1=0,a1=0,s2=0,a2=0;
            for(int l=0;l<32;l++){
                float xa=xb[j+l], xc=xb[j+32+l];
                s1 += (float)((ql[l]&0xF)+((qh[l]&u1)?16:0))*xa; a1 += xa;
                s2 += (float)((ql[l]>>4) +((qh[l]&u2)?16:0))*xc; a2 += xc;
            }
            acc += d1*s1 - m1*a1 + d2*s2 - m2*a2;
            ql+=32; is+=2; u1<<=2; u2<<=2;
        }
    }
    return acc;
}

/* The i-quant dots are the real "compute on the compressed representation" path:
 * a lattice point is looked up, sign-flipped, and contracted against x — the expert
 * row is never reconstructed anywhere. */
static float kq_dot_iq2_xs(const void *vw,const float *x,int64_t n){
    const kq_iq2_xs *w=(const kq_iq2_xs*)vw; float acc=0.0f;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const float d=kq_half(w[i].d); float db[2];
        const float *xb=x+i*KQ_QK_K;
        for(int ib32=0;ib32<KQ_QK_K/32;ib32++){
            db[0]=d*(0.5f+(float)(w[i].scales[ib32]&0xF))*0.25f;
            db[1]=d*(0.5f+(float)(w[i].scales[ib32]>>4)) *0.25f;
            for(int l=0;l<4;l++){
                const uint8_t *grid=(const uint8_t*)(iq2xs_grid+(w[i].qs[4*ib32+l]&511));
                const uint8_t signs=ksigns_iq2xs[w[i].qs[4*ib32+l]>>9];
                const float *xx=xb+ib32*32+l*8;
                float s=0.0f;
                for(int j=0;j<8;j++)
                    s += (float)grid[j]*((signs&kmask_iq2xs[j])?-1.f:1.f)*xx[j];
                acc += db[l/2]*s;
            }
        }
    }
    return acc;
}

static float kq_dot_iq3_xxs(const void *vw,const float *x,int64_t n){
    const kq_iq3_xxs *w=(const kq_iq3_xxs*)vw; float acc=0.0f; uint32_t aux32;
    for(int64_t i=0;i<n/KQ_QK_K;i++){
        const float d=kq_half(w[i].d);
        const uint8_t *qs=w[i].qs, *ss=qs+KQ_QK_K/4;
        const float *xb=x+i*KQ_QK_K;
        for(int ib32=0;ib32<KQ_QK_K/32;ib32++){
            memcpy(&aux32,ss+4*ib32,4);
            const float db=d*(0.5f+(float)(aux32>>28))*0.5f;
            for(int l=0;l<4;l++){
                const uint8_t signs=ksigns_iq2xs[(aux32>>(7*l))&127];
                const uint8_t *g1=(const uint8_t*)(iq3xxs_grid+qs[2*l+0]);
                const uint8_t *g2=(const uint8_t*)(iq3xxs_grid+qs[2*l+1]);
                const float *xx=xb+ib32*32+l*8;
                float s=0.0f;
                for(int j=0;j<4;j++){
                    s += (float)g1[j]*((signs&kmask_iq2xs[j+0])?-1.f:1.f)*xx[j+0];
                    s += (float)g2[j]*((signs&kmask_iq2xs[j+4])?-1.f:1.f)*xx[j+4];
                }
                acc += db*s;
            }
            qs+=8;
        }
    }
    return acc;
}

/* Generic dot dispatch.
 *
 * The hot types (routed experts, dense projections) get hand-fused kernels that never
 * materialize the row. Everything else falls through to a correct block-at-a-time
 * dequant-then-dot — slower, but only ever hit by the handful of tensors Unsloth bumps
 * to a rare type (5 of ~1900 in GLM-5.2-UD-Q2_K_XL).
 *
 * There is deliberately NO "return 0" path: a type this build cannot decode must be
 * caught loudly by kq_supported() at load time, never turn into a silent zero row here. */
static float kq_dot(int t,const void *w,const float *x,int64_t n){
    switch(t){
        case KQ_Q2_K: return kq_dot_q2_K(w,x,n);
        case KQ_Q4_K: return kq_dot_q4_K(w,x,n);
        case KQ_Q5_K: return kq_dot_q5_K(w,x,n);
        case KQ_Q6_K: return kq_dot_q6_K(w,x,n);
        case KQ_IQ2_XS:  return kq_dot_iq2_xs(w,x,n);
        case KQ_IQ3_XXS: return kq_dot_iq3_xxs(w,x,n);
        default: break;
    }
    float acc=0.0f;
    if(t==KQ_F32){ const float *f=(const float*)w; for(int64_t i=0;i<n;i++) acc+=f[i]*x[i]; return acc; }
    if(t==KQ_Q8_0){
        const kq_q8_0 *q=(const kq_q8_0*)w;
        for(int64_t i=0;i<n/32;i++){ float d=kq_half(q[i].d), s=0;
            for(int l=0;l<32;l++) s+=(float)q[i].qs[l]*x[i*32+l];
            acc+=d*s; }
        return acc;
    }
    if(t==KQ_F16){ const uint16_t *h=(const uint16_t*)w;
        for(int64_t i=0;i<n;i++) acc+=kq_half(h[i])*x[i]; return acc; }
    if(t==KQ_BF16){ const uint16_t *h=(const uint16_t*)w;
        for(int64_t i=0;i<n;i++){ uint32_t b=(uint32_t)h[i]<<16; float f; memcpy(&f,&b,4); acc+=f*x[i]; }
        return acc; }

    /* remaining block types (Q3_K, IQ4_XS): decode one super-block at a time.
     * If the type is not decodable we must NOT fall through with a stale buffer — a row of
     * garbage weights produces plausible logits and corrupts the model silently. Loading
     * already rejects unknown types (kq_supported), so reaching here is a bug, not input. */
    const int bs = kq_blocksize(t);
    const int ts = kq_typesize(t);
    float buf[KQ_QK_K] = {0};
    for(int64_t b=0; b<n/bs; b++){
        if(!kq_dequant_row(t,(const uint8_t*)w + b*ts, buf, bs)) return 0.0f/0.0f;  /* NaN, loudly */
        for(int j=0;j<bs;j++) acc += buf[j]*x[b*bs+j];
    }
    return acc;
}

/* y[S,O] = x[S,I] @ W^T, with W[O,I] stored as ggml quants, row-major, one row per output.
 * Rows are contracted in place — W is never reconstructed anywhere, which is what lets a
 * streamed expert be consumed straight out of the read buffer.
 *
 * Serial: for callers already inside a parallel region (glm.c's moe() parallelizes over
 * experts, so nesting here would oversubscribe). */
static void kq_gemm_serial(float *y,const float *x,const void *W,int gtype,int S,int I,int O){
    const int64_t rb = kq_row_bytes(gtype, I);        /* bytes of ONE row of W */
    for(int o=0;o<O;o++){
        const uint8_t *w = (const uint8_t*)W + (int64_t)o*rb;
        for(int s=0;s<S;s++)
            y[(int64_t)s*O+o] = kq_dot(gtype, w, x+(int64_t)s*I, I);
    }
}

/* Parallel over output rows. One expert's gate projection is 2048 rows of 6144 weights;
 * decoding a lattice quant costs enough that this is worth spreading over every core —
 * and in the disk-bound regime those cores are idle anyway. */
static void kq_gemm(float *y,const float *x,const void *W,int gtype,int S,int I,int O){
#ifdef _OPENMP
    if(omp_in_parallel()){ kq_gemm_serial(y,x,W,gtype,S,I,O); return; }
    const int64_t rb = kq_row_bytes(gtype, I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w = (const uint8_t*)W + (int64_t)o*rb;
        for(int s=0;s<S;s++)
            y[(int64_t)s*O+o] = kq_dot(gtype, w, x+(int64_t)s*I, I);
    }
#else
    kq_gemm_serial(y,x,W,gtype,S,I,O);
#endif
}

#endif /* QWEN_KQUANT_H */
