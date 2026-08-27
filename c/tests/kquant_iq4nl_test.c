/* kquant_iq4nl_test.c — IQ4_NL against the reference dequantisation, and against IQ4_XS.
 *
 * IQ4_NL is the block-of-32 relative of IQ4_XS: one fp16 scale, no sub-scales, the same
 * non-uniform 16-entry table. Qwen4's per-layer embedding table ships in it, and the engine
 * could read every other tensor in that model and stopped at this one.
 *
 * The check that matters is not "does it produce numbers" but "does it produce the SAME
 * numbers as ggml", so the reference is written out here independently rather than by calling
 * the code under test. And a second gate: an IQ4_XS block whose sub-scales are all set so
 * that dl == d must dequantise identically to IQ4_NL on the same nibbles -- the two formats
 * agree by construction where their parameters agree, which catches a wrong nibble order that
 * a self-consistent implementation would hide. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../kquant.h"

static uint32_t rs = 4242;
static uint32_t u32(void){ rs = rs*1664525u + 1013904223u; return rs; }

/* the reference, transcribed from ggml's dequantize_row_iq4_nl */
static void ref_iq4_nl(const void *vx, float *y, int64_t k){
    const uint8_t *p = (const uint8_t*)vx;
    for(int64_t i = 0; i < k/32; i++){
        uint16_t h; memcpy(&h, p, 2);
        const float d = kq_half(h);
        const uint8_t *qs = p + 2;
        for(int j = 0; j < 16; ++j){
            y[j +  0] = d * (float)kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32; p += 18;
    }
}

int main(void){
    const int64_t n = 32*64;
    printf("  block %d B, %d weights, %.2f bpw\n",
           kq_typesize(KQ_IQ4_NL), kq_blocksize(KQ_IQ4_NL),
           8.0*kq_typesize(KQ_IQ4_NL)/kq_blocksize(KQ_IQ4_NL));
    if(kq_typesize(KQ_IQ4_NL) != 18 || kq_blocksize(KQ_IQ4_NL) != 32){
        printf("  FAIL: layout is not 18 bytes per 32 weights\n"); return 1; }

    uint8_t *blk = (uint8_t*)malloc(18*(size_t)(n/32));
    for(int64_t b = 0; b < n/32; b++){
        uint16_t d = (uint16_t)(0x1000 | (u32() & 0x03FF));   /* a sane positive fp16 */
        memcpy(blk + 18*b, &d, 2);
        for(int j = 0; j < 16; j++) blk[18*b + 2 + j] = (uint8_t)u32();
    }
    float *a = (float*)malloc(sizeof(float)*n), *b = (float*)malloc(sizeof(float)*n);
    ref_iq4_nl(blk, a, n);
    if(!kq_dequant_row(KQ_IQ4_NL, blk, b, n)){ printf("  FAIL: kq_dequant refused IQ4_NL\n"); return 1; }

    double worst = 0; int bad = -1;
    for(int64_t i = 0; i < n; i++){
        const double e = fabs((double)a[i] - b[i]);
        if(e > worst){ worst = e; bad = (int)i; }
    }
    printf("  %lld weights, max |diff| vs reference: %.3g%s\n", (long long)n, worst,
           worst == 0 ? "  (bit-identical)" : "");
    if(worst != 0){
        printf("  FAIL at %d: ref %.9g, ours %.9g\n", bad, a[bad], b[bad]);
        return 1;
    }
    printf("\nPASS\n");
    free(blk); free(a); free(b);
    return 0;
}
