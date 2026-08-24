/* zexp_test.c — does compressing Q2_K experts actually pay?
 *
 * Two questions decide it, and neither can be answered from theory:
 *   1. RATIO. Q2_K is already 2.625 bpw. If zstd finds nothing, the whole idea dies.
 *   2. DECODE THROUGHPUT. It must beat the NVMe it is hiding behind, or it becomes
 *      the new bottleneck. The bar is this machine's disk: ~3.2 GB/s (measured, O_DIRECT).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../kquant.h"
#include "../zexp.h"

void ggml_abort(const char *f, int l, const char *fmt, ...);
void ggml_abort(const char *f, int l, const char *fmt, ...){ (void)fmt;
    fprintf(stderr, "ggml_abort %s:%d\n", f, l); exit(1); }
size_t ggml_row_size(int t, int64_t n);
size_t ggml_row_size(int t, int64_t n){ return (size_t)kq_row_bytes(t, n); }
size_t ggml_type_size(int t);
size_t ggml_type_size(int t){ return (size_t)kq_typesize(t); }
const char *ggml_type_name(int t);
const char *ggml_type_name(int t){ return kq_name(t); }
void quantize_row_q2_K_ref(const float *x, void *y, int64_t k);

static double now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void){
    /* One DeepSeek-V4-Flash expert: 3 matrices (gate/up/down) of 4096 x 2048. */
    const int64_t NW = 3LL * 4096 * 2048;
    const int64_t NB = kq_row_bytes(KQ_Q2_K, NW);
    printf("one expert (deepseek4: 3 x 4096 x 2048 = %.1fM weights)\n", NW / 1e6);
    printf("  Q2_K on disk today: %.2f MB (%.3f bit/W)\n\n", NB / 1e6, 8.0 * NB / NW);

    float *w = malloc(NW * sizeof(float));
    uint8_t *q = malloc(NB);
    if(!w || !q){ printf("OOM\n"); return 1; }

    /* realistic weights, then ggml's own quantizer -> the exact bytes Unsloth ships */
    srand(7);
    for(int64_t i = 0; i < NW; i++){
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0), u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        w[i] = (float)(0.02 * sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2));
    }
    quantize_row_q2_K_ref(w, q, NW);
    free(w);

    uint8_t *cont = malloc(zexp_bound(NB));
    uint8_t *back = malloc(NB);
    if(!cont || !back){ printf("OOM\n"); return 1; }

    printf("zstd over the Q2_K byte stream\n");
    for(int lvl = 1; lvl <= 9; lvl += 4){
        double t0 = now();
        size_t cs = zexp_pack(q, NB, cont, lvl);
        double tp = now() - t0;
        if(!cs){ printf("  pack failed\n"); return 1; }

        memset(back, 0xAA, NB);
        double t1 = now();
        int ok = zexp_unpack(cont, cs, back, NB);
        double td = now() - t1;

        if(!ok || memcmp(back, q, NB) != 0){ printf("  FAIL: round-trip corrupted at level %d\n", lvl); return 1; }

        double bpw = 8.0 * cs / NW;
        printf("  lvl %d: %.2f MB -> %.2f MB  ratio %.3fx  (%.3f bit/W)  "
               "pack %.1f MB/s  DECODE %.2f GB/s\n",
               lvl, NB / 1e6, cs / 1e6, (double)NB / cs, bpw,
               NB / 1e6 / tp, NB / 1e9 / td);
    }

    /* What it means for the engine: 43 layers x 6 routed experts per token. */
    printf("\nper token (43 layers x 6 routed experts), against this box's 3.2 GB/s NVMe\n");
    size_t cs = zexp_pack(q, NB, cont, 3);
    double gb_plain = 43.0 * 6.0 * NB / 1e9;
    double gb_zstd  = 43.0 * 6.0 * cs / 1e9;
    printf("  Q2_K plain : %5.2f GB/token -> %.2f tok/s ceiling\n", gb_plain, 3.2 / gb_plain);
    printf("  Q2_K + zexp: %5.2f GB/token -> %.2f tok/s ceiling   (%.2fx)\n",
           gb_zstd, 3.2 / gb_zstd, gb_plain / gb_zstd);

    memset(back, 0, NB);
    double t1 = now();
    zexp_unpack(cont, cs, back, NB);
    double td = now() - t1;
    double dec_gbs = NB / 1e9 / td;
    printf("  decode runs at %.2f GB/s vs 3.2 GB/s disk -> decode %s the I/O shadow\n",
           dec_gbs, dec_gbs > 3.2 ? "STAYS INSIDE" : "BREAKS OUT OF");

    free(cont); free(back); free(q);
    printf("\nround-trip verified byte-identical at every level\n");
    return 0;
}
