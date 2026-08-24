/* server_stubs.c — the four ggml reference quantizers, as abort-stubs.
 *
 * They are only reached when a recipe REQUANTIZES the attention (ATTN_Q=hard/medium/vramfit).
 * On a 16 GB card the unquantized weights fit in VRAM resident, so the default ATTN_Q=shipped
 * never calls them — and stubbing lets the whole engine build without the ggml source tree.
 * If a requantizing recipe is ever selected on this build, it aborts loudly rather than
 * silently producing garbage. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define STUB(name) void name(const float*a,void*b,int64_t n){ (void)a;(void)b;(void)n; \
    fprintf(stderr,"%s: requantization not built on this host — use ATTN_Q=shipped\n",#name); \
    exit(1); }
STUB(quantize_row_q4_K_ref)
STUB(quantize_row_q5_K_ref)
STUB(quantize_row_q3_K_ref)
STUB(quantize_row_q2_K_ref)
