#ifndef QWEN_KQUANT_CUDA_H
#define QWEN_KQUANT_CUDA_H
/* kquant_cuda.h — host interface to the GPU GEMV that runs on GGUF quant bytes.
 *
 * The contract: quantized expert bytes go to the device exactly as they came off NVMe.
 * They are decoded in registers inside the dot product and never materialized as a
 * weight matrix — not on the host, not in VRAM. That is what makes an 8 GB card usable
 * against a 254 GB model. */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ggml type ids, same numbering as kquant.h */
enum {
    KQ_CU_Q8_0    = 8,
    KQ_CU_Q2_K    = 10,
    KQ_CU_Q3_K    = 11,
    KQ_CU_Q4_K    = 12,
    KQ_CU_Q5_K    = 13,
    KQ_CU_Q6_K    = 14,
    KQ_CU_IQ2_XS  = 17,
    KQ_CU_IQ3_XXS = 18,
    KQ_CU_IQ4_XS  = 23
};

/* Push the lattice codebooks into __constant__ once at startup (~5 KB total). */
int kq_cu_upload_tables(const uint64_t *iq2xs_grid, const uint32_t *iq3xxs_grid,
                        const uint8_t *ksigns, const uint8_t *kmask,
                        const int8_t *kvalues_iq4nl);

#define KQ_CU_SMAX 8      /* one register accumulator per token, per lane */

/* y[S,O] = x[S,I] . W[O,I]^T, W in `gtype`. All pointers are device pointers.
 * `type_size` is the bytes per 256-weight super-block.
 *
 * S > 1 matters more than it looks: decoding an i-quant lattice point costs far more than
 * using it, so a batch decodes each weight ONCE and reuses it for every token — and any
 * tokens routed to the same expert share its ~12 MB read instead of paying for it each. */
struct CUstream_st;
const char *kq_cu_last_error(void);

int kq_cu_gemm(float *y_dev, const float *x_dev, const uint8_t *W_dev,
               int gtype, int S, int I, int O, int type_size, struct CUstream_st *stream);

/* S=1 shorthand. */
int kq_cu_gemv(float *y_dev, const float *x_dev, const uint8_t *W_dev,
               int gtype, int I, int O, int type_size, struct CUstream_st *stream);

/* ---- the int8 path (IQ2_XS / IQ3_XXS only) ----
 *
 * The fp32 kernels reach ~490 GFLOP/s; ggml's int8 ones reach ~1300 on the same tensor.
 * A lattice weight is a signed byte, so if the activation is quantized to int8 the dot
 * becomes an integer dot and __dp4a does four of them per instruction.
 *
 * Quantize ONCE per layer (every routed expert of a layer sees the same x), then feed the
 * result to as many experts as the router picked:
 *
 *     kq_cu_quant_act(xq, xd, x, S, I, stream);            // once
 *     kq_cu_gemm_i8(y, xq, xd, W_expert, type, S, I, O, stream);   // per expert
 *
 * xq needs S*I int8, xd needs S*(I/32) floats.
 */
int kq_cu_quant_act(signed char *qs_dev, float *d_dev, const float *x_dev,
                    int S, int I, struct CUstream_st *stream);

int kq_cu_gemm_i8(float *y_dev, const signed char *xq_dev, const float *xd_dev,
                  const unsigned char *W_dev, int gtype, int S, int I, int O,
                  struct CUstream_st *stream);

/* Keep the whole expert FFN on the device: g = silu(g)*u, acc += w*v, acc = 0.
 * Bouncing gate/up back to the host for a sigmoid would cost more in PCIe latency than
 * the matmuls themselves. */
/* y[h] = W[h] . x[h] for all n_head heads, in ONE launch. MLA's k_b and v_b are per-head
 * matrices and there are 64 of them; a launch (and a sync) per head costs ~15000 round trips
 * per token and starves the drive. */
int kq_cu_gemv_heads(float *y, const float *x, const unsigned char *W,
                     int gtype, int I, int O, int n_head,
                     long wstride, int xstride, int ystride,
                     struct CUstream_st *stream);

void kq_cu_swiglu(float *g, const float *u, int n, struct CUstream_st *stream);
void kq_cu_axpy(float *acc, const float *v, float a, int n, struct CUstream_st *stream);
void kq_cu_zero(float *v, int n, struct CUstream_st *stream);

#ifdef __cplusplus
}
#endif
#endif
