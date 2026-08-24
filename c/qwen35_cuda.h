#ifndef COLIBRI_QWEN35_CUDA_H
#define COLIBRI_QWEN35_CUDA_H
/* qwen35_cuda.h — the device half of the engine.
 *
 * WHAT GOES TO THE GPU AND WHY
 *
 * The split is not "GPU is faster", it is "these bytes must not be read at 36 GB/s". Per
 * generated token the model reads 15.6 GiB of weights. 9.65 of those are FFN and stay in
 * RAM, because 9.65 GiB does not fit in 8 GiB of VRAM and streaming it over a x8 link at
 * 25 GB/s would be slower than reading it in place. The other 5.08 GiB — gdn 3.222, attn
 * 0.889, output 0.971 — do fit, and once they are resident they are read at VRAM speed and
 * cost the DRAM controller nothing at all. That is where the 37% comes from: not overlap,
 * but bytes that stop being read on the slow side.
 *
 * THE INTERFACE IS PRIMITIVE ON PURPOSE
 *
 * Every call here takes device pointers and plain scalars. Nothing in this file knows what
 * a Q35Model is. The orchestration — which layer runs where, when the KV window spills to
 * the host — lives in qwen35_hetero.h, in C, next to the CPU reference it has to agree with.
 * A kernel that cannot be called in isolation cannot be diffed against its CPU counterpart,
 * and on this project every silent bug came from an optimization with nothing to compare to.
 *
 * FAILURE POLICY
 *
 * Nothing here falls back silently. If the device is missing, out of memory, or handed a
 * type it does not decode, the call returns 0 and q35cu_error() says why. "CUDA didn't help"
 * and "CUDA didn't run" look identical from the outside, and that has already cost 1.84x
 * once on this codebase — see COLIBRI_KERNEL_LOG.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- device lifecycle ---------------- */

int         q35cu_init(int device);        /* 1 = ready, 0 = unavailable (see q35cu_error) */
void        q35cu_shutdown(void);
int         q35cu_ok(void);
const char *q35cu_error(void);
const char *q35cu_name(void);
int         q35cu_mem(size_t *free_bytes, size_t *total_bytes);
double      q35cu_bandwidth(void);         /* measured GB/s, 0 until q35cu_measure_bw runs */
double      q35cu_measure_bw(void);        /* streaming read benchmark, ~100 ms */

/* ---------------- memory ---------------- */

void *q35cu_alloc(size_t n);               /* device; NULL on failure */
void  q35cu_free(void *p);
void *q35cu_host_alloc(size_t n);          /* pinned host, for real async DMA */
void  q35cu_host_free(void *p);

int q35cu_h2d(void *dst, const void *src, size_t n);
int q35cu_d2h(void *dst, const void *src, size_t n);
int q35cu_h2d_async(void *dst, const void *src, size_t n);   /* on the copy stream */
int q35cu_d2h_async(void *dst, const void *src, size_t n);
int q35cu_copy_sync(void);
int  q35cu_host_register(void *p, size_t n);
void q35cu_host_unregister(void *p);
int  q35cu_h2d_compute(void *d, const void *s, size_t n);
int  q35cu_h2d_2d_async(void *dst, size_t dpitch, const void *src, size_t spitch,
                        size_t width, size_t height);
void q35cu_copy_mark(int slot);
void q35cu_stream_join(int slot);
int  q35cu_d2h_compute(void *d, const void *s, size_t n);                 /* wait for the copy stream only */
int q35cu_sync(void);                      /* wait for the compute stream */
int q35cu_zero(void *p, size_t n);

/* ---------------- matmul over GGUF quant bytes ----------------
 * W is row-major with ne[0]=I (the contracted dim) and ne[1]=O, exactly as the file stores
 * it. Weights are decoded in registers inside the dot product and never materialized. */

int q35cu_gemv(float *y, const float *x, const void *W, int gtype, int I, int O);
int q35cu_gemm(float *y, const float *x, const void *W, int gtype, int I, int O, int S);
int q35cu_type_ok(int gtype);

/* ---------------- elementwise ---------------- */

void q35cu_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void q35cu_add(float *x, const float *t, int n);
void q35cu_rmsnorm_rows(float *out, const float *x, const float *w, int n, int S, float eps);
int  q35cu_d2d(void *dst, const void *src, size_t n);
void q35cu_swiglu(float *g, const float *u, int n);   /* g = silu(g)*u */
void q35cu_scale_add(float *x, const float *t, float a, int n);

/* ---------------- gated delta net ----------------
 * conv1d weight is [d_conv, conv_dim] with d_conv FASTEST — the taps of one channel are
 * contiguous. Reading it transposed is silent: still in bounds, still fluent, still wrong. */

void q35cu_gdn_conv(float *qkv, float *convstate, const float *w, int conv_dim, int d_conv);
void q35cu_gdn_gates(float *beta, float *gdec, const float *dt_bias, const float *ssm_a, int nv);
void q35cu_gdn_l2norm(float *qk, int n_head, int dk, float eps);
void q35cu_gdn_delta(float *S, float *o, const float *q, const float *k, const float *v,
                     const float *beta, const float *gdec,
                     int nv, int nk, int dk, int dv, int tile);
void q35cu_gdn_norm_gate(float *o, const float *z, const float *w, int nv, int dv, float eps);

/* ---------------- full attention ----------------
 * KV formats match kv_tier.h's: 1 = fp16, 2 = q8_0. The device window uses the SAME packed
 * layout as the host tier, so a spilled chunk can be DMA'd in without a conversion pass. */

void q35cu_attn_qk(float *qg, float *k, const float *q_norm, const float *k_norm,
                   int nh, int nkv, int dh, int n_rot, float rope_base, int pos, float eps);
void q35cu_kv_store(void *Kc, void *Vc, const float *k, const float *v,
                    int64_t koff, int64_t voff, int n_kv, int fmt);

/* Flash-decode over T timesteps of the device window.
 *   o[nh][dh]  = sum_t softmax(s)_t * V_t     (already normalized)
 *   lse[nh]    = log sum_t exp(s_t)           (for the split-context merge, see §7.4)
 * `scratch` must hold q35cu_attn_scratch(nh, dh, T) floats. */
size_t q35cu_attn_scratch(int nh, int dh, int T);
void   q35cu_attn_flash(float *o, float *lse, const float *qg,
                        const void *Kc, const void *Vc,
                        int nh, int nkv, int dh, int T, int64_t k_stride, int64_t v_stride,
                        float scale, int fmt, float *scratch);
void   q35cu_attn_gate(float *o, const float *qg, int nh, int dh);

/* Merge a second (output, lse) pair computed elsewhere — the CPU's half of a split context.
 * Standard flash-decoding combine; exact up to rounding. */
void q35cu_lse_merge(float *o, float *lse, const float *o2, const float *lse2, int nh, int dh);

/* ---------------- instrumentation ----------------
 * COLIBRI_KERNEL_LOG=1 makes every component announce the device it actually ran on. */
void q35cu_log(const char *component, const char *detail);
void q35cu_mark_begin(void);
void q35cu_mark(const char *name);
int  q35cu_mark_read(int i, const char **name, float *ms);
void q35cu_mark_end(void);
int  q35cu_log_on(void);

#ifdef __cplusplus
}
#endif
#endif
