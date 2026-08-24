/* gguf_test.c — parse real GGUF files, then prove the expert-slice math.
 *
 * Run against:
 *   1. a complete small GGUF (tensor directory, offsets, dense mmap reads)
 *   2. Unsloth's DeepSeek-V4-Flash metadata shard (72 KV, 0 tensors — the split layout
 *      that a naive reader chokes on)
 *
 * The load must be O(header): if opening a 97 GB model touches 97 GB, the design is wrong.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "../gguf.h"

static double now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int fails = 0;
static void ck(int cond, const char *msg){
    printf("  %s %s\n", cond ? "ok  " : "FAIL", msg);
    if(!cond) fails++;
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <file.gguf> [meta_shard.gguf]\n", argv[0]); return 2; }

    gguf_model m;
    double t0 = now();
    if(!gguf_open(&m, argv[1])){ printf("FAIL: could not open %s\n", argv[1]); return 1; }
    double dt = now() - t0;

    printf("%s\n", argv[1]);
    printf("  %d shard(s), %d tensors, %d kv, opened in %.1f ms\n",
           m.n_shards, m.n_tensors, m.n_kv, dt * 1e3);
    printf("  arch = %s\n", gguf_str(&m, "general.architecture", "?"));

    int64_t total = 0;
    for(int i = 0; i < m.n_tensors; i++) total += m.t[i].bytes;
    printf("  tensor payload: %.2f MB\n", total / 1e6);

    if(m.n_tensors){
        /* a dense read must come straight out of the mapping and decode sanely */
        const gguf_tensor *t = &m.t[0];
        printf("  first tensor: %-28s %-7s ne=[%lld,%lld,%lld]\n", t->name, kq_name(t->type),
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
        const void *p = gguf_data(&m, t);
        ck(p != NULL, "dense tensor resolves to a mapped pointer");

        float buf[256];
        int n = (int)(t->ne[0] < 256 ? t->ne[0] : 256);
        n -= n % kq_blocksize(t->type);
        if(n > 0){
            int okd = kq_dequant_row(t->type, p, buf, n);
            int finite = 1;
            for(int i = 0; i < n; i++) if(!(buf[i] == buf[i])) finite = 0;
            ck(okd && finite, "first row decodes to finite values");
        }
        ck(gguf_find(&m, t->name) == t, "hash lookup returns the same tensor");
        ck(gguf_find(&m, "no.such.tensor") == NULL, "unknown name returns NULL");
    }

    /* --- the part that matters for streaming: expert slices --- */
    const gguf_tensor *ex = NULL;
    for(int i = 0; i < m.n_tensors; i++)
        if(m.t[i].n_dims == 3 && strstr(m.t[i].name, "_exps.")){ ex = &m.t[i]; break; }

    if(ex){
        printf("\n  expert tensor: %s\n", ex->name);
        printf("    %s  ne=[%lld,%lld,%lld]  %.2f MB total\n", kq_name(ex->type),
               (long long)ex->ne[0], (long long)ex->ne[1], (long long)ex->ne[2],
               ex->bytes / 1e6);
        int fd; uint64_t off0, off1; int64_t bytes, rows, cols;
        int a = gguf_expert_slice(&m, ex, 0, &fd, &off0, &bytes, &rows, &cols);
        int b = gguf_expert_slice(&m, ex, 1, &fd, &off1, &bytes, &rows, &cols);
        ck(a && b, "expert 0 and 1 both resolve");
        ck(off1 - off0 == (uint64_t)bytes, "consecutive experts are contiguous (one pread each)");
        printf("    one expert = %.2f MB, rows=%lld cols=%lld\n",
               bytes / 1e6, (long long)rows, (long long)cols);
        int bad = gguf_expert_slice(&m, ex, (int)ex->ne[2], &fd, &off0, &bytes, &rows, &cols);
        ck(!bad, "out-of-range expert index is rejected");
    } else {
        printf("\n  (no 3-D expert tensor in this file)\n");
    }
    gguf_close(&m);

    /* --- Unsloth's metadata-only shard --- */
    if(argc >= 3){
        printf("\n%s\n", argv[2]);
        gguf_model d;
        t0 = now();
        if(!gguf_open(&d, argv[2])){ printf("FAIL: could not open %s\n", argv[2]); return 1; }
        dt = now() - t0;
        printf("  %d tensors, %d kv, opened in %.1f ms\n", d.n_tensors, d.n_kv, dt * 1e3);
        ck(d.n_tensors == 0, "metadata shard carries no tensors (as Unsloth ships it)");

        const char *arch = gguf_str(&d, "general.architecture", "?");
        printf("  arch = %s   quantized_by = %s\n", arch,
               gguf_str(&d, "general.quantized_by", "?"));
        ck(strcmp(arch, "deepseek4") == 0, "architecture reads back as deepseek4");

        uint64_t nl  = gguf_u64(&d, "deepseek4.block_count", 0);
        uint64_t ne  = gguf_u64(&d, "deepseek4.expert_count", 0);
        uint64_t nu  = gguf_u64(&d, "deepseek4.expert_used_count", 0);
        uint64_t emb = gguf_u64(&d, "deepseek4.embedding_length", 0);
        uint64_t ffl = gguf_u64(&d, "deepseek4.expert_feed_forward_length", 0);
        printf("  %llu layers, %llu experts, top-%llu, d_model=%llu, d_ff=%llu\n",
               (unsigned long long)nl, (unsigned long long)ne, (unsigned long long)nu,
               (unsigned long long)emb, (unsigned long long)ffl);
        ck(nl == 43 && ne == 256 && nu == 6, "MoE geometry matches the published config");

        /* the numbers that decide tok/s */
        double bpw_gu = 8.0 * kq_typesize(KQ_IQ2_XS)  / (double)kq_blocksize(KQ_IQ2_XS);
        double bpw_d  = 8.0 * kq_typesize(KQ_IQ3_XXS) / (double)kq_blocksize(KQ_IQ3_XXS);
        double gb = nl * nu * ((2.0 * emb * ffl * bpw_gu) + (1.0 * emb * ffl * bpw_d)) / 8.0 / 1e9;
        printf("\n  routed-expert traffic: %.2f GB/token (gate+up IQ2_XS, down IQ3_XXS)\n", gb);
        printf("  on this box's 3.2 GB/s NVMe -> %.2f tok/s ceiling\n", 3.2 / gb);
        gguf_close(&d);
    }

    printf("\n%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
