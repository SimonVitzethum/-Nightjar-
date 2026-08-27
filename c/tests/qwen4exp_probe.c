/* qwen4exp_probe.c — read a Qwen4-preview GGUF and say what this engine can and cannot run.
 *
 * The point is to settle the port's scope against the file instead of against a guess. It
 * reads the config, checks every tensor family the architecture names, and reports which are
 * already served by kernels this engine has and which are not. No model needs to be loaded
 * and no GPU is required -- the header is a few megabytes at the front of the first shard. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include "../models.h"
#include "../qwen4exp.h"
#include "../nj_arch.h"

static int have(const gguf_model *m, const char *fmt, int il){
    char n[256]; snprintf(n, sizeof n, fmt, il);
    return gguf_find(m, n) != NULL;
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : nj_model_path("Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf");
    char arch[64];
    const NjArch a = nj_arch_of(path, arch, sizeof arch);
    printf("  file: %s\n  architecture: %s -> engine %s\n\n", path, arch, nj_engine_for(a));

    gguf_model m;
    if(!gguf_open(&m, path)){ printf("  cannot read\n"); return 1; }
    Q4XCfg c;
    if(!q4x_cfg(&m, &c)){ printf("  not a qwen4exp config\n"); return 1; }
    q4x_report(&m, &c);

    /* Which tensor families are present. Shard 1 of a split carries metadata only, so absence
     * here means "not in this shard", not "not in the model" -- say so rather than imply a
     * missing weight. */
    printf("\n  tensor families in THIS shard (%llu tensors):\n",
           (unsigned long long)m.n_tensors);
    const struct { const char *fmt, *what, *status; } fam[] = {
        { "blk.%d.attn_q.weight",        "attention q/k/v/out",  "kernel exists" },
        { "blk.%d.ssm_conv1d.weight",    "gated delta net",      "kernel exists" },
        { "blk.%d.ffn_gate_exps.weight", "routed experts",       "kernel exists" },
        { "blk.%d.ffn_gate_shexp.weight","shared expert",        "kernel exists" },
        { "blk.%d.hc_attn_up.weight",    "hyper-connections",    "TO BUILD" },
        { "blk.%d.indexer.q_proj.weight","sparse attn indexer",  "TO BUILD" },
        { "blk.%d.ple_key.weight",       "per-layer embeddings", "TO BUILD" },
    };
    for(size_t i = 0; i < sizeof fam/sizeof *fam; i++){
        int n = 0;
        for(int il = 0; il < c.n_layer; il++) if(have(&m, fam[i].fmt, il)) n++;
        printf("    %-22s %3d/%d blocks   %s\n", fam[i].what, n, c.n_layer, fam[i].status);
    }
    printf("    %-22s %s\n", "per_layer_token_embd",
           gguf_find(&m, "per_layer_token_embd.weight") ? "present" : "not in this shard");
    gguf_close(&m);
    return 0;
}
