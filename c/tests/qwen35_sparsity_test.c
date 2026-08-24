/* qwen35_sparsity_test.c — is there exploitable activation sparsity in a SwiGLU FFN?
 *
 * The MoE-conversion route is out (it needs training, and MoEfication additionally assumes
 * ReLU's ~90% hard zeros, which SwiGLU does not have). The runtime alternative is threshold
 * sparsity: compute gate fully, keep only the neurons whose activation matters, and read only
 * those rows of up and down. Bytes per FFN become 1/3 + 2/3*s.
 *
 * That rests on a premise this test measures rather than assumes: that the hidden vector
 * h = silu(gate) * up concentrates its mass in a small fraction of its 17408 neurons. If it
 * does not, the whole approach is dead and no amount of kernel work saves it.
 *
 * Two things are reported per layer:
 *   - what fraction of neurons carries 99% / 99.9% of the L2 mass of h
 *   - the relative error in h if the smallest neurons are dropped to hit a target sparsity
 *
 * The second is the number that matters: it bounds the FFN output error, because
 * ||down*(h - h_s)|| <= ||down|| * ||h - h_s||.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../qwen35_cpu.h"

static int cmp_desc(const void *a, const void *b){
    const float x = *(const float*)a, y = *(const float*)b;
    return x < y ? 1 : (x > y ? -1 : 0);
}

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : "/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf";
    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    Q35Resident RES;
    if(!q35_reside_anon(&M, &RES, Q35_RES_ALL)) return 1;

    Q35State R; q35_state_init(&R, &M, 64); q35_state_reset(&R);
    const Q35Cfg *c = &M.c;
    const int F = c->d_ff;

    /* a real prompt, so the activations are the ones the model actually sees */
    int prompt[8] = {760,6511,314,9338,369,11751,13,198};
    float *logits = malloc(sizeof(float)*c->vocab);
    for(int i = 0; i < 8; i++) q35_forward(&R, prompt[i], i, i==7?logits:NULL);

    /* Re-run the last token layer by layer, capturing h at each FFN. q35_ffn leaves
     * silu(gate)*up in R->ffn_g, so the trunk is walked here rather than calling q35_forward. */
    float *mag = malloc(sizeof(float)*F);
    /* Low active fractions, which is what a fine-grained MoE actually asks for: E=200 with
     * top-4 activates 2% of the neurons. */
    const float targets[8] = {0.50f, 0.30f, 0.20f, 0.10f, 0.05f, 0.02f, 0.01f, 0.005f};
    double acc99[80], acc999[80], err[8] = {0,0,0,0,0,0,0,0};
    /* Group test: split the 17408 neurons into E CONTIGUOUS groups and ask how much of a
     * token's mass the top-k groups hold. Contiguous grouping is a LOWER bound — real
     * MoEfication clusters by co-activation and does better. The per-token oracle above is
     * the UPPER bound no static partition can beat. If the two bracket badly, routing is
     * dead regardless of how clever the clustering is. */
    const int EE[3] = {20, 64, 200};
    double grp[3][3] = {{0}};      /* [E][top-k index], k = E/10, E/20, E/50 */
    int n_layer_seen = 0;

    float *x = R.x, *xn = R.xn, *t = R.tmp;
    q35_embed(&R, prompt[7], x);
    for(int il = 0; il < c->n_layer; il++){
        const Q35Layer *L = &M.L[il];
        q35_rmsnorm(xn, x, L->attn_norm, c->d_model, c->eps);
        if(L->kind == Q35_LAYER_ATTN) q35_attn(&R, L, il, xn, 8, t);
        else                          q35_gdn (&R, L, il, xn, t);
        for(int i = 0; i < c->d_model; i++) x[i] += t[i];

        q35_rmsnorm(xn, x, L->post_attn_norm, c->d_model, c->eps);
        q35_ffn(&R, L, xn, t);          /* leaves h in R.ffn_g */
        for(int i = 0; i < c->d_model; i++) x[i] += t[i];

        /* --- the measurement --- */
        double tot = 0;
        for(int i = 0; i < F; i++){ mag[i] = fabsf(R.ffn_g[i]); tot += (double)mag[i]*mag[i]; }
        qsort(mag, F, sizeof(float), cmp_desc);
        double run = 0; int k99 = F, k999 = F;
        for(int i = 0; i < F; i++){
            run += (double)mag[i]*mag[i];
            if(k99  == F && run >= 0.990*tot) k99  = i+1;
            if(k999 == F && run >= 0.999*tot) k999 = i+1;
        }
        acc99[n_layer_seen]  = (double)k99/F;
        acc999[n_layer_seen] = (double)k999/F;
        /* error from keeping the top s fraction */
        for(int ti = 0; ti < 8; ti++){
            const int keep = (int)(targets[ti]*F);
            double kept = 0;
            for(int i = 0; i < keep && i < F; i++) kept += (double)mag[i]*mag[i];
            err[ti] += sqrt((tot-kept)/tot);
        }
        /* contiguous-group masses, using the UNSORTED h (mag was sorted in place, so recompute) */
        for(int ei = 0; ei < 3; ei++){
            const int E = EE[ei], per = F/E;
            double *gm = (double*)calloc(E, sizeof(double));
            for(int i = 0; i < F; i++){
                const int g = i/per < E ? i/per : E-1;
                gm[g] += (double)R.ffn_g[i]*R.ffn_g[i];
            }
            /* sort group masses descending */
            for(int a = 0; a < E; a++) for(int b = a+1; b < E; b++)
                if(gm[b] > gm[a]){ double t2 = gm[a]; gm[a] = gm[b]; gm[b] = t2; }
            const int ks[3] = { E/10, E/20, E/50 };
            for(int kk = 0; kk < 3; kk++){
                const int K = ks[kk] > 0 ? ks[kk] : 1;
                double c = 0;
                for(int i = 0; i < K; i++) c += gm[i];
                grp[ei][kk] += c/tot;
            }
            free(gm);
        }
        n_layer_seen++;
    }

    double m99 = 0, m999 = 0, w99 = 0;
    for(int i = 0; i < n_layer_seen; i++){
        m99 += acc99[i]; m999 += acc999[i];
        if(acc99[i] > w99) w99 = acc99[i];
    }
    printf("\nSwiGLU hidden h = silu(gate)*up, %d neurons, %d layers, real prompt\n", F, n_layer_seen);
    printf("  neurons carrying 99%%   of L2 mass: mean %.1f%%  (worst layer %.1f%%)\n",
           100*m99/n_layer_seen, 100*w99);
    printf("  neurons carrying 99.9%% of L2 mass: mean %.1f%%\n", 100*m999/n_layer_seen);
    printf("\n  PER-TOKEN ORACLE (upper bound: no static partition can beat this)\n");
    printf("    keep top s of neurons -> mass captured / relative L2 error in h:\n");
    for(int ti = 0; ti < 8; ti++){
        const double e = err[ti]/n_layer_seen;
        printf("      s = %.3f   mass %5.1f%%   err %.4f\n", targets[ti], 100*(1.0-e*e), e);
    }
    printf("\n  CONTIGUOUS GROUPS (lower bound: real clustering does better than this)\n");
    printf("    %-6s %10s %10s %10s\n", "E", "top E/10", "top E/20", "top E/50");
    for(int ei = 0; ei < 3; ei++)
        printf("    %-6d %9.1f%% %9.1f%% %9.1f%%\n", EE[ei],
               100*grp[ei][0]/n_layer_seen, 100*grp[ei][1]/n_layer_seen, 100*grp[ei][2]/n_layer_seen);
    printf("\n    a UNIFORM h would give exactly 10%%, 5%%, 2%% — that is the no-structure baseline\n");

    free(mag); free(logits);
    q35_state_free(&R); q35_close(&M);
    return 0;
}
