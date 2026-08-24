/* qwen35_mtp_test.c — how often does the shipped draft head guess right?
 *
 * Speculative decoding is worth exactly its ACCEPTANCE RATE and nothing else. A draft head
 * that agrees 80% of the time turns one pass over 15.6 GiB into 1.8 tokens; one that agrees
 * 20% of the time is a slower decoder with extra steps. The rate is a property of these
 * weights on this kind of text, so it is measured, not assumed.
 *
 * Method: generate greedily with the main model. At each step, ask the draft head what it
 * thinks the NEXT token will be, then let the main model actually produce it and compare.
 * That is precisely the accept test a speculative loop performs, without needing the
 * rollback machinery to be correct first.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../qwen35_mtp.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

int main(int argc, char **argv){
    const char *path = argc > 1 ? argv[1]
        : "/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf";
    const int n_gen = argc > 2 ? atoi(argv[2]) : 24;

    Q35Model M;
    if(!q35_open(&M, path)){ printf("open failed\n"); return 1; }
    if(M.c.n_layer_mtp < 1){ printf("model has no MTP block\n"); return 1; }

    Q35Resident RES;
    if(!q35_reside_anon(&M, &RES, Q35_RES_ALL)) return 1;
    q35_reside_report(&RES, stdout);

    Q35State R; q35_state_init(&R, &M, n_gen + 32); q35_state_reset(&R);
    Q35Mtp P;
    if(!q35_mtp_init(&P, &R)){ printf("mtp init failed\n"); return 1; }

    const int V = M.c.vocab;
    float *lg = malloc(sizeof(float)*V), *dr = malloc(sizeof(float)*V), *h = malloc(sizeof(float)*M.c.d_model);

    /* "The capital of France is" */
    int prompt[5] = {760,6511,314,9338,369};
    int pos = 0;
    for(int i = 0; i < 5; i++) q35_forward(&R, prompt[i], pos++, i==4 ? lg : NULL);

    int tok = q35_argmax(lg, V);
    int hits = 0, tries = 0, top5 = 0;
    double t_main = 0, t_draft = 0;

    printf("\n  step  main_tok  draft_tok  hit  draft_rank\n");
    for(int i = 0; i < n_gen; i++){
        /* the draft head sees the state that produced `tok`, and guesses what follows it */
        q35_hidden(&R, h);
        double t0 = now();
        q35_mtp_draft(&P, h, tok, pos, dr);
        t_draft += now()-t0;
        const int guess = q35_argmax(dr, V);

        /* the main model then actually produces it */
        t0 = now();
        q35_forward(&R, tok, pos++, lg);
        t_main += now()-t0;
        const int truth = q35_argmax(lg, V);

        /* where did the truth rank in the draft's distribution? */
        int rank = 0;
        for(int j = 0; j < V; j++) if(dr[j] > dr[truth]) rank++;

        const int hit = (guess == truth);
        hits += hit; tries++;
        if(rank < 5) top5++;
        if(i < 12) printf("  %4d  %8d  %9d  %3s  %10d\n", i, truth, guess, hit?"yes":"no", rank);
        tok = truth;
    }

    const double a = (double)hits/tries;
    printf("\n  acceptance (draft argmax == main argmax): %d/%d = %.1f%%\n", hits, tries, 100*a);
    printf("  main token in draft top-5              : %d/%d = %.1f%%\n", top5, tries, 100.0*top5/tries);
    printf("  cost: main %.3f s/token, draft %.3f s/token (%.1f%% of a main pass)\n",
           t_main/tries, t_draft/tries, 100.0*(t_draft/tries)/(t_main/tries));

    /* Expected speedup for 1-token lookahead: a verification batch of 2 costs about what a
     * batch of 1 does (both read the weights once), plus the draft and the state snapshot. */
    const double c_main = t_main/tries, c_draft = t_draft/tries;
    const double c_round = c_main*1.06 + c_draft + 0.003;   /* batch-of-2 + draft + snapshot */
    printf("\n  projected: %.2f tokens per %.3f s round -> %.2f tok/s vs %.2f baseline (%.2fx)\n",
           1.0+a, c_round, (1.0+a)/c_round, 1.0/c_main, ((1.0+a)/c_round)/(1.0/c_main));

    /* ---- the real thing: run the speculative loop and compare against greedy ---- */
    {
        Q35State R2; q35_state_init(&R2, &M, n_gen + 64); q35_state_reset(&R2);
        Q35Mtp P2;   q35_mtp_init(&P2, &R2);
        Q35Batch B2; q35_batch_init(&B2, &R2, 4);

        int p2 = 0;
        for(int i = 0; i < 5; i++) q35_forward(&R2, prompt[i], p2++, i==4 ? lg : NULL);
        int start = q35_argmax(lg, V);

        int *out = malloc(sizeof(int)*n_gen);
        Q35SpecStats st; memset(&st, 0, sizeof st);
        double t0 = now();
        const int n = q35_spec_generate(&P2, &B2, start, p2, out, n_gen, -1, &st);
        const double dt = now()-t0;

        printf("\n  speculative loop: %d tokens in %.2f s = %.2f tok/s\n", n, dt, n/dt);
        printf("    rounds %lld, accepted %lld (%.1f%%), %.2f tokens/round\n",
               (long long)st.rounds, (long long)st.accepts,
               st.rounds ? 100.0*st.accepts/st.rounds : 0.0,
               st.rounds ? (double)st.tokens/st.rounds : 0.0);
        printf("    draft %.3f s, verify %.3f s, rollback %.3f s\n",
               st.t_draft, st.t_verify, st.t_rollback);
        printf("    vs greedy baseline %.2f tok/s  ->  %.2fx\n",
               1.0/c_main, (n/dt)/(1.0/c_main));

        /* correctness: greedy and speculative must produce the SAME tokens. Speculation is
         * only sound if a rejected draft leaves no trace — this is the test for that. */
        printf("\n    first 10 greedy vs speculative:\n      greedy:");
        /* regenerate greedily for comparison */
        Q35State R3; q35_state_init(&R3, &M, n_gen + 64); q35_state_reset(&R3);
        int p3 = 0;
        for(int i = 0; i < 5; i++) q35_forward(&R3, prompt[i], p3++, i==4 ? lg : NULL);
        int t3 = q35_argmax(lg, V), mism = 0;
        for(int i = 0; i < n && i < n_gen; i++){
            if(i < 10) printf(" %d", t3);
            if(i < n && out[i] != t3) mism++;
            q35_forward(&R3, t3, p3++, lg);
            t3 = q35_argmax(lg, V);
        }
        printf("\n      spec  :");
        for(int i = 0; i < 10 && i < n; i++) printf(" %d", out[i]);
        printf("\n    mismatches: %d %s\n", mism, mism ? "<- FAIL: speculation changed the output" : "(identical)");

        free(out);
        q35_batch_free(&B2); q35_mtp_free(&P2); q35_state_free(&R2); q35_state_free(&R3);
        if(mism) { free(lg); free(dr); free(h); q35_mtp_free(&P); q35_state_free(&R); q35_close(&M); return 1; }
    }

    free(lg); free(dr); free(h);
    q35_mtp_free(&P); q35_state_free(&R); q35_close(&M);
    return 0;
}
