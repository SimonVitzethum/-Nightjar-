/* qwen35_run.c — the engine as a program: tokenizer, chat template, sampling, streaming.
 *
 * Everything in front of this file is a kernel or a schedule; this is what makes it usable.
 * It does four things the tests do not:
 *
 *   - builds the tokenizer out of the GGUF, with the qwen35 pre-tokenizer variant (see
 *     tok_set_pre — two alternatives differ from cl100k and both fail silently);
 *   - applies the ChatML template the model was trained on, literally, using CONTROL tokens
 *     that must never go through BPE;
 *   - orders startup so the FFN layers that went to VRAM are NOT also copied into RAM, which
 *     is 1.5 GiB of KV budget at long context;
 *   - grows the context instead of fixing it, and stops the process rather than allocating
 *     past the floor.
 *
 * Usage:
 *   qwen35_run <model.gguf> [-p "prompt"] [-n 256] [--temp 0.7] [--top-p 0.8] [--top-k 20]
 *              [--ctx N] [--raw] [--no-think] [--cpu] [--stats]
 *   With no -p it reads a prompt per line from stdin and keeps the conversation.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/prctl.h>

#include "qwen35_cpu.h"
#include "qwen35_sample.h"
#ifndef QWEN_NO_CUDA
#include "qwen35_cu_spec.h"
#endif
#include "qwen35_cupti.h"
#include "tok.h"
#include "tok_gguf.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

/* libgomp reads OMP_WAIT_POLICY and friends when it initializes, which happens before we
 * could set them. Measured on this machine they are worth 20%: without an active wait the
 * threads sleep between layers and every FFN pays a wake-up. So set them and re-exec once.
 * QWEN_ENV_SET marks that we already did, so this cannot loop. */
static void q35_fix_omp_env(char **argv){
    /* execv("/proc/self/exe") renames the process to "exe" — that is the basename of the path
     * handed to execve, not of the binary. pgrep, pkill, ps and every "is it running" check
     * then miss it, including this project's own `make stop`. Say the name explicitly. */
    prctl(PR_SET_NAME, "qwen35_run", 0, 0, 0);
    q35_env_compat();          /* COLIBRI_* still works */
    if(getenv("QWEN_ENV_SET")) return;
    setenv("QWEN_ENV_SET", "1", 1);
    if(!getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY", "active", 1);
    if(!getenv("OMP_PROC_BIND"))   setenv("OMP_PROC_BIND", "close", 1);
    if(!getenv("OMP_PLACES"))      setenv("OMP_PLACES", "cores", 1);
    if(!getenv("GOMP_SPINCOUNT"))  setenv("GOMP_SPINCOUNT", "200000", 1);
    if(!getenv("OMP_NUM_THREADS")){
        /* 16 measured best on this 10-core/20-thread part: past that the k-quant decode
         * stops scaling and the hyperthread pairs just contend for the same ports. */
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        char b[16]; snprintf(b, sizeof b, "%ld", n > 16 ? 16 : (n > 1 ? n : 1));
        setenv("OMP_NUM_THREADS", b, 1);
    }
    execv("/proc/self/exe", argv);          /* if this fails we simply run unteuned */
}

static volatile sig_atomic_t g_stop = 0;
static void on_int(int s){ (void)s; g_stop = 1; }

/* ---------------- sampling ---------------- */

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static double urand(void){
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11)/9007199254740992.0;
}
/* Cand, and the selection sampler behind it, live in qwen35_sample.h — sorting a
 * 248320-entry vocabulary once per token cost more than the entire forward pass. */
static int sample(float *logits, int V, float temp, float top_p, int top_k, Cand *buf){
    return q35_sample(logits, V, temp, top_p, top_k, buf, urand);
}

/* ---------------- chat template ----------------
 * <|im_start|>role\n content <|im_end|>\n , then an open assistant turn. The role markers are
 * CONTROL tokens: tok_encode matches them literally and never merges them. */
typedef struct { char *buf; size_t n, cap; } Str;
static void sadd(Str *s, const char *p, size_t n){
    if(s->n + n + 1 > s->cap){ s->cap = (s->n + n + 1)*2; s->buf = (char*)realloc(s->buf, s->cap); }
    memcpy(s->buf + s->n, p, n); s->n += n; s->buf[s->n] = 0;
}
static void sfmt(Str *s, const char *fmt, ...){
    char tmp[8192]; va_list ap; va_start(ap, fmt);
    const int n = vsnprintf(tmp, sizeof tmp, fmt, ap); va_end(ap);
    if(n > 0) sadd(s, tmp, (size_t)(n < (int)sizeof tmp ? n : (int)sizeof tmp - 1));
}

int main(int argc, char **argv){
    q35cupti_init();               /* before any CUDA context exists, so nothing is missed */
    if(getenv("QWEN_PROFILE")) g_q35_stage_timing = 1;
    const char *path = NULL, *prompt = NULL, *sysmsg = NULL;
    int n_pred = 256, top_k = -1, n_ctx = 0, raw = 0, cpu_only = 0, stats = 0, no_think = 0;
    int spec = 1;
    float temp = -1, top_p = -1;
    for(int i = 1; i < argc; i++){
        const char *a = argv[i];
        if(a[0] != '-' && !path) path = a;
        else if(!strcmp(a,"-p") && i+1 < argc) prompt = argv[++i];
        else if(!strcmp(a,"-s") && i+1 < argc) sysmsg = argv[++i];
        else if(!strcmp(a,"-n") && i+1 < argc) n_pred = atoi(argv[++i]);
        else if(!strcmp(a,"--temp") && i+1 < argc) temp = (float)atof(argv[++i]);
        else if(!strcmp(a,"--top-p") && i+1 < argc) top_p = (float)atof(argv[++i]);
        else if(!strcmp(a,"--top-k") && i+1 < argc) top_k = atoi(argv[++i]);
        else if(!strcmp(a,"--ctx") && i+1 < argc) n_ctx = atoi(argv[++i]);
        else if(!strcmp(a,"--seed") && i+1 < argc) g_rng = (uint64_t)strtoull(argv[++i],NULL,10)|1;
        else if(!strcmp(a,"--raw")) raw = 1;
        else if(!strcmp(a,"--no-think")) no_think = 1;
        else if(!strcmp(a,"--cpu")) cpu_only = 1;
        else if(!strcmp(a,"--no-spec")) spec = 0;
        else if(!strcmp(a,"--stats")) stats = 1;
        else if(!strcmp(a,"-h") || !strcmp(a,"--help")){
            fprintf(stderr, "usage: %s <model.gguf> [-p prompt] [-s system] [-n 256]"
                    " [--temp t] [--top-p p] [--top-k k] [--ctx n] [--raw] [--no-think]"
                    " [--cpu] [--no-spec] [--stats]\n", argv[0]);
            return 0;
        }
    }
    if(!path){ fprintf(stderr, "qwen35_run: no model given (-h for usage)\n"); return 1; }
    q35_fix_omp_env(argv);
    signal(SIGINT, on_int);

    Q35Model M;
    if(!q35_open(&M, path)){ fprintf(stderr, "qwen35_run: cannot open %s\n", path); return 1; }
    const Q35Cfg *c = &M.c;
    if(temp  < 0) temp  = c->temp  > 0 ? c->temp  : 0.7f;
    if(top_p < 0) top_p = c->top_p > 0 ? c->top_p : 0.8f;
    if(top_k < 0) top_k = c->top_k > 0 ? c->top_k : 20;

    Tok T; TokIds ids;
    tok_set_pre(&M.m);
    if(!tok_from_gguf(&T, &M.m, &ids)){ fprintf(stderr, "qwen35_run: tokenizer failed\n"); return 1; }
    const int im_start = tok_id_of(&T, "<|im_start|>");
    const int im_end   = tok_id_of(&T, "<|im_end|>");
    const int eos      = im_end >= 0 ? im_end : ids.eos;

    /* Context: start at what the prompt needs and let kvt_grow take it from there. The KV
     * tier grows until MemAvailable drops under the floor, and then q35_oom stops the
     * process instead of asking for more. */
    if(n_ctx <= 0) n_ctx = 8192;

    fprintf(stderr, "qwen35_run: %s\n  %d layers (%d gdn + %d attn), d_model %d, vocab %d\n",
            path, c->n_layer, c->n_layer - q35_n_attn_layers(c), q35_n_attn_layers(c),
            c->d_model, c->vocab);
    fprintf(stderr, "  sampling: temp %.2f  top_p %.2f  top_k %d\n", temp, top_p, top_k);

    Q35State R;
    const int64_t kvb = q35_kv_bytes_per_token(c, Q35_KV_Q8_0)*(int64_t)n_ctx + (32<<20);
    if(!q35_state_init_ex(&R, &M, n_ctx, Q35_KV_Q8_0, kvb, getenv("QWEN_KV_SPILL"), 4096)){
        fprintf(stderr, "qwen35_run: kv init failed\n"); return 1; }

#ifndef QWEN_NO_CUDA
    Q35Cu G; int use_gpu = 0;
    if(!cpu_only){
        const double t0 = now();
        if(q35cu_model_init(&G, &M, n_ctx, R.kv.fmt)){
            q35cu_align_win(&G, R.kv.chunk);
            /* BEFORE residency: the FFN layers now in VRAM must not be copied to RAM too. */
            q35cu_mark_resident_elsewhere(&G);
            use_gpu = 1;
            fprintf(stderr, "  gpu upload: %.1f s\n", now()-t0);
        } else {
            fprintf(stderr, "  gpu unavailable (%s) — CPU only\n", q35cu_error());
        }
    }
#else
    const int use_gpu = 0;
#endif

    Q35Resident RES;
    q35_reside_anon(&M, &RES, use_gpu ? Q35_RES_FFN : Q35_RES_ALL);
#ifndef QWEN_NO_CUDA
    if(use_gpu){
        q35cu_pin_weights(&G, &M);           /* after residency: the mapping exists only now */
        q35cu_report(&G, stderr);            /* ...and report after it, so the line is true */
    }
#endif
    if(stats) q35_reside_report(&RES, stderr);
    else fprintf(stderr, "  weights in RAM: %.2f GiB (%.0f%% resident)\n",
                 RES.resident/1073741824.0, RES.want ? 100.0*RES.resident/RES.want : 0.0);

    q35_state_reset(&R);
#ifndef QWEN_NO_CUDA
    Q35CuBatch Z; Q35Mtp P; int use_spec = 0;
    if(use_gpu){
        q35cu_state_reset(&G);
        if(spec && q35cu_batch_init(&Z, &G, &R, 4) && q35_mtp_init(&P, &R)){
            use_spec = 1;
            fprintf(stderr, "  speculative decode: MTP draft head, +%.0f MiB VRAM\n",
                    Z.vram/1048576.0);
        } else if(spec){
            fprintf(stderr, "  speculative decode unavailable — plain decode\n");
        }
    }
#endif

    float *logits = (float*)malloc(sizeof(float)*c->vocab);
    Cand  *cbuf   = (Cand*)malloc(sizeof(Cand)*c->vocab);
    int    tok_cap = n_ctx > (1<<20) ? n_ctx : (1<<20);
    int   *toks   = (int*)malloc(sizeof(int)*(size_t)tok_cap);
    char   piece[512];
    if(!logits || !cbuf || !toks){ fprintf(stderr, "qwen35_run: OOM\n"); return 1; }

    Str hist = {0};
    char line[1<<16];
    int pos = 0, turn = 0;

    for(;;){
        const char *user = prompt;
        if(!user){
            if(turn == 0) fprintf(stderr, "\n> ");
            else          fprintf(stderr, "\n> ");
            if(!fgets(line, sizeof line, stdin)) break;
            size_t L = strlen(line);
            while(L && (line[L-1]=='\n' || line[L-1]=='\r')) line[--L] = 0;
            if(!L) continue;
            if(!strcmp(line, "/quit") || !strcmp(line, "/exit")) break;
            user = line;
        }

        Str turn_s = {0};
        if(raw){
            sadd(&turn_s, user, strlen(user));
        } else {
            if(turn == 0 && sysmsg) sfmt(&turn_s, "<|im_start|>system\n%s<|im_end|>\n", sysmsg);
            sfmt(&turn_s, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
            if(no_think) sadd(&turn_s, "<think>\n\n</think>\n\n", 20);
        }
        sadd(&hist, turn_s.buf, turn_s.n);

        int nt = tok_encode(&T, turn_s.buf, (int)turn_s.n, toks, tok_cap - 8);
        free(turn_s.buf);
        /* Grow the store to hold the prompt rather than truncating it. kvt_grow stops and
         * q35_oom terminates the process if RAM falls under the floor — the context is never
         * silently clipped and never allocated past the point of no return. */
        if(pos + nt + 8 > R.n_ctx){
            if(!kvt_grow(&R.kv, pos + nt + 256)){
                fprintf(stderr, "qwen35_run: cannot grow context to %d\n", pos + nt + 256);
                break;
            }
            R.n_ctx = R.kv.n_ctx;
            R.att = (float*)realloc(R.att, sizeof(float)*(size_t)R.n_ctx*c->n_head);
            if(!R.att){ fprintf(stderr, "qwen35_run: OOM growing score buffer\n"); break; }
        }
        if(nt <= 0){ fprintf(stderr, "qwen35_run: nothing to encode\n"); if(prompt) break; continue; }

        const double tp0 = now();
        int i = 0;
#ifndef QWEN_NO_CUDA
        /* BATCHED PREFILL. A prompt token costs the same weight pass as a generated one only
         * if you feed them one at a time; in a batch the 9.65 GiB is read once for all of
         * them, and on the streamed layers the PCIe traffic is shared too. This is what makes
         * a long prompt affordable at all — it is the difference between hours and minutes at
         * 100k tokens. */
        if(use_spec){
            const int SB = Z.S_max;
            float *xall = (float*)malloc(sizeof(float)*(size_t)SB*c->d_model);
            float *hh   = (float*)malloc(sizeof(float)*c->d_model);
            /* stop one token short: the final position must go through the single-token
             * path, because that is the call that produces `logits`. Batching straight to
             * the end leaves logits from the previous turn. */
            for(; i + SB <= nt - 1 && !g_stop; i += SB){
                q35_forward_cu_batch_ex(&Z, toks + i, SB, pos, NULL, 0, xall);
                /* keep the draft head's own KV in step with the trunk */
                for(int s2 = 0; s2 < SB; s2++){
                    q35_rmsnorm(hh, xall + (size_t)s2*c->d_model, M.output_norm,
                                c->d_model, c->eps);
                    q35_mtp_prefill_kv(&P, hh, toks[i+s2], pos + s2);
                }
                pos += SB;
                if(((i/SB) % 16) == 0 || i + 2*SB > nt)
                    fprintf(stderr, "\r  prefill %d/%d  %.1f tok/s   ", i+SB, nt, (i+SB)/(now()-tp0));
            }
            free(xall); free(hh);
        }
#endif
        for(; i < nt && !g_stop; i++){
            const int last = (i == nt-1);
#ifndef QWEN_NO_CUDA
            if(use_gpu) q35_forward_cu(&G, &R, toks[i], pos, last ? logits : NULL);
            else
#endif
                        q35_forward(&R, toks[i], pos, last ? logits : NULL);
#ifndef QWEN_NO_CUDA
            if(use_spec){
                float hbuf[8192];
                q35_hidden(&R, hbuf);
                q35_mtp_prefill_kv(&P, hbuf, toks[i], pos);
            }
#endif
            pos++;
            if(nt > 8 && (i % 64 == 0 || last))
                fprintf(stderr, "\r  prefill %d/%d  %.1f tok/s   ", i+1, nt, (i+1)/(now()-tp0));
        }
        const double tprefill = now()-tp0;
        if(nt > 8) fprintf(stderr, "\r%*s\r", 48, "");

        const double td0 = now();
        q35cupti_reset();          /* window the timeline to decode: prefill is a different regime */
        int ngen = 0;
#ifndef QWEN_NO_CUDA
        if(use_spec){
            /* One round drafts a token with the MTP head and verifies BOTH positions in a
             * single batched pass. The true token is always sampled from the trunk's own
             * logits, so accepting only on an exact match leaves the output distribution
             * untouched — it is the same sampler, run at two positions for one pass. */
            Q35SpecStats st; memset(&st, 0, sizeof st);
            int *buf = (int*)malloc(sizeof(int)*(size_t)(n_pred+4));
            int cur = sample(logits, c->vocab, temp, top_p, top_k, cbuf);
            while(ngen < n_pred && !g_stop){
                const int room = n_pred - ngen;
                const int k = q35_spec_generate_cu(&Z, &P, cur, pos, buf, room < 2 ? 2 : room,
                                                  eos, &st, temp, top_p, top_k,
                                                  (int(*)(float*,int,float,float,int,void*))sample,
                                                  cbuf);
                if(k <= 0) break;
                int hit_eos = 0;
                for(int i = 0; i < k && ngen < n_pred; i++){
                    if(buf[i] == eos || buf[i] == ids.eos){ hit_eos = 1; break; }
                    const int np2 = tok_decode(&T, &buf[i], 1, piece, sizeof piece);
                    if(np2 > 0){ fwrite(piece, 1, (size_t)np2, stdout); sadd(&hist, piece, (size_t)np2); }
                    ngen++;
                }
                fflush(stdout);
                pos = R.pos;
                if(hit_eos || ngen >= n_pred) break;
                cur = buf[k-1];
                break;                     /* q35_spec_generate_cu already looped to n_out */
            }
            free(buf);
            if(stats)
                fprintf(stderr, "\n  [spec: %lld rounds, %lld accepts (%.0f%%), draft %.2fs,"
                        " verify %.2fs, rollback %.3fs]\n",
                        (long long)st.rounds, (long long)st.accepts,
                        st.rounds ? 100.0*st.accepts/st.rounds : 0.0,
                        st.t_draft, st.t_verify, st.t_rollback);
        } else
#endif
        for(; ngen < n_pred && !g_stop; ngen++){
            const int id = sample(logits, c->vocab, temp, top_p, top_k, cbuf);
            if(id == eos || id == ids.eos) break;
            const int np = tok_decode(&T, &id, 1, piece, sizeof piece);
            if(np > 0){ fwrite(piece, 1, (size_t)np, stdout); fflush(stdout); }
            sadd(&hist, piece, (size_t)(np > 0 ? np : 0));
            if(pos + 1 >= R.n_ctx && !kvt_grow(&R.kv, pos + 256)){
                fprintf(stderr, "\n[context full]\n"); break;
            }
            if(pos + 1 > R.n_ctx) R.n_ctx = R.kv.n_ctx;
#ifndef QWEN_NO_CUDA
            if(use_gpu) q35_forward_cu(&G, &R, id, pos, logits);
            else
#endif
                        q35_forward(&R, id, pos, logits);
            pos++;
        }
        const double tdec = now()-td0;
        q35cupti_report(ngen);
        printf("\n");
        fprintf(stderr, "  [prefill %d tok in %.2fs = %.1f tok/s | decode %d tok in %.2fs = %.2f tok/s | ctx %d]\n",
                nt, tprefill, nt/tprefill, ngen, tdec, ngen > 0 ? ngen/tdec : 0.0, pos);
        if(getenv("QWEN_PROFILE")){
            const double tot = g_t_embd+g_t_attn+g_t_gdn+g_t_ffn+g_t_out;
            fprintf(stderr, "  [profile per run: embd %.3f  attn %.3f  gdn %.3f  ffn %.3f  out %.3f  (sum %.3fs)]\n",
                    g_t_embd, g_t_attn, g_t_gdn, g_t_ffn, g_t_out, tot);
            if(tot>0) fprintf(stderr, "  [share: attn %.0f%%  gdn %.0f%%  ffn %.0f%%  out %.0f%%]\n",
                    100*g_t_attn/tot, 100*g_t_gdn/tot, 100*g_t_ffn/tot, 100*g_t_out/tot);
        }
        if(!raw) sadd(&hist, "<|im_end|>\n", 11);
        turn++;
        g_stop = 0;
        if(prompt) break;
    }

    free(logits); free(cbuf); free(toks); free(hist.buf);
#ifndef QWEN_NO_CUDA
    if(use_gpu && G.moe && G.moe_pre_tot)
        fprintf(stderr, "  moe pre-route: %.1f%% of the guess is truly routed (%llu probes)\n",
                100.0*G.moe_pre_hit/G.moe_pre_tot, (unsigned long long)G.moe_pre_tot);
    if(use_gpu && G.moe){ const uint64_t h=G.moe_hit,m=G.moe_miss,cp=G.moe_cpu,tot=h+m+cp;
        if(tot) fprintf(stderr,"  moe experts: %.1f%% VRAM-hit, %.1f%% streamed, %.1f%% CPU  (%llu reads)\n",
            100.0*h/tot,100.0*m/tot,100.0*cp/tot,(unsigned long long)tot); }
    if(use_spec){ q35_mtp_free(&P); q35cu_batch_free(&Z); }
    if(use_gpu) q35cu_model_free(&G);
#endif
    q35_state_free(&R);
    q35_close(&M);
    return 0;
}
