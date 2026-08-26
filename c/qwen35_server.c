/* qwen35_server.c — the engine behind an OpenAI-compatible HTTP API and a web chat UI.
 *
 * WHY THIS SHAPE
 *
 * Two clients, one endpoint. The browser UI and opencode both speak the same protocol, so
 * there is exactly one code path to keep correct: /v1/chat/completions with SSE. Everything
 * client-specific (markdown, thinking blocks, model picker) lives in the page, not here.
 *
 * THE ONE THING THAT MAKES IT USABLE: PREFIX CACHING
 *
 * OpenAI's protocol is stateless — every turn resends the whole conversation. Prefilling it
 * from scratch each time is O(n^2) in the conversation length, and at 8.7 tok/s of prefill a
 * ten-turn chat would spend minutes re-reading its own history. So the server keeps the token
 * sequence currently in the KV cache and, when the next request EXTENDS it, prefills only the
 * delta.
 *
 * When it does not extend it, the state is thrown away and rebuilt — there is no partial
 * rewind. That is not laziness: the gated delta net's state is decayed and rank-1 updated at
 * every step, which is not invertible, so "rewind to token k" does not exist. Attention KV
 * alone could be truncated; the recurrent half cannot.
 *
 * SERIALIZED ON PURPOSE
 *
 * One engine, one state, one request at a time. A second concurrent generation would need a
 * second 146 MiB recurrent state and a second KV window, and this card has neither.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <arpa/inet.h>
#include <pthread.h>

#include "qwen35_cpu.h"
#ifndef QWEN_NO_CUDA
#include "qwen35_cu_spec.h"
#endif
#include "qwen35_cache.h"
#ifndef QWEN_NO_CUDA
#endif
#include "tok.h"
#include "tok_gguf.h"
#include "json.h"

#include "strbuf.h"

/* The tool half of the agent loop. It needs Str, s_json and now(), which is why it is included
 * here and not at the top with the rest. */
#include "harness.h"
#include "chatstore.h"

/* WEIGHT BYTES PER GENERATED TOKEN, from the placement plan.
 *
 * Decoding one token reads every weight once: the tensors resident in VRAM, the FFN (whether the
 * CPU reads it in place or the GPU pulls it over PCIe — the bytes leave DRAM either way), and one
 * row of the embedding table. So a live GB/s is bytes_per_token x tok/s.
 *
 * It is DERIVED, not measured on the bus, and the display says so. What makes it worth showing
 * anyway is that it is the number this engine is built around: 59 GB/s against a DRAM controller
 * that tops out at 64.8. A tok/s figure alone does not say whether you are near the wall or
 * leaving the machine idle. This does. */
static int64_t g_bytes_per_tok = 0;   /* everything a token touches */
static int64_t g_bytes_dram    = 0;   /* the half that crosses the DRAM controller */
static int64_t g_bytes_vram    = 0;   /* the half already on the card */
static int     g_prefill_sb    = 8;   /* weights are read once per batch, not once per token */

/* THE TURN OUTLIVES THE CONNECTION.
 *
 * A turn here takes minutes. Tying it to the socket means a page reload throws away work that
 * cannot be redone cheaply — and reloading is the first thing anyone does when a UI looks
 * stuck. So a turn that names its conversation is held HERE: the text goes into this buffer as
 * it is generated, the browser can come back and read it from the start, and only an explicit
 * cancel stops the engine.
 *
 * That is the exact opposite of the disconnect rule one screen up, and both are right for their
 * case. A client with no chat id — opencode, curl — cannot come back for anything, so its
 * disconnect still aborts. A client that CAN come back gets to.
 *
 * One turn at a time, because there is one engine. */
typedef struct {
    pthread_mutex_t mu;
    char     chat[96];
    int      live, cancel, done, had_error;
    uint64_t seq;
    Str      content, reasoning, tools;
    int      gen, prompt, cached, pre_done;
    double   pre_s, dec_s, started, pre_tps;
} LiveTurn;
static LiveTurn g_live = { PTHREAD_MUTEX_INITIALIZER, {0}, 0,0,0,0, 0, {0},{0},{0}, 0,0,0, 0,0,0 };

static void live_begin(const char *chat){
    pthread_mutex_lock(&g_live.mu);
    snprintf(g_live.chat, sizeof g_live.chat, "%s", chat ? chat : "");
    g_live.content.n = g_live.reasoning.n = g_live.tools.n = 0;
    g_live.live = 1; g_live.cancel = 0; g_live.done = 0; g_live.had_error = 0;
    g_live.gen = g_live.prompt = g_live.cached = g_live.pre_done = 0;
    g_live.pre_tps = 0;
    g_live.pre_s = g_live.dec_s = 0; g_live.started = now();
    g_live.seq++;
    pthread_mutex_unlock(&g_live.mu);
}
static void live_add(int reasoning, const char *b, size_t n){
    pthread_mutex_lock(&g_live.mu);
    if(g_live.live) s_add(reasoning ? &g_live.reasoning : &g_live.content, b, n);
    pthread_mutex_unlock(&g_live.mu);
}
static void live_prefill(int done, int total, double tps){
    pthread_mutex_lock(&g_live.mu);
    g_live.pre_done = done; g_live.prompt = total; g_live.pre_tps = tps;
    pthread_mutex_unlock(&g_live.mu);
}
static void live_stats(int gen, double dec_s, int prompt, int cached, double pre_s){
    pthread_mutex_lock(&g_live.mu);
    g_live.gen = gen; g_live.dec_s = dec_s;
    g_live.prompt = prompt; g_live.cached = cached; g_live.pre_s = pre_s;
    pthread_mutex_unlock(&g_live.mu);
}
static void live_end(const char *tools_json){
    pthread_mutex_lock(&g_live.mu);
    g_live.tools.n = 0;
    if(tools_json) s_cat(&g_live.tools, tools_json);
    g_live.live = 0; g_live.done = 1;
    pthread_mutex_unlock(&g_live.mu);
}
static int live_cancelled(void){
    pthread_mutex_lock(&g_live.mu);
    const int c = g_live.cancel;
    pthread_mutex_unlock(&g_live.mu);
    return c;
}

/* EVERY EXIT FROM A TURN IS AN ENDING, INCLUDING THE ONES THAT GIVE UP.
 *
 * live_begin() is called before the prompt is even built, and only the two paths that reach the
 * bottom of the generation loop called live_end(). The early ones — malloc failing, a context
 * that will not fit — went straight to `done:` and left g_live.live set with no turn behind it.
 * Nothing else ever clears that flag, so /v1/live answers `live:true` forever, and the page
 * believes it:
 *
 *   attachLive() parks in its `for(;;)` poll with busy = true, so the send button is dead;
 *   a reload runs resumeProject() -> attachLive() and walks straight back in;
 *   /v1/live/cancel only sets g_live.cancel, which is read by a generation loop that is gone.
 *
 * The one advertised way out is the one that cannot work, and the chat stays frozen until the
 * server is restarted. That is the "eingefroren" report, and it is the same shape as §24: a
 * client-visible state owned by a thread that can leave without clearing it.
 *
 * Idempotent on purpose. The normal path has already published its tool calls through
 * live_end(); this must not overwrite them, so it only acts on a turn still marked live. */
static void live_abandon(void){
    pthread_mutex_lock(&g_live.mu);
    if(g_live.live){ g_live.live = 0; g_live.done = 1; g_live.had_error = 1; }
    pthread_mutex_unlock(&g_live.mu);
}

/* ---------------- global engine state ---------------- */

static Q35Model  g_M;
static Q35State  g_R;
static Tok       g_T;
static TokIds    g_ids;
static int       g_im_start = -1, g_im_end = -1, g_eos = -1;
static char     *g_ui = NULL;          /* the web page, read once at startup */
static size_t    g_ui_len = 0;
static const char *g_model_name = "qwen3.5-27b";
/* THE SWITCHABLE MODEL REGISTRY. One server, several GGUFs, one loaded at a time (they do not
 * fit in RAM together). --model NAME:PATH registers each; the web UI lists them and asks for a
 * switch, which is a re-exec (see model_switch). */
static struct { char name[64]; char path[PATH_MAX]; } g_models[8];
static int          g_nmodels = 0, g_cur_model = 0;
static const char  *g_default_model = NULL;
static char       **g_argv = NULL;
static int          g_srv_fd = -1;
static const char *g_api_key = NULL;
static int       g_n_ctx = 8192;
static int       g_verbose = 1;
static volatile sig_atomic_t g_abort = 0;

#ifndef QWEN_NO_CUDA
static Q35Cu      g_G;
static Q35CuBatch g_Z;
static Q35Mtp     g_P;
static int        g_gpu = 0, g_spec = 0;
#endif

static int q35_oom_score_adj(void){
    int v = 0; FILE *f = fopen("/proc/self/oom_score_adj", "r");
    if(f){ if(fscanf(f, "%d", &v) != 1) v = 0; fclose(f); }
    return v;
}

static Q35Cache g_C;
#define ctx_push(t) q35_cache_push(&g_C, (t))

/* WHERE A CHAT'S CHECKPOINT LIVES.
 *
 * Beside the transcripts, not inside a project — same rule as chatstore.h, and it matters more
 * here: these are gigabytes, and an agent that finds four of them in its own workspace will
 * eventually tidy one away. The name is the chat id, which is already validated as a path
 * component by cs_id_ok, so deleting a chat can delete its checkpoint without knowing which
 * project it was in. */
static char g_ck_root[PATH_MAX] = "";
static int ck_path(const char *id, char *out, size_t cap){
    if(!g_ck_root[0] || !cs_id_ok(id)) return 0;
    return snprintf(out, cap, "%s/%s.ck", g_ck_root, id) < (int)cap;
}

/* ---------------- sampling ---------------- */

typedef struct { float p; int id; } Cand;
static int cmp_desc(const void *a, const void *b){
    const float x = ((const Cand*)a)->p, y = ((const Cand*)b)->p;
    return x < y ? 1 : (x > y ? -1 : 0);
}
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double urand(void){
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11)/9007199254740992.0;
}
static int sample(float *logits, int V, float temp, float top_p, int top_k, Cand *buf){
    if(temp <= 0.0f){
        int best = 0;
        for(int i = 1; i < V; i++) if(logits[i] > logits[best]) best = i;
        return best;
    }
    float mx = -INFINITY;
    for(int i = 0; i < V; i++) if(logits[i] > mx) mx = logits[i];
    double s = 0;
    for(int i = 0; i < V; i++){
        const double e = exp((double)(logits[i]-mx)/temp);
        buf[i].p = (float)e; buf[i].id = i; s += e;
    }
    const float inv = (float)(1.0/s);
    for(int i = 0; i < V; i++) buf[i].p *= inv;
    qsort(buf, V, sizeof(Cand), cmp_desc);
    int n = V;
    if(top_k > 0 && top_k < n) n = top_k;
    if(top_p > 0 && top_p < 1.0f){
        double acc = 0; int m = 0;
        for(; m < n; m++){ acc += buf[m].p; if(acc >= top_p){ m++; break; } }
        n = m ? m : 1;
    }
    double tot = 0;
    for(int i = 0; i < n; i++) tot += buf[i].p;
    double r = urand()*tot, c = 0;
    for(int i = 0; i < n; i++){ c += buf[i].p; if(r <= c) return buf[i].id; }
    return buf[0].id;
}

/* ---------------- HTTP plumbing ---------------- */

static int sock_write(int fd, const char *b, size_t n){
    while(n){
        const ssize_t w = send(fd, b, n, MSG_NOSIGNAL);
        if(w <= 0){ if(errno == EINTR) continue; return 0; }
        b += w; n -= (size_t)w;
    }
    return 1;
}
static int sock_str(int fd, const char *z){ return sock_write(fd, z, strlen(z)); }

/* THE ENGINE THREAD MUST NEVER BLOCK ON A SOCKET.
 *
 * sock_write loops on send() until everything is out, and send() on a blocking socket waits
 * when the peer's window is closed. That is fine on a connection thread and catastrophic on the
 * engine thread: one browser that reads its SSE stream slower than the model writes it — a
 * backgrounded tab is enough — fills the socket buffer, and the generation loop stops inside
 * send() with 0% CPU, 0% GPU, no error and no log line. Every later request then queues behind
 * it. That is the stall that was reported three times and misdiagnosed twice.
 *
 * So the streaming writes are non-blocking with a short bounded wait. If the client will not
 * take the bytes in 250 ms it is not keeping up, and the right answer is to drop the frame and
 * mark the stream dead — not to stop the engine. For a resumable turn the text is in the live
 * buffer anyway and the page can fetch it whole; for everyone else a stalled reader was going
 * to lose the stream regardless. */
static int sock_write_nb(int fd, const char *b, size_t n){
    const double t0 = now();
    while(n){
        const ssize_t w = send(fd, b, n, MSG_NOSIGNAL | MSG_DONTWAIT);
        if(w > 0){ b += w; n -= (size_t)w; continue; }
        if(w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)){
            if((now() - t0) * 1000.0 > 250) return 0;      /* not keeping up */
            struct pollfd p = { fd, POLLOUT, 0 };
            poll(&p, 1, 20);
            continue;
        }
        if(w < 0 && errno == EINTR) continue;
        return 0;
    }
    return 1;
}

/* HAS THE CLIENT GONE AWAY?
 *
 * Nothing asked this before, and the consequence was the bug that produced this function. The
 * page aborts a turn (the user pressed stop, or switched chat); the browser closes the socket;
 * the server notices nothing and keeps prefilling — for FIVE AND A HALF MINUTES at 11k tokens.
 * The engine runs one request at a time, so the next prompt sits in engine_submit behind a
 * generation whose answer nobody will ever read. From the page it looks like "it said aborted
 * and then stopped responding", which is exactly what it was: an abort that aborted only the
 * half of the work that was not doing anything.
 *
 * POLLRDHUP is the half-close a browser's abort produces; POLLERR and POLLHUP cover the rest.
 * A request that is still sending its body would read as readable, so this is only called
 * AFTER the body has been consumed, when nothing more is expected from the client. */
static int client_gone(int fd){
    struct pollfd p = { fd, POLLRDHUP | POLLERR | POLLHUP, 0 };
    if(poll(&p, 1, 0) <= 0) return 0;
    return (p.revents & (POLLRDHUP | POLLERR | POLLHUP)) != 0;
}

#define CORS "Access-Control-Allow-Origin: *\r\n" \
             "Access-Control-Allow-Headers: *\r\n" \
             "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"

static void http_send(int fd, int code, const char *status, const char *ctype,
                      const char *body, size_t blen){
    char head[512];
    const int n = snprintf(head, sizeof head,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        CORS "Connection: close\r\n\r\n", code, status, ctype, blen);
    if(n > 0) sock_write(fd, head, (size_t)n);   /* snprintf returns <0 on error */
    if(blen) sock_write(fd, body, blen);
}
static void http_err(int fd, int code, const char *status, const char *msg){
    Str b = {0};
    s_cat(&b, "{\"error\":{\"message\":");
    s_json(&b, msg, strlen(msg));
    s_fmt(&b, ",\"type\":\"invalid_request_error\",\"code\":%d}}", code);
    http_send(fd, code, status, "application/json", b.p, b.n);
    s_free(&b);
}

/* ---------------- the chat template ----------------
 *
 * Transcribed from tokenizer.chat_template, literally. The role markers are CONTROL tokens:
 * tok_encode matches them against the input and never lets BPE touch them, which is the
 * whole reason they work as delimiters. Getting one of them wrong does not fail — the model
 * simply stops seeing role boundaries and answers as if there were none. */

static const char *TOOL_FORMAT =
"\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
"</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
"that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
"Reminder:\n- Function calls MUST follow the specified format: an inner <function=...>"
"</function> block must be nested within <tool_call></tool_call> XML tags\n"
"- Required parameters MUST be specified\n"
"- You may provide optional reasoning for your function call in natural language BEFORE the "
"function call, but NOT after\n"
"- If there is no function call available, answer the question like normal with your current "
"knowledge and do not tell the user about function calls\n</IMPORTANT>";

/* OpenAI lets `content` be a string OR an array of {type,text} parts. Both appear in the
 * wild; opencode sends the array form for anything multi-part. */
static void render_content(Str *out, jval *c){
    if(!c) return;
    if(c->t == J_STR){ s_cat(out, c->str); return; }
    if(c->t == J_ARR){
        for(int i = 0; i < c->len; i++){
            jval *part = c->kids[i];
            jval *tx = json_get(part, "text");
            if(tx && tx->t == J_STR) s_cat(out, tx->str);
        }
    }
}

/* messages[] + tools[] -> the exact string the model was trained on */
static void build_prompt(Str *out, jval *msgs, jval *tools, int think){
    jval *first = (msgs && msgs->len) ? msgs->kids[0] : NULL;
    jval *frole = first ? json_get(first, "role") : NULL;
    const int has_sys = frole && frole->t == J_STR && !strcmp(frole->str, "system");

    if(tools && tools->t == J_ARR && tools->len > 0){
        s_cat(out, "<|im_start|>system\n");
        s_cat(out, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for(int i = 0; i < tools->len; i++){ s_cat(out, "\n"); j_emit(out, tools->kids[i]); }
        s_cat(out, "\n</tools>");
        s_cat(out, TOOL_FORMAT);
        if(has_sys){
            Str c = {0}; render_content(&c, json_get(first, "content"));
            if(c.n){ s_cat(out, "\n\n"); s_add(out, c.p, c.n); }
            s_free(&c);
        }
        s_cat(out, "<|im_end|>\n");
    } else if(has_sys){
        Str c = {0}; render_content(&c, json_get(first, "content"));
        if(c.n){ s_cat(out, "<|im_start|>system\n"); s_add(out, c.p, c.n); s_cat(out, "<|im_end|>\n"); }
        s_free(&c);
    }

    int prev_tool = 0;
    for(int i = has_sys ? 1 : 0; i < (msgs ? msgs->len : 0); i++){
        jval *m = msgs->kids[i];
        jval *r = json_get(m, "role");
        const char *role = (r && r->t == J_STR) ? r->str : "user";
        Str c = {0}; render_content(&c, json_get(m, "content"));

        if(!strcmp(role, "tool")){
            /* consecutive tool results are grouped into ONE user turn, per the template */
            if(!prev_tool) s_cat(out, "<|im_start|>user");
            s_cat(out, "\n<tool_response>\n");
            s_add(out, c.p, c.n);
            s_cat(out, "\n</tool_response>");
            const int last = (i == msgs->len - 1);
            int next_tool = 0;
            if(!last){
                jval *nr = json_get(msgs->kids[i+1], "role");
                next_tool = nr && nr->t == J_STR && !strcmp(nr->str, "tool");
            }
            if(last || !next_tool) s_cat(out, "<|im_end|>\n");
            prev_tool = 1;
        } else if(!strcmp(role, "assistant")){
            prev_tool = 0;
            s_cat(out, "<|im_start|>assistant\n");
            /* THE GLUE IS THE MODEL'S, NOT THE TEMPLATE'S — AND THAT COST 25 MINUTES A TURN.
             *
             * The template reads `<think>\n{reasoning}\n</think>\n\n{content}`, so this wrote
             * those newlines itself. But `reasoning_content` is captured between the two
             * CONTROL tokens and `content` after them, so the newline before `</think>` and the
             * blank line after it were generated by the model and are ALREADY inside the
             * captured strings. Writing them again produced `\n\n</think>\n\n\n\n` where the
             * engine had run `\n</think>\n\n`.
             *
             * Ten tokens of difference, at the very end of the turn. And `live` in
             * q35_cache_begin is all-or-nothing — one token off discards the whole prefix — so
             * every single agent step fell back to a checkpoint, which §23.2 caps at 8192.
             * Measured on a real session: `19504 prompt (8192 cached) in 1497.43s`. Twenty-five
             * minutes of re-prefill, per step, for two newlines.
             *
             * So: emit the glue only where it is not already present. For a turn this engine
             * generated, every branch below is a no-op and the render is byte-identical to what
             * ran — which is the whole point. For a client that sends a trimmed
             * `reasoning_content` (opencode does), the canonical form is still produced. */
            jval *rc = json_get(m, "reasoning_content");
            const char *rs = (rc && rc->t == J_STR && rc->str) ? rc->str : "";
            const size_t rn = strlen(rs);
            s_cat(out, "<think>\n");
            if(rn){
                s_add(out, rs, rn);
                if(rs[rn-1] != '\n') s_cat(out, "\n");
                s_cat(out, "</think>");
                if(!(c.n && c.p[0] == '\n')) s_cat(out, "\n\n");
            } else {
                s_cat(out, "\n</think>\n\n");
            }
            s_add(out, c.p, c.n);
            jval *tc = json_get(m, "tool_calls");
            if(tc && tc->t == J_ARR){
                for(int k = 0; k < tc->len; k++){
                    jval *f = json_get(tc->kids[k], "function");
                    if(!f) f = tc->kids[k];
                    jval *nm = json_get(f, "name"), *ar = json_get(f, "arguments");
                    if(!nm || nm->t != J_STR) continue;
                    /* same rule as the think block: `<tool_call>` is a control token, so
                     * whatever the model wrote before it is already the tail of `content`. */
                    s_cat(out, (k == 0 && c.n && c.p[c.n-1] != '\n') ? "\n\n<tool_call>\n<function=" :
                               (k == 0 ? "<tool_call>\n<function=" : "\n<tool_call>\n<function="));
                    s_cat(out, nm->str);
                    s_cat(out, ">\n");
                    /* arguments arrive as a JSON STRING; re-parse so each key becomes its
                     * own <parameter=...> block the way the template writes them */
                    if(ar && ar->t == J_STR && ar->str[0]){
                        char *arena = NULL;
                        jval *a = json_parse(ar->str, &arena);
                        if(a && a->t == J_OBJ){
                            for(int q = 0; q < a->len; q++){
                                s_fmt(out, "<parameter=%s>\n", a->keys[q]);
                                if(a->kids[q]->t == J_STR) s_cat(out, a->kids[q]->str);
                                else j_emit(out, a->kids[q]);
                                s_cat(out, "\n</parameter>\n");
                            }
                        }
                        free(arena);
                    }
                    s_cat(out, "</function>\n</tool_call>");
                }
            }
            s_cat(out, "<|im_end|>\n");
        } else {
            prev_tool = 0;
            s_cat(out, "<|im_start|>user\n");
            s_add(out, c.p, c.n);
            s_cat(out, "<|im_end|>\n");
        }
        s_free(&c);
    }
    s_cat(out, "<|im_start|>assistant\n");
    s_cat(out, think ? "<think>\n" : "<think>\n\n</think>\n\n");
}

/* ---------------- generation ----------------
 *
 * <think>, </think>, <tool_call> and </tool_call> are CONTROL tokens with fixed ids, so the
 * state machine below switches on ids and never string-matches the output. That matters: the
 * model can and does write the literal text "</think>" inside a code block, and a text-based
 * detector would end the reasoning block there. */
static int T_THINK_O, T_THINK_C, T_TOOL_O, T_TOOL_C;

static size_t utf8_complete(const char *b, size_t n){
    size_t i = 0, last = 0;
    while(i < n){
        const unsigned char c = (unsigned char)b[i];
        const int need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2
                       : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if(i + need > n) break;
        i += (size_t)need; last = i;
    }
    return last;
}

typedef struct {
    char *name;
    Str   args;          /* already-built JSON object */
} ToolCall;

/* <function=NAME>\n<parameter=K>\nV\n</parameter>...\n</function> -> name + JSON arguments */
static int parse_tool_call(const char *s, size_t n, ToolCall *tc){
    memset(tc, 0, sizeof *tc);
    const char *end = s + n;
    const char *f = memmem(s, n, "<function=", 10);
    if(!f) return 0;
    f += 10;
    const char *gt = memchr(f, '>', (size_t)(end - f));
    if(!gt) return 0;
    tc->name = strndup(f, (size_t)(gt - f));
    s_cat(&tc->args, "{");
    const char *p = gt + 1;
    int first = 1;
    for(;;){
        const char *k = memmem(p, (size_t)(end - p), "<parameter=", 11);
        if(!k) break;
        k += 11;
        const char *kg = memchr(k, '>', (size_t)(end - k));
        if(!kg) break;
        const char *v = kg + 1;
        if(v < end && *v == '\n') v++;
        const char *ve = memmem(v, (size_t)(end - v), "</parameter>", 12);
        if(!ve) break;
        size_t vlen = (size_t)(ve - v);
        while(vlen && (v[vlen-1] == '\n' || v[vlen-1] == '\r')) vlen--;

        if(!first) s_cat(&tc->args, ",");
        first = 0;
        s_json(&tc->args, k, (size_t)(kg - k));
        s_cat(&tc->args, ":");
        /* the template writes non-string arguments through tojson, so anything that parses
         * as JSON goes back verbatim and everything else is a string. Quoting a number here
         * makes strict tool schemas reject the call. */
        /* json.h is lenient — handed "Berlin" it returns a number 0, and the tool then gets
         * {"city":0} instead of {"city":"Berlin"}. So gate on the FIRST CHARACTER, which is
         * what actually distinguishes a JSON scalar from prose, and only then parse. */
        const char f0 = vlen ? v[0] : 0;
        const int looks_json = vlen && (f0 == '{' || f0 == '[' || f0 == '-' ||
                                        (f0 >= '0' && f0 <= '9') ||
                                        (vlen == 4 && !memcmp(v, "true", 4)) ||
                                        (vlen == 5 && !memcmp(v, "false", 5)) ||
                                        (vlen == 4 && !memcmp(v, "null", 4)));
        char *tmp = looks_json ? strndup(v, vlen) : NULL, *arena = NULL;
        jval *jv = tmp ? json_parse(tmp, &arena) : NULL;
        if(jv && (jv->t == J_NUM || jv->t == J_BOOL || jv->t == J_OBJ || jv->t == J_ARR || jv->t == J_NULL))
            j_emit(&tc->args, jv);
        else
            s_json(&tc->args, v, vlen);
        free(tmp); free(arena);
        p = ve + 12;
    }
    s_cat(&tc->args, "}");
    return 1;
}

typedef struct {
    int   fd, stream, sent_role, dead, resumable;
    char  id[48];
    long  created;
    Str   content, reasoning, toolbuf;
    Str   pend_c, pend_r;          /* bytes waiting for a UTF-8 boundary */
    int   in_think, in_tool;
    ToolCall tc[8];
    int   n_tc;
} Gen;

static void sse_delta(Gen *g, const char *field, const char *b, size_t n){
    Str f = {0};
    s_fmt(&f, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
              "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{", g->id, g->created, g_model_name);
    if(!g->sent_role){ s_cat(&f, "\"role\":\"assistant\","); g->sent_role = 1; }
    s_fmt(&f, "\"%s\":", field);
    s_json(&f, b, n);
    s_cat(&f, "},\"finish_reason\":null}]}\n\n");
    if(!sock_write_nb(g->fd, f.p, f.n)) g->dead = 1;
    s_free(&f);
}

/* An extra key on an otherwise ordinary chunk. Clients that do not know it ignore it — which
 * is every client but this page — so the live readout costs no second request and no protocol
 * of its own. */
/* The prefill has no tokens to report, so it reports its own progress on the same stream. Also
 * the thing that keeps the response from being silent for minutes. */
static Gen *g_pf_gen = NULL;
static void sse_prefill(int done, int total, int cached, double secs){
    Gen *g = g_pf_gen;
    if(!g || !g->stream) return;
    Str f = {0};
    s_fmt(&f, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
              "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}],"
              "\"qwen\":{\"phase\":\"prefill\",\"pre_done\":%d,\"prompt\":%d,\"cached\":%d,"
              "\"pre_tok_s\":%.2f,\"elapsed\":%.1f,\"gen\":0,\"tok_s\":0,\"gbps\":%.2f}}\n\n",
          g->id, g->created, g_model_name, done, total, cached,
          secs > 0 ? (done - cached) / secs : 0.0, secs,
          secs > 0 ? g_bytes_dram * ((done - cached) / secs) / (double)g_prefill_sb / 1e9 : 0.0);
    if(!sock_write_nb(g->fd, f.p, f.n)) g->dead = 1;
    s_free(&f);
}

static void sse_stats(Gen *g, int gen, double dec_s, int prompt, int cached, double pre_s){
    if(!g->stream || dec_s <= 0) return;
    const double tps = gen / dec_s;
    Str f = {0};
    s_fmt(&f, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
              "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}],"
              "\"qwen\":{\"gen\":%d,\"decode_s\":%.3f,\"tok_s\":%.3f,\"gbps\":%.2f,"
              "\"gbps_vram\":%.2f,\"prompt\":%d,\"cached\":%d,\"prefill_s\":%.3f,"
              "\"phase\":\"decode\"}}\n\n",
          g->id, g->created, g_model_name, gen, dec_s, tps,
          g_bytes_dram * tps / 1e9, g_bytes_vram * tps / 1e9, prompt, cached, pre_s);
    if(!sock_write_nb(g->fd, f.p, f.n)) g->dead = 1;
    s_free(&f);
}

static void emit(Gen *g, Str *acc, Str *pend, const char *field, const char *b, size_t n){
    s_add(acc, b, n);
    if(g->resumable) live_add(field[0] == 'r', b, n);
    if(!g->stream) return;
    s_add(pend, b, n);
    const size_t cut = utf8_complete(pend->p, pend->n);
    if(!cut) return;
    sse_delta(g, field, pend->p, cut);
    memmove(pend->p, pend->p + cut, pend->n - cut);
    pend->n -= cut;
    pend->p[pend->n] = 0;
}

static void forward1(int tok, int pos, float *logits){
#ifndef QWEN_NO_CUDA
    if(g_gpu){ q35_forward_cu(&g_G, &g_R, tok, pos, logits); return; }
#endif
    q35_forward(&g_R, tok, pos, logits);
}

/* Grow the KV store to hold `need` tokens, or return 0. kvt_grow stops before the memory
 * floor and q35_oom terminates rather than allocating past it — the context is never
 * silently clipped. */
static int ensure_ctx(int need){
    if(need + 8 <= g_R.n_ctx) return 1;
    if(!kvt_grow(&g_R.kv, need + 512)) return 0;
    g_R.n_ctx = g_R.kv.n_ctx;
    g_R.att = (float*)realloc(g_R.att, sizeof(float)*(size_t)g_R.n_ctx*g_M.c.n_head);
    return g_R.att != NULL;
}

static int prefill(int fd, int resumable, const int *req, int n, float *logits,
                   double *t_out, int *reused){
    #define PF_STOP() (resumable ? live_cancelled() : client_gone(fd))
    const int start = q35_cache_begin(&g_C, req, n, reused);
    /* ensure_ctx grows the KV store, and for a 20k prompt that is seconds of allocation and
     * spilling before a single token is computed. Checked on both sides of it, which is as
     * close as this gets: the growth ITSELF is not interruptible.
     *
     * Measured, same request six times: with no growth needed the prefill stops at 6.0 s —
     * the exact second the client left — and the next prompt is served 4.0 s later. When the
     * store had to grow by 9k tokens it stopped at 19.1 s instead, and the extra 13 s is that
     * one uninterruptible call. Worth naming rather than rounding away: the window exists, it
     * is bounded by how much the store has to grow, and it shrinks to nothing once the context
     * has reached its working size. */
    if(PF_STOP()) return -1;
    if(!ensure_ctx(n + 4)) return 0;
    if(PF_STOP()) return -1;
    const double t0 = now();
    int i = start;
#ifndef QWEN_NO_CUDA
    if(g_spec){
        const int SB = g_Z.S_max;
        float *xall = (float*)malloc(sizeof(float)*(size_t)SB*g_M.c.d_model);
        float *hh   = (float*)malloc(sizeof(float)*g_M.c.d_model);
        /* checked once per batch of 8, so a five-minute prefill for a client that left ends
         * within a fraction of a second rather than at the end */
        for(; i + SB <= n - 1 && !g_abort; i += SB){
            if(PF_STOP()) break;
            if((i & 127) < SB){
                const double el = now() - t0;
                if(resumable) live_prefill(i, n, (i - start) / (el + 1e-9));
                sse_prefill(i, n, start, el);
            }
            q35_forward_cu_batch_ex(&g_Z, (int*)req + i, SB, i, NULL, 0, xall);
            for(int s = 0; s < SB; s++){
                q35_rmsnorm(hh, xall + (size_t)s*g_M.c.d_model, g_M.output_norm,
                            g_M.c.d_model, g_M.c.eps);
                q35_mtp_prefill_kv(&g_P, hh, req[i+s], i+s);
            }
            for(int s = 0; s < SB; s++) ctx_push(req[i+s]);
            q35_cache_mark(&g_C, i + SB);
        }
        free(xall); free(hh);
    }
#endif
    int aborted = 0;
    for(; i < n && !g_abort; i++){
        if((i & 63) == 0 && PF_STOP()){ aborted = 1; break; }
        forward1(req[i], i, (i == n-1) ? logits : NULL);
#ifndef QWEN_NO_CUDA
        if(g_spec){ float h[8192]; q35_hidden(&g_R, h); q35_mtp_prefill_kv(&g_P, h, req[i], i); }
#endif
        ctx_push(req[i]);
        q35_cache_mark(&g_C, i + 1);
    }
    *t_out = now() - t0;
    /* AN ABORTED PREFILL MUST NOT LOOK LIKE A FINISHED ONE.
     *
     * The first version broke out of the loop and returned 1 all the same — but `logits` is only
     * written at i == n-1, so the caller went straight into sample() on a malloc'd buffer that
     * had never been touched. It did not crash; it drew a token from uninitialised memory and
     * fed it to the speculative decoder, which then ran a batched forward with a snapshot
     * against a state that meant nothing. The engine wedged afterwards — 0% CPU, GPU idle, every
     * thread in futex, no error anywhere — and each later request queued behind it forever.
     *
     * Reading uninitialised memory is the kind of defect nothing complains about, which is why
     * this returns a distinct code and the caller stops rather than carries on. */
    #undef PF_STOP
    return (aborted || i < n) ? -1 : 1;
}

/* One generated token through the state machine. Control tokens switch mode and are never
 * emitted as text; everything else goes to reasoning, to content, or into the tool buffer. */
static void handle_token(Gen *g, int tok, char *piece, size_t pcap){
    if(tok == T_THINK_C){ g->in_think = 0; return; }
    if(tok == T_THINK_O){ g->in_think = 1; return; }
    if(tok == T_TOOL_O){ g->in_tool = 1; g->toolbuf.n = 0; if(g->toolbuf.p) g->toolbuf.p[0]=0; return; }
    if(tok == T_TOOL_C){
        g->in_tool = 0;
        if(g->n_tc < (int)(sizeof g->tc/sizeof *g->tc) && g->toolbuf.n)
            if(parse_tool_call(g->toolbuf.p, g->toolbuf.n, &g->tc[g->n_tc])) g->n_tc++;
        return;
    }
    const int np = tok_decode(&g_T, &tok, 1, piece, (int)pcap);
    if(np <= 0) return;
    if(g->in_tool)       s_add(&g->toolbuf, piece, (size_t)np);
    else if(g->in_think) emit(g, &g->reasoning, &g->pend_r, "reasoning_content", piece, (size_t)np);
    else                 emit(g, &g->content,   &g->pend_c, "content",           piece, (size_t)np);
}

static void handle_chat(int fd, const char *body, size_t blen){
    (void)blen;
    char *arena = NULL;
    jval *root = json_parse(body, &arena);
    if(!root || root->t != J_OBJ){ free(arena); http_err(fd, 400, "Bad Request", "malformed JSON body"); return; }

    jval *msgs = json_get(root, "messages");
    if(!msgs || msgs->t != J_ARR || msgs->len == 0){
        free(arena); http_err(fd, 400, "Bad Request", "messages[] is required"); return;
    }
    jval *jt;
    const int stream   = (jt = json_get(root, "stream"))      ? (jt->t == J_BOOL ? jt->boolean : 0) : 0;
    float temp         = (jt = json_get(root, "temperature")) && jt->t == J_NUM ? (float)jt->num : 0.7f;
    float top_p        = (jt = json_get(root, "top_p"))       && jt->t == J_NUM ? (float)jt->num : 0.95f;
    int   top_k        = (jt = json_get(root, "top_k"))       && jt->t == J_NUM ? (int)jt->num   : 20;
    int   max_tokens   = (jt = json_get(root, "max_tokens"))  && jt->t == J_NUM ? (int)jt->num   : 2048;
    if((jt = json_get(root, "max_completion_tokens")) && jt->t == J_NUM) max_tokens = (int)jt->num;
    if((jt = json_get(root, "seed")) && jt->t == J_NUM) g_rng = (uint64_t)jt->num | 1ull;
    int think = 1;
    if((jt = json_get(root, "reasoning")) && jt->t == J_BOOL) think = jt->boolean;
    if((jt = json_get(root, "enable_thinking")) && jt->t == J_BOOL) think = jt->boolean;
    if(max_tokens <= 0 || max_tokens > 32768) max_tokens = 2048;
    jval *tools = json_get(root, "tools");

    /* WHICH CONVERSATION THIS IS, for the prefix cache.
     *
     * An extra field, ignored by every other client, and that is the point: without it two
     * conversations in the same project look like one drifting sequence to the checkpoint
     * eviction, which then drops one chat's foothold to "widen a gap" between points that were
     * never on the same line. opencode and curl send nothing and share bucket 0, exactly as
     * before. */
    const char *chat_id = (jt = json_get(root, "chat")) && jt->t == J_STR ? jt->str : NULL;
    q35_cache_use_chat(&g_C, chat_id);
    if(chat_id && *chat_id) live_begin(chat_id);

    Str prompt = {0};
    build_prompt(&prompt, msgs, tools, think);

    const int V = g_M.c.vocab;
    int *req = (int*)malloc(sizeof(int)*(size_t)(prompt.n + 64));
    const int np = tok_encode(&g_T, prompt.p, (int)prompt.n, req, (int)prompt.n + 60);
    if(np <= 0){ free(req); s_free(&prompt); free(arena);
                 http_err(fd, 400, "Bad Request", "prompt tokenized to nothing"); return; }

    float *logits = (float*)malloc(sizeof(float)*V);
    float *lg2    = (float*)malloc(sizeof(float)*2*(size_t)V);
    Cand  *cbuf   = (Cand*)malloc(sizeof(Cand)*V);
    char   piece[512];
    if(!logits || !lg2 || !cbuf){ http_err(fd, 500, "Internal Server Error", "out of memory"); goto done; }

    Gen g; memset(&g, 0, sizeof g);
    g.fd = fd; g.stream = stream; g.in_think = think; g.created = (long)time(NULL);
    g.resumable = chat_id && *chat_id;
    snprintf(g.id, sizeof g.id, "chatcmpl-%08lx%04x", (unsigned long)time(NULL), rand() & 0xffff);

    /* THE HEADERS GO OUT BEFORE THE PREFILL, NOT AFTER.
     *
     * They used to be written here, after prefill returned — which for a 12k-token prompt is
     * SEVEN MINUTES during which the client received nothing at all. Not a partial response:
     * not even a status line. Every HTTP client eventually gives up on that, so the page said
     * "abgebrochen" at a point that moved with the prompt length, which is what "it broke at
     * almost the same place again" means. Measured in the log right before this fix:
     *
     *     11738 prompt (8192 cached) in 407.85s
     *
     * So: status line first, then a keep-alive comment and a progress frame every second while
     * the prefill runs. The response is alive from the first millisecond and the page can show
     * how far along it is instead of guessing.
     *
     * The cost of moving it: a 507 can no longer be a status code once the headers are out, so
     * a failure after this point is delivered as an SSE error frame. */
    if(stream){
        { static const char H[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                  "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n" CORS
                                  "Connection: close\r\n\r\n: prefill\n\n";
          sock_write_nb(fd, H, sizeof H - 1); }
    }

    /* THE CHECKPOINT ON DISK, BEFORE ANYTHING IS PREFILLED.
     *
     * Only when it reaches FURTHER than the live state already does: restoring costs a few
     * seconds of disk and overwrites the engine, and doing that to land where we already stood
     * would be pure loss. After a successful load the live sequence IS the file's prefix, so
     * q35_cache_begin's `live` path picks it up with no special case. */
    if(chat_id && *chat_id && g_ck_root[0]){
        char cp[PATH_MAX];
        if(ck_path(chat_id, cp, sizeof cp)){
            int livem = 0;
            while(livem < np && livem < g_C.ctx_n && req[livem] == g_C.ctx[livem]) livem++;
            if(livem != g_C.ctx_n) livem = 0;
            if(q35_ck_peek(&g_C, cp) > livem){
                double cs = 0;
                const int got = q35_ck_load(&g_C, cp, req, np, &cs);
                if(g_verbose)
                    fprintf(stderr, got ? "  checkpoint: %d tokens restored from disk in %.2fs\n"
                                        : "  checkpoint: on disk but not this conversation (%d)\n",
                            got ? got : 0, cs);
            }
        }
    }

    double t_pre = 0; int reused = 0;
    g_pf_gen = &g;
    const int pf = prefill(fd, chat_id && *chat_id, req, np, logits, &t_pre, &reused);
    g_pf_gen = NULL;
    if(pf < 0){
        if(g_verbose) fprintf(stderr, "  %d of %d prompt tokens in %.2fs, then it was dropped\n",
                              g_C.ctx_n, np, t_pre);
        if(chat_id && *chat_id) live_end("[]");
        goto done;
    }
    if(!pf){
        if(stream){
            sock_str(fd, "data: {\"error\":{\"message\":\"context does not fit in memory\"}}\n\n"
                         "data: [DONE]\n\n");
        } else http_err(fd, 507, "Insufficient Storage", "context does not fit in memory");
        goto done;
    }

    const double t_dec0 = now();
    int pos = np, ngen = 0, hit_eos = 0;
    long long rounds = 0, accepts = 0;
    int tok = sample(logits, V, temp, top_p, top_k, cbuf);

    int gone = 0;
    while(ngen < max_tokens && !g_abort && !gone){
        if(tok == g_eos || tok == g_ids.eos){ hit_eos = 1; break; }
        handle_token(&g, tok, piece, sizeof piece);
        ngen++; ctx_push(tok);
        if(ngen >= max_tokens) break;
        if(!ensure_ctx(pos + 8)) break;
        /* g.dead is set by the SSE writer the first time a send() fails; client_gone catches
         * the non-streaming case, where nothing is written until the end. */
        if((ngen & 7) == 0){
            sse_stats(&g, ngen, now() - t_dec0, np, reused, t_pre);
            if(g.resumable) live_stats(ngen, now() - t_dec0, np, reused, t_pre);
        }
        if(g.resumable){
            /* the page may be reloading; keep going and let it come back for the text */
            if(live_cancelled()){ gone = 1; break; }
        } else if(g.dead || ((ngen & 15) == 0 && client_gone(fd))){ gone = 1; break; }

#ifndef QWEN_NO_CUDA
        /* One speculative round: the MTP head drafts, and the trunk verifies BOTH positions
         * in a single pass over the weights. The true token is always sampled from the
         * trunk's own logits, so accepting only on an exact match leaves the output
         * distribution untouched. Below 25% acceptance a round costs more than it saves, so
         * the loop stops speculating. */
        const int spec_ok = g_spec && (rounds < 12 || (double)accepts/(double)rounds >= 0.25);
        if(spec_ok){
            float h[8192];
            q35_hidden(&g_R, h);
            q35_mtp_hidden(&g_P, h, tok, pos);
            q35cu_h2d(g_G.xn, g_P.xn, sizeof(float)*g_M.c.d_model);
            q35cu_gemv(g_G.logits, g_G.xn, g_G.output, g_M.output->type, g_M.c.d_model, V);
            q35cu_sync();
            q35cu_d2h(lg2, g_G.logits, sizeof(float)*V);
            const int draft = q35_argmax(lg2, V);

            const int pair[2] = { tok, draft };
            g_Z.snap_after = 0;
            q35_forward_cu_batch(&g_Z, pair, 2, pos, lg2, 1);
            g_Z.snap_after = -1;
            rounds++;

            const int truth = sample(lg2, V, temp, top_p, top_k, cbuf);
            if(truth == draft){
                accepts++;
                pos += 2;
                if(draft == g_eos || draft == g_ids.eos){ hit_eos = 1; break; }
                handle_token(&g, draft, piece, sizeof piece);
                ngen++; ctx_push(draft);
                tok = sample(lg2 + V, V, temp, top_p, top_k, cbuf);
            } else {
                q35cu_spec_restore(&g_Z);
                pos += 1;
                tok = truth;
            }
            continue;
        }
#endif
        forward1(tok, pos++, logits);
        tok = sample(logits, V, temp, top_p, top_k, cbuf);
    }
    const double t_dec = now() - t_dec0;

    /* flush whatever was still short of a UTF-8 boundary */
    if(stream){
        if(g.pend_r.n) sse_delta(&g, "reasoning_content", g.pend_r.p, g.pend_r.n);
        if(g.pend_c.n) sse_delta(&g, "content", g.pend_c.p, g.pend_c.n);
    }
    const char *finish = gone ? "abort"
                       : g.n_tc ? "tool_calls" : (hit_eos ? "stop" : "length");

    Str out = {0};
    if(stream){
        if(g.n_tc){
            s_fmt(&out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
                        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                        g.id, g.created, g_model_name);
            for(int i = 0; i < g.n_tc; i++){
                if(i) s_cat(&out, ",");
                s_fmt(&out, "{\"index\":%d,\"id\":\"call_%s_%d\",\"type\":\"function\",\"function\":{\"name\":", i, g.id+9, i);
                s_json(&out, g.tc[i].name, strlen(g.tc[i].name));
                s_cat(&out, ",\"arguments\":");
                s_json(&out, g.tc[i].args.p, g.tc[i].args.n);
                s_cat(&out, "}}");
            }
            s_cat(&out, "]},\"finish_reason\":null}]}\n\n");
        }
        s_fmt(&out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
                    "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
                    "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
                    "\"prompt_tokens_cached\":%d,\"prefill_s\":%.3f,\"decode_s\":%.3f}}\n\n"
                    "data: [DONE]\n\n",
                    g.id, g.created, g_model_name, finish, np, ngen, np + ngen,
                    reused, t_pre, t_dec);
        sock_write_nb(fd, out.p, out.n);
    } else {
        s_fmt(&out, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\","
                    "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":",
                    g.id, g.created, g_model_name);
        s_json(&out, g.content.p ? g.content.p : "", g.content.n);
        if(g.reasoning.n){ s_cat(&out, ",\"reasoning_content\":"); s_json(&out, g.reasoning.p, g.reasoning.n); }
        if(g.n_tc){
            s_cat(&out, ",\"tool_calls\":[");
            for(int i = 0; i < g.n_tc; i++){
                if(i) s_cat(&out, ",");
                s_fmt(&out, "{\"id\":\"call_%s_%d\",\"type\":\"function\",\"function\":{\"name\":", g.id+9, i);
                s_json(&out, g.tc[i].name, strlen(g.tc[i].name));
                s_cat(&out, ",\"arguments\":");
                s_json(&out, g.tc[i].args.p, g.tc[i].args.n);
                s_cat(&out, "}}");
            }
            s_cat(&out, "]");
        }
        s_fmt(&out, "},\"finish_reason\":\"%s\"}],"
                    "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
                    "\"prompt_tokens_cached\":%d,\"prefill_s\":%.3f,\"decode_s\":%.3f}}",
                    finish, np, ngen, np + ngen, reused, t_pre, t_dec);
        http_send(fd, 200, "OK", "application/json", out.p, out.n);
    }
    s_free(&out);

    if(g.resumable){
        Str tj = {0};
        s_cat(&tj, "[");
        for(int i = 0; i < g.n_tc; i++){
            if(i) s_cat(&tj, ",");
            s_fmt(&tj, "{\"id\":\"call_%s_%d\",\"type\":\"function\",\"function\":{\"name\":", g.id+9, i);
            s_json(&tj, g.tc[i].name, strlen(g.tc[i].name));
            s_cat(&tj, ",\"arguments\":");
            s_json(&tj, g.tc[i].args.p, g.tc[i].args.n);
            s_cat(&tj, "}}");
        }
        s_cat(&tj, "]");
        live_stats(ngen, t_dec, np, reused, t_pre);
        live_end(tj.p);
        s_free(&tj);
    }

    /* THE TURN, MADE DURABLE. Written after the answer is out, so it never delays a reply, and
     * only for a turn that actually finished — a checkpoint of an aborted generation would
     * resume into the middle of a sentence nobody chose. */
    if(chat_id && *chat_id && g_ck_root[0] && !gone && !g_abort){
        char cp[PATH_MAX];
        if(ck_path(chat_id, cp, sizeof cp)){
            double cs = 0;
            const int wrote = q35_ck_save(&g_C, cp, &cs);
            if(g_verbose){
                if(wrote) fprintf(stderr, "  checkpoint: %d tokens to disk in %.2fs (%.2f GiB)\n",
                                  wrote, cs, (double)ck_bytes(&g_C, wrote)/1073741824.0);
                else fprintf(stderr, "  checkpoint: NOT written — %d tokens is past the %d the "
                                     "per-chat budget holds (QWEN_CHAT_CKPT_GB)\n",
                             g_C.ctx_n, q35_ck_max_pos(&g_C));
            }
        }
    }

    if(g_verbose){
        /* On an abort the prompt was NOT prefilled — only as far as g_C.ctx_n got. Reporting
         * `np` there printed "20513 prompt in 19.34s = 1060.7 tok/s" for 129 tokens of actual
         * work: a rate three orders out, in a log this project reads to make decisions. Same
         * class as every other number here that described something other than what happened. */
        const int did = g_C.ctx_n < np ? g_C.ctx_n : np;
        fprintf(stderr, "  %d prompt (%d cached%s) in %.2fs = %.1f tok/s | %d gen in %.2fs = %.2f tok/s"
                        " | accept %.0f%% | ctx %d | %s\n",
                did, reused, did < np ? ", ABORTED partway" : "",
                t_pre, reused < did ? (did-reused)/(t_pre > 0 ? t_pre : 1) : 0.0,
                ngen, t_dec, t_dec > 0 ? ngen/t_dec : 0.0,
                rounds ? 100.0*(double)accepts/(double)rounds : 0.0, g_C.ctx_n, finish);
    }

    for(int i = 0; i < g.n_tc; i++){ free(g.tc[i].name); s_free(&g.tc[i].args); }
    s_free(&g.content); s_free(&g.reasoning); s_free(&g.toolbuf);
    s_free(&g.pend_c); s_free(&g.pend_r);
done:
    if(chat_id && *chat_id) live_abandon();   /* no turn may outlive the flag that says it runs */
    free(logits); free(lg2); free(cbuf); free(req);
    s_free(&prompt); free(arena);
}

/* ---------------- routing ----------------
 *
 * A connection gets its own thread so the page and /health answer while a generation runs —
 * without that, loading the UI mid-generation simply times out.
 *
 * But the ENGINE never runs on a connection thread. It runs on ONE dedicated thread, always
 * the same one, and connection threads hand it the request.
 *
 * A mutex alone was not enough, and the failure was not the one a mutex prevents. Only one
 * request was ever inside the engine, but each arrived on a DIFFERENT pthread, and the engine
 * enters `omp parallel` regions a few thousand times per request. libgomp builds a fresh
 * thread team per encountering thread; churning teams across short-lived pthreads segfaulted
 * inside libc on an OpenMP worker after ~1300 tokens of prefill. The same prompt through the
 * single-threaded CLI was fine, which is what localized it.
 *
 * One engine thread also means the OpenMP team is built once and reused for the life of the
 * process, which is what the active-wait tuning assumes anyway. */
typedef struct { int fd; char *body; size_t len; } Job;

static pthread_mutex_t g_qmu  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_qcv  = PTHREAD_COND_INITIALIZER;
static Job             g_job;
static int             g_job_pending = 0;

static void *engine_thread(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&g_qmu);
        while(!g_job_pending) pthread_cond_wait(&g_qcv, &g_qmu);
        Job j = g_job;
        g_job_pending = 0;
        pthread_mutex_unlock(&g_qmu);

        handle_chat(j.fd, j.body, j.len);
        close(j.fd);
        free(j.body);
    }
    return NULL;
}

/* Hands the request over and returns; the engine thread owns the fd from here. */
static void engine_submit(int fd, const char *body, size_t len){
    char *copy = (char*)malloc(len + 1);
    if(!copy){ http_err(fd, 503, "Service Unavailable", "out of memory"); close(fd); return; }
    memcpy(copy, body, len); copy[len] = 0;
    pthread_mutex_lock(&g_qmu);
    while(g_job_pending){                    /* one at a time: one engine, one state */
        pthread_mutex_unlock(&g_qmu);
        usleep(2000);
        pthread_mutex_lock(&g_qmu);
    }
    g_job.fd = fd; g_job.body = copy; g_job.len = len;
    g_job_pending = 1;
    pthread_cond_signal(&g_qcv);
    pthread_mutex_unlock(&g_qmu);
}

static void handle_models(int fd){
    Str b = {0};
    s_cat(&b, "{\"object\":\"list\",\"data\":[");
    for(int i = 0; i < g_nmodels; i++){
        if(i) s_cat(&b, ",");
        s_cat(&b, "{\"id\":"); s_json(&b, g_models[i].name, strlen(g_models[i].name));
        s_fmt(&b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"qwenengine\",\"current\":%s}",
              (long)time(NULL), i == g_cur_model ? "true" : "false");
    }
    s_cat(&b, "]}");
    http_send(fd, 200, "OK", "application/json", b.p, b.n);
    s_free(&b);
}

/* SWITCHING MODELS IS RE-EXEC, NOT TEARDOWN. The engine's global state — model, CUDA context,
 * ~18 GiB of resident experts — is intricate to unwind and cheap to recreate, and the startup
 * path that builds it is the one that has been debugged. So a switch re-execs this same binary
 * with the chosen model named in the environment and the listening socket handed across intact,
 * so the port never drops. exec frees the old address space (its RAM and, lazily, its VRAM);
 * the new process retries the upload while the driver catches up. */
static void model_switch(int idx){
    setenv("QWEN_SELECT_MODEL", g_models[idx].name, 1);
    char fds[16]; snprintf(fds, sizeof fds, "%d", g_srv_fd);
    setenv("QWEN_LISTEN_FD", fds, 1);
    if(g_srv_fd >= 0) fcntl(g_srv_fd, F_SETFD, 0);   /* survive exec; the new process re-arms CLOEXEC */
    web_stop();                                      /* no orphaned Firefox across the switch */
    execv("/proc/self/exe", g_argv);
    _exit(1);                                        /* only reached if exec failed */
}

static void handle_health(int fd){
    Str b = {0};
    size_t vf = 0, vt = 0;
#ifndef QWEN_NO_CUDA
    if(g_gpu) q35cu_mem(&vf, &vt);
#endif
    s_fmt(&b, "{\"status\":\"ok\",\"model\":\"%s\",\"n_ctx\":%d,\"ctx_used\":%d,"
              "\"gpu\":%s,\"speculative\":%s,\"vram_free_gib\":%.2f}",
          g_model_name, g_R.n_ctx, g_C.ctx_n,
#ifndef QWEN_NO_CUDA
          g_gpu ? "true" : "false", g_spec ? "true" : "false",
#else
          "false", "false",
#endif
          vf/1073741824.0);
    http_send(fd, 200, "OK", "application/json", b.p, b.n);
    s_free(&b);
}

/* Percent-decoding, because a path query carries spaces, plus signs and non-ASCII names. Not
 * a general URL decoder: it decodes into a fixed buffer and stops, and everything it produces
 * still has to survive h_resolve afterwards. */
static void url_decode(char *s){
    char *w = s;
    for(char *r = s; *r; r++){
        if(*r == '%' && isxdigit((unsigned char)r[1]) && isxdigit((unsigned char)r[2])){
            char h[3] = { r[1], r[2], 0 };
            *w++ = (char)strtol(h, NULL, 16);
            r += 2;
        } else if(*r == '+') *w++ = ' ';
        else *w++ = *r;
    }
    *w = 0;
}

/* ?project=NAME out of a request target. Only what the caller can legally send: the name is
 * validated in the harness, so nothing here needs to decode anything clever. */
static void query_arg(const char *path, const char *key, char *out, size_t cap){
    out[0] = 0;
    const char *q = strchr(path, '?');
    if(!q) return;
    char pat[32];
    const int pl = snprintf(pat, sizeof pat, "%s=", key);
    for(const char *p = q + 1; p && *p; ){
        if(!strncmp(p, pat, (size_t)pl)){
            const char *v = p + pl;
            size_t n = strcspn(v, "& ");
            if(n >= cap) n = cap - 1;
            memcpy(out, v, n); out[n] = 0;
            return;
        }
        const char *amp = strchr(p, '&');
        p = amp ? amp + 1 : NULL;
    }
}

static int path_is(const char *p, const char *want){
    const size_t n = strlen(want);
    return !strncmp(p, want, n) && (p[n] == 0 || p[n] == '?' || p[n] == ' ');
}

static int serve(int fd){   /* returns 1 if the fd was handed off and must not be closed */
    Str req = {0};
    char buf[16384];
    size_t hdr_end = 0;
    /* headers first */
    for(;;){
        const ssize_t r = recv(fd, buf, sizeof buf, 0);
        if(r <= 0){ s_free(&req); return 0; }
        s_add(&req, buf, (size_t)r);
        const char *e = memmem(req.p, req.n, "\r\n\r\n", 4);
        if(e){ hdr_end = (size_t)(e - req.p) + 4; break; }
        if(req.n > (1u<<20)){ http_err(fd, 431, "Request Header Fields Too Large", "headers too long");
                              s_free(&req); return 0; }
    }
    /* body, if Content-Length says so */
    size_t clen = 0;
    { char *h = req.p;
      for(char *l = h; l && l < req.p + hdr_end; ){
          if(!strncasecmp(l, "Content-Length:", 15)) clen = (size_t)strtoul(l+15, NULL, 10);
          char *nl = memchr(l, '\n', hdr_end - (size_t)(l - h));
          l = nl ? nl + 1 : NULL;
      } }
    while(req.n < hdr_end + clen){
        const ssize_t r = recv(fd, buf, sizeof buf, 0);
        if(r <= 0) break;
        s_add(&req, buf, (size_t)r);
    }

    char method[16] = {0}, path[1024] = {0};
    sscanf(req.p, "%15s %1023s", method, path);

    if(!strcmp(method, "OPTIONS")){
        sock_str(fd, "HTTP/1.1 204 No Content\r\n" CORS "Content-Length: 0\r\nConnection: close\r\n\r\n");
        s_free(&req); return 0;
    }
    if(g_api_key){
        const char *a = memmem(req.p, hdr_end, "Bearer ", 7);
        if(!a || strncmp(a + 7, g_api_key, strlen(g_api_key))){
            http_err(fd, 401, "Unauthorized", "invalid api key"); s_free(&req); return 0;
        }
    }

    if(!strcmp(method, "GET")){
        if(path_is(path, "/") || path_is(path, "/index.html")){
            if(g_ui) http_send(fd, 200, "OK", "text/html; charset=utf-8", g_ui, g_ui_len);
            else     http_err(fd, 404, "Not Found", "no web UI was loaded (see --www)");
        }
        else if(path_is(path, "/v1/models") || path_is(path, "/models")) handle_models(fd);
        else if(path_is(path, "/v1/tools")  || path_is(path, "/tools")){
            char proj[96]; query_arg(path, "project", proj, sizeof proj);
            const char *perr = NULL;
            if(!h_use_project(proj, &perr)){ http_err(fd, 400, "Bad Request", perr); }
            else { Str b = {0}; h_tools_json(&b);
                   http_send(fd, 200, "OK", "application/json", b.p, b.n); s_free(&b); }
        }
        else if(path_is(path, "/v1/projects") || path_is(path, "/projects")){
            Str b = {0}; h_projects_json(&b);
            http_send(fd, 200, "OK", "application/json", b.p, b.n); s_free(&b);
        }
        else if(path_is(path, "/v1/chats")){
            char proj[96]; query_arg(path, "project", proj, sizeof proj);
            Str b = {0}; cs_list_json(&b, proj);
            http_send(fd, 200, "OK", "application/json", b.p, b.n); s_free(&b);
        }
        else if(path_is(path, "/v1/chat")){
            char proj[96], id[96];
            query_arg(path, "project", proj, sizeof proj);
            query_arg(path, "id", id, sizeof id);
            Str b = {0};
            if(cs_get(&b, proj, id)) http_send(fd, 200, "OK", "application/json", b.p, b.n);
            else                     http_err(fd, 404, "Not Found", "no such chat");
            s_free(&b);
        }
        /* The file view. Same h_resolve, same fence, same per-thread project as the tools —
         * a second reader with its own idea of "inside" would be the second door. */
        /* Whatever the live turn has produced so far. The whole text each time rather than a
         * delta: a turn is a few KB, this is localhost, and an offset protocol over two
         * interleaved streams would be more to get wrong than it saves. */
        else if(path_is(path, "/v1/live")){
            char want[96]; query_arg(path, "chat", want, sizeof want);
            Str b = {0};
            pthread_mutex_lock(&g_live.mu);
            const int mine = want[0] && !strcmp(want, g_live.chat);
            s_fmt(&b, "{\"seq\":%llu,\"live\":%s,\"done\":%s,\"mine\":%s,\"chat\":",
                  (unsigned long long)g_live.seq,
                  g_live.live ? "true" : "false", g_live.done ? "true" : "false",
                  mine ? "true" : "false");
            s_json(&b, g_live.chat, strlen(g_live.chat));
            if(mine){
                s_cat(&b, ",\"content\":");
                s_json(&b, g_live.content.p ? g_live.content.p : "", g_live.content.n);
                s_cat(&b, ",\"reasoning\":");
                s_json(&b, g_live.reasoning.p ? g_live.reasoning.p : "", g_live.reasoning.n);
                s_fmt(&b, ",\"tool_calls\":%s", g_live.tools.n ? g_live.tools.p : "[]");
                const double tps = g_live.dec_s > 0 ? g_live.gen / g_live.dec_s : 0;
                s_fmt(&b, ",\"gen\":%d,\"decode_s\":%.3f,\"tok_s\":%.3f,\"gbps\":%.2f,"
                          "\"gbps_vram\":%.2f,\"prompt\":%d,\"cached\":%d,\"prefill_s\":%.3f,"
                          "\"elapsed\":%.1f,\"pre_done\":%d,\"pre_tok_s\":%.2f,\"phase\":\"%s\"",
                      g_live.gen, g_live.dec_s, tps, g_bytes_dram * tps / 1e9,
                      g_bytes_vram * tps / 1e9, g_live.prompt, g_live.cached, g_live.pre_s,
                      now() - g_live.started, g_live.pre_done, g_live.pre_tps,
                      g_live.live ? (g_live.gen ? "decode" : "prefill") : "done");
            }
            s_cat(&b, "}");
            pthread_mutex_unlock(&g_live.mu);
            http_send(fd, 200, "OK", "application/json", b.p, b.n);
            s_free(&b);
        }
        else if(path_is(path, "/v1/term")){
            char sn[32]; query_arg(path, "since", sn, sizeof sn);
            Str b = {0}; h_term_json(&b, sn[0] ? strtoull(sn, NULL, 10) : 0);
            http_send(fd, 200, "OK", "application/json", b.p, b.n); s_free(&b);
        }
        else if(path_is(path, "/v1/fs/raw")){
            char proj[96], rel[1024];
            query_arg(path, "project", proj, sizeof proj);
            query_arg(path, "path", rel, sizeof rel);
            url_decode(rel);
            const char *err = "tools are disabled on this server", *mime = NULL;
            size_t n = 0; char *b = NULL;
            if(g_tmode != TM_OFF && h_use_project(proj, &err))
                b = h_fs_raw(rel, &n, &mime, &err);
            if(b){ http_send(fd, 200, "OK", mime, b, n); free(b); }
            else   http_err(fd, 400, "Bad Request", err);
        }
        else if(path_is(path, "/v1/fs/list") || path_is(path, "/v1/fs/read")){
            char proj[96], rel[1024];
            query_arg(path, "project", proj, sizeof proj);
            query_arg(path, "path", rel, sizeof rel);
            url_decode(rel);
            const char *err = "no tools on this server";
            Str b = {0};
            int ok = 0;
            if(g_tmode == TM_OFF) err = "tools are disabled on this server";
            else if(!h_use_project(proj, &err)) ok = 0;
            else ok = path_is(path, "/v1/fs/list") ? h_fs_list(&b, rel, &err)
                                                   : h_fs_read(&b, rel, &err);
            if(ok) http_send(fd, 200, "OK", "application/json", b.p, b.n);
            else   http_err(fd, 400, "Bad Request", err);
            s_free(&b);
        }
        else if(path_is(path, "/health") || path_is(path, "/v1/health"))  handle_health(fd);
        else http_err(fd, 404, "Not Found", "no such path");
        s_free(&req); return 0;
    }
    if(!strcmp(method, "POST")){
        if(path_is(path, "/v1/chat/completions") || path_is(path, "/chat/completions")
           || path_is(path, "/api/chat/completions")){
            if(g_verbose) fprintf(stderr, "[%s] %s %s (%zu B)\n",
                                  g_api_key ? "auth" : "open", method, path, clen);
            engine_submit(fd, req.p + hdr_end, clen);
            s_free(&req);
            return 1;                    /* the engine thread owns fd now */
        } else if(path_is(path, "/v1/debug/prefix")){
            /* WHERE DOES THE PREFIX BREAK.
             *
             * `live` in q35_cache_begin is all-or-nothing: one token of divergence anywhere
             * discards the entire prefix and the turn re-prefills from a checkpoint instead.
             * The only figure the API reports is prompt_tokens_cached, and a 0 there says
             * "it did not match" without saying WHERE — which is the one thing worth knowing,
             * because the answer is a template detail measured in single newlines.
             *
             * So: build the prompt exactly as handle_chat would, tokenize it, and diff it
             * against the token sequence the engine actually ran. Read-only, and only
             * meaningful while the engine is idle. */
            char *body = strndup(req.p + hdr_end, clen);
            char *arena = NULL;
            jval *r = body ? json_parse(body, &arena) : NULL;
            jval *msgs = r ? json_get(r, "messages") : NULL;
            if(!msgs || msgs->t != J_ARR || msgs->len == 0){
                http_err(fd, 400, "Bad Request", "messages[] is required");
            } else {
                jval *jt; int think = 1;
                if((jt = json_get(r, "reasoning")) && jt->t == J_BOOL) think = jt->boolean;
                if((jt = json_get(r, "enable_thinking")) && jt->t == J_BOOL) think = jt->boolean;
                Str p = {0};
                build_prompt(&p, msgs, json_get(r, "tools"), think);
                int *tk = (int*)malloc(sizeof(int)*(size_t)(p.n + 64));
                const int n = tk ? tok_encode(&g_T, p.p, (int)p.n, tk, (int)p.n + 60) : 0;
                int i = 0;
                while(i < n && i < g_C.ctx_n && tk[i] == g_C.ctx[i]) i++;
                Str b = {0};
                s_fmt(&b, "{\"prompt\":%d,\"engine_ctx\":%d,\"match\":%d,\"live_reuse\":%s",
                      n, g_C.ctx_n, i,
                      (g_C.ctx_n > 0 && i == g_C.ctx_n) ? "true" : "false");
                #define DBG_DEC(arr, from, to) do{ Str t = {0};                                \
                    for(int q = (from); q < (to); q++){ int id = (arr)[q]; char pc[512];        \
                        const int np = tok_decode(&g_T, &id, 1, pc, (int)sizeof pc);            \
                        if(np > 0) s_add(&t, pc, (size_t)np); }                                 \
                    s_json(&b, t.p ? t.p : "", t.n); s_free(&t); }while(0)
                { const int lo = i > 12 ? i - 12 : 0;
                  s_cat(&b, ",\"common_tail\":");  DBG_DEC(tk, lo, i);
                  s_cat(&b, ",\"prompt_next\":");  DBG_DEC(tk, i, i + 10 < n ? i + 10 : n);
                  s_cat(&b, ",\"engine_next\":");
                  DBG_DEC(g_C.ctx, i, i + 10 < g_C.ctx_n ? i + 10 : g_C.ctx_n); }
                #undef DBG_DEC
                s_cat(&b, "}");
                http_send(fd, 200, "OK", "application/json", b.p, b.n);
                s_free(&b); s_free(&p); free(tk);
            }
            free(body); free(arena);
        } else if(path_is(path, "/v1/projects") || path_is(path, "/projects")){
            char *body = strndup(req.p + hdr_end, clen);
            char *arena = NULL;
            jval *r = body ? json_parse(body, &arena) : NULL;
            const char *nm = r ? a_str(r, "name", NULL) : NULL, *perr = "no name given";
            if(nm && h_make_project(nm, &perr)){
                Str b = {0}; h_projects_json(&b);
                http_send(fd, 200, "OK", "application/json", b.p, b.n); s_free(&b);
            } else http_err(fd, 400, "Bad Request", perr);
            free(arena); free(body);
        } else if(path_is(path, "/v1/chat/save") || path_is(path, "/v1/chat/delete")){
            char *body = strndup(req.p + hdr_end, clen);
            char *arena = NULL;
            jval *r = body ? json_parse(body, &arena) : NULL;
            const char *proj = r ? a_str(r, "project", "") : "";
            const char *id   = r ? a_str(r, "id", NULL) : NULL;
            int ok = 0;
            if(id && path_is(path, "/v1/chat/delete")){
                /* the transcript and its checkpoint go together — leaving a 4 GiB file behind
                 * for a conversation the user deleted is both a leak and a surprise */
                char cp[PATH_MAX];
                if(ck_path(id, cp, sizeof cp)) unlink(cp);
                ok = cs_delete(proj, id);
            }
            else if(id){
                jval *m = json_get(r, "messages");
                Str mj = {0};
                if(m) j_emit(&mj, m); else s_cat(&mj, "[]");
                ok = cs_save(proj, id, a_str(r, "title", ""), m ? m->len : 0, mj.p);
                s_free(&mj);
            }
            free(arena); free(body);
            if(ok) http_send(fd, 200, "OK", "application/json", "{\"ok\":true}", 13);
            else   http_err(fd, 400, "Bad Request", "could not store the chat");
        } else if(path_is(path, "/v1/live/cancel")){
            /* The explicit stop, which is now the ONLY thing that ends a resumable turn. */
            pthread_mutex_lock(&g_live.mu);
            g_live.cancel = 1;
            pthread_mutex_unlock(&g_live.mu);
            http_send(fd, 200, "OK", "application/json", "{\"ok\":true}", 13);
        } else if(path_is(path, "/v1/model/select")){
            /* Switch the loaded model. The reply goes out FIRST, because model_switch re-execs
             * and never returns; the page then polls /health and reloads when the server is back. */
            char *body = strndup(req.p + hdr_end, clen); char *arena = NULL;
            jval *r = body ? json_parse(body, &arena) : NULL;
            const char *want = r ? a_str(r, "model", "") : "";
            int idx = -1;
            for(int i = 0; i < g_nmodels; i++) if(!strcmp(g_models[i].name, want)) idx = i;
            if(idx < 0){ http_err(fd, 400, "Bad Request", "unknown model"); }
            else if(idx == g_cur_model){
                const char *m = "{\"ok\":true,\"same\":true}";
                http_send(fd, 200, "OK", "application/json", m, strlen(m));
            } else {
                const char *m = "{\"ok\":true,\"switching\":true}";
                http_send(fd, 200, "OK", "application/json", m, strlen(m));
                free(body); free(arena);
                model_switch(idx);          /* re-execs; does not return */
            }
            free(body); free(arena);
        } else if(path_is(path, "/v1/term/input") || path_is(path, "/v1/term/signal")){
            /* Whatever arrives here goes down the PTY and nowhere else. It is not logged, not
             * echoed, and not put anywhere the model can read it: when it carries a sudo
             * password the only correct number of copies is zero. */
            char *body = strndup(req.p + hdr_end, clen);
            char *arena = NULL;
            jval *r = body ? json_parse(body, &arena) : NULL;
            int ok = 0;
            if(path_is(path, "/v1/term/signal")){
                const char *sg = r ? a_str(r, "sig", "int") : "int";
                ok = h_term_signal(!strcmp(sg, "kill") ? SIGKILL : SIGINT);
            } else {
                jval *d = r ? json_get(r, "data") : NULL;
                const int secret = r && a_bool(r, "secret", 0);
                if(d && d->t == J_STR) ok = h_term_input(d->str, strlen(d->str), secret);
            }
            free(arena);
            if(body) memset(body, 0, clen);      /* do not leave it in the heap either */
            free(body);
            http_send(fd, 200, "OK", "application/json",
                      ok ? "{\"ok\":true}" : "{\"ok\":false}", ok ? 13 : 14);
        } else if(path_is(path, "/v1/tools/exec") || path_is(path, "/tools/exec")){
            /* Deliberately NOT on the engine thread. A tool call holds no model state, so
             * running it here lets the page fetch, list and grep while a generation streams —
             * and a `make` that takes two minutes does not stall the engine queue behind it. */
            Str b = {0};
            char *body = strndup(req.p + hdr_end, clen);
            if(g_verbose){
                char *arena = NULL; jval *r = json_parse(body, &arena);
                fprintf(stderr, "  tool %s\n", r ? a_str(r, "name", "?") : "?");
                free(arena);
            }
            h_exec_json(&b, body ? body : "{}");
            free(body);
            http_send(fd, 200, "OK", "application/json", b.p, b.n);
            s_free(&b);
        } else {
            http_err(fd, 404, "Not Found",
                     "POST is /v1/chat/completions or /v1/tools/exec");
        }
        s_free(&req); return 0;
    }
    http_err(fd, 405, "Method Not Allowed", "use GET or POST");
    s_free(&req);
    return 0;
}

/* ---------------- startup ---------------- */

static void on_sig(int s){ (void)s; g_abort = 1; }

/* Firefox is a child process, and a child process that outlives its parent is a half-gigabyte
 * leak nobody will connect to this server an hour later. */
static void on_exit_cleanup(void){ web_stop(); }

static void *conn_thread(void *arg){
    const int fd = (int)(intptr_t)arg;
    if(!serve(fd)) close(fd);        /* a handed-off fd belongs to the engine thread */
    return NULL;
}

static void fix_omp_env(char **argv){
    /* execv("/proc/self/exe") renames the process to "exe" — that is the basename of the path
     * handed to execve, not of the binary. pgrep, pkill, ps and every "is it running" check
     * then miss it, including this project's own `make stop`. Say the name explicitly. */
    prctl(PR_SET_NAME, "qwen35_server", 0, 0, 0);
    q35_env_compat();          /* COLIBRI_* still works */
    if(getenv("QWEN_ENV_SET")) return;
    setenv("QWEN_ENV_SET", "1", 1);
    if(!getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY", "active", 1);
    if(!getenv("OMP_PROC_BIND"))   setenv("OMP_PROC_BIND", "close", 1);
    if(!getenv("OMP_PLACES"))      setenv("OMP_PLACES", "cores", 1);
    if(!getenv("GOMP_SPINCOUNT"))  setenv("GOMP_SPINCOUNT", "200000", 1);
    if(!getenv("OMP_NUM_THREADS")){
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        char b[16]; snprintf(b, sizeof b, "%ld", n > 16 ? 16 : (n > 1 ? n : 1));
        setenv("OMP_NUM_THREADS", b, 1);
    }
    execv("/proc/self/exe", argv);
}

static char *read_file(const char *p, size_t *len){
    FILE *f = fopen(p, "rb");
    if(!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if(n < 0){ fclose(f); return NULL; }        /* ftell returns -1 on error */
    char *b = (char*)malloc((size_t)n + 1);
    if(!b || fread(b, 1, (size_t)n, f) != (size_t)n){ free(b); fclose(f); return NULL; }
    b[n] = 0; *len = (size_t)n; fclose(f);
    return b;
}

int main(int argc, char **argv){
    const char *path = NULL, *host = "127.0.0.1", *www = NULL, *workspace = NULL,
               *projects = NULL;
    /* 8192, not 32768. The KV tier is sized from n_ctx and the engine already holds 9.3 GiB
     * of resident weights; at 32768 the process reached 14.5 GiB RSS on a 31 GiB machine and
     * was OOM-killed mid-request. The context still GROWS past this — kvt_grow raises it on
     * demand and stops at the memory floor — so this is a starting point, not a ceiling. */
    int port = 8080, cpu_only = 0;
    for(int i = 1; i < argc; i++){
        const char *a = argv[i];
        if(a[0] != '-' && !path) path = a;
        else if(!strcmp(a,"--port") && i+1 < argc) port = atoi(argv[++i]);
        else if(!strcmp(a,"--host") && i+1 < argc) host = argv[++i];
        else if(!strcmp(a,"--ctx")  && i+1 < argc) g_n_ctx = atoi(argv[++i]);
        else if(!strcmp(a,"--www")  && i+1 < argc) www = argv[++i];
        else if(!strcmp(a,"--api-key") && i+1 < argc) g_api_key = argv[++i];
        else if(!strcmp(a,"--name") && i+1 < argc) g_model_name = argv[++i];
        else if(!strcmp(a,"--model") && i+1 < argc){        /* NAME:PATH, repeatable */
            const char *spec = argv[++i]; const char *colon = strchr(spec, ':');
            if(colon && g_nmodels < (int)(sizeof g_models/sizeof g_models[0])){
                int nl = (int)(colon - spec); if(nl > 63) nl = 63;
                memcpy(g_models[g_nmodels].name, spec, (size_t)nl); g_models[g_nmodels].name[nl] = 0;
                snprintf(g_models[g_nmodels].path, sizeof g_models[g_nmodels].path, "%s", colon+1);
                g_nmodels++;
            } else fprintf(stderr, "--model: expected NAME:PATH\n");
        }
        else if(!strcmp(a,"--default") && i+1 < argc) g_default_model = argv[++i];
        else if(!strcmp(a,"--cpu")) cpu_only = 1;
        else if(!strcmp(a,"--workspace") && i+1 < argc) workspace = argv[++i];
        else if(!strcmp(a,"--projects")  && i+1 < argc) projects  = argv[++i];
        else if(!strcmp(a,"--tools") && i+1 < argc){
            if(!h_mode_parse(argv[++i], &g_tmode)){
                fprintf(stderr, "--tools: expected off | ro | workspace | full\n"); return 1; }
        }
        else if(!strcmp(a,"--tool-timeout") && i+1 < argc) g_ttmo = atoi(argv[++i]);
        else if(!strcmp(a,"--tool-output")  && i+1 < argc) g_tmax = (size_t)atol(argv[++i]);
        else if(!strcmp(a,"--web") && i+1 < argc) g_web_on = strcmp(argv[++i], "off") != 0;
        else if(!strcmp(a,"--quiet")) g_verbose = 0;
        else if(!strcmp(a,"-h") || !strcmp(a,"--help")){
            fprintf(stderr,
                "usage: %s <model.gguf> [--host 127.0.0.1] [--port 8080] [--ctx 8192]\n"
                "          [--www webui.html] [--api-key KEY] [--name ID] [--cpu] [--quiet]\n"
                "          [--tools MODE] [--projects DIR | --workspace DIR]\n"
                "          [--tool-timeout MS] [--tool-output N]\n\n"
                "  --ctx N        INITIAL context, not a limit. It grows on demand and stops\n"
                "                 only at the memory floor (QWEN_MEM_FLOOR_GB, 2 GiB).\n"
                "  --tools MODE   what the agent may do. The default is `workspace`.\n"
                "                   off        no tools at all; a plain chat server\n"
                "                   ro         read, grep, glob, list — nothing that writes\n"
                "                   workspace  also write, confined to --workspace;\n"
                "                              a shell command needs a human to approve it\n"
                "                   full       everything, anywhere, unapproved\n"
                "  --projects D   root holding one subdirectory per project; the client picks\n"
                "                 one by NAME. Default $HOME/QwenEngine/projects.\n"
                "  --web on|off   the headless-Firefox tools. Off removes them from the prompt\n"
                "                 entirely, which is ~250 tokens a turn.\n"
                "  --workspace D  a single fixed workspace instead of projects.\n"
                "                 Either way paths are resolved with realpath before they are\n"
                "                 compared to the root, so .. and symlinks cannot leave it.\n\n"
                "  GET  /                     the chat UI\n"
                "  GET  /v1/models            model list\n"
                "  GET  /v1/tools[?project=N] tool schemas + policy + the agent system prompt\n"
                "  GET  /v1/projects          the projects that exist\n"
                "  GET  /v1/live?chat=ID      the turn in flight, so a reload can pick it up\n"
                "  POST /v1/live/cancel       stop it — the only thing that does\n"
                "  GET  /v1/chats?project=N   the conversations of a project\n"
                "  GET  /v1/chat?project=&id= one conversation\n"
                "  POST /v1/chat/save         {project, id, title, messages}\n"
                "  POST /v1/chat/delete       {project, id}\n"
                "  GET  /v1/fs/list           directory listing for the file view\n"
                "  GET  /v1/fs/read           one file, for the file view\n"
                "  GET  /v1/fs/raw            the bytes, for images the view can show\n"
                "  GET  /v1/term?since=N      the live terminal transcript\n"
                "  POST /v1/term/input        type into the running command (sudo password)\n"
                "  POST /v1/term/signal       {sig:\"int\"|\"kill\"}\n"
                "  POST /v1/projects          create one: {name}\n"
                "  POST /v1/chat/completions  OpenAI-compatible, streaming and not\n"
                "  POST /v1/tools/exec        run one tool  {name, arguments, approved}\n"
                "  GET  /health               engine status\n", argv[0]);
            return 0;
        }
    }
    if(!path && g_nmodels == 0){ fprintf(stderr, "qwen35_server: no model (path or --model NAME:PATH)\n"); return 1; }

    /* The workspace is resolved ONCE, here, and every path a tool is handed is compared to the
     * result. Resolving it per call would let a symlink swapped in mid-session change what
     * "inside" means. */
    /* Projects: one engine, many workspaces, each a direct subdirectory of this root. The
     * default is ~/QwenEngine/projects — outside any checkout, because a coding agent whose
     * workspace is the directory holding its own source is a footgun waiting for the first
     * "clean up this repo". /home itself is root-owned, so "a folder in Home" means $HOME. */
    static char pdefault[PATH_MAX];
    if(!projects && !workspace){
        const char *home = getenv("HOME");
        if(home && *home){
            snprintf(pdefault, sizeof pdefault, "%s/QwenEngine/projects", home);
            mkdir(pdefault, 0755);                  /* parent may not exist; checked below */
            struct stat st;
            if(stat(pdefault, &st) || !S_ISDIR(st.st_mode)){
                char up[PATH_MAX];
                snprintf(up, sizeof up, "%s/QwenEngine", home);
                mkdir(up, 0755); mkdir(pdefault, 0755);
            }
            projects = pdefault;
        }
    }
    if(projects){
        if(!realpath(projects, g_proot)){
            fprintf(stderr, "--projects %s: %s\n", projects, strerror(errno)); return 1; }
        struct stat st;
        if(stat(g_proot, &st) || !S_ISDIR(st.st_mode)){
            fprintf(stderr, "--projects %s is not a directory\n", g_proot); return 1; }
        snprintf(g_root, sizeof g_root, "%s", g_proot);   /* the default when none is chosen */
        /* the engine's own directory: one level up from projects/. It is not a fence — it is
         * how the consent question knows whether it is asking about a neighbour or about the
         * rest of the machine. */
        { char up[PATH_MAX]; snprintf(up, sizeof up, "%s/..", g_proot);
          if(!realpath(up, g_home)) g_home[0] = 0; }
        /* Beside the projects, never inside one: the agent works in there, and a folder of its
         * own transcripts is noise in every listing and something it can corrupt. */
        if(g_home[0]){ snprintf(g_chat_root, sizeof g_chat_root, "%s/chats", g_home);
                       mkdir(g_chat_root, 0700);
                       snprintf(g_ck_root, sizeof g_ck_root, "%s/ckpt", g_home);
                       mkdir(g_ck_root, 0700); }
    }
    if(workspace){                                   /* explicit single-workspace mode */
        g_proot[0] = 0;
        if(!realpath(workspace, g_root)){
            fprintf(stderr, "--workspace %s: %s\n", workspace, strerror(errno)); return 1; }
        struct stat ws;
        if(stat(g_root, &ws) || !S_ISDIR(ws.st_mode)){
            fprintf(stderr, "--workspace %s is not a directory\n", g_root); return 1; }
    }
    if(!g_root[0] && !realpath(".", g_root)){
        fprintf(stderr, "cannot resolve the current directory\n"); return 1; }

    /* FAIL CLOSED. Tools on a non-loopback address with no key is remote code execution on
     * this machine for anyone who can route to the port — and `full` is that for anyone at
     * all. Refusing to start is the only honest answer; a warning printed above a running
     * server is one nobody reads. */
    const int loopback = !strncmp(host, "127.", 4) || !strcmp(host, "localhost");
    if(g_tmode != TM_OFF && !loopback && !g_api_key){
        fprintf(stderr,
            "\n  refusing to start: --tools %s on %s with no --api-key.\n"
            "  Anything that can reach %s:%d could run commands here. Pick one:\n"
            "    --api-key KEY   require a key, or\n"
            "    --tools off     serve chat only, or\n"
            "    --host 127.0.0.1\n\n", h_mode_name(), host, host, port);
        return 1;
    }
    if(g_tmode == TM_FULL)
        fprintf(stderr, "\n  ** --tools full: no path confinement and no approval. The agent can\n"
                        "     run and change anything this user can. **\n");

    g_argv = argv;
    /* Best-effort: keep our own pages out of swap. RLIMIT_MEMLOCK is usually tiny (8 MiB here),
     * so this mostly fails and the systemd scope (serve.sh, MemorySwapMax=0) is the real fence;
     * it still helps where the limit was raised. */
    { struct rlimit rl; if(!getrlimit(RLIMIT_MEMLOCK, &rl) && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < (1ULL<<34)){
          rl.rlim_cur = rl.rlim_max; setrlimit(RLIMIT_MEMLOCK, &rl); }
      if(mlockall(MCL_CURRENT | MCL_FUTURE) != 0 && g_verbose)
          fprintf(stderr, "  note: mlockall failed (%s) — relying on the cgroup swap fence\n", strerror(errno)); }
    fix_omp_env(argv);
    signal(SIGINT, on_sig);
    signal(SIGPIPE, SIG_IGN);
    atexit(on_exit_cleanup);

    double t_boot = now(), t_ph = t_boot;
#define PHASE(name) do{ const double _n = now(); \
    fprintf(stderr, "  [%6.2fs] %-22s %.2fs\n", _n - t_boot, (name), _n - t_ph); \
    t_ph = _n; }while(0)

    /* Resolve which registered model to load now: the switch that asked for it
     * (QWEN_SELECT_MODEL, carried across the re-exec), then --default, then the first. A bare
     * positional path with no --model is registered as the sole model. */
    if(g_nmodels == 0 && path){
        snprintf(g_models[0].name, sizeof g_models[0].name, "%s", g_model_name);
        snprintf(g_models[0].path, sizeof g_models[0].path, "%s", path);
        g_nmodels = 1;
    }
    if(g_nmodels == 0){ fprintf(stderr, "no model: give a path or --model NAME:PATH\n"); return 1; }
    {
        const char *sel = getenv("QWEN_SELECT_MODEL");
        if(!sel) sel = g_default_model;
        int idx = 0;
        if(sel) for(int i = 0; i < g_nmodels; i++) if(!strcmp(g_models[i].name, sel)){ idx = i; break; }
        g_cur_model = idx; path = g_models[idx].path; g_model_name = g_models[idx].name;
    }
    if(!q35_open(&g_M, path)){ fprintf(stderr, "cannot open %s\n", path); return 1; }
    /* ---- THE ENGINE REFUSES TO START WITHOUT ROOM FOR ITS OWN CEILING ----
     *
     * Every silent death this engine has had on this machine looked the same from the outside:
     * no core, no log line, no error, because the kernel does not ask. The kernel log is
     * unambiguous about what actually happened —
     *
     *   oom-kill:constraint=CONSTRAINT_NONE, global_oom, task=qwen35_server
     *   Killed process 50230  anon-rss:10847436kB  oom_score_adj:200
     *
     * — global_oom, meaning the MACHINE ran out, not this process; and oom_score_adj 200,
     * meaning this process is the one the kernel prefers to kill when that happens. So it is
     * not enough for the engine to fit at startup. It has to fit at its own ceiling, with room
     * left over for everything else, or it is merely queued to be killed later — after twenty
     * minutes of prefill, which is exactly when losing it costs the most.
     *
     * The ceiling is the cgroup's memory.max when serve.sh was given MEMMAX, otherwise
     * QWEN_MAX_RAM_GB. The headroom is the same floor everything else here respects
     * (QWEN_MEM_FLOOR_GB, 2 GiB). Failing HERE, in a second, with a number and a reason, is the
     * whole point: a refusal that names the shortfall is worth more than a start that succeeds
     * and dies at token 19712. */
    {
        int64_t cap = 0;
        const char *e = getenv("QWEN_MAX_RAM_GB");
        if(e && *e) cap = (int64_t)(atof(e)*1073741824.0);
        if(cap <= 0){
            /* our own cgroup: /proc/self/cgroup gives "0::<path>" under cgroup v2 */
            char line[512] = {0};
            FILE *f = fopen("/proc/self/cgroup", "r");
            if(f){ if(!fgets(line, sizeof line, f)) line[0] = 0; fclose(f); }
            const char *rel = strstr(line, "0::");
            if(rel){
                char path[768]; rel += 3;
                size_t k = strcspn(rel, "\n");
                if(k && k < 600){
                    char sub[640]; memcpy(sub, rel, k); sub[k] = 0;
                    snprintf(path, sizeof path, "/sys/fs/cgroup%s/memory.max", sub);
                    FILE *m = fopen(path, "r");
                    if(m){ char v[64] = {0};
                           if(fgets(v, sizeof v, m) && strncmp(v, "max", 3) != 0)
                               cap = strtoll(v, NULL, 10);
                           fclose(m); }
                }
            }
        }
        if(cap > 0){
            const int64_t need  = cap + q35_mem_floor();
            const int64_t avail = q35_ram_avail_bytes();
            fprintf(stderr, "  memory ceiling %.2f GiB + %.2f GiB floor = %.2f GiB needed, "
                            "%.2f GiB available\n",
                    cap/1073741824.0, q35_mem_floor()/1073741824.0,
                    need/1073741824.0, avail/1073741824.0);
            if(avail < need){
                fprintf(stderr,
                  "\n  REFUSING TO START: %.2f GiB short.\n"
                  "  This engine is marked oom_score_adj %d, so when the machine runs out it is\n"
                  "  the process the kernel kills — and it would do that after the prefill, not\n"
                  "  before it. Starting now would only move the failure to the expensive end.\n"
                  "  Free memory, lower the ceiling (MEMMAX / QWEN_MAX_RAM_GB), or lower the\n"
                  "  floor (QWEN_MEM_FLOOR_GB) if you know what you are trading away.\n",
                  (need - avail)/1073741824.0, q35_oom_score_adj());
                return 1;
            }
        }
    }

    PHASE("open gguf");
    const Q35Cfg *c = &g_M.c;
    tok_set_pre(&g_M.m);
    if(!tok_from_gguf(&g_T, &g_M.m, &g_ids)){ fprintf(stderr, "tokenizer failed\n"); return 1; }
    PHASE("tokenizer");
    g_im_start = tok_id_of(&g_T, "<|im_start|>");
    g_im_end   = tok_id_of(&g_T, "<|im_end|>");
    g_eos      = g_im_end >= 0 ? g_im_end : g_ids.eos;
    T_THINK_O  = tok_id_of(&g_T, "<think>");
    T_THINK_C  = tok_id_of(&g_T, "</think>");
    T_TOOL_O   = tok_id_of(&g_T, "<tool_call>");
    T_TOOL_C   = tok_id_of(&g_T, "</tool_call>");

    /* THE CONTEXT IS NOT CAPPED, AND --ctx IS NOT A LIMIT.
     *
     * --ctx sizes the FIRST allocation. From there the store grows on demand: kvt_grow
     * extends it, checks q35_mem_room() before every extension, and refuses — keeping the
     * context where it is and saying so — the moment MemAvailable would drop under the floor
     * (QWEN_MEM_FLOOR_GB, 2 GiB by default). Nothing else stops it.
     *
     * What --ctx really decides is how much of the KV stays in RAM rather than on the spill
     * tier, because the hot arena is sized once at open. So it is chosen from the memory that
     * will actually be spare AFTER residency — which is known here, since the weights that are
     * about to become resident are exactly the FFN and the embedding table. Sizing it from
     * MemAvailable as measured now would over-promise by 9.3 GiB and is how this process got
     * OOM-killed once already. */
    Q35Bytes BY; q35_bytes(&g_M, &BY);
    const int64_t per_tok  = q35_kv_bytes_per_token(c, Q35_KV_Q8_0);
    const int64_t will_use = BY.ffn + BY.embd + BY.mtp;
    int64_t spare = q35_ram_avail_bytes() - will_use - q35_reserve_bytes() - q35_mem_floor();
    int64_t kvb   = spare > 0 ? spare/2 : (64<<20);
    const int64_t kv_250k = per_tok*250000;
    if(kvb > kv_250k) kvb = kv_250k;           /* no point reserving past the plan's horizon */
    /* QWEN_KV_RAM_GB — the same knob qwen35_plan.h documents, honoured here too, and a CAP
     * rather than a raise for the same reason it is there: this number is taken from RAM the
     * moment the store opens, and MemAvailable at startup counts page cache that the weights
     * are about to evict. The auto figure is a good default and a bad promise on a machine
     * that is already in swap — and being OOM-killed mid-request is silent, because the kernel
     * does not ask. Pin it when the box is tight, or to exercise the disk path deliberately. */
    { const char *e = getenv("QWEN_KV_RAM_GB");
      if(e && *e){ const int64_t cap = (int64_t)(atof(e)*1073741824.0);
                   if(cap > 0 && cap < kvb) kvb = cap; } }
    if(kvb < per_tok*(int64_t)g_n_ctx + (32<<20)) kvb = per_tok*(int64_t)g_n_ctx + (32<<20);
    if(!q35_state_init_ex(&g_R, &g_M, g_n_ctx, Q35_KV_Q8_0, kvb, getenv("QWEN_KV_SPILL"), 4096)){
        fprintf(stderr, "kv init failed\n"); return 1; }
    PHASE("kv store");
    /* THE ARENA, NOT THE BUDGET. These were the same line once, and they were not the same
     * number: the arena used to be clamped to the chunk index of the INITIAL context, so this
     * printed 23091 tokens while 8192 was what it held. A figure that comes from the store
     * itself cannot drift from it again. */
    { const KvTier *T = &g_R.kv;
      const long long hot_tok = (long long)T->ram_chunks*T->chunk/(T->n_layer ? T->n_layer : 1);
      fprintf(stderr, "  context: %d initial, UNBOUNDED — grows on demand and stops only when\n"
                      "           MemAvailable reaches the %.2f GiB floor. Hot KV tier %.2f GiB"
                      " (%d chunks, %lld tokens%s), the rest spills to %s\n",
              g_n_ctx, q35_mem_floor()/1073741824.0,
              T->arena_bytes/1073741824.0, T->ram_chunks, hot_tok,
              getenv("QWEN_KV_RAM_GB") ? ", pinned by QWEN_KV_RAM_GB" : "",
              getenv("QWEN_KV_SPILL") ? getenv("QWEN_KV_SPILL") : "the spill tier"); }

#ifndef QWEN_NO_CUDA
    if(!cpu_only){
        /* After a switch the previous process's VRAM may still be draining — the driver frees a
         * context lazily — so the first upload can fail for room that is about to exist. Retry
         * briefly, but only when a switch actually just happened. */
        int tries = getenv("QWEN_SELECT_MODEL") ? 15 : 1, ok = 0;
        for(int a = 0; a < tries; a++){
            if(q35cu_model_init(&g_G, &g_M, g_n_ctx, g_R.kv.fmt)){ ok = 1; break; }
            if(a+1 < tries){ fprintf(stderr, "  VRAM busy (%s) — retry %d/%d\n", q35cu_error(), a+1, tries); sleep(1); }
        }
        if(ok){
            q35cu_align_win(&g_G, g_R.kv.chunk);
            q35cu_mark_resident_elsewhere(&g_G);
            g_gpu = 1;
            PHASE("vram upload");
        } else fprintf(stderr, "  gpu unavailable (%s) — CPU only\n", q35cu_error());
    }
#else
    (void)cpu_only;
#endif
    Q35Resident RES;
    q35_reside_anon(&g_M, &RES,
#ifndef QWEN_NO_CUDA
        g_gpu ? Q35_RES_FFN : Q35_RES_ALL
#else
        Q35_RES_ALL
#endif
    );
    PHASE("ram residency");
#ifndef QWEN_NO_CUDA
    if(g_gpu){
        q35cu_pin_weights(&g_G, &g_M);
        PHASE("pin for DMA");
        q35cu_report(&g_G, stderr);
        q35cu_state_reset(&g_G);
        if(g_M.c.n_layer_mtp > 0 && q35cu_batch_init(&g_Z, &g_G, &g_R, 8) && q35_mtp_init(&g_P, &g_R)) g_spec = 1;
        else if(g_M.c.n_layer_mtp == 0) fprintf(stderr, "  no MTP head in this model — plain decode\n");
        /* A switch to halve the problem with, and an escape hatch while a fault in the
         * speculative path is being found: decoding without it is slower and correct. */
        if(getenv("QWEN_NO_SPEC")){ g_spec = 0; fprintf(stderr, "  speculation OFF (QWEN_NO_SPEC)\n"); }
        PHASE("batch + mtp");
    }
#endif
    q35_state_reset(&g_R);
    /* The cache holds the recurrent state at spaced positions. 6 slots x 150 MiB by default;
     * it shrinks itself rather than push the machine under the memory floor. */
#ifndef QWEN_NO_CUDA
    q35_cache_init(&g_C, &g_R, g_gpu ? &g_G : NULL, 6, 512);
#else
    q35_cache_init(&g_C, &g_R, NULL, 6, 512);
#endif
    PHASE("prefix cache");
    q35_cache_report(&g_C, stderr);
    /* TWO NUMBERS, NOT ONE. Summing them gives 84 GB/s at 5.3 tok/s, which reads like the
     * machine is doing 84 — and there is no bus here that does 84. The FFN half crosses the
     * DRAM controller (whether the CPU reads it in place or the GPU pulls it over PCIe) and
     * that is the half with the 64.8 GB/s ceiling this engine is built against. The rest is
     * already on the card at 283.7. One combined figure hides exactly the thing worth
     * watching. */
    { Q35Bytes B2; q35_bytes(&g_M, &B2);
      g_bytes_dram = B2.ffn + B2.embd / (g_M.c.vocab > 0 ? g_M.c.vocab : 1);
      g_bytes_vram = 0;
#ifndef QWEN_NO_CUDA
      if(g_gpu){ g_bytes_vram = g_G.vram_weights; g_prefill_sb = g_Z.S_max > 0 ? g_Z.S_max : 8; }
      else
#endif
      g_bytes_dram += B2.gdn + B2.attn + B2.out + B2.norms;
      g_bytes_per_tok = g_bytes_dram + g_bytes_vram;
      fprintf(stderr, "  per generated token: %.2f GiB from DRAM/PCIe (ceiling 64.8 GB/s), "
                      "%.2f GiB from VRAM\n",
              g_bytes_dram/1073741824.0, g_bytes_vram/1073741824.0); }

    if(!www){
        static char guess[1024];
        const char *slash = strrchr(argv[0], '/');
        snprintf(guess, sizeof guess, "%.*swebui.html",
                 slash ? (int)(slash - argv[0] + 1) : 0, slash ? argv[0] : "");
        www = guess;
    }
    g_ui = read_file(www, &g_ui_len);
    if(!g_ui) fprintf(stderr, "  note: no web UI at %s — the API still works\n", www);

    /* SOCK_CLOEXEC, not a bare socket(). Everything this server forks — a shell command, a
     * whole Firefox — inherits its descriptors otherwise, and a child holding the listening
     * socket keeps the port bound long after the server is gone. */
    int srv;
    const char *inh = getenv("QWEN_LISTEN_FD");
    if(inh && *inh){
        srv = atoi(inh);                                     /* handed across a model-switch re-exec */
        fcntl(srv, F_SETFD, FD_CLOEXEC);                     /* re-arm the protection cleared for exec */
        fprintf(stderr, "  listening socket inherited across model switch (fd %d)\n", srv);
    } else {
        srv = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        int one = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
        sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)port);
        sa.sin_addr.s_addr = !strcmp(host, "0.0.0.0") ? INADDR_ANY : inet_addr(host);
        if(bind(srv, (struct sockaddr*)&sa, sizeof sa) < 0){
            fprintf(stderr, "bind %s:%d failed: %s\n", host, port, strerror(errno)); return 1; }
        listen(srv, 16);
    }
    g_srv_fd = srv;

    fprintf(stderr,
        "\n  qwen35_server ready\n"
        "    model    %s  (%d layers, ctx from %d and growing, %s)\n"
        "    web UI   http://%s:%d/\n"
        "    API      http://%s:%d/v1   (OpenAI-compatible)\n"
        "    tools    %s%s%s\n"
        "    opencode  point a provider at the /v1 URL above; any api key works%s\n\n",
        g_model_name, c->n_layer, g_n_ctx,
#ifndef QWEN_NO_CUDA
        g_spec ? "gpu + speculative" : (g_gpu ? "gpu" : "cpu"),
#else
        "cpu",
#endif
        host, port, host, port, h_mode_name(),
        g_proot[0] ? ", projects under " : " in ", g_proot[0] ? g_proot : g_root,
        g_api_key ? " except this one, which is required" : "");

    { pthread_t eng; pthread_create(&eng, NULL, engine_thread, NULL); pthread_detach(eng); }

    while(!g_abort){
        struct sockaddr_in ca; socklen_t cl = sizeof ca;
        const int fd = accept4(srv, (struct sockaddr*)&ca, &cl, SOCK_CLOEXEC);
        if(fd < 0){ if(errno == EINTR) continue; break; }
        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof nodelay);
        pthread_t th;
        if(pthread_create(&th, NULL, conn_thread, (void*)(intptr_t)fd) == 0)
            pthread_detach(th);
        else { if(!serve(fd)) close(fd); }
    }
    fprintf(stderr, "\n  shutting down\n");
    web_stop();
    close(srv);
    return 0;
}
