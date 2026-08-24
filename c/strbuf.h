/* strbuf.h — the growable byte buffer everything in the server appends to, and the clock.
 *
 * It lives in its own file for one reason: the tool harness and its test must escape JSON with
 * the SAME function the HTTP layer uses. A second copy in the test would be a test that passes
 * while the server ships a different escaper. */
#ifndef QWEN_STRBUF_H
#define QWEN_STRBUF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

/* ---------------- small dynamic string ---------------- */

typedef struct { char *p; size_t n, cap; } Str;
static void s_reserve(Str *s, size_t extra){
    if(s->n + extra + 1 <= s->cap) return;
    size_t c = s->cap ? s->cap : 256;
    while(c < s->n + extra + 1) c *= 2;
    s->p = (char*)realloc(s->p, c); s->cap = c;
}
static void s_add(Str *s, const char *b, size_t n){
    s_reserve(s, n); memcpy(s->p + s->n, b, n); s->n += n; s->p[s->n] = 0;
}
static void s_cat(Str *s, const char *z){ s_add(s, z, strlen(z)); }
static void s_fmt(Str *s, const char *fmt, ...){
    va_list ap; va_start(ap, fmt);
    char tmp[4096];
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if(n > 0) s_add(s, tmp, (size_t)(n < (int)sizeof tmp ? n : (int)sizeof tmp - 1));
}
static void s_free(Str *s){ free(s->p); s->p = NULL; s->n = s->cap = 0; }

/* JSON string escaping. Control characters MUST be escaped or the client's parser rejects the
 * whole SSE frame — and a model emitting a raw newline inside a delta is the normal case. */
static void s_json(Str *s, const char *z, size_t n){
    s_reserve(s, n*6 + 2);
    s_add(s, "\"", 1);
    for(size_t i = 0; i < n; i++){
        const unsigned char c = (unsigned char)z[i];
        switch(c){
            case '"':  s_cat(s, "\\\""); break;
            case '\\': s_cat(s, "\\\\"); break;
            case '\n': s_cat(s, "\\n");  break;
            case '\r': s_cat(s, "\\r");  break;
            case '\t': s_cat(s, "\\t");  break;
            case '\b': s_cat(s, "\\b");  break;
            case '\f': s_cat(s, "\\f");  break;
            default:
                if(c < 0x20) s_fmt(s, "\\u%04x", c);
                else         s_add(s, (const char*)&c, 1);
        }
    }
    s_add(s, "\"", 1);
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }


#endif /* QWEN_STRBUF_H */
