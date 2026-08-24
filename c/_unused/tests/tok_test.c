/* tok_test.c — the tokenizer, against llama.cpp.
 *
 * A tokenizer that is 99% right is not 99% useful: one wrong split shifts every token after
 * it, and the model answers a question nobody asked. The reference is hard and specific —
 * llama.cpp on this exact GGUF turned "Hallo" into exactly one token, 77451.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../gguf.h"
#include "../tok_gguf.h"

static int fails = 0;
static void ck(int ok, const char *msg){
    printf("  %s %s\n", ok ? "ok  " : "FAIL", msg);
    if(!ok) fails++;
}

static void show(const char *label, const int *ids, int n, Tok *T){
    printf("  %-24s [", label);
    for(int i=0;i<n;i++) printf("%s%d", i?", ":"", ids[i]);
    printf("]  ->  ");
    char buf[4096];
    int m = tok_decode(T, ids, n, buf, sizeof(buf));
    fwrite(buf, 1, m, stdout);
    printf("\n");
}

int main(int argc, char **argv){
    if(argc < 2){ printf("usage: %s <shard1.gguf>\n", argv[0]); return 2; }
    gguf_model m;
    if(!gguf_open(&m, argv[1])){ printf("open failed\n"); return 1; }

    Tok T; TokIds ids;
    if(!tok_from_gguf(&T, &m, &ids)){ printf("tokenizer build failed\n"); return 1; }
    printf("\n  bos %d  eos %d  eot %d  eom %d\n\n", ids.bos, ids.eos, ids.eot, ids.eom);

    int out[512];

    /* the one hard reference we have */
    int n = tok_encode(&T, "Hallo", 5, out, 512);
    show("\"Hallo\"", out, n, &T);
    ck(n == 1 && out[0] == 77451, "matches llama.cpp: one token, id 77451");

    /* round trips */
    const char *cases[] = {
        "def add(a, b):\n    return a + b\n",
        "Der Fluss fließt schnell.",
        "  leading   spaces\tand\ttabs",
        "int x = 42; // Kommentar",
        "\xe4\xbd\xa0\xe5\xa5\xbd",                 /* 你好 */
    };
    printf("\n  round trips (encode -> decode must be byte-identical)\n");
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
        int len = (int)strlen(cases[i]);
        n = tok_encode(&T, cases[i], len, out, 512);
        char back[4096];
        int bl = tok_decode(&T, out, n, back, sizeof(back));
        int ok = (bl == len) && memcmp(back, cases[i], len) == 0;
        printf("  %s %2d tokens  %.40s%s\n", ok ? "ok  " : "FAIL", n,
               cases[i], len > 40 ? "..." : "");
        if(!ok){
            fails++;
            printf("       got back: %.*s\n", bl, back);
        }
    }

    /* control tokens must survive whole — this is the one that breaks chat silently */
    printf("\n  control tokens\n");
    const char *chat = "<|user|>Hallo<|assistant|>";
    n = tok_encode(&T, chat, (int)strlen(chat), out, 512);
    show("chat markers", out, n, &T);
    int has_user = 0, has_asst = 0;
    for(int i=0;i<n;i++){
        const char *s = T.id2str[out[i]];
        if(s && !strcmp(s, "<|user|>"))      has_user = 1;
        if(s && !strcmp(s, "<|assistant|>")) has_asst = 1;
    }
    ck(has_user && has_asst, "<|user|> and <|assistant|> stay single tokens (not BPE'd apart)");

    /* the FIM markers a coding assistant depends on */
    const char *fimstr = "<|code_prefix|>x = <|code_suffix|>\n<|code_middle|>";
    n = tok_encode(&T, fimstr, (int)strlen(fimstr), out, 512);
    int fim = 0;
    for(int i=0;i<n;i++){
        const char *s = T.id2str[out[i]];
        if(s && (!strcmp(s,"<|code_prefix|>") || !strcmp(s,"<|code_suffix|>") ||
                 !strcmp(s,"<|code_middle|>"))) fim++;
    }
    ck(fim == 3, "the three FIM markers survive as single tokens");

    const char *tc = "<tool_call>get_weather<arg_key>city</arg_key></tool_call>";
    n = tok_encode(&T, tc, (int)strlen(tc), out, 512);
    int tool = 0;
    for(int i=0;i<n;i++){
        const char *s2 = T.id2str[out[i]];
        if(s2 && (!strcmp(s2,"<tool_call>") || !strcmp(s2,"</tool_call>") ||
                  !strcmp(s2,"<arg_key>")   || !strcmp(s2,"</arg_key>"))) tool++;
    }
    show("tool call", out, n, &T);
    ck(tool == 4, "<tool_call>/<arg_key> survive too (USER_DEFINED, not CONTROL)");

    printf("\n%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
