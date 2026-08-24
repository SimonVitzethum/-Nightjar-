#ifndef COLIBRI_TOK_GGUF_H
#define COLIBRI_TOK_GGUF_H
/* tok_gguf.h — build colibri's tokenizer straight out of a GGUF.
 *
 * Nothing new is invented here. colibri's tok.h already implements exactly what GLM-5.2
 * needs: byte-level BPE with the cl100k pre-tokenizer regex, which is character for
 * character the regex llama.cpp uses for its "glm4" pre-tokenizer type:
 *
 *   (?:'[sS]|'[tT]|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|...
 *
 * All that was missing is the source of the vocabulary. tok_load() reads colibri's own
 * tokenizer.json; this reads the same three arrays out of the GGUF metadata instead:
 *
 *   tokenizer.ggml.tokens      154880 strings, already byte-level encoded (Ġ for space)
 *   tokenizer.ggml.merges      321649 "left right" pairs, IN RANK ORDER — the index IS the
 *                              rank, which is the whole of BPE: always apply the lowest-
 *                              ranked merge that still applies
 *   tokenizer.ggml.token_type  1 = normal, 3 = control/special
 *
 * The one thing that must not be got wrong: CONTROL tokens (type 3) are matched literally
 * against the input and never split by BPE. If <|user|> went through the merge table it
 * would come out as a fistful of ordinary pieces, the chat template would silently stop
 * working, and the model would answer as if there were no roles at all.
 */
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "tok.h"

enum { GGUF_TOKTYPE_NORMAL = 1, GGUF_TOKTYPE_CONTROL = 3, GGUF_TOKTYPE_USER_DEFINED = 4 };

/* Both CONTROL and USER_DEFINED must be matched literally and never fed to BPE. GLM-5.2
 * splits them in a way that is easy to miss: the chat and FIM markers (<|user|>,
 * <|code_prefix|>, ...) are CONTROL, but <tool_call>, <think> and the <arg_key>/<arg_value>
 * pair are USER_DEFINED. Handle only CONTROL and BPE quietly shreds exactly the markers
 * tool calling and reasoning hang on — the model keeps talking, the tool calls just stop
 * parsing. */
static inline int tok_is_special(int type){
    return type == GGUF_TOKTYPE_CONTROL || type == GGUF_TOKTYPE_USER_DEFINED;
}

/* tok.h's hashmap masks with (cap-1), so cap MUST be a power of two — pass it 309760 and
 * the probe sequence wraps wrong and never terminates. Nothing in hm_init checks this. */
static int tok_pow2(uint64_t want){
    int cap = 16;
    while((uint64_t)cap < want) cap <<= 1;
    return cap;
}

typedef struct {
    int bos, eos, eot, eom, pad, unk;
} TokIds;

/* tokenizer.ggml.pre names the pre-tokenizer variant. "qwen35" differs from cl100k in
 * exactly two alternatives (see pretok_chunk): the letter run is [\p{L}\p{M}]+ rather than
 * \p{L}+, and digits are taken ONE at a time rather than in groups of up to three. Both are
 * silent when wrong -- the text still tokenizes, into pieces the model never saw. */
static void tok_set_pre(const gguf_model *m){
    const gguf_kv *k = gguf_kv_get(m, "tokenizer.ggml.pre");
    g_tok_qwen35 = (k && k->s && !strcmp(k->s, "qwen35"));
}

static int tok_from_gguf(Tok *T, const gguf_model *m, TokIds *ids){
    memset(T, 0, sizeof(*T));

    char **toks = NULL, **merges = NULL;
    const int32_t *types = NULL;
    uint64_t n_tok = gguf_strs(m, "tokenizer.ggml.tokens", &toks);
    uint64_t n_mrg = gguf_strs(m, "tokenizer.ggml.merges", &merges);
    uint64_t n_typ = gguf_i32s(m, "tokenizer.ggml.token_type", &types);
    if(!n_tok || !toks){ fprintf(stderr, "tok: no vocabulary in the GGUF\n"); return 0; }
    if(n_typ && n_typ != n_tok){ fprintf(stderr, "tok: token_type length mismatch\n"); return 0; }

    tk_build_bytemap(T);

    T->n_ids    = (int)n_tok;
    T->id2str   = (char**)calloc(n_tok, sizeof(char*));
    T->id_added = (int*)  calloc(n_tok, sizeof(int));
    if(!T->id2str || !T->id_added) return 0;

    hm_init(&T->vocab, tok_pow2(n_tok*2));
    int n_ctrl = 0;
    for(uint64_t i = 0; i < n_tok; i++){
        char *s = toks[i];
        T->id2str[i] = s;                                  /* owned by the gguf_model */
        hm_put(&T->vocab, s, (int)strlen(s), (int)i);
        if(types && tok_is_special(types[i])){
            T->id_added[i] = 1;                            /* emit literally, never merge */
            n_ctrl++;
        }
    }

    /* control tokens, longest first: <|end_of_image|> must win over <|end... */
    T->nsp = n_ctrl;
    T->sp  = (Special*)calloc(n_ctrl ? n_ctrl : 1, sizeof(Special));
    for(uint64_t i = 0, j = 0; i < n_tok; i++){
        if(!T->id_added[i]) continue;
        T->sp[j].str = T->id2str[i];
        T->sp[j].len = (int)strlen(T->id2str[i]);
        T->sp[j].id  = (int)i;
        j++;
    }
    qsort(T->sp, T->nsp, sizeof(Special), cmp_sp_len);

    /* merges: "left right" -> rank. The array's ORDER is the rank; BPE always applies the
     * lowest-ranked applicable merge, so getting this order wrong tokenizes everything
     * plausibly and nothing correctly. */
    hm_init(&T->merges, tok_pow2(n_mrg*2));
    for(uint64_t i = 0; i < n_mrg; i++){
        char *sp = strchr(merges[i], ' ');
        if(!sp) continue;
        int ll = (int)(sp - merges[i]);
        int rl = (int)strlen(sp + 1);
        char *key = (char*)malloc((size_t)ll + 1 + rl + 1);
        memcpy(key, merges[i], ll);
        key[ll] = 0;                                        /* tok.h keys on "left\0right" */
        memcpy(key + ll + 1, sp + 1, rl);
        key[ll + 1 + rl] = 0;
        hm_put(&T->merges, key, ll + 1 + rl, (int)i);
    }

    if(ids){
        ids->bos = (int)gguf_u64(m, "tokenizer.ggml.bos_token_id", -1);
        ids->eos = (int)gguf_u64(m, "tokenizer.ggml.eos_token_id", -1);
        ids->eot = (int)gguf_u64(m, "tokenizer.ggml.eot_token_id", -1);
        ids->eom = (int)gguf_u64(m, "tokenizer.ggml.eom_token_id", -1);
        ids->pad = (int)gguf_u64(m, "tokenizer.ggml.padding_token_id", -1);
        ids->unk = (int)gguf_u64(m, "tokenizer.ggml.unknown_token_id", -1);
    }
    fprintf(stderr, "[tok] %llu tokens (%d special), %llu merges\n",
            (unsigned long long)n_tok, n_ctrl, (unsigned long long)n_mrg);
    return 1;
}

#endif /* COLIBRI_TOK_GGUF_H */
