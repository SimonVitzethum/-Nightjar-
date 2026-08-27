/* nj_arch.h — which engine a GGUF needs, decided before anything is loaded.
 *
 * The runtime holds several models and switches between them by re-exec, so exactly one is
 * resident at any moment and two engines never share an address space. That makes the
 * dispatch a startup decision rather than an object model: read general.architecture out of
 * the header, pick an engine, done. No registry of vtables, no dynamic loading.
 *
 * Reading the header is cheap -- gguf_open maps the file and parses metadata without touching
 * tensor data -- so this costs a few milliseconds against a 21 GB file.
 */
#ifndef NIGHTJAR_ARCH_H
#define NIGHTJAR_ARCH_H

#include <string.h>
#include "gguf.h"

typedef enum {
    NJ_ARCH_UNKNOWN = 0,
    NJ_ARCH_QWEN35,        /* dense hybrid GDN/attention  — Qwen 3.8 27B */
    NJ_ARCH_QWEN35MOE,     /* the same trunk, sparse FFN  — Ornith 1.5 35B-A3B */
    NJ_ARCH_QWEN4EXP,      /* Qwen4 preview               — Qwen3.8 Flash Next, 512x56B */
    NJ_ARCH_GLM5,          /* GLM 5.2 744B                — Colibri engine */
    NJ_ARCH_KIMI,          /* Kimi K3 2.8T                — Colibri engine */
    NJ_ARCH_DSV4           /* DeepSeek V4 Flash 284B      — Colibri engine */
} NjArch;

static const struct { const char *key; NjArch arch; const char *engine; } NJ_ARCHS[] = {
    { "qwen35",    NJ_ARCH_QWEN35,    "nightjar" },
    { "qwen35moe", NJ_ARCH_QWEN35MOE, "nightjar" },
    { "qwen4exp",  NJ_ARCH_QWEN4EXP,  "nightjar" },
    { "glm5",      NJ_ARCH_GLM5,      "colibri"  },
    { "glm5moe",   NJ_ARCH_GLM5,      "colibri"  },
    { "kimi",      NJ_ARCH_KIMI,      "colibri"  },
    { "kimik3",    NJ_ARCH_KIMI,      "colibri"  },
    { "deepseek4", NJ_ARCH_DSV4,      "colibri"  },
    { "dsv4",      NJ_ARCH_DSV4,      "colibri"  },
};

static NjArch nj_arch_from_name(const char *arch){
    if(!arch) return NJ_ARCH_UNKNOWN;
    for(size_t i = 0; i < sizeof NJ_ARCHS/sizeof *NJ_ARCHS; i++)
        if(!strcmp(arch, NJ_ARCHS[i].key)) return NJ_ARCHS[i].arch;
    return NJ_ARCH_UNKNOWN;
}
static const char *nj_engine_for(NjArch a){
    for(size_t i = 0; i < sizeof NJ_ARCHS/sizeof *NJ_ARCHS; i++)
        if(NJ_ARCHS[i].arch == a) return NJ_ARCHS[i].engine;
    return "unknown";
}

/* Peek at a GGUF's architecture without loading it. Writes the raw string into `name` (which
 * is what an unsupported model should be REPORTED as -- "architecture glm5 needs the colibri
 * engine" is actionable; "failed to load" is not). */
static NjArch nj_arch_of(const char *path, char *name, size_t cap){
    if(name && cap) name[0] = 0;
    gguf_model g;
    if(!gguf_open(&g, path)) return NJ_ARCH_UNKNOWN;
    const char *a = gguf_str(&g, "general.architecture", "");
    if(name && cap) snprintf(name, cap, "%s", a);
    const NjArch r = nj_arch_from_name(a);
    gguf_close(&g);
    return r;
}

#endif
