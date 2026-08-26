/* models.h — where model files live.
 *
 * The path to a GGUF was hardcoded in nine places: the Makefile, serve.sh twice, and five
 * tests, each carrying one developer's home directory. That is fine until anyone else builds
 * it. Resolution order, first hit wins:
 *
 *   1. $NIGHTJAR_MODELS   — explicit, for anyone who keeps them elsewhere
 *   2. /models            — the intended location: one directory, outside any checkout,
 *                          shared by every engine in the runtime
 *   3. $HOME/Models       — fallback so an existing developer setup keeps working
 *
 * nj_model_path() joins that root with a relative name. An argument that is already absolute
 * is returned untouched, so `qwen35_run /elsewhere/x.gguf` still does what it says.
 */
#ifndef NIGHTJAR_MODELS_H
#define NIGHTJAR_MODELS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *nj_models_root(void){
    static char root[1024];
    if(root[0]) return root;
    const char *e = getenv("NIGHTJAR_MODELS");
    if(e && *e){ snprintf(root, sizeof root, "%s", e); return root; }
    struct stat st;
    if(stat("/models", &st) == 0 && S_ISDIR(st.st_mode)){ snprintf(root, sizeof root, "/models"); return root; }
    const char *h = getenv("HOME");
    snprintf(root, sizeof root, "%s/Models", h ? h : ".");
    return root;
}

/* Joins the root with `rel`, or passes an absolute path straight through. The buffer is
 * per-slot static so a couple of paths can be live at once (two models in one registry). */
static const char *nj_model_path(const char *rel){
    static char buf[4][2048];
    static int slot = 0;
    if(!rel || !*rel) return rel;
    if(rel[0] == '/') return rel;
    char *b = buf[slot++ & 3];
    snprintf(b, 2048, "%s/%s", nj_models_root(), rel);
    return b;
}


/* The trained MTP draft head, if one is installed beside the model.
 *
 * Ornith-1.5 ships mtp.* tensors that are randomly initialised rather than trained (~13%
 * acceptance, i.e. chance), so a usable head has to come from elsewhere -- see
 * THIRD-PARTY-NOTICES.md. Convention: <model dir>/mtp.gguf, or $NIGHTJAR_MTP for an explicit
 * one. Returns NULL when absent, which is not an error: the engine simply drafts differently
 * or not at all. */
static const char *nj_mtp_path(const char *model_path){
    static char buf[2048];
    const char *e = getenv("NIGHTJAR_MTP");
    if(e && *e) return e;
    if(!model_path) return NULL;
    const char *slash = strrchr(model_path, '/');
    const size_t dirlen = slash ? (size_t)(slash - model_path) : 1;
    snprintf(buf, sizeof buf, "%.*s/mtp.gguf", (int)dirlen, slash ? model_path : ".");
    struct stat st;
    return (stat(buf, &st) == 0 && S_ISREG(st.st_mode)) ? buf : NULL;
}

#endif
