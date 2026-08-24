/* chatstore.h — one conversation per project, on disk.
 *
 * WHY THE SERVER KEEPS IT AND NOT THE BROWSER
 *
 * localStorage would have been three lines. But a conversation here is not a chat log, it is
 * the thing the KV cache is keyed on: the engine's checkpoints hold the recurrent state for a
 * specific token prefix, and that prefix IS this message list. If the history lives only in one
 * browser tab then a reload, a second tab, or a restart produces a different prefix and throws
 * away work that costs 110 seconds a thousand tokens to rebuild. Keeping it next to the engine
 * is what makes "come back to this project tomorrow" mean something.
 *
 * WHERE IT DOES NOT LIVE: inside the project directory. The agent reads and writes in there,
 * and a .qwen/ folder full of its own transcripts is both noise in every `list_dir` and
 * something it can quietly corrupt. Chats live beside the projects, not in them.
 *
 * The server does not parse messages. It stores the JSON array the page sends and hands it
 * back — the message SHAPE is the chat template's business, in one place, and a second
 * interpretation here would be a second thing to keep correct.
 */
#ifndef QWEN_CHATSTORE_H
#define QWEN_CHATSTORE_H

static char g_chat_root[PATH_MAX] = "";

/* Same whitelist rule as project names, for the same reason: an id becomes a path component,
 * and anything that is not plainly a name is refused rather than escaped. */
static int cs_id_ok(const char *n){
    if(!n || !*n || strlen(n) > 64) return 0;
    if(n[0] == '.') return 0;
    for(const char *p = n; *p; p++)
        if(!(isalnum((unsigned char)*p) || *p == '.' || *p == '_' || *p == '-')) return 0;
    return 1;
}

static int cs_dir(const char *project, char *out, size_t cap, int make){
    if(!g_chat_root[0]) return 0;
    const char *p = (project && *project) ? project : "_default";
    if(!cs_id_ok(p)) return 0;
    if(snprintf(out, cap, "%s/%s", g_chat_root, p) >= (int)cap) return 0;
    if(make) mkdir(out, 0700);
    return 1;
}
static int cs_path(const char *project, const char *id, char *out, size_t cap, int make){
    char dir[PATH_MAX];
    if(!cs_id_ok(id)) return 0;
    if(!cs_dir(project, dir, sizeof dir, make)) return 0;
    return snprintf(out, cap, "%s/%s.json", dir, id) < (int)cap;
}

/* The listing carries enough to draw a sidebar and no more: reading every transcript to count
 * its messages would mean parsing megabytes to render a list. Title and counts are stored at
 * save time, in the head of the file, and read back with a bounded read. */
typedef struct { char id[80]; char title[200]; long updated; int n; } CsRow;

static int cs_row(const char *path, CsRow *r){
    FILE *f = fopen(path, "rb");
    if(!f) return 0;
    char head[2048];
    const size_t n = fread(head, 1, sizeof head - 1, f);
    fclose(f);
    head[n] = 0;
    char *arena = NULL;
    /* The head is a truncated document, so it will not parse. The three fields are written
     * first, deliberately, and pulled out by hand — a full parse of a 2 MB transcript per row
     * would make opening the sidebar cost more than the answer it is listing. */
    const char *t = strstr(head, "\"title\":\"");
    r->title[0] = 0;
    if(t){
        t += 9;
        size_t k = 0;
        while(*t && *t != '"' && k < sizeof r->title - 1){
            if(*t == '\\' && t[1]){ t++; if(*t=='n') r->title[k++]=' '; else r->title[k++]=*t; }
            else r->title[k++] = *t;
            t++;
        }
        r->title[k] = 0;
    }
    const char *u = strstr(head, "\"updated\":");
    r->updated = u ? strtol(u + 10, NULL, 10) : 0;
    const char *c = strstr(head, "\"n\":");
    r->n = c ? (int)strtol(c + 4, NULL, 10) : 0;
    free(arena);
    return 1;
}

static int cs_cmp(const void *a, const void *b){
    const CsRow *x = (const CsRow*)a, *y = (const CsRow*)b;
    return x->updated < y->updated ? 1 : x->updated > y->updated ? -1 : 0;   /* newest first */
}

static void cs_list_json(Str *o, const char *project){
    char dir[PATH_MAX];
    s_cat(o, "{\"data\":[");
    if(cs_dir(project, dir, sizeof dir, 1)){
        DIR *d = opendir(dir);
        CsRow rows[256]; int n = 0;
        if(d){
            struct dirent *e;
            while((e = readdir(d)) && n < 256){
                const size_t L = strlen(e->d_name);
                if(L < 6 || strcmp(e->d_name + L - 5, ".json")) continue;
                char full[PATH_MAX];
                if(snprintf(full, sizeof full, "%s/%s", dir, e->d_name) >= (int)sizeof full) continue;
                if(!cs_row(full, &rows[n])) continue;
                snprintf(rows[n].id, sizeof rows[n].id, "%.*s", (int)(L - 5), e->d_name);
                n++;
            }
            closedir(d);
        }
        qsort(rows, (size_t)n, sizeof *rows, cs_cmp);
        for(int i = 0; i < n; i++){
            if(i) s_cat(o, ",");
            s_cat(o, "{\"id\":");
            s_json(o, rows[i].id, strlen(rows[i].id));
            s_cat(o, ",\"title\":");
            s_json(o, rows[i].title, strlen(rows[i].title));
            s_fmt(o, ",\"updated\":%ld,\"n\":%d}", rows[i].updated, rows[i].n);
        }
    }
    s_cat(o, "]}");
}

static int cs_get(Str *o, const char *project, const char *id){
    char path[PATH_MAX];
    if(!cs_path(project, id, path, sizeof path, 0)) return 0;
    FILE *f = fopen(path, "rb");
    if(!f) return 0;
    char buf[65536];
    size_t k;
    while((k = fread(buf, 1, sizeof buf, f)) > 0) s_add(o, buf, k);
    fclose(f);
    return 1;
}

/* Written to a temporary file, FSYNCED, and then renamed. A conversation is overwritten on every
 * turn, and a crash halfway through a 2 MB write would otherwise leave a truncated JSON
 * document — which loses not the last turn but the whole history, including the prefix the KV
 * cache is keyed on.
 *
 * rename() is atomic for VISIBILITY and not for DURABILITY, and on ext4 the difference is easy
 * to miss. Without the fsync on the temporary file BEFORE the rename, a power cut can leave a
 * directory entry pointing at an inode whose data blocks were never written — the name is
 * there, the content is zeroes. A process crash is survived by the rename alone; a machine
 * crash is not. The directory itself is synced afterwards, because the entry that now points at
 * good data is only a promise until it too is on disk.
 *
 * Same argument as the raw write(2) in q35_oom: a path that exists to survive a failure must not
 * depend on machinery that is failing at that moment. */
static int cs_save(const char *project, const char *id, const char *title,
                   int n_msgs, const char *messages_json){
    char path[PATH_MAX], tmp[PATH_MAX];
    if(!cs_path(project, id, path, sizeof path, 1)) return 0;
    if(snprintf(tmp, sizeof tmp, "%s.tmp", path) >= (int)sizeof tmp) return 0;
    FILE *f = fopen(tmp, "wb");
    if(!f) return 0;
    Str head = {0};
    s_cat(&head, "{\"id\":");
    s_json(&head, id, strlen(id));
    s_cat(&head, ",\"title\":");
    s_json(&head, title ? title : "", title ? strlen(title) : 0);
    s_fmt(&head, ",\"updated\":%ld,\"n\":%d,\"project\":", (long)time(NULL), n_msgs);
    s_json(&head, project ? project : "", project ? strlen(project) : 0);
    s_cat(&head, ",\"messages\":");
    int ok = fwrite(head.p, 1, head.n, f) == head.n
          && fputs(messages_json ? messages_json : "[]", f) >= 0
          && fputs("}", f) >= 0;
    s_free(&head);
    if(ok) ok = fflush(f) == 0 && fsync(fileno(f)) == 0;
    fclose(f);
    if(!ok || rename(tmp, path) != 0){ unlink(tmp); return 0; }
    { char dir[PATH_MAX];
      snprintf(dir, sizeof dir, "%s", path);
      char *slash = strrchr(dir, '/');
      if(slash){ *slash = 0;
                 const int dfd = open(dir, O_RDONLY | O_DIRECTORY);
                 if(dfd >= 0){ fsync(dfd); close(dfd); } } }
    return 1;
}

static int cs_delete(const char *project, const char *id){
    char path[PATH_MAX];
    if(!cs_path(project, id, path, sizeof path, 0)) return 0;
    return unlink(path) == 0;
}

#endif /* QWEN_CHATSTORE_H */
