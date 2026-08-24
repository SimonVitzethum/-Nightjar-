/* harness.h — the tool half of the agent loop: ONE registry, ONE gate, ONE dispatch.
 *
 * WHY IT IS SHAPED LIKE THIS
 *
 * The design comes from the same place every other decision in this engine does — the failure
 * mode, not the feature list. Three properties, in order of how much they matter:
 *
 *   1. NO SECOND DOOR. Every tool the model can reach goes through h_dispatch(), and every
 *      call through h_dispatch() passes h_policy() first. The browser cannot invent a tool,
 *      cannot pass one that is not in the registry, and cannot skip the gate — it only ever
 *      sends a name and an argument object. The schemas the model sees are generated FROM
 *      this registry (GET /v1/tools), so the prompt and the executor cannot drift apart.
 *      A hand-written schema list in the page would be exactly the kind of silent
 *      duplicate-truth bug that cost this project a day already.
 *
 *   2. FAIL CLOSED WHEN THERE IS NOBODY TO ASK. A policy decision of ASK is not "run it and
 *      tell someone"; it is a refusal that carries a question. The caller must come back with
 *      approved=1. A headless client that ignores the question gets nothing done — which is
 *      the correct behaviour, not a limitation.
 *
 *   3. CONFINEMENT IS RESOLVED, NOT TEXTUAL. Every path argument goes through realpath()
 *      before it is compared to the workspace root, so "../../etc/passwd", a symlink out of
 *      the tree, and /proc/self/mem all land outside and are refused. A strncmp on the
 *      argument as typed would pass all three.
 *
 * OUTPUT IS A BUDGET, NOT A STREAM. This engine prefills at ~10 tok/s. A tool that returns
 * 400 KiB of build log costs the next turn two minutes before the model says a word. So every
 * result is truncated to g_tmax bytes at line boundaries, head and tail, with the gap named.
 * Being unable to see the middle of a log is cheap; waiting two minutes to see all of it is
 * not.
 *
 * DEPENDS ON, and must be included after: Str + s_add/s_cat/s_fmt/s_json/s_free, j_emit(),
 * json.h, and now(). It is a part of qwen35_server.c that happens to live in its own file.
 */
#ifndef QWEN_HARNESS_H
#define QWEN_HARNESS_H

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fnmatch.h>
#include <regex.h>
#include <limits.h>
#include <poll.h>
#include <sys/wait.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef enum { TM_OFF = 0, TM_RO, TM_WS, TM_FULL } ToolMode;
typedef enum { TCL_READ = 0, TCL_WRITE, TCL_EXEC } ToolClass;
typedef enum { TD_ALLOW = 0, TD_ASK, TD_DENY }    ToolDecision;

static ToolMode g_tmode = TM_WS;
static char     g_root[PATH_MAX]  = "";   /* the default workspace */
static char     g_proot[PATH_MAX] = "";   /* the projects root; "" = single-workspace mode */

/* THE ACTIVE WORKSPACE IS PER REQUEST, AND THEREFORE PER THREAD.
 *
 * Tool calls run on connection threads and two can be in flight at once, so the workspace a
 * call is confined to cannot be a plain global — a second request switching project mid-call
 * would move the fence under the first one. It is thread-local, set from the request, and
 * falls back to the default.
 *
 * The client never sends a PATH for this. It sends a project NAME, which the server joins to
 * the projects root and resolves; a name with a slash or a dot-dot in it is rejected before it
 * gets that far. Accepting a path here would hand the caller the one thing the confinement
 * exists to withhold. */
static __thread char t_root[PATH_MAX] = "";
static const char *h_root(void){ return t_root[0] ? t_root : g_root; }
/* THE OUTPUT BUDGET IS SET BY PREFILL, NOT BY TASTE.
 *
 * Measured on this machine: 10.5 tok/s of prefill. A tool result of B bytes is about B/3.6
 * tokens, so it costs B/37.8 seconds before the model says a word on the next turn — ONCE,
 * because the prefix cache then holds it, but once is enough to notice:
 *
 *     16 KiB -> 4551 tokens -> 7.2 min      4 KiB -> 1138 tokens -> 1.8 min
 *      8 KiB -> 2276 tokens -> 3.6 min      2 KiB ->  569 tokens -> 0.9 min
 *
 * 16 KiB is the number the other harnesses use and it is wrong here by an order of magnitude.
 * 4 KiB costs about a minute and a half, the same order as the generation step it sits between
 * (~90 tokens at 5.4 tok/s = 17 s) rather than twenty times it. Raise it with --tool-output on
 * a machine that prefills faster; the arithmetic above is the whole argument. */
static size_t   g_tmax  = 4096;      /* bytes of tool output the model is allowed to see */
static int      g_ttmo  = 120000;    /* default command timeout, ms */

static const char *h_mode_name(void){
    switch(g_tmode){
        case TM_OFF:  return "off";
        case TM_RO:   return "read-only";
        case TM_WS:   return "workspace-write";
        case TM_FULL: return "danger-full-access";
    }
    return "off";
}
static int h_mode_parse(const char *s, ToolMode *m){
    if(!strcmp(s,"off"))                                     { *m = TM_OFF;  return 1; }
    if(!strcmp(s,"ro")   || !strcmp(s,"read-only"))          { *m = TM_RO;   return 1; }
    if(!strcmp(s,"ws")   || !strcmp(s,"workspace")
       || !strcmp(s,"workspace-write"))                      { *m = TM_WS;   return 1; }
    if(!strcmp(s,"full") || !strcmp(s,"danger-full-access")) { *m = TM_FULL; return 1; }
    return 0;
}

/* The gate. Read is free; writing is free but confined; running a command is the one that
 * needs a human, because it is the only one whose blast radius is not visible in its
 * arguments — "make" can do anything at all. */
static ToolDecision h_policy(ToolClass c, const char **why){
    switch(g_tmode){
        case TM_OFF:
            *why = "tools are disabled on this server (--tools off)";
            return TD_DENY;
        case TM_RO:
            if(c == TCL_READ) return TD_ALLOW;
            *why = "the server runs read-only (--tools ro): this call would change the machine";
            return TD_DENY;
        case TM_WS:
            if(c == TCL_EXEC){ *why = "runs a shell command"; return TD_ASK; }
            return TD_ALLOW;
        case TM_FULL:
            return TD_ALLOW;
    }
    *why = "unknown mode";
    return TD_DENY;
}

/* ---------------- paths ---------------- */

/* Set when a call was refused for leaving the workspace rather than for failing. It is a
 * REFUSAL, not an error, and the difference is worth carrying: the policy gate cannot make
 * this decision because it does not see the arguments, so without this the two arrive at the
 * caller looking identical — a red tool error next to a green "allow". Thread-local because
 * tool calls run on connection threads and two can be in flight at once. */
static __thread int h_out_of_tree = 0;

static int h_under_root(const char *abs){
    const char *root = h_root();
    const size_t n = strlen(root);
    if(!n) return 0;
    if(strncmp(abs, root, n)) return 0;
    return abs[n] == 0 || abs[n] == '/';
}

/* Resolve a path argument to an absolute, symlink-free path inside the workspace.
 * need_exists=0 means the target itself may be absent — then its PARENT must exist and be
 * inside, which is what a write to a new file needs. */
static int h_resolve(const char *rel, int need_exists, char *out, const char **err){
    static char msg[PATH_MAX + 160];
    char joined[PATH_MAX];
    if(!rel || !*rel){ *err = "path is empty"; return 0; }
    if(rel[0] == '/'){ if(snprintf(joined, sizeof joined, "%s", rel) >= (int)sizeof joined){
                           *err = "path too long"; return 0; } }
    else             { if(snprintf(joined, sizeof joined, "%s/%s", h_root(), rel) >= (int)sizeof joined){
                           *err = "path too long"; return 0; } }

    if(realpath(joined, out) == NULL){
        /* THE CONFINEMENT DECIDES FIRST, EVEN WHEN THE TARGET IS NOT THERE.
         *
         * The obvious order — report ENOENT for a missing file, check the fence only for one
         * that exists — builds an existence oracle across the fence: "../secrets/x" answers
         * "No such file or directory" when it is absent and "outside the workspace" when it is
         * present, which tells the caller exactly what it was not allowed to learn. So the
         * parent is resolved and checked in BOTH cases, and ENOENT is only ever reported for a
         * path that was inside to begin with. */
        const int missing = need_exists;
        /* split off the last component and resolve the directory instead */
        char dir[PATH_MAX];
        snprintf(dir, sizeof dir, "%s", joined);
        char *slash = strrchr(dir, '/');
        if(!slash){ *err = "cannot resolve path"; return 0; }
        const char *base = slash + 1;
        if(!*base || !strcmp(base, ".") || !strcmp(base, "..")){
            *err = "path does not name a file"; return 0; }
        char basecopy[256];
        snprintf(basecopy, sizeof basecopy, "%s", base);
        *slash = 0;
        char rdir[PATH_MAX];
        if(realpath(dir[0] ? dir : "/", rdir) == NULL){
            snprintf(msg, sizeof msg, "%s: parent directory does not exist", rel);
            *err = msg; return 0;
        }
        if(snprintf(out, PATH_MAX, "%s/%s", rdir, basecopy) >= PATH_MAX){
            *err = "path too long"; return 0; }
        if(g_tmode != TM_FULL && !h_under_root(out)){
            snprintf(msg, sizeof msg,
                     "%s resolves to %s, which is outside the workspace %s — refused",
                     rel, out, h_root());
            *err = msg; h_out_of_tree = 1; return 0;
        }
        if(missing){                      /* inside, and genuinely not there */
            snprintf(msg, sizeof msg, "%s: %s", rel, strerror(ENOENT));
            *err = msg; return 0;
        }
    }
    if(g_tmode != TM_FULL && !h_under_root(out)){
        snprintf(msg, sizeof msg,
                 "%s resolves to %s, which is outside the workspace %s — refused",
                 rel, out, h_root());
        *err = msg; h_out_of_tree = 1; return 0;
    }
    return 1;
}

/* Path as the model should see it: relative to the workspace when it is inside. */
static const char *h_rel(const char *abs){
    const char *root = h_root();
    const size_t n = strlen(root);
    if(!strncmp(abs, root, n) && abs[n] == '/') return abs + n + 1;
    return abs;
}

/* ---------------- projects ----------------
 *
 * One engine, many workspaces. A project is a direct subdirectory of the projects root and is
 * addressed by NAME, never by path — the validation below is the whole security boundary for
 * that, and it is deliberately a whitelist: anything not in [A-Za-z0-9._-] is refused rather
 * than escaped or stripped. Stripping produces a different name that still resolves somewhere,
 * which is how a "sanitizer" turns into a directory traversal. */
static int h_pname_ok(const char *n){
    if(!n || !*n || strlen(n) > 64) return 0;
    if(n[0] == '.') return 0;                      /* no ".", "..", no hidden directories */
    for(const char *p = n; *p; p++)
        if(!(isalnum((unsigned char)*p) || *p == '.' || *p == '_' || *p == '-')) return 0;
    return 1;
}

/* Point this thread at a project for the duration of one request. Returns 0 and leaves the
 * thread on the default workspace if the name is bad or the directory is not there. */
static int h_use_project(const char *name, const char **err){
    static __thread char msg[PATH_MAX + 96];
    t_root[0] = 0;
    if(!name || !*name) return 1;                  /* no project asked for: the default */
    if(!g_proot[0]){ *err = "this server runs a single workspace, not projects"; return 0; }
    if(!h_pname_ok(name)){ *err = "a project name is letters, digits, dot, dash, underscore"; return 0; }
    char joined[PATH_MAX], real[PATH_MAX];
    if(snprintf(joined, sizeof joined, "%s/%s", g_proot, name) >= (int)sizeof joined){
        *err = "name too long"; return 0; }
    if(!realpath(joined, real)){
        snprintf(msg, sizeof msg, "no project \"%s\"", name); *err = msg; return 0; }
    /* resolved, then checked: a symlink in the projects root pointing at / would otherwise
     * make the whole filesystem a "project" */
    const size_t n = strlen(g_proot);
    if(strncmp(real, g_proot, n) || real[n] != '/'){
        snprintf(msg, sizeof msg, "project \"%s\" resolves outside the projects root", name);
        *err = msg; return 0; }
    struct stat st;
    if(stat(real, &st) || !S_ISDIR(st.st_mode)){ *err = "not a directory"; return 0; }
    snprintf(t_root, sizeof t_root, "%s", real);
    return 1;
}

static int h_make_project(const char *name, const char **err){
    static __thread char msg[PATH_MAX + 96];
    if(!g_proot[0]){ *err = "this server runs a single workspace, not projects"; return 0; }
    if(!h_pname_ok(name)){ *err = "a project name is letters, digits, dot, dash, underscore"; return 0; }
    char joined[PATH_MAX];
    if(snprintf(joined, sizeof joined, "%s/%s", g_proot, name) >= (int)sizeof joined){
        *err = "name too long"; return 0; }
    if(mkdir(joined, 0755) != 0 && errno != EEXIST){
        snprintf(msg, sizeof msg, "%s: %s", name, strerror(errno)); *err = msg; return 0; }
    return 1;
}

/* GET /v1/projects — what exists, and how big each one is in files. Sorted, because a list
 * that reorders itself between reloads is a list nobody trusts. */
static int h_pcmp(const void *a, const void *b){ return strcmp(*(const char**)a, *(const char**)b); }
static void h_projects_json(Str *o){
    s_cat(o, "{\"root\":");
    s_json(o, g_proot, strlen(g_proot));
    s_cat(o, ",\"data\":[");
    if(g_proot[0]){
        DIR *d = opendir(g_proot);
        char *names[512]; int n = 0;
        if(d){
            struct dirent *e;
            while((e = readdir(d)) && n < 512){
                if(e->d_name[0] == '.') continue;
                char full[PATH_MAX];
                if(snprintf(full, sizeof full, "%s/%s", g_proot, e->d_name) >= (int)sizeof full) continue;
                struct stat st;
                if(stat(full, &st) || !S_ISDIR(st.st_mode)) continue;
                names[n++] = strdup(e->d_name);
            }
            closedir(d);
        }
        qsort(names, (size_t)n, sizeof *names, h_pcmp);
        for(int i = 0; i < n; i++){
            char full[PATH_MAX];
            snprintf(full, sizeof full, "%s/%s", g_proot, names[i]);
            int files = 0;
            DIR *pd = opendir(full);
            if(pd){ struct dirent *e;
                    while((e = readdir(pd)) && files < 10000){ if(e->d_name[0] != '.') files++; }
                    closedir(pd); }
            if(i) s_cat(o, ",");
            s_cat(o, "{\"name\":");
            s_json(o, names[i], strlen(names[i]));
            s_fmt(o, ",\"entries\":%d}", files);
            free(names[i]);
        }
    }
    s_cat(o, "]}");
}

/* ---------------- output budget ---------------- */

static size_t h_line_fwd(const char *p, size_t n, size_t at){    /* end of the line at `at` */
    while(at < n && p[at] != '\n') at++;
    return at < n ? at + 1 : n;
}
static size_t h_line_back(const char *p, size_t at){             /* start of the line at `at` */
    while(at > 0 && p[at-1] != '\n') at--;
    return at;
}
static void h_trunc(Str *s){
    if(s->n <= g_tmax) return;
    const size_t head = h_line_fwd(s->p, s->n, g_tmax * 3 / 4);
    size_t tail = h_line_back(s->p, s->n - g_tmax / 4);
    if(tail < head) tail = head;
    Str o = {0};
    s_add(&o, s->p, head);
    s_fmt(&o, "\n… %zu bytes cut here (output limit %zu; narrow the command or read a range) …\n\n",
          tail - head, g_tmax);
    s_add(&o, s->p + tail, s->n - tail);
    s_free(s);
    *s = o;
}

/* ---------------- running a command ---------------- */

/* fork/exec rather than popen, for three reasons popen cannot give: a process GROUP that can
 * be killed as a unit (a shell that spawned children leaks them otherwise), a timeout at all,
 * and stderr merged into the same stream in the order it was written.
 *
 * The output is kept as HEAD PLUS A RING OF THE TAIL, not as the first N bytes. The first
 * version kept the first 64 KiB and dropped the rest, which is precisely backwards for the
 * command this tool exists to run: a build prints its warnings for a minute and its ERRORS in
 * the last twenty lines. Keeping the head only means the model is handed a log that ends
 * before anything went wrong, and it then reports success.
 *
 * The pipe is drained to the end either way, whether the bytes are kept or not — stopping the
 * reads would block the child on a full pipe until the timeout killed it, turning "long
 * output" into "hangs for two minutes". */
static int h_run(const char *cmd, Str *out, int timeout_ms, int *code){
    int pfd[2];
    if(pipe(pfd) != 0) return 0;
    const pid_t pid = fork();
    if(pid < 0){ close(pfd[0]); close(pfd[1]); return 0; }
    if(pid == 0){
        setpgid(0, 0);
        close(pfd[0]);
        dup2(pfd[1], 1); dup2(pfd[1], 2); close(pfd[1]);
        const int nul = open("/dev/null", O_RDONLY);
        if(nul >= 0){ dup2(nul, 0); close(nul); }
        if(chdir(h_root()) != 0) _exit(126);
        setenv("TERM", "dumb", 1);          /* no colour escapes in the model's context */
        setenv("PAGER", "cat", 1);
        setenv("GIT_PAGER", "cat", 1);
        setenv("NO_COLOR", "1", 1);
        execl("/bin/sh", "sh", "-c", cmd, (char*)NULL);
        _exit(127);
    }
    close(pfd[1]);

    const size_t hcap = g_tmax * 3 / 4;
    const size_t tcap = g_tmax / 4 > 300 ? g_tmax / 4 - 200 : 256;
    char  *ring = (char*)malloc(tcap);
    size_t rpos = 0, seen_tail = 0;         /* seen_tail counts everything past the head */
    const double t0 = now();
    int killed = 0;
    for(;;){
        const int left = timeout_ms - (int)((now() - t0) * 1000.0);
        if(left <= 0){ kill(-pid, SIGKILL); killed = 1; break; }
        struct pollfd p = { pfd[0], POLLIN, 0 };
        const int r = poll(&p, 1, left > 200 ? 200 : left);
        if(r > 0){
            char b[8192];
            const ssize_t k = read(pfd[0], b, sizeof b);
            if(k <= 0) break;
            const char *q = b; size_t n = (size_t)k;
            if(out->n < hcap){
                const size_t take = hcap - out->n < n ? hcap - out->n : n;
                s_add(out, q, take);
                q += take; n -= take;
            }
            if(n && ring){                  /* the rest goes round and round */
                seen_tail += n;
                if(n >= tcap){ memcpy(ring, q + n - tcap, tcap); rpos = 0; }
                else {
                    const size_t first = tcap - rpos < n ? tcap - rpos : n;
                    memcpy(ring + rpos, q, first);
                    if(n > first) memcpy(ring, q + first, n - first);
                    rpos = (rpos + n) % tcap;
                }
            }
        } else if(r < 0 && errno != EINTR) break;
    }
    close(pfd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    if(killed) kill(-pid, SIGKILL);

    if(seen_tail && ring){
        Str t = {0};
        if(seen_tail >= tcap){ s_add(&t, ring + rpos, tcap - rpos); s_add(&t, ring, rpos); }
        else                   s_add(&t, ring, seen_tail);
        size_t at = 0;                      /* never start the tail mid-line */
        if(seen_tail >= tcap){ while(at < t.n && t.p[at] != '\n') at++; if(at < t.n) at++; }
        if(seen_tail > t.n - at)
            s_fmt(out, "\n… %zu bytes cut here (output limit %zu — narrow the command) …\n",
                  seen_tail - (t.n - at), g_tmax);
        s_add(out, t.p + at, t.n - at);
        s_free(&t);
    }
    free(ring);

    *code = killed ? 124
          : WIFEXITED(st)   ? WEXITSTATUS(st)
          : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1;
    if(killed) s_fmt(out, "\n[killed after %d ms]\n", timeout_ms);
    return 1;
}

/* ---------------- argument helpers ---------------- */

static const char *a_str(jval *a, const char *k, const char *def){
    jval *v = json_get(a, k);
    return (v && v->t == J_STR) ? v->str : def;
}
static double a_num(jval *a, const char *k, double def){
    jval *v = json_get(a, k);
    if(v && v->t == J_NUM)  return v->num;
    if(v && v->t == J_STR)  return strtod(v->str, NULL);   /* models quote numbers */
    return def;
}
static int a_bool(jval *a, const char *k, int def){
    jval *v = json_get(a, k);
    if(v && v->t == J_BOOL) return v->boolean;
    if(v && v->t == J_NUM)  return v->num != 0;
    if(v && v->t == J_STR)  return !strcmp(v->str, "true") || !strcmp(v->str, "1");
    return def;
}

static char *h_slurp(const char *abs, size_t *len, size_t cap, const char **err){
    static char msg[PATH_MAX + 64];
    FILE *f = fopen(abs, "rb");
    if(!f){ snprintf(msg, sizeof msg, "%s: %s", h_rel(abs), strerror(errno)); *err = msg; return NULL; }
    Str b = {0};
    char buf[65536];
    size_t k;
    while((k = fread(buf, 1, sizeof buf, f)) > 0){
        s_add(&b, buf, k);
        if(b.n > cap) break;
    }
    fclose(f);
    if(!b.p){ b.p = (char*)calloc(1,1); b.n = 0; }
    *len = b.n;
    return b.p;
}

/* ---------------- the tools ---------------- */

static int t_bash(jval *a, Str *o, int *code, const char **err){
    const char *cmd = a_str(a, "command", NULL);
    if(!cmd || !*cmd){ *err = "command is required"; return 0; }
    int tmo = (int)a_num(a, "timeout_ms", g_ttmo);
    if(tmo < 100) tmo = 100;
    if(tmo > 600000) tmo = 600000;
    if(!h_run(cmd, o, tmo, code)){ *err = "fork failed"; return 0; }
    if(o->n == 0) s_cat(o, *code == 0 ? "(no output)\n" : "");
    if(*code != 0) s_fmt(o, "\n[exit %d]\n", *code);
    return 1;
}

static int t_read(jval *a, Str *o, int *code, const char **err){
    (void)code;
    char abs[PATH_MAX];
    if(!h_resolve(a_str(a, "path", NULL), 1, abs, err)) return 0;
    struct stat st;
    if(!stat(abs, &st) && S_ISDIR(st.st_mode)){ *err = "that is a directory — use list_dir"; return 0; }
    size_t n = 0;
    char *b = h_slurp(abs, &n, 8u<<20, err);
    if(!b) return 0;
    const long off = (long)a_num(a, "offset", 1);          /* 1-based, like an editor */
    long lim = (long)a_num(a, "limit", 1500);
    if(lim <= 0) lim = 1500;
    /* Stop at a LINE when the budget runs out, rather than letting the generic truncation cut
     * a hole in the middle. A hole in a log is survivable; a hole in the middle of a function
     * is a file the model will then edit against text it never saw. */
    const size_t soft = g_tmax > 512 ? g_tmax - 256 : g_tmax;
    long line = 1, shown = 0;
    size_t i = 0;
    while(i < n && shown < lim && o->n < soft){
        size_t e = i;
        while(e < n && b[e] != '\n') e++;
        if(line >= off){
            s_fmt(o, "%6ld\t", line);
            size_t len = e - i;
            if(len > 2000) len = 2000;                     /* one pathological line, not the file */
            s_add(o, b + i, len);
            if(len < e - i) s_cat(o, " …");
            s_cat(o, "\n");
            shown++;
        }
        i = e < n ? e + 1 : n;
        line++;
    }
    long total = line - 1;                    /* a fact, not an estimate */
    for(size_t k = i; k < n; k++) if(b[k] == '\n') total++;
    if(shown == 0) s_fmt(o, "(no lines at offset %ld; the file has %ld)\n", off, total);
    else if(i < n) s_fmt(o, "\n… %ld more lines — continue with offset=%ld …\n",
                         total - (off + shown - 1), off + shown);
    free(b);
    return 1;
}

static int t_write(jval *a, Str *o, int *code, const char **err){
    (void)code;
    char abs[PATH_MAX];
    if(!h_resolve(a_str(a, "path", NULL), 0, abs, err)) return 0;
    jval *c = json_get(a, "content");
    const char *content = (c && c->t == J_STR) ? c->str : "";
    struct stat st;
    const int existed = stat(abs, &st) == 0;
    if(existed && S_ISDIR(st.st_mode)){ *err = "that is a directory"; return 0; }
    FILE *f = fopen(abs, "wb");
    if(!f){ static char m[PATH_MAX+64]; snprintf(m, sizeof m, "%s: %s", h_rel(abs), strerror(errno));
            *err = m; return 0; }
    const size_t len = strlen(content);
    const size_t w = len ? fwrite(content, 1, len, f) : 0;
    fclose(f);
    if(w != len){ *err = "short write (disk full?)"; return 0; }
    long lines = len ? 1 : 0;
    for(size_t i = 0; i < len; i++) if(content[i] == '\n' && i + 1 < len) lines++;
    s_fmt(o, "%s %s: %zu bytes, %ld lines\n", existed ? "overwrote" : "created", h_rel(abs), len, lines);
    return 1;
}

/* Exact-string replacement, and it REFUSES an ambiguous match instead of guessing. The
 * alternative — replace the first occurrence — silently edits the wrong call site in a file
 * where the same three lines appear twice, and the model has no way to notice. */
static int t_edit(jval *a, Str *o, int *code, const char **err){
    (void)code;
    static char msg[PATH_MAX + 256];
    char abs[PATH_MAX];
    if(!h_resolve(a_str(a, "path", NULL), 1, abs, err)) return 0;
    jval *jo = json_get(a, "old_string"), *jn = json_get(a, "new_string");
    if(!jo || jo->t != J_STR || !*jo->str){ *err = "old_string is required"; return 0; }
    const char *os = jo->str, *ns = (jn && jn->t == J_STR) ? jn->str : "";
    const size_t ol = strlen(os), nl = strlen(ns);
    const int all = a_bool(a, "replace_all", 0);
    size_t n = 0;
    char *b = h_slurp(abs, &n, 32u<<20, err);
    if(!b) return 0;

    int hits = 0;
    for(size_t i = 0; i + ol <= n; ){
        if(!memcmp(b + i, os, ol)){ hits++; i += ol; } else i++;
    }
    if(hits == 0){
        snprintf(msg, sizeof msg, "old_string does not appear in %s — read the file and copy "
                                  "the exact text, whitespace included", h_rel(abs));
        *err = msg; free(b); return 0;
    }
    if(hits > 1 && !all){
        snprintf(msg, sizeof msg, "old_string appears %d times in %s — include surrounding "
                                  "lines to make it unique, or pass replace_all=true",
                 hits, h_rel(abs));
        *err = msg; free(b); return 0;
    }
    Str out = {0};
    long at_line = 0, line = 1;
    for(size_t i = 0; i < n; ){
        if(i + ol <= n && !memcmp(b + i, os, ol) && (all || !at_line)){
            if(!at_line) at_line = line;
            s_add(&out, ns, nl);
            for(size_t k = 0; k < ol; k++) if(os[k] == '\n') line++;
            i += ol;
        } else {
            if(b[i] == '\n') line++;
            s_add(&out, b + i, 1);
            i++;
        }
    }
    FILE *f = fopen(abs, "wb");
    if(!f){ snprintf(msg, sizeof msg, "%s: %s", h_rel(abs), strerror(errno)); *err = msg;
            free(b); s_free(&out); return 0; }
    const size_t w = out.n ? fwrite(out.p, 1, out.n, f) : 0;
    fclose(f);
    free(b);
    if(w != out.n){ s_free(&out); *err = "short write"; return 0; }
    s_fmt(o, "%s: %d replacement%s at line %ld (%zu → %zu bytes)\n",
          h_rel(abs), all ? hits : 1, (all && hits > 1) ? "s" : "", at_line, n, out.n);
    s_free(&out);
    return 1;
}

static int t_ls(jval *a, Str *o, int *code, const char **err){
    (void)code;
    char abs[PATH_MAX];
    const char *p = a_str(a, "path", ".");
    if(!h_resolve(p, 1, abs, err)) return 0;
    DIR *d = opendir(abs);
    if(!d){ static char m[PATH_MAX+64]; snprintf(m, sizeof m, "%s: %s", h_rel(abs), strerror(errno));
            *err = m; return 0; }
    struct dirent *e;
    int n = 0;
    Str dirs = {0}, files = {0};
    while((e = readdir(d)) && n < 800){
        if(!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char full[PATH_MAX];
        if(snprintf(full, sizeof full, "%s/%s", abs, e->d_name) >= (int)sizeof full) continue;
        struct stat st;
        if(lstat(full, &st) != 0) continue;
        n++;
        if(S_ISDIR(st.st_mode))       s_fmt(&dirs,  "  %s/\n", e->d_name);
        else if(S_ISLNK(st.st_mode))  s_fmt(&files, "  %s -> (symlink)\n", e->d_name);
        else                          s_fmt(&files, "  %s  %lld B\n", e->d_name, (long long)st.st_size);
    }
    closedir(d);
    s_fmt(o, "%s/\n", h_rel(abs));
    if(dirs.n)  s_add(o, dirs.p, dirs.n);
    if(files.n) s_add(o, files.p, files.n);
    if(!n) s_cat(o, "  (empty)\n");
    if(n >= 800) s_cat(o, "  … more entries not listed …\n");
    s_free(&dirs); s_free(&files);
    return 1;
}

/* Directories that are never worth walking. Not a security measure — a token measure: one
 * node_modules is 30k files and the model pays for every path it is shown. */
static int h_skip(const char *n){
    static const char *sk[] = { ".git", "node_modules", "__pycache__", ".mypy_cache",
                                ".pytest_cache", ".venv", "venv", ".cache", ".ccache", NULL };
    for(int i = 0; sk[i]; i++) if(!strcmp(n, sk[i])) return 1;
    return 0;
}

typedef struct {
    Str  *out;
    int   hits, max, depth_max, visited, budget_hit;
    const char *pat;                  /* glob */
    regex_t    *re;                   /* grep */
    const char *fglob;                /* grep: which files */
} Walk;

static int h_glob_match(const char *pat, const char *rel){
    if(!strchr(pat, '/')){
        const char *base = strrchr(rel, '/');
        return fnmatch(pat, base ? base + 1 : rel, 0) == 0;
    }
    if(!fnmatch(pat, rel, 0)) return 1;
    /* a leading double-star should also match at the root itself, which fnmatch will not */
    if(!strncmp(pat, "**/", 3)) return fnmatch(pat + 3, rel, 0) == 0;
    return 0;
}

static void h_walk(const char *abs, const char *rel, int depth, Walk *w,
                   void (*hit)(const char *abs, const char *rel, Walk *w)){
    if(w->out->n > g_tmax) w->budget_hit = 1;
    if(depth > w->depth_max || w->hits >= w->max || w->visited > 200000 || w->budget_hit) return;
    DIR *d = opendir(abs);
    if(!d) return;
    struct dirent *e;
    while((e = readdir(d))){
        if(w->hits >= w->max) break;
        if(e->d_name[0] == '.' && (!e->d_name[1] || !strcmp(e->d_name, ".."))) continue;
        if(h_skip(e->d_name)) continue;
        char cabs[PATH_MAX], crel[PATH_MAX];
        /* a truncated path is a WRONG path, not a cosmetic problem: it would be reported to
         * the model as a real file name and every later call on it would fail elsewhere */
        if(snprintf(cabs, sizeof cabs, "%s/%s", abs, e->d_name) >= (int)sizeof cabs) continue;
        if(snprintf(crel, sizeof crel, "%s%s%s", rel, *rel ? "/" : "", e->d_name) >= (int)sizeof crel) continue;
        struct stat st;
        if(lstat(cabs, &st) != 0) continue;
        w->visited++;
        if(S_ISDIR(st.st_mode))       h_walk(cabs, crel, depth + 1, w, hit);   /* lstat: never follows a link */
        else if(S_ISREG(st.st_mode))  hit(cabs, crel, w);
    }
    closedir(d);
}

static void h_glob_hit(const char *abs, const char *rel, Walk *w){
    (void)abs;
    if(h_glob_match(w->pat, rel)){ s_fmt(w->out, "%s\n", rel); w->hits++; }
}

static int t_glob(jval *a, Str *o, int *code, const char **err){
    (void)code;
    const char *pat = a_str(a, "pattern", NULL);
    if(!pat || !*pat){ *err = "pattern is required"; return 0; }
    char abs[PATH_MAX];
    if(!h_resolve(a_str(a, "path", "."), 1, abs, err)) return 0;
    Walk w = { o, 0, (int)a_num(a, "max_results", 300), 24, 0, 0, pat, NULL, NULL };
    h_walk(abs, "", 0, &w, h_glob_hit);
    if(!w.hits) s_fmt(o, "no file under %s matches %s\n", h_rel(abs), pat);
    else        s_fmt(o, "\n%d match%s\n", w.hits, w.hits == 1 ? "" : "es");
    return 1;
}

static void h_grep_hit(const char *abs, const char *rel, Walk *w){
    if(w->fglob && !h_glob_match(w->fglob, rel)) return;
    FILE *f = fopen(abs, "rb");
    if(!f) return;
    char *buf = (char*)malloc(1u<<20);
    if(!buf){ fclose(f); return; }
    const size_t n = fread(buf, 1, (1u<<20) - 1, f);
    fclose(f);
    buf[n] = 0;
    for(size_t i = 0; i < n && i < 4096; i++) if(!buf[i]){ free(buf); return; }   /* binary */
    long line = 1;
    for(size_t i = 0; i < n && w->hits < w->max && !w->budget_hit; ){
        if(w->out->n > g_tmax){ w->budget_hit = 1; break; }
        size_t e = i;
        while(e < n && buf[e] != '\n') e++;
        const char save = buf[e];
        buf[e] = 0;
        if(regexec(w->re, buf + i, 0, NULL, 0) == 0){
            size_t len = e - i;
            if(len > 300) len = 300;
            s_fmt(w->out, "%s:%ld: ", rel, line);
            s_add(w->out, buf + i, len);
            if(len < e - i) s_cat(w->out, " …");
            s_cat(w->out, "\n");
            w->hits++;
        }
        buf[e] = save;
        i = e < n ? e + 1 : n;
        line++;
    }
    free(buf);
}

static int t_grep(jval *a, Str *o, int *code, const char **err){
    (void)code;
    static char msg[256];
    const char *pat = a_str(a, "pattern", NULL);
    if(!pat || !*pat){ *err = "pattern is required"; return 0; }
    char abs[PATH_MAX];
    if(!h_resolve(a_str(a, "path", "."), 1, abs, err)) return 0;
    regex_t re;
    const int flags = REG_EXTENDED | REG_NEWLINE | (a_bool(a, "ignore_case", 0) ? REG_ICASE : 0);
    const int rc = regcomp(&re, pat, flags);
    if(rc){ regerror(rc, &re, msg, sizeof msg); *err = msg; regfree(&re); return 0; }
    struct stat st;
    Walk w = { o, 0, (int)a_num(a, "max_results", 120), 24, 0, 0, NULL, &re, a_str(a, "glob", NULL) };
    if(!stat(abs, &st) && S_ISREG(st.st_mode)) h_grep_hit(abs, h_rel(abs), &w);
    else                                       h_walk(abs, "", 0, &w, h_grep_hit);
    regfree(&re);
    if(!w.hits) s_fmt(o, "no match for /%s/ under %s\n", pat, h_rel(abs));
    else if(w.budget_hit)    s_fmt(o, "\n%d matches shown, then the output budget ran out — "
                                      "narrow the pattern or pass a glob\n", w.hits);
    else if(w.hits >= w.max) s_fmt(o, "\n%d matches, stopped at the limit — narrow the pattern\n", w.hits);
    else s_fmt(o, "\n%d match%s\n", w.hits, w.hits == 1 ? "" : "es");
    return 1;
}

/* ---------------- the registry ----------------
 *
 * The descriptions are part of the engine, not documentation: they are the only thing that
 * tells a 27B model when to reach for edit_file instead of rewriting a file whole, and every
 * word of them is prefilled on every turn. Short, imperative, and about WHEN — not WHAT. */

typedef struct {
    const char *name;
    ToolClass   cls;
    const char *desc;
    const char *schema;
    int (*fn)(jval *, Str *, int *, const char **);
} Tool;

static const Tool g_tools[] = {
{ "bash", TCL_EXEC,
  "Run a shell command in the workspace. Use it to build, test, run git, or anything the "
  "other tools do not cover. stdout and stderr come back merged, with the exit code. Prefer "
  "grep/glob/read_file for searching and reading — they are cheaper and their output is shaped "
  "for you.",
  "{\"type\":\"object\",\"properties\":{"
  "\"command\":{\"type\":\"string\",\"description\":\"the command line, run through /bin/sh\"},"
  "\"timeout_ms\":{\"type\":\"integer\",\"description\":\"kill it after this long (default 120000)\"}"
  "},\"required\":[\"command\"]}", t_bash },

{ "read_file", TCL_READ,
  "Read a text file with line numbers. Pass offset (1-based line) and limit to page through a "
  "long file instead of pulling all of it.",
  "{\"type\":\"object\",\"properties\":{"
  "\"path\":{\"type\":\"string\",\"description\":\"relative to the workspace, or absolute\"},"
  "\"offset\":{\"type\":\"integer\",\"description\":\"first line to show, 1-based\"},"
  "\"limit\":{\"type\":\"integer\",\"description\":\"how many lines (default 2000)\"}"
  "},\"required\":[\"path\"]}", t_read },

{ "write_file", TCL_WRITE,
  "Create a file, or replace one completely. For a change to an existing file use edit_file "
  "instead — rewriting a file you have only partly read loses the rest of it.",
  "{\"type\":\"object\",\"properties\":{"
  "\"path\":{\"type\":\"string\"},"
  "\"content\":{\"type\":\"string\",\"description\":\"the entire new contents\"}"
  "},\"required\":[\"path\",\"content\"]}", t_write },

{ "edit_file", TCL_WRITE,
  "Replace an exact string in a file. old_string must match the file byte for byte, including "
  "indentation, and must be UNIQUE — the call is refused rather than guessing, so include the "
  "surrounding lines. Read the file first.",
  "{\"type\":\"object\",\"properties\":{"
  "\"path\":{\"type\":\"string\"},"
  "\"old_string\":{\"type\":\"string\",\"description\":\"exact text to replace\"},"
  "\"new_string\":{\"type\":\"string\",\"description\":\"what it becomes; empty deletes it\"},"
  "\"replace_all\":{\"type\":\"boolean\",\"description\":\"replace every occurrence\"}"
  "},\"required\":[\"path\",\"old_string\",\"new_string\"]}", t_edit },

{ "list_dir", TCL_READ,
  "List one directory. Use it to orient yourself before guessing at paths.",
  "{\"type\":\"object\",\"properties\":{"
  "\"path\":{\"type\":\"string\",\"description\":\"default is the workspace root\"}"
  "},\"required\":[]}", t_ls },

{ "glob", TCL_READ,
  "Find files by name pattern, recursively. A pattern without a slash matches the file name "
  "(\"*.c\"); one with a slash matches the path from the search root (\"src/**/*.h\").",
  "{\"type\":\"object\",\"properties\":{"
  "\"pattern\":{\"type\":\"string\"},"
  "\"path\":{\"type\":\"string\",\"description\":\"where to start; default the workspace root\"},"
  "\"max_results\":{\"type\":\"integer\"}"
  "},\"required\":[\"pattern\"]}", t_glob },

{ "grep", TCL_READ,
  "Search file CONTENTS with a POSIX extended regex and get path:line:text back. This is the "
  "right first move in an unfamiliar codebase — cheaper than reading files to look for "
  "something.",
  "{\"type\":\"object\",\"properties\":{"
  "\"pattern\":{\"type\":\"string\",\"description\":\"POSIX extended regex\"},"
  "\"path\":{\"type\":\"string\",\"description\":\"file or directory; default the workspace root\"},"
  "\"glob\":{\"type\":\"string\",\"description\":\"only search files matching this, e.g. *.c\"},"
  "\"ignore_case\":{\"type\":\"boolean\"},"
  "\"max_results\":{\"type\":\"integer\"}"
  "},\"required\":[\"pattern\"]}", t_grep },
};
static const int g_ntools = (int)(sizeof g_tools / sizeof *g_tools);

static const Tool *h_find(const char *name){
    if(!name) return NULL;
    for(int i = 0; i < g_ntools; i++) if(!strcmp(g_tools[i].name, name)) return &g_tools[i];
    return NULL;
}
static const char *h_cls_name(ToolClass c){
    return c == TCL_READ ? "read" : c == TCL_WRITE ? "write" : "exec";
}
static const char *h_dec_name(ToolDecision d){
    return d == TD_ALLOW ? "allow" : d == TD_ASK ? "ask" : "deny";
}

static void h_system_prompt(Str *o);

/* GET /v1/tools — the schemas the client hands straight back in `tools`, plus what the gate
 * will do with each one, so the page can label a call before it makes it. Kept OUT of the
 * function objects: everything inside those is prefilled verbatim into the prompt. */
static void h_tools_json(Str *o){
    s_cat(o, "{\"object\":\"list\",\"workspace\":");
    s_json(o, h_root(), strlen(h_root()));
    s_fmt(o, ",\"mode\":\"%s\",\"max_output\":%zu,\"data\":[", h_mode_name(), g_tmax);
    int first = 1;
    for(int i = 0; i < g_ntools; i++){
        const char *why = NULL;
        if(h_policy(g_tools[i].cls, &why) == TD_DENY) continue;    /* not offered at all */
        if(!first) s_cat(o, ",");
        first = 0;
        s_cat(o, "{\"type\":\"function\",\"function\":{\"name\":");
        s_json(o, g_tools[i].name, strlen(g_tools[i].name));
        s_cat(o, ",\"description\":");
        s_json(o, g_tools[i].desc, strlen(g_tools[i].desc));
        s_fmt(o, ",\"parameters\":%s}}", g_tools[i].schema);
    }
    s_cat(o, "],\"policy\":{");
    for(int i = 0; i < g_ntools; i++){
        const char *why = NULL;
        const ToolDecision d = h_policy(g_tools[i].cls, &why);
        s_fmt(o, "%s\"%s\":{\"class\":\"%s\",\"decision\":\"%s\"}",
              i ? "," : "", g_tools[i].name, h_cls_name(g_tools[i].cls), h_dec_name(d));
    }
    s_cat(o, "}");
    { Str sp = {0}; h_system_prompt(&sp);
      s_cat(o, ",\"system\":"); s_json(o, sp.p ? sp.p : "", sp.n); s_free(&sp); }
    s_cat(o, "}");
}

/* POST /v1/tools/exec — the only way in. {"name":..,"arguments":{..}|"{..}","approved":bool} */
static void h_exec_json(Str *o, const char *body){
    char *arena = NULL;
    jval *req = json_parse(body, &arena);
    const char *name = req ? a_str(req, "name", NULL) : NULL;
    const Tool *t = h_find(name);
    if(!t){
        s_cat(o, "{\"decision\":\"deny\",\"ok\":false,\"error\":");
        s_json(o, "no such tool", 12);
        s_cat(o, "}");
        free(arena);
        return;
    }
    /* Pick the workspace BEFORE the gate: a denial should name the project it was denied in,
     * and a bad project name must not run anything anywhere. */
    const char *perr = NULL;
    if(!h_use_project(a_str(req, "project", NULL), &perr)){
        s_cat(o, "{\"decision\":\"deny\",\"ok\":false,\"reason\":");
        s_json(o, perr, strlen(perr));
        s_cat(o, "}");
        free(arena);
        return;
    }
    const char *why = NULL;
    const ToolDecision d = h_policy(t->cls, &why);
    const int approved = a_bool(req, "approved", 0);
    if(d == TD_DENY || (d == TD_ASK && !approved)){
        s_fmt(o, "{\"decision\":\"%s\",\"ok\":false,\"tool\":\"%s\",\"class\":\"%s\",\"reason\":",
              h_dec_name(d), t->name, h_cls_name(t->cls));
        s_json(o, why ? why : "refused", strlen(why ? why : "refused"));
        s_cat(o, "}");
        free(arena);
        return;
    }

    /* arguments may be an object or, as OpenAI tool_calls carry them, a JSON string */
    jval *args = json_get(req, "arguments");
    char *arena2 = NULL;
    if(args && args->t == J_STR) args = json_parse(args->str, &arena2);
    if(!args || args->t != J_OBJ){
        static jval empty;
        empty.t = J_OBJ; empty.len = 0; empty.kids = NULL; empty.keys = NULL;
        args = &empty;
    }

    Str out = {0};
    int code = 0;
    const char *err = NULL;
    h_out_of_tree = 0;
    const double t0 = now();
    const int ok = t->fn(args, &out, &code, &err);
    const double ms = (now() - t0) * 1000.0;
    h_trunc(&out);

    s_fmt(o, "{\"decision\":\"%s\",\"ok\":%s,\"tool\":\"%s\",\"class\":\"%s\","
             "\"exit\":%d,\"ms\":%.1f,\"content\":",
          h_out_of_tree ? "deny" : "allow",
          ok ? "true" : "false", t->name, h_cls_name(t->cls), code, ms);
    if(ok) s_json(o, out.p ? out.p : "", out.n);
    else { const char *e = err ? err : "failed"; s_json(o, e, strlen(e)); }
    s_cat(o, "}");
    s_free(&out);
    free(arena2);
    free(arena);
}

/* The agent's own instructions. Deliberately short: it is prefilled on every single turn, and
 * at this engine's prefill rate a 700-token preamble is seven seconds of every request. What
 * it has to establish is only what the model cannot see from the schemas — where it is, that
 * it should look before it edits, and that it must stop talking and act. */
static void h_system_prompt(Str *o){
    s_fmt(o,
      "You are a coding agent with real access to this machine, working in %s.\n\n"
      "How to work:\n"
      "- Look before you act: grep or glob to find the place, read_file to see it, then edit.\n"
      "- Change files with edit_file. Use write_file only for a new file or a full rewrite.\n"
      "- After changing code, build or test it with bash and read the actual output.\n"
      "- One tool call per step. Wait for the result before deciding the next one.\n"
      "- Say in one short line what you are about to do, then call the tool. No essays.\n"
      "- When a tool fails, read the error — it usually says exactly what to fix.\n"
      "- When the task is done, say what changed and stop calling tools.\n",
      h_root());
}

#endif /* QWEN_HARNESS_H */
