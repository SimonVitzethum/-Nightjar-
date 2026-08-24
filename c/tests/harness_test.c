/* harness_test.c — THE GATE for the tool layer.
 *
 * The tools are the part of this server that can destroy something. Every other component
 * here fails by producing a wrong number; these fail by writing to the wrong file or running
 * a command outside the workspace, and neither of those shows up in a perplexity check. So
 * this test spends most of its assertions not on whether the tools work but on whether they
 * REFUSE: three ways out of the workspace, an ambiguous edit, a write in read-only mode, a
 * shell command with no approval.
 *
 * It builds a throwaway workspace under /tmp, so it needs no model and runs in a second.
 */
#define _GNU_SOURCE
#include "json.h"
#include "strbuf.h"
#include "harness.h"
#include "chatstore.h"
#include <netinet/in.h>
#include <arpa/inet.h>

static int pass = 0, fail = 0;
static void ok(int cond, const char *what){
    if(cond) pass++;
    else { fail++; printf("  FAIL  %s\n", what); return; }
    printf("  ok    %s\n", what);
}

/* Every call goes through the real endpoint body, so the test exercises the policy gate and
 * the JSON escaping too — not just the tool function underneath them. */
static jval *call(const char *json, char **arena, Str *raw){
    raw->n = 0; if(raw->p) raw->p[0] = 0;
    h_exec_json(raw, json);
    jval *r = json_parse(raw->p, arena);
    if(!r) printf("  !! response is not JSON: %.200s\n", raw->p);
    return r;
}
static const char *field(jval *r, const char *k){
    jval *v = r ? json_get(r, k) : NULL;
    return (v && v->t == J_STR) ? v->str : "";
}
static int has(jval *r, const char *k, const char *want){
    return strstr(field(r, k), want) != NULL;
}

/* Types a secret into whatever command is running, once it has had a moment to start and turn
 * echo off. A helper thread, because h_run blocks until the command exits — which is the same
 * shape the real thing has: the browser posts to /v1/term/input on a different connection. */
static int secret_flag = 0;
static void *feed_secret(void *arg){
    (void)arg;
    for(int i = 0; i < 300; i++){
        struct timespec ts = { 0, 10*1000*1000 };
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&g_term.mu);
        const int live = g_term.live;
        pthread_mutex_unlock(&g_term.mu);
        if(live) break;
    }
    { struct timespec ts = { 0, 250*1000*1000 }; nanosleep(&ts, NULL); }
    h_term_input("hunter2-secret\n", 15, secret_flag);
    return NULL;
}

static void wr(const char *rel, const char *body){
    char p[PATH_MAX]; snprintf(p, sizeof p, "%s/%s", g_root, rel);
    FILE *f = fopen(p, "wb"); if(f){ fputs(body, f); fclose(f); }
}
static char *rd(const char *rel){
    char p[PATH_MAX]; snprintf(p, sizeof p, "%s/%s", g_root, rel);
    FILE *f = fopen(p, "rb"); if(!f) return NULL;
    static char b[65536]; const size_t n = fread(b, 1, sizeof b - 1, f); b[n] = 0; fclose(f);
    return b;
}

int main(void){
    char tmpl[] = "/tmp/qwen_harness_XXXXXX";
    const char *dir = mkdtemp(tmpl);
    if(!dir){ perror("mkdtemp"); return 1; }
    if(!realpath(dir, g_root)){ perror("realpath"); return 1; }
    printf("workspace %s\n\n", g_root);

    char sub[PATH_MAX]; snprintf(sub, sizeof sub, "%s/src", g_root); mkdir(sub, 0755);
    wr("hello.c",     "#include <stdio.h>\nint main(void){\n    puts(\"hi\");\n    return 0;\n}\n");
    wr("src/util.h",  "#ifndef UTIL_H\n#define UTIL_H\nint twice(int x);\n#endif\n");
    wr("src/util.c",  "#include \"util.h\"\nint twice(int x){ return x + x; }\n");
    wr("twice.txt",   "same line\nsame line\nother\n");

    Str raw = {0}; char *ar = NULL; jval *r;

    /* ---- 1. the registry is the single source of truth ---- */
    puts("1. registry and schemas");
    { Str t = {0}; h_tools_json(&t);
      char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      ok(j && j->t == J_OBJ, "GET /v1/tools is valid JSON");
      jval *d = j ? json_get(j, "data") : NULL;
      ok(d && d->t == J_ARR && d->len == g_ntools, "every registered tool is offered");
      jval *sys = j ? json_get(j, "system") : NULL;
      ok(sys && sys->t == J_STR && strstr(sys->str, g_root), "the system prompt names the workspace");
      /* the schemas are prefilled verbatim, so a malformed one would corrupt the prompt */
      int schemas_ok = 1;
      for(int i = 0; i < g_ntools; i++){
          char *a3 = NULL; jval *sc = json_parse(g_tools[i].schema, &a3);
          if(!sc || sc->t != J_OBJ) { schemas_ok = 0; printf("     bad schema: %s\n", g_tools[i].name); }
          free(a3);
      }
      ok(schemas_ok, "every parameter schema parses as a JSON object");
      free(a2); s_free(&t); }

    /* ---- 2. reading ---- */
    puts("\n2. reading");
    r = call("{\"name\":\"list_dir\",\"arguments\":{}}", &ar, &raw);
    ok(has(r, "content", "src/") && has(r, "content", "hello.c"), "list_dir shows dirs and files");
    free(ar); ar = NULL;

    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"hello.c\"}}", &ar, &raw);
    ok(has(r, "content", "     3\t    puts(\"hi\");"), "read_file numbers lines and keeps indentation");
    free(ar); ar = NULL;

    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"hello.c\",\"offset\":4,\"limit\":1}}", &ar, &raw);
    ok(has(r, "content", "     4\t") && !has(r, "content", "stdio"), "read_file honours offset and limit");
    free(ar); ar = NULL;

    r = call("{\"name\":\"grep\",\"arguments\":{\"pattern\":\"twice\",\"glob\":\"*.h\"}}", &ar, &raw);
    ok(has(r, "content", "src/util.h:3") && !has(r, "content", "util.c"), "grep filters by glob and reports path:line");
    free(ar); ar = NULL;

    r = call("{\"name\":\"glob\",\"arguments\":{\"pattern\":\"*.c\"}}", &ar, &raw);
    ok(has(r, "content", "hello.c") && has(r, "content", "src/util.c"), "a bare glob matches by file name at any depth");
    free(ar); ar = NULL;

    r = call("{\"name\":\"glob\",\"arguments\":{\"pattern\":\"src/*.h\"}}", &ar, &raw);
    ok(has(r, "content", "src/util.h") && !has(r, "content", "hello.c"), "a glob with a slash matches the path");
    free(ar); ar = NULL;

    /* ---- 3. writing ---- */
    puts("\n3. writing");
    r = call("{\"name\":\"write_file\",\"arguments\":{\"path\":\"new/deep.txt\",\"content\":\"x\"}}", &ar, &raw);
    ok(has(r, "content", "parent directory does not exist"), "write_file will not invent directories");
    free(ar); ar = NULL;

    r = call("{\"name\":\"write_file\",\"arguments\":{\"path\":\"src/gen.c\",\"content\":\"int g(void){return 1;}\\n\"}}", &ar, &raw);
    ok(has(r, "content", "created src/gen.c"), "write_file creates a file");
    ok(rd("src/gen.c") && !strcmp(rd("src/gen.c"), "int g(void){return 1;}\n"), "the bytes on disk are exactly what was sent");
    free(ar); ar = NULL;

    r = call("{\"name\":\"edit_file\",\"arguments\":{\"path\":\"hello.c\",\"old_string\":\"puts(\\\"hi\\\")\",\"new_string\":\"puts(\\\"bye\\\")\"}}", &ar, &raw);
    ok(has(r, "content", "1 replacement at line 3"), "edit_file reports the line it changed");
    ok(rd("hello.c") && strstr(rd("hello.c"), "puts(\"bye\")"), "edit_file wrote the replacement");
    free(ar); ar = NULL;

    r = call("{\"name\":\"edit_file\",\"arguments\":{\"path\":\"twice.txt\",\"old_string\":\"same line\",\"new_string\":\"X\"}}", &ar, &raw);
    ok(has(r, "content", "appears 2 times"), "an AMBIGUOUS edit is refused, not guessed");
    ok(rd("twice.txt") && !strstr(rd("twice.txt"), "X"), "and it changed nothing");
    free(ar); ar = NULL;

    r = call("{\"name\":\"edit_file\",\"arguments\":{\"path\":\"twice.txt\",\"old_string\":\"same line\",\"new_string\":\"X\",\"replace_all\":true}}", &ar, &raw);
    ok(has(r, "content", "2 replacements"), "replace_all takes both");
    free(ar); ar = NULL;

    r = call("{\"name\":\"edit_file\",\"arguments\":{\"path\":\"hello.c\",\"old_string\":\"not in there\",\"new_string\":\"y\"}}", &ar, &raw);
    ok(has(r, "content", "does not appear"), "a missing old_string is an error, not a no-op");
    free(ar); ar = NULL;

    /* ---- 4. THE FENCE. Leaving the workspace is a QUESTION now, not a verdict — but the
     *         question has to be asked about the RESOLVED path, the answer has to be bound to
     *         that path, and nothing may happen until it comes. ---- */
    puts("\n4. the fence, and consent");
    { char home[PATH_MAX]; snprintf(home, sizeof home, "%s", g_root); realpath(home, g_home); }

    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"../../../etc/passwd\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "ask"), "..  out of the tree asks instead of refusing");
    ok(!strcmp(field(r, "kind"), "path"), "and says it is a path question, not a shell one");
    ok(!strcmp(field(r, "path"), "/etc/passwd"), "the RESOLVED path is what is put to the human");
    ok(!has(r, "content", "root:"), "and nothing was read while asking");
    free(ar); ar = NULL;

    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"/etc/passwd\"},"
             "\"allow_path\":\"/etc/passwd\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "allow") && has(r, "content", "root:"),
       "with consent to exactly that path, it reads");
    free(ar); ar = NULL;

    /* the whole point of binding: yes to one thing is not yes to the neighbourhood */
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"/etc/hostname\"},"
             "\"allow_path\":\"/etc/passwd\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "ask"), "consent to one path is not consent to its sibling");
    free(ar); ar = NULL;

    /* a symlink out of the tree resolves first, so the human is asked about where it GOES */
    { char link[PATH_MAX]; snprintf(link, sizeof link, "%s/escape", g_root);
      ok(symlink("/etc", link) == 0, "(made a symlink to /etc inside the workspace)"); }
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"escape/passwd\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "ask") && !strcmp(field(r, "path"), "/etc/passwd"),
       "a symlink is asked about by its TARGET, not by its name");
    free(ar); ar = NULL;

    r = call("{\"name\":\"write_file\",\"arguments\":{\"path\":\"../escaped.txt\",\"content\":\"no\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "ask"), "a write above the root asks");
    { char probe[PATH_MAX]; snprintf(probe, sizeof probe, "%s/../escaped.txt", g_root);
      struct stat st;
      ok(stat(probe, &st) != 0, "and wrote nothing while asking"); }
    free(ar); ar = NULL;

    /* the answer must not depend on whether the target exists — that difference is an oracle */
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"../../../nowhere/at/all\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "ask"), "a path whose parents do not exist asks the same way");
    free(ar); ar = NULL;
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"inside-but-absent\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "allow") && has(r, "content", "No such file"),
       "while a missing path INSIDE is simply missing");
    free(ar); ar = NULL;

    /* and consent is about the PATH; it never upgrades the policy class */
    g_tmode = TM_RO;
    r = call("{\"name\":\"write_file\",\"arguments\":{\"path\":\"/tmp/nope.txt\",\"content\":\"x\"},"
             "\"allow_path\":\"/tmp/nope.txt\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "consent to a path does not buy a write in read-only mode");
    free(ar); ar = NULL;
    g_tmode = TM_WS;

    /* ---- 5. the gate ---- */
    puts("\n5. the policy gate");
    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"echo hi\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "ask"), "workspace mode ASKS before running a command");
    ok(!has(r, "content", "hi"), "and does not run it while asking");
    free(ar); ar = NULL;

    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"echo hi\"},\"approved\":true}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "allow") && has(r, "content", "hi"), "approved, it runs");
    free(ar); ar = NULL;

    g_tmode = TM_RO;
    r = call("{\"name\":\"write_file\",\"arguments\":{\"path\":\"ro.txt\",\"content\":\"no\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "read-only mode denies a write");
    free(ar); ar = NULL;
    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"echo hi\"},\"approved\":true}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "read-only mode denies a command EVEN IF APPROVED");
    free(ar); ar = NULL;
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"hello.c\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "allow"), "read-only mode still reads");
    free(ar); ar = NULL;
    { Str t = {0}; h_tools_json(&t); char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      jval *d = json_get(j, "data");
      ok(d && d->len < g_ntools, "and a denied tool is not even offered to the model");
      free(a2); s_free(&t); }

    g_tmode = TM_OFF;
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"hello.c\"}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "--tools off denies everything");
    free(ar); ar = NULL;

    g_tmode = TM_WS;
    r = call("{\"name\":\"no_such_tool\",\"arguments\":{}}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "an unregistered name cannot be invoked");
    free(ar); ar = NULL;

    /* ---- 6. the output budget and the timeout ---- */
    puts("\n6. budget and timeout");
    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"seq 1 200000\"},\"approved\":true}", &ar, &raw);
    { const size_t n = strlen(field(r, "content"));
      ok(n <= g_tmax + 512, "a huge output is cut to the budget");
      ok(has(r, "content", "cut here"), "and the cut is named, not silent");
      /* the head is the START of the output and the tail its END — the point of the ring.
       * Checking only the length would pass on an implementation that keeps the first 16 KiB
       * and throws away the errors at the bottom of a build log, which is what the first one
       * did. */
      const char *c = field(r, "content");
      ok(!strncmp(c, "1\n2\n3\n", 6), "the head is the beginning of the output");
      ok(n > 32 && strstr(c + n - 16, "200000"), "and the tail is its END, not the end of the first chunk"); }
    free(ar); ar = NULL;

    { const double t0 = now();
      r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"sleep 30\",\"timeout_ms\":400},\"approved\":true}", &ar, &raw);
      const double el = now() - t0;
      jval *ex = json_get(r, "exit");
      ok(ex && ex->t == J_NUM && (int)ex->num == 124, "a hung command is killed and reports 124");
      ok(el < 3.0, "and the kill actually happens on time"); }
    free(ar); ar = NULL;

    /* a shell that spawned children must not leave them behind */
    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"sleep 30 & echo started\",\"timeout_ms\":400},\"approved\":true}", &ar, &raw);
    ok(has(r, "content", "started"), "output before the kill is kept");
    free(ar); ar = NULL;

    /* and ordinary, non-secret input is left alone — suppressing it would make every
     * interactive prompt unreadable */
    { pthread_t th; secret_flag = 0;
      pthread_create(&th, NULL, feed_secret, NULL);
      Str t0 = {0}; h_term_json(&t0, ~0ull); char *a3 = NULL; jval *j3 = json_parse(t0.p, &a3);
      unsigned long long m2 = 0;
      { jval *at = j3 ? json_get(j3, "at") : NULL; if(at) m2 = (unsigned long long)at->num; }
      free(a3); s_free(&t0);
      r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"read q; echo q=$q\","
               "\"timeout_ms\":6000},\"approved\":true}", &ar, &raw);
      pthread_join(th, NULL);
      ok(has(r, "content", "q=hunter2-secret"), "plain input is passed through untouched");
      free(ar); ar = NULL;
      Str t = {0}; h_term_json(&t, m2); char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      ok(strstr(field(j, "data"), "hunter2-secret") != NULL,
         "and stays visible in the transcript when it is not marked secret");
      free(a2); s_free(&t); }

    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"exit 3\"},\"approved\":true}", &ar, &raw);
    { jval *ex = json_get(r, "exit");
      ok(ex && (int)ex->num == 3, "the exit code comes back"); }
    ok(has(r, "content", "[exit 3]"), "and the model is told about it in the text");
    free(ar); ar = NULL;

    /* A file bigger than the budget must be PAGED, not holed. A gap in the middle of a log is
     * survivable; a gap in the middle of a function is text the model will then edit against
     * having never seen it. */
    { Str big = {0};
      for(int i = 1; i <= 4000; i++) s_fmt(&big, "line %d of the file\n", i);
      wr("big.txt", big.p); s_free(&big); }
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"big.txt\"}}", &ar, &raw);
    ok(!has(r, "content", "cut here"), "read_file pages instead of cutting a hole in the middle");
    ok(has(r, "content", "continue with offset="), "and says exactly where to continue");
    { const char *c = field(r, "content");
      ok(strlen(c) <= g_tmax + 512, "and stays inside the budget"); }
    free(ar); ar = NULL;

    /* ---- 6a. WHAT A CHILD MUST NOT INHERIT.
     *
     * This is here because it happened. Firefox is forked by the server and inherited its
     * LISTENING SOCKET, then outlived it and kept port 8080 bound:
     *
     *   LISTEN 127.0.0.1:8080  users:(("firefox",pid=1712201,fd=39))
     *
     * `make stop` reported success, `make serve` failed with "port 8080 is still held by
     * something else", and nothing named qwen35_server was running. The same inheritance pins
     * the model file and every in-flight HTTP connection. ---- */
    puts("\n6a. inheritance");
    {
        const int ls = socket(AF_INET, SOCK_STREAM, 0);          /* deliberately NOT CLOEXEC:
                                                                  * the child must close it
                                                                  * even when the parent forgot */
        struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
        sa.sin_family = AF_INET; sa.sin_addr.s_addr = inet_addr("127.0.0.1"); sa.sin_port = 0;
        int bound = ls >= 0 && bind(ls, (struct sockaddr*)&sa, sizeof sa) == 0 && listen(ls, 4) == 0;
        ok(bound, "(opened a listening socket in the parent)");
        r = call("{\"name\":\"bash\",\"arguments\":{"
                 "\"command\":\"ls -l /proc/self/fd | grep -c socket || true\"},"
                 "\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "0"), "a forked command inherits NO socket from the server");
        free(ar); ar = NULL;
        r = call("{\"name\":\"bash\",\"arguments\":{"
                 "\"command\":\"ls /proc/self/fd | wc -l\"},\"approved\":true}", &ar, &raw);
        { const char *c = field(r, "content");
          const int nfd = atoi(c);
          ok(nfd > 0 && nfd <= 5, "and only its own three plus the listing itself"); }
        free(ar); ar = NULL;
        if(ls >= 0) close(ls);
    }

    /* ---- 6b. PROJECTS. A second confinement, and a different one: the caller no longer
     *          names a directory at all, it names a project, and everything that is not a
     *          plain name has to bounce before it reaches the filesystem. ---- */
    puts("\n6b. projects");
    { char pr[PATH_MAX];
      snprintf(pr, sizeof pr, "%s/projects", g_root);
      mkdir(pr, 0755);
      realpath(pr, g_proot);
      char a1[PATH_MAX], a2[PATH_MAX];
      snprintf(a1, sizeof a1, "%s/alpha", g_proot); mkdir(a1, 0755);
      snprintf(a2, sizeof a2, "%s/beta",  g_proot); mkdir(a2, 0755);
      FILE *f = fopen("/dev/null", "r"); if(f) fclose(f);
      char af[PATH_MAX]; snprintf(af, sizeof af, "%s/only-in-alpha.txt", a1);
      f = fopen(af, "wb"); if(f){ fputs("alpha\n", f); fclose(f); } }

    r = call("{\"name\":\"list_dir\",\"arguments\":{},\"project\":\"alpha\"}", &ar, &raw);
    ok(has(r, "content", "only-in-alpha.txt"), "a project scopes the workspace to itself");
    free(ar); ar = NULL;
    r = call("{\"name\":\"list_dir\",\"arguments\":{},\"project\":\"beta\"}", &ar, &raw);
    ok(!has(r, "content", "only-in-alpha.txt"), "and the next project cannot see it");
    free(ar); ar = NULL;
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"../alpha/only-in-alpha.txt\"},"
             "\"project\":\"beta\"}", &ar, &raw);
    /* a sibling project is out of tree, so it ASKS — and because g_home is the projects root
     * here, the question says the target is still inside the engine directory rather than out
     * on the wider machine. The distinction is the whole reason inside_home exists. */
    ok(!strcmp(field(r, "decision"), "ask"), "one project cannot silently reach its sibling");
    { jval *ih = json_get(r, "inside_home");
      ok(ih && ih->t == J_BOOL && ih->boolean, "and the ask says it is a neighbour, not the machine"); }
    ok(!has(r, "content", "alpha"), "nothing was read while asking");
    free(ar); ar = NULL;
    r = call("{\"name\":\"list_dir\",\"arguments\":{},\"project\":\"../..\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "a dot-dot project name is refused");
    free(ar); ar = NULL;
    r = call("{\"name\":\"list_dir\",\"arguments\":{},\"project\":\"a/b\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "a project name with a slash is refused");
    free(ar); ar = NULL;
    r = call("{\"name\":\"list_dir\",\"arguments\":{},\"project\":\"nope\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "a project that does not exist is refused");
    free(ar); ar = NULL;
    /* a symlink IN the projects root pointing at / would otherwise make everything a project */
    { char lk[PATH_MAX]; snprintf(lk, sizeof lk, "%s/sneaky", g_proot);
      ok(symlink("/etc", lk) == 0, "(made a project that is a symlink to /etc)"); }
    r = call("{\"name\":\"list_dir\",\"arguments\":{},\"project\":\"sneaky\"}", &ar, &raw);
    ok(!strcmp(field(r, "decision"), "deny"), "a project symlinked out of the root is refused");
    free(ar); ar = NULL;
    { Str t = {0}; h_projects_json(&t); char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      jval *d = j ? json_get(j, "data") : NULL;
      ok(d && d->t == J_ARR && d->len >= 2, "the project list is JSON and finds them");
      free(a2); s_free(&t); }
    g_proot[0] = 0; t_root[0] = 0;

    /* ---- 6c. THE TERMINAL. A PTY, a live transcript, and stdin that works — which is the
     *          whole reason sudo can be answered at all. ---- */
    puts("\n6c. the terminal");
    unsigned long long mark = 0;
    { Str t = {0}; h_term_json(&t, ~0ull); char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      jval *at = j ? json_get(j, "at") : NULL;
      if(at && at->t == J_NUM) mark = (unsigned long long)at->num;
      free(a2); s_free(&t); }
    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"echo eins; echo zwei\"},\"approved\":true}", &ar, &raw);
    free(ar); ar = NULL;
    { Str t = {0}; h_term_json(&t, mark); char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      const char *d = j ? field(j, "data") : "";
      ok(strstr(d, "$ echo eins") != NULL, "the transcript shows the command that was run");
      ok(strstr(d, "eins") && strstr(d, "zwei"), "and its output");
      ok(strstr(d, "[exit 0]") != NULL, "and how it ended");
      ok(!strstr(d, "\r"), "with no carriage returns from the line discipline");
      free(a2); s_free(&t); }

    /* Is it really a terminal? Ask something that can only tell the truth on one. */
    r = call("{\"name\":\"bash\",\"arguments\":{\"command\":\"test -t 1 && echo TTY || echo PIPE\"},"
             "\"approved\":true}", &ar, &raw);
    ok(has(r, "content", "TTY"), "stdout IS a tty — without this sudo refuses to read a password");
    free(ar); ar = NULL;

    /* And the property that makes a password safe to type: with echo off, what is typed does
     * not come back. But the command here does NOT hide anything: `stty -echo; read p` echoes
     * in bash, because bash's read re-enables echo unless given -s. That is exactly the case a
     * harness has to survive — a model writing the obvious idiom and getting it wrong must not
     * leak the password anyway — so the input is marked secret and the harness strips it. */
    { pthread_t th;
      secret_flag = 1;
      pthread_create(&th, NULL, feed_secret, NULL);
      r = call("{\"name\":\"bash\",\"arguments\":{"
               "\"command\":\"stty -echo; read p; stty echo; echo len=${#p}\","
               "\"timeout_ms\":6000},\"approved\":true}", &ar, &raw);
      pthread_join(th, NULL);
      ok(has(r, "content", "len=14"), "input typed into the terminal reaches the command");
      ok(!has(r, "content", "hunter2-secret"), "and with echo off it is NOT in what the model sees");
      free(ar); ar = NULL;
      Str t = {0}; h_term_json(&t, mark); char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      ok(!strstr(field(j, "data"), "hunter2-secret"), "nor anywhere in the transcript");
      ok(strstr(field(j, "data"), "[Eingabe verborgen]") != NULL,
         "but the transcript DOES say that something was typed");
      free(a2); s_free(&t); }

    /* ---- 6d. THE BROWSER. Against a local file:// page, so the gate is deterministic and
     *          needs no network: what is being tested is whether the model can operate a form,
     *          not whether some website is up today. ---- */
    puts("\n6d. the browser");
    if(system("command -v firefox >/dev/null 2>&1") != 0){
        puts("  --    firefox not installed, skipped");
    } else {
      wr("form.html",
        "<html><head><title>Testformular</title></head><body>"
        "<h1>Anmeldung</h1>"
        "<form id=f onsubmit=\"document.getElementById('out').textContent="
        "  'GESENDET:'+name.value+'/'+ort.value+'/'+magst.checked; return false\">"
        "<label for=name>Ihr Name</label><input id=name name=name placeholder=\"Name\">"
        "<select id=ort><option>Berlin</option><option>Hamburg</option><option>München</option></select>"
        "<input id=magst type=checkbox>"
        "<textarea id=nach placeholder=\"Nachricht\"></textarea>"
        "<input id=datei type=file>"
        "<button id=go type=submit>Absenden</button>"
        "</form><div id=out></div>"
        "<div id=slider style=\"width:300px;height:30px;background:#eee\"></div>"
        "<script>var s=document.getElementById('slider');var down=false,x0=0;"
        "s.onmousedown=function(e){down=true;x0=e.clientX};"
        "document.onmouseup=function(e){if(down){down=false;"
        " document.getElementById('out').textContent+=' GEZOGEN:'+Math.round(e.clientX-x0)}};"
        "document.getElementById('datei').onchange=function(e){"
        " document.getElementById('out').textContent+=' DATEI:'+e.target.files[0].name};"
        "</script></body></html>");
      wr("hochladen.txt", "inhalt\n");

      char url[PATH_MAX + 16];
      snprintf(url, sizeof url, "file://%s/form.html", g_root);
      Str req = {0};
      s_cat(&req, "{\"name\":\"web_open\",\"arguments\":{\"url\":");
      s_json(&req, url, strlen(url));
      s_cat(&req, "},\"approved\":true}");
      r = call(req.p, &ar, &raw);
      s_free(&req);
      const int up = !strcmp(field(r, "decision"), "allow") && has(r, "content", "Testformular");
      ok(up, "a headless firefox opens a page and reports its title");
      if(!up) printf("     (%s)\n", (field(r,"content")[0] ? field(r,"content") : field(r,"reason")));
      ok(has(r, "content", "Anmeldung"), "the VISIBLE TEXT comes back, not the HTML");
      ok(has(r, "content", "[0] text"), "and the operable elements are numbered");
      ok(has(r, "content", "Berlin | Hamburg"), "a select lists its options");
      free(ar); ar = NULL;

      if(up){
        /* filling in — the thing this exists for */
        r = call("{\"name\":\"web_do\",\"arguments\":{\"action\":\"type\",\"target\":0,"
                 "\"text\":\"Simon\"},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "=Simon"), "typing into a field lands in its value");
        free(ar); ar = NULL;

        r = call("{\"name\":\"web_do\",\"arguments\":{\"action\":\"select\",\"target\":1,"
                 "\"text\":\"München\"},\"approved\":true}", &ar, &raw);
        ok(!has(r, "content", "no option"), "an option is picked by its text");
        free(ar); ar = NULL;

        r = call("{\"name\":\"web_do\",\"arguments\":{\"action\":\"click\",\"target\":2},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "[x]"), "a checkbox can be clicked");
        free(ar); ar = NULL;

        r = call("{\"name\":\"web_do\",\"arguments\":{\"action\":\"type\",\"target\":3,"
                 "\"text\":\"Hallo Welt\"},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "=Hallo Welt"), "a textarea too");
        free(ar); ar = NULL;

        /* upload: the workspace fence applies to the local side of it */
        r = call("{\"name\":\"web_file\",\"arguments\":{\"action\":\"upload\",\"target\":4,"
                 "\"path\":\"hochladen.txt\"},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "DATEI:hochladen.txt"), "a file input takes a file and the page reacts");
        free(ar); ar = NULL;
        r = call("{\"name\":\"web_file\",\"arguments\":{\"action\":\"upload\",\"target\":4,"
                 "\"path\":\"/etc/passwd\"},\"approved\":true}", &ar, &raw);
        ok(!strcmp(field(r, "decision"), "ask"),
           "and uploading from outside the workspace asks first");
        free(ar); ar = NULL;

        /* submit, and check the PAGE agrees about what it received */
        r = call("{\"name\":\"web_do\",\"arguments\":{\"action\":\"click\",\"target\":5},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "GESENDET:Simon/München/true"),
           "the form submits with everything that was filled in");
        free(ar); ar = NULL;

        /* MOVING. A press, a move and a release, which is the only thing a drag listener sees;
         * assigning a property from script produces no events at all. The slider here is a bare
         * div with a handler attached from JS — it has no semantics for the numbering to find,
         * which is exactly why web_do also takes a CSS selector. */
        r = call("{\"name\":\"web_do\",\"arguments\":{\"action\":\"drag\","
                 "\"selector\":\"#slider\",\"dx\":120,\"dy\":0},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "GEZOGEN:"), "a drag on a bare div produces real pointer events");
        free(ar); ar = NULL;

        r = call("{\"name\":\"web_file\",\"arguments\":{\"action\":\"screenshot\","
                 "\"path\":\"schuss.png\"},\"approved\":true}", &ar, &raw);
        ok(has(r, "content", "bytes PNG"), "a screenshot is written into the workspace");
        { char png[PATH_MAX]; snprintf(png, sizeof png, "%s/schuss.png", g_root);
          FILE *pf = fopen(png, "rb"); unsigned char sig[8] = {0};
          if(pf){ if(fread(sig, 1, 8, pf)){} fclose(pf); }
          ok(sig[0] == 0x89 && sig[1] == 'P' && sig[2] == 'N' && sig[3] == 'G',
             "and it really is a PNG, not base64 text"); }
        free(ar); ar = NULL;
      }
      web_stop();
    }

    /* ---- 6e. THE CHAT STORE. It holds the thing the KV cache is keyed on, so a truncated
     *          write does not lose the last turn — it loses the prefix and with it a hundred
     *          seconds of prefill. ---- */
    puts("\n6e. chats");
    { char cr[PATH_MAX]; snprintf(cr, sizeof cr, "%s/chats", g_root);
      mkdir(cr, 0700); realpath(cr, g_chat_root); }
    ok(cs_save("proj", "c1", "Erster Chat", 2,
               "[{\"role\":\"user\",\"content\":\"hallo\"}]"), "a chat is stored");
    ok(cs_save("proj", "c2", "Zweiter", 1, "[]"), "and a second one");
    { Str t = {0}; cs_list_json(&t, "proj");
      char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      jval *d = j ? json_get(j, "data") : NULL;
      ok(d && d->t == J_ARR && d->len == 2, "the listing finds both");
      /* the title and count are read from the HEAD of the file, not by parsing a 2 MB
       * transcript per row — a sidebar that costs more than the answer it lists is a sidebar
       * nobody opens */
      ok(d && d->len == 2 && !strcmp(a_str(d->kids[0], "title", ""), "Zweiter"),
         "newest first, with its title");
      ok(d && d->len == 2 && (int)a_num(d->kids[1], "n", 0) == 2, "and its message count");
      free(a2); s_free(&t); }
    { Str t = {0};
      ok(cs_get(&t, "proj", "c1"), "a chat reads back");
      char *a2 = NULL; jval *j = json_parse(t.p, &a2);
      jval *m = j ? json_get(j, "messages") : NULL;
      ok(m && m->t == J_ARR && m->len == 1, "with its messages intact as JSON");
      ok(j && !strcmp(a_str(j, "project", ""), "proj"), "and the project it belongs to");
      free(a2); s_free(&t); }
    /* an id becomes a path component, so it gets the same whitelist as a project name */
    ok(!cs_save("proj", "../../../etc/passwd", "x", 0, "[]"), "a traversing id is refused");
    ok(!cs_save("proj", "a/b", "x", 0, "[]"), "an id with a slash is refused");
    ok(!cs_save("proj", ".hidden", "x", 0, "[]"), "a dotfile id is refused");
    ok(!cs_save("../../etc", "c1", "x", 0, "[]"), "and so is a traversing project");
    { /* nothing may be left behind by a refused save */
      char probe[PATH_MAX]; snprintf(probe, sizeof probe, "%s/chats/proj/c1.json.tmp", g_root);
      struct stat st;
      ok(stat(probe, &st) != 0, "a save leaves no temporary file behind"); }
    ok(cs_delete("proj", "c1"), "a chat deletes");
    { Str t = {0};
      ok(!cs_get(&t, "proj", "c1"), "and is then gone");
      s_free(&t); }
    g_chat_root[0] = 0;

    /* ---- 7. the escaper: a tool result goes into a JSON body ---- */
    puts("\n7. escaping");
    wr("nasty.txt", "quote \" backslash \\ newline\ttab\nsecond\x01line\n");
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"nasty.txt\"}}", &ar, &raw);
    ok(r != NULL, "a file full of quotes, backslashes and control bytes still parses back");
    ok(has(r, "content", "backslash \\"), "and its contents survive the round trip");
    free(ar); ar = NULL;

    /* ---- 8. the workspace is not left behind ---- */
    s_free(&raw);
    char rm[PATH_MAX + 32]; snprintf(rm, sizeof rm, "rm -rf '%s'", g_root);
    if(system(rm)){}

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
