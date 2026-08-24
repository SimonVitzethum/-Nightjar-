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
    char tmpl[] = "/tmp/colibri_harness_XXXXXX";
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

    /* ---- 4. THE CONFINEMENT. Three ways out, all of which a strncmp on the argument as
     *         typed would let through. ---- */
    puts("\n4. confinement");
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"../../../etc/passwd\"}}", &ar, &raw);
    ok(has(r, "content", "outside the workspace"), "..  cannot climb out");
    /* and it must READ as a refusal, not as a tool that happened to fail — the caller renders
     * those differently and the model is told a different thing */
    ok(!strcmp(field(r, "decision"), "deny"), "leaving the workspace is reported as a refusal");
    free(ar); ar = NULL;

    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"/etc/passwd\"}}", &ar, &raw);
    ok(has(r, "content", "outside the workspace"), "an absolute path outside is refused");
    free(ar); ar = NULL;

    { char link[PATH_MAX]; snprintf(link, sizeof link, "%s/escape", g_root);
      ok(symlink("/etc", link) == 0, "(made a symlink to /etc inside the workspace)"); }
    r = call("{\"name\":\"read_file\",\"arguments\":{\"path\":\"escape/passwd\"}}", &ar, &raw);
    ok(has(r, "content", "outside the workspace"), "a symlink out of the tree is refused");
    free(ar); ar = NULL;

    r = call("{\"name\":\"write_file\",\"arguments\":{\"path\":\"../escaped.txt\",\"content\":\"no\"}}", &ar, &raw);
    ok(has(r, "content", "outside the workspace"), "a write above the root is refused");
    free(ar); ar = NULL;

    r = call("{\"name\":\"glob\",\"arguments\":{\"pattern\":\"*\",\"path\":\"/\"}}", &ar, &raw);
    ok(has(r, "content", "outside the workspace"), "the search root itself is confined");
    free(ar); ar = NULL;

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
