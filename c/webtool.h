/* webtool.h — a headless Firefox the agent can actually operate.
 *
 * WHY MARIONETTE AND NOT SELENIUM, PLAYWRIGHT OR GECKODRIVER
 *
 * Marionette is Firefox's own remote protocol and it is already in the browser you have:
 * length-prefixed JSON over a TCP socket, "<bytes>:<json>". Nothing to install, no driver
 * binary to version-match, no Python or Node in the runtime of an engine that has neither.
 * This file speaks it directly with the socket code and the JSON parser that were already
 * here.
 *
 * THREE TOOLS, NOT FIFTEEN, AND THAT IS A MEASUREMENT
 *
 * Every tool schema is prefilled on every turn. §12.3: this engine prefills at ~10.5 tok/s, so
 * a byte in a schema is a real cost paid over and over. Fifteen browser verbs would be ~2 KB
 * = ~550 tokens = the better part of a minute added to every single request in the session,
 * whether or not the browser is ever used. So the verbs collapse into an `action` enum on one
 * tool, which is a worse API in the abstract and the right one here.
 *
 * THE PAGE IS RETURNED AS A NUMBERED INDEX, NOT AS HTML
 *
 * Handing a model raw HTML is hopeless: a login page is 200 KB of markup around four
 * interesting elements, which at 4 KiB of budget it will never see. After every action the page
 * is digested into visible text plus a numbered list of the things that can be operated, and
 * the model addresses them by number. One tool call per action, no element handles crossing the
 * boundary, and the numbering is refreshed by the same call that changed the page — because an
 * index that survives a navigation is an index that clicks the wrong button.
 *
 * Depends on strbuf.h, json.h and harness.h (for h_resolve and the workspace fence).
 */
#ifndef QWEN_WEBTOOL_H
#define QWEN_WEBTOOL_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static pid_t g_ff_pid = 0;
static int   g_ff_fd  = -1;
static int   g_ff_id  = 0;
static int   g_ff_port = 0;
static char  g_ff_prof[PATH_MAX] = "";
static char  g_ff_dl[PATH_MAX]   = "";
static char  g_web_err[512]      = "";
static int   g_web_on = 1;              /* --web off to drop the tools entirely */
static pthread_mutex_t g_ff_mu = PTHREAD_MUTEX_INITIALIZER;

#define MAR_EK "element-6066-11e4-a52e-4f735466cecf"

/* ---------------- the wire ---------------- */

static int mar_write(const char *json, size_t n){
    char hdr[32];
    const int hn = snprintf(hdr, sizeof hdr, "%zu:", n);
    if(send(g_ff_fd, hdr, (size_t)hn, MSG_NOSIGNAL) != hn) return 0;
    size_t done = 0;
    while(done < n){
        const ssize_t w = send(g_ff_fd, json + done, n - done, MSG_NOSIGNAL);
        if(w <= 0) return 0;
        done += (size_t)w;
    }
    return 1;
}

/* One frame, with a deadline. A page that never finishes loading must not wedge the server. */
static int mar_read(Str *out, int timeout_ms){
    out->n = 0;
    char lenbuf[24];
    size_t li = 0;
    const double t0 = now();
    for(;;){
        if((now() - t0) * 1000.0 > timeout_ms) return 0;
        struct pollfd p = { g_ff_fd, POLLIN, 0 };
        const int r = poll(&p, 1, 200);
        if(r <= 0){ if(r < 0 && errno != EINTR) return 0; continue; }
        char c;
        if(recv(g_ff_fd, &c, 1, 0) != 1) return 0;
        if(c == ':') break;
        if(li + 1 >= sizeof lenbuf || c < '0' || c > '9') return 0;
        lenbuf[li++] = c;
    }
    lenbuf[li] = 0;
    long need = strtol(lenbuf, NULL, 10);
    if(need <= 0 || need > (32<<20)) return 0;
    char buf[16384];
    while(need > 0){
        if((now() - t0) * 1000.0 > timeout_ms) return 0;
        struct pollfd p = { g_ff_fd, POLLIN, 0 };
        const int r = poll(&p, 1, 200);
        if(r <= 0){ if(r < 0 && errno != EINTR) return 0; continue; }
        const ssize_t k = recv(g_ff_fd, buf, need < (long)sizeof buf ? (size_t)need : sizeof buf, 0);
        if(k <= 0) return 0;
        s_add(out, buf, (size_t)k);
        need -= k;
    }
    return 1;
}

/* Send a command, return its `result` as JSON text. Marionette answers [1, id, error, result];
 * an error is an object with .message, which is worth passing to the model verbatim — "element
 * not interactable" tells it exactly what to do differently. */
static int mar_cmd(const char *name, const char *params, Str *result, int timeout_ms){
    g_web_err[0] = 0;
    if(g_ff_fd < 0){ snprintf(g_web_err, sizeof g_web_err, "browser is not running"); return 0; }
    Str req = {0};
    s_fmt(&req, "[0,%d,", ++g_ff_id);
    s_json(&req, name, strlen(name));
    s_fmt(&req, ",%s]", params && *params ? params : "{}");
    const int okw = mar_write(req.p, req.n);
    s_free(&req);
    if(!okw){ snprintf(g_web_err, sizeof g_web_err, "browser connection closed"); return 0; }

    Str resp = {0};
    if(!mar_read(&resp, timeout_ms)){
        s_free(&resp);
        snprintf(g_web_err, sizeof g_web_err, "browser did not answer within %d ms", timeout_ms);
        return 0;
    }
    char *arena = NULL;
    jval *v = json_parse(resp.p, &arena);
    int ok = 0;
    if(v && v->t == J_ARR && v->len >= 4){
        jval *e = v->kids[2];
        if(e && e->t == J_OBJ){
            jval *m = json_get(e, "message");
            snprintf(g_web_err, sizeof g_web_err, "%s",
                     (m && m->t == J_STR) ? m->str : "browser refused the command");
        } else {
            if(result){ result->n = 0; j_emit(result, v->kids[3]); }
            ok = 1;
        }
    } else snprintf(g_web_err, sizeof g_web_err, "malformed answer from the browser");
    free(arena);
    s_free(&resp);
    return ok;
}

/* ExecuteScript, with the result unwrapped from Marionette's {"value":…} envelope. */
static int mar_js(const char *script, Str *out, int timeout_ms){
    Str p = {0};
    s_cat(&p, "{\"script\":");
    s_json(&p, script, strlen(script));
    s_cat(&p, ",\"args\":[]}");
    Str r = {0};
    const int ok = mar_cmd("WebDriver:ExecuteScript", p.p, &r, timeout_ms);
    s_free(&p);
    if(ok && out){
        char *arena = NULL;
        jval *v = json_parse(r.p, &arena);
        jval *val = v ? json_get(v, "value") : NULL;
        out->n = 0;
        if(val && val->t == J_STR) s_add(out, val->str, strlen(val->str));
        else if(val)              j_emit(out, val);
        free(arena);
    }
    s_free(&r);
    return ok;
}

/* ---------------- the browser ---------------- */

static void web_stop(void){
    pthread_mutex_lock(&g_ff_mu);
    if(g_ff_fd >= 0){ mar_cmd("Marionette:Quit", "{\"flags\":[\"eForceQuit\"]}", NULL, 4000);
                      close(g_ff_fd); g_ff_fd = -1; }
    if(g_ff_pid > 0){
        kill(-g_ff_pid, SIGTERM);
        for(int i = 0; i < 40; i++){
            if(waitpid(g_ff_pid, NULL, WNOHANG) == g_ff_pid) break;
            struct timespec ts = { 0, 100*1000*1000 }; nanosleep(&ts, NULL);
        }
        kill(-g_ff_pid, SIGKILL);
        waitpid(g_ff_pid, NULL, WNOHANG);
        g_ff_pid = 0;
        if(g_ff_prof[0]){
            char pf[PATH_MAX];
            snprintf(pf, sizeof pf, "%.*s/firefox.pid",
                     (int)(strrchr(g_ff_prof, '/') - g_ff_prof), g_ff_prof);
            unlink(pf);
        }
    }
    pthread_mutex_unlock(&g_ff_mu);
}

static int web_write_profile(void){
    char js[PATH_MAX + 2048];
    char path[PATH_MAX];
    if(snprintf(path, sizeof path, "%s/user.js", g_ff_prof) >= (int)sizeof path) return 0;
    FILE *f = fopen(path, "wb");
    if(!f) return 0;
    /* A PRIVATE profile on a NON-DEFAULT port, with --no-remote. Without all three this
     * attaches to the Firefox the user already has open and starts driving their tabs. */
    snprintf(js, sizeof js,
        "user_pref(\"marionette.port\", %d);\n"
        "user_pref(\"browser.shell.checkDefaultBrowser\", false);\n"
        "user_pref(\"browser.startup.homepage_override.mstone\", \"ignore\");\n"
        "user_pref(\"toolkit.telemetry.enabled\", false);\n"
        "user_pref(\"datareporting.policy.dataSubmissionEnabled\", false);\n"
        "user_pref(\"app.update.enabled\", false);\n"
        "user_pref(\"extensions.update.enabled\", false);\n"
        "user_pref(\"browser.download.folderList\", 2);\n"
        "user_pref(\"browser.download.dir\", \"%s\");\n"
        "user_pref(\"browser.download.useDownloadDir\", true);\n"
        "user_pref(\"browser.download.manager.showWhenStarting\", false);\n"
        "user_pref(\"browser.download.alwaysOpenPanel\", false);\n"
        "user_pref(\"pdfjs.disabled\", true);\n"
        "user_pref(\"browser.helperApps.neverAsk.saveToDisk\", \"application/octet-stream,"
        "application/pdf,application/zip,application/x-gzip,text/csv,text/plain,image/png,"
        "image/jpeg,application/json,application/gzip,application/x-tar\");\n",
        g_ff_port, g_ff_dl);
    fputs(js, f);
    fclose(f);
    return 1;
}

/* Started on first use, not at boot: Firefox costs half a gigabyte and several seconds, and
 * most sessions never touch it. */
static int web_start(void){
    if(g_ff_fd >= 0) return 1;
    if(!g_web_on){ snprintf(g_web_err, sizeof g_web_err, "the browser tools are off (--web off)"); return 0; }

    char base[PATH_MAX];
    if(g_home[0]) snprintf(base, sizeof base, "%s/browser", g_home);
    else          snprintf(base, sizeof base, "/tmp/qwen-browser-%d", (int)getuid());
    mkdir(base, 0700);
    snprintf(g_ff_prof, sizeof g_ff_prof, "%s/profile", base);
    snprintf(g_ff_dl,   sizeof g_ff_dl,   "%s/downloads", base);
    mkdir(g_ff_prof, 0700);
    mkdir(g_ff_dl, 0700);

    /* a free port above the default, so a Firefox the user is already debugging is untouched */
    g_ff_port = 0;
    for(int p = 2929; p < 2949; p++){
        const int t = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
        sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)p);
        sa.sin_addr.s_addr = inet_addr("127.0.0.1");
        const int one = 1;
        setsockopt(t, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        const int free_ = bind(t, (struct sockaddr*)&sa, sizeof sa) == 0;
        close(t);
        if(free_){ g_ff_port = p; break; }
    }
    if(!g_ff_port){ snprintf(g_web_err, sizeof g_web_err, "no free marionette port"); return 0; }
    if(!web_write_profile()){ snprintf(g_web_err, sizeof g_web_err, "cannot write the browser profile"); return 0; }

    const pid_t pid = fork();
    if(pid < 0){ snprintf(g_web_err, sizeof g_web_err, "fork failed"); return 0; }
    if(pid == 0){
        setsid();
        const int nul = open("/dev/null", O_RDWR);
        if(nul >= 0){ dup2(nul, 0); dup2(nul, 1); dup2(nul, 2); if(nul > 2) close(nul); }
        /* Firefox outlives every request and would otherwise hold the listening socket, the
         * model file and any connection that was open at the moment it started. It held port
         * 8080 for exactly this reason. */
        h_close_inherited();
        setenv("MOZ_HEADLESS", "1", 1);
        execlp("firefox", "firefox", "--headless", "--marionette", "--no-remote",
               "--profile", g_ff_prof, "about:blank", (char*)NULL);
        _exit(127);
    }
    g_ff_pid = pid;
    { char pf[PATH_MAX];
      snprintf(pf, sizeof pf, "%s/firefox.pid", base);
      FILE *f = fopen(pf, "w");
      if(f){ fprintf(f, "%d\n", (int)pid); fclose(f); } }

    for(int i = 0; i < 160; i++){          /* a cold profile takes seconds to come up */
        struct timespec ts = { 0, 250*1000*1000 }; nanosleep(&ts, NULL);
        if(waitpid(pid, NULL, WNOHANG) == pid){
            g_ff_pid = 0;
            snprintf(g_web_err, sizeof g_web_err, "firefox exited during startup (is it installed?)");
            return 0;
        }
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
        sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)g_ff_port);
        sa.sin_addr.s_addr = inet_addr("127.0.0.1");
        if(connect(fd, (struct sockaddr*)&sa, sizeof sa) == 0){ g_ff_fd = fd; break; }
        close(fd);
    }
    if(g_ff_fd < 0){ snprintf(g_web_err, sizeof g_web_err, "firefox did not open its marionette port");
                     web_stop(); return 0; }

    Str hello = {0};                        /* the server greets first */
    if(!mar_read(&hello, 20000)){ s_free(&hello); snprintf(g_web_err, sizeof g_web_err,
                                  "no handshake from the browser"); web_stop(); return 0; }
    s_free(&hello);
    Str r = {0};
    const int ok = mar_cmd("WebDriver:NewSession", "{\"capabilities\":{}}", &r, 60000);
    s_free(&r);
    if(!ok){ web_stop(); return 0; }
    mar_cmd("WebDriver:SetTimeouts", "{\"pageLoad\":45000,\"script\":30000,\"implicit\":0}", NULL, 5000);
    return 1;
}

/* ---------------- the page, as the model sees it ----------------
 *
 * One script does everything: it numbers the operable elements, tags them so a later call can
 * find the same one, and returns a digest. Doing it in one round trip matters because the
 * alternative is a WebDriver call per element and this model costs a minute a turn. */
static const char *WEB_DIGEST_JS =
"var out=[],n=0;"
"document.querySelectorAll('[data-q35]').forEach(function(e){e.removeAttribute('data-q35')});"
/* Draggable things usually have no semantics at all — a slider is a div with a mousedown
 * handler set from script, a canvas is a rectangle. They are included by shape (canvas,
 * range, draggable, cursor:grab) because otherwise "move that" has nothing to address. */
"var sel='a[href],button,input,select,textarea,summary,canvas,[role=button],[role=link],"
"[role=checkbox],[role=tab],[role=slider],[onclick],[onmousedown],[draggable=\"true\"],"
"[contenteditable=\"true\"]';"
"var els=document.querySelectorAll(sel);"
"for(var i=0;i<els.length&&n<80;i++){var e=els[i];"
" var r=e.getBoundingClientRect();"
" var vis=r.width>0&&r.height>0&&getComputedStyle(e).visibility!=='hidden'&&getComputedStyle(e).display!=='none';"
" if(!vis)continue;"
" e.setAttribute('data-q35',n);"
" var t=e.tagName.toLowerCase();"
" var lab=(e.getAttribute('aria-label')||e.placeholder||e.title||"
"   (e.labels&&e.labels[0]&&e.labels[0].innerText)||e.value||e.innerText||e.name||'').trim().replace(/\\s+/g,' ');"
" if(lab.length>70)lab=lab.slice(0,70)+'…';"
" var kind=t==='input'?(e.type||'text'):t;"
" var cur=getComputedStyle(e).cursor;"
" if(cur==='grab'||cur==='move'||e.draggable)kind+=' (ziehbar)';"
" var extra='';"
" if(t==='a'&&e.getAttribute('href'))extra=' -> '+e.getAttribute('href').slice(0,60);"
" if(t==='select'){var o=[];for(var k=0;k<e.options.length&&k<12;k++)o.push(e.options[k].text);"
"   extra=' {'+o.join(' | ')+'}';}"
" if((t==='input'||t==='textarea')&&e.value)extra=' ='+String(e.value).slice(0,40);"
" if(e.checked)extra+=' [x]';"
" out.push('['+n+'] '+kind+(lab?' \"'+lab+'\"':'')+extra);"
" n++;}"
"var txt=(document.body?document.body.innerText:'').replace(/\\n{3,}/g,'\\n\\n').trim();"
"if(txt.length>1800)txt=txt.slice(0,1800)+'\\n… (Seite gekürzt) …';"
"return JSON.stringify({u:location.href,t:document.title,x:txt,e:out.join('\\n'),"
" sy:Math.round(window.scrollY),sh:Math.round(document.body?document.body.scrollHeight:0),"
" ih:Math.round(window.innerHeight)});";

static int web_digest(Str *o, const char **err){
    Str raw = {0};
    if(!mar_js(WEB_DIGEST_JS, &raw, 30000)){ s_free(&raw); *err = g_web_err; return 0; }
    char *arena = NULL;
    jval *v = json_parse(raw.p, &arena);
    if(!v || v->t != J_OBJ){ free(arena); s_free(&raw); *err = "could not read the page"; return 0; }
    const char *u = a_str(v, "u", ""), *t = a_str(v, "t", "");
    const char *x = a_str(v, "x", ""), *e = a_str(v, "e", "");
    const double sy = a_num(v, "sy", 0), sh = a_num(v, "sh", 0), ih = a_num(v, "ih", 0);
    s_fmt(o, "%s\n%s\n", t && *t ? t : "(kein Titel)", u);
    if(sh > ih + 20)
        s_fmt(o, "[scroll %d%% of %d px]\n", (int)(100.0*sy/(sh - ih + 1)), (int)sh);
    s_cat(o, "\n");
    if(*x){ s_cat(o, x); s_cat(o, "\n"); }
    if(*e){ s_cat(o, "\nBedienbare Elemente:\n"); s_cat(o, e); s_cat(o, "\n"); }
    else    s_cat(o, "\n(keine bedienbaren Elemente gefunden)\n");
    free(arena);
    s_free(&raw);
    return 1;
}

/* By CSS selector, for the things the numbering cannot see: a slider that is a bare div, a
 * canvas region, anything a page renders without semantics. The numbered index covers the
 * ordinary case; this covers the rest without turning every action into a selector. */
static int web_handle_sel(const char *sel, char *out, size_t cap, const char **err){
    static __thread char msg[256];
    Str p = {0};
    s_cat(&p, "{\"using\":\"css selector\",\"value\":");
    s_json(&p, sel, strlen(sel));
    s_cat(&p, "}");
    Str r = {0};
    const int sent = mar_cmd("WebDriver:FindElement", p.p, &r, 15000);
    s_free(&p);
    if(!sent){
        s_free(&r);
        snprintf(msg, sizeof msg, "nothing matches \"%s\" on this page", sel);
        *err = msg; return 0;
    }
    char *arena = NULL;
    jval *v = json_parse(r.p, &arena);
    jval *val = v ? json_get(v, "value") : NULL;
    jval *h = val ? json_get(val, MAR_EK) : NULL;
    int ok = 0;
    if(h && h->t == J_STR){ snprintf(out, cap, "%s", h->str); ok = 1; }
    else { snprintf(msg, sizeof msg, "\"%s\" has no handle", sel); *err = msg; }
    free(arena);
    s_free(&r);
    return ok;
}

/* Resolve "element number N" to a WebDriver handle. The number came from the digest, which
 * tagged the node — so this finds the SAME node, not the n-th match of a fresh query that a
 * re-render may have reordered. */
static int web_handle(int idx, char *out, size_t cap, const char **err){
    static __thread char msg[128];
    char sel[64];
    snprintf(sel, sizeof sel, "{\"using\":\"css selector\",\"value\":\"[data-q35=\\\"%d\\\"]\"}", idx);
    Str r = {0};
    if(!mar_cmd("WebDriver:FindElement", sel, &r, 15000)){
        s_free(&r);
        snprintf(msg, sizeof msg, "no element [%d] on this page — read the page again", idx);
        *err = msg; return 0;
    }
    char *arena = NULL;
    jval *v = json_parse(r.p, &arena);
    jval *val = v ? json_get(v, "value") : NULL;
    jval *h = val ? json_get(val, MAR_EK) : NULL;
    int ok = 0;
    if(h && h->t == J_STR){ snprintf(out, cap, "%s", h->str); ok = 1; }
    else { snprintf(msg, sizeof msg, "element [%d] has no handle", idx); *err = msg; }
    free(arena);
    s_free(&r);
    return ok;
}

/* ---------------- base64, for the screenshot ---------------- */
static size_t web_unb64(const char *in, unsigned char *out){
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int v = 0, bits = 0; size_t n = 0;
    for(const char *p = in; *p; p++){
        if(*p == '=' ) break;
        const char *q = strchr(T, *p);
        if(!q) continue;                       /* newlines and stray bytes */
        v = (v << 6) | (int)(q - T); bits += 6;
        if(bits >= 8){ bits -= 8; out[n++] = (unsigned char)((v >> bits) & 0xff); }
    }
    return n;
}

/* Marionette wants the actual key CHARACTER for special keys, from the WebDriver private-use
 * block. A model will write "Enter", not U+E007. */
static const char *web_key(const char *name){
    if(!name) return NULL;
    if(!strcasecmp(name, "enter") || !strcasecmp(name, "return")) return "\xee\x80\x87";
    if(!strcasecmp(name, "tab"))       return "\xee\x80\x84";
    if(!strcasecmp(name, "escape") || !strcasecmp(name, "esc")) return "\xee\x80\x8c";
    if(!strcasecmp(name, "backspace")) return "\xee\x80\x83";
    if(!strcasecmp(name, "delete"))    return "\xee\x80\x87";
    if(!strcasecmp(name, "up"))        return "\xee\x80\x98";
    if(!strcasecmp(name, "down"))      return "\xee\x80\x95";
    if(!strcasecmp(name, "left"))      return "\xee\x80\x92";
    if(!strcasecmp(name, "right"))     return "\xee\x80\x94";
    if(!strcasecmp(name, "pageup"))    return "\xee\x80\x8e";
    if(!strcasecmp(name, "pagedown"))  return "\xee\x80\x8f";
    if(!strcasecmp(name, "home"))      return "\xee\x80\x90";
    if(!strcasecmp(name, "end"))       return "\xee\x80\x91";
    return NULL;
}

/* ---------------- the three tools ---------------- */

static int t_web_open(jval *a, Str *o, int *code, const char **err){
    (void)code;
    const char *url = a_str(a, "url", NULL);
    if(!url || !*url){ *err = "url is required"; return 0; }
    pthread_mutex_lock(&g_ff_mu);
    if(!web_start()){ pthread_mutex_unlock(&g_ff_mu); *err = g_web_err; return 0; }
    char full[2200];
    if(!strstr(url, "://")) snprintf(full, sizeof full, "https://%s", url);
    else                    snprintf(full, sizeof full, "%s", url);
    Str p = {0};
    s_cat(&p, "{\"url\":");
    s_json(&p, full, strlen(full));
    s_cat(&p, "}");
    const int ok = mar_cmd("WebDriver:Navigate", p.p, NULL, 60000);
    s_free(&p);
    if(!ok){ pthread_mutex_unlock(&g_ff_mu); *err = g_web_err; return 0; }
    const int okd = web_digest(o, err);
    pthread_mutex_unlock(&g_ff_mu);
    return okd;
}

static int t_web_do(jval *a, Str *o, int *code, const char **err){
    (void)code;
    static __thread char msg[256];
    const char *act = a_str(a, "action", NULL);
    if(!act || !*act){ *err = "action is required"; return 0; }
    const int  idx  = (int)a_num(a, "target", -1);
    const int  to   = (int)a_num(a, "to", -1);
    const char *txt = a_str(a, "text", "");
    const char *sel = a_str(a, "selector", NULL);
    const char *sel2= a_str(a, "to_selector", NULL);
    /* target OR selector; the caller may use whichever the page makes possible */
    #define WEB_GRAB(dst) (sel ? web_handle_sel(sel, dst, sizeof dst, err) \
                               : web_handle(idx, dst, sizeof dst, err))
    #define WEB_HAS_EL    (sel || idx >= 0)

    pthread_mutex_lock(&g_ff_mu);
    if(g_ff_fd < 0){ pthread_mutex_unlock(&g_ff_mu);
                     *err = "no page is open — use web_open first"; return 0; }
    int ok = 1;
    char h[128], h2[128];
    Str p = {0};

    if(!strcmp(act, "click")){
        if(!WEB_HAS_EL){ *err = "click needs target or selector"; ok = 0; }
        else if(!WEB_GRAB(h)) ok = 0;
        else { s_fmt(&p, "{\"id\":\"%s\"}", h);
               if(!mar_cmd("WebDriver:ElementClick", p.p, NULL, 30000)){ *err = g_web_err; ok = 0; } }
    } else if(!strcmp(act, "type")){
        if(!WEB_HAS_EL){ *err = "type needs target or selector"; ok = 0; }
        else if(!WEB_GRAB(h)) ok = 0;
        else {
            /* Clear first unless asked not to: a model filling a form nearly always means
             * "this is the value", not "append to whatever was there". */
            if(!a_bool(a, "append", 0)){
                s_fmt(&p, "{\"id\":\"%s\"}", h);
                mar_cmd("WebDriver:ElementClear", p.p, NULL, 15000);
                p.n = 0;
            }
            s_fmt(&p, "{\"id\":\"%s\",\"text\":", h);
            s_json(&p, txt, strlen(txt));
            s_cat(&p, "}");
            /* REAL key events, not el.value = x. Sites built on React ignore a value that was
             * assigned rather than typed, and the form then submits empty. */
            if(!mar_cmd("WebDriver:ElementSendKeys", p.p, NULL, 30000)){ *err = g_web_err; ok = 0; }
            else if(a_bool(a, "enter", 0)){
                p.n = 0;
                s_fmt(&p, "{\"id\":\"%s\",\"text\":\"\\ue007\"}", h);
                mar_cmd("WebDriver:ElementSendKeys", p.p, NULL, 30000);
            }
        }
    } else if(!strcmp(act, "select")){
        if(idx < 0){ *err = "select needs target"; ok = 0; }
        else {
            Str js = {0};
            s_fmt(&js, "var e=document.querySelector('[data-q35=\"%d\"]');if(!e)return 'no';"
                       "var w=%s;var hit=-1;"
                       "for(var i=0;i<e.options.length;i++){var t=e.options[i].text.trim();"
                       " if(t===w||e.options[i].value===w){hit=i;break}}"
                       "if(hit<0)for(var i=0;i<e.options.length;i++)"
                       " if(e.options[i].text.toLowerCase().indexOf(w.toLowerCase())>=0){hit=i;break}"
                       "if(hit<0)return 'miss';e.selectedIndex=hit;"
                       "e.dispatchEvent(new Event('input',{bubbles:true}));"
                       "e.dispatchEvent(new Event('change',{bubbles:true}));return 'ok';", idx, "W");
            /* the option text is data, so it goes in as a JSON string, not as source */
            Str js2 = {0};
            s_cat(&js2, "var W=");
            s_json(&js2, txt, strlen(txt));
            s_cat(&js2, ";");
            s_add(&js2, js.p, js.n);
            Str r = {0};
            if(!mar_js(js2.p, &r, 15000)){ *err = g_web_err; ok = 0; }
            else if(!strcmp(r.p ? r.p : "", "miss")){
                snprintf(msg, sizeof msg, "no option matching \"%s\" in element [%d]", txt, idx);
                *err = msg; ok = 0;
            } else if(!strcmp(r.p ? r.p : "", "no")){
                snprintf(msg, sizeof msg, "no element [%d]", idx); *err = msg; ok = 0;
            }
            s_free(&js); s_free(&js2); s_free(&r);
        }
    } else if(!strcmp(act, "press")){
        const char *k = web_key(txt);
        if(!k){ snprintf(msg, sizeof msg, "unknown key \"%s\"", txt); *err = msg; ok = 0; }
        else if(WEB_HAS_EL && WEB_GRAB(h)){
            s_fmt(&p, "{\"id\":\"%s\",\"text\":", h);
            s_json(&p, k, strlen(k));
            s_cat(&p, "}");
            if(!mar_cmd("WebDriver:ElementSendKeys", p.p, NULL, 20000)){ *err = g_web_err; ok = 0; }
        } else if(WEB_HAS_EL) ok = 0;
        else {
            s_cat(&p, "{\"actions\":[{\"type\":\"key\",\"id\":\"k\",\"actions\":"
                      "[{\"type\":\"keyDown\",\"value\":");
            s_json(&p, k, strlen(k));
            s_cat(&p, "},{\"type\":\"keyUp\",\"value\":");
            s_json(&p, k, strlen(k));
            s_cat(&p, "}]}]}");
            if(!mar_cmd("WebDriver:PerformActions", p.p, NULL, 20000)){ *err = g_web_err; ok = 0; }
        }
    } else if(!strcmp(act, "scroll")){
        long dy = (long)a_num(a, "amount", 0);
        if(!dy){
            if(!strcasecmp(txt, "up"))        dy = -600;
            else if(!strcasecmp(txt, "top"))  dy = -100000;
            else if(!strcasecmp(txt, "bottom")) dy = 100000;
            else                              dy = 600;
        }
        Str js = {0};
        if(idx >= 0) s_fmt(&js, "var e=document.querySelector('[data-q35=\"%d\"]');"
                                "if(e)e.scrollIntoView({block:'center'});return '';", idx);
        else         s_fmt(&js, "window.scrollBy(0,%ld);return '';", dy);
        if(!mar_js(js.p, NULL, 15000)){ *err = g_web_err; ok = 0; }
        s_free(&js);
    } else if(!strcmp(act, "drag")){
        /* A real pointer press, move and release — which is the only thing that works on a
         * slider, a canvas or an HTML5 drag target. Setting properties from script does not
         * produce the events those listen for. */
        const long dx0 = (long)a_num(a, "x", -1), dy0 = (long)a_num(a, "y", -1);
        if(!WEB_HAS_EL && dx0 < 0){ *err = "drag needs target, selector, or x/y"; ok = 0; }
        else if(WEB_HAS_EL && !WEB_GRAB(h)) ok = 0;
        else {
            const int from_el = WEB_HAS_EL;
            const int has_to = sel2 ? web_handle_sel(sel2, h2, sizeof h2, err)
                             : (to >= 0 && web_handle(to, h2, sizeof h2, err));
            const long dx = (long)a_num(a, "dx", 0), dyv = (long)a_num(a, "dy", 0);
            s_cat(&p, "{\"actions\":[{\"type\":\"pointer\",\"id\":\"m\","
                      "\"parameters\":{\"pointerType\":\"mouse\"},\"actions\":[");
            if(from_el)
                s_fmt(&p, "{\"type\":\"pointerMove\",\"duration\":60,\"x\":0,\"y\":0,"
                          "\"origin\":{\"%s\":\"%s\"}},", MAR_EK, h);
            else
                s_fmt(&p, "{\"type\":\"pointerMove\",\"duration\":60,\"x\":%ld,\"y\":%ld,"
                          "\"origin\":\"viewport\"},", dx0, dy0);
            s_cat(&p, "{\"type\":\"pointerDown\",\"button\":0},");
            if(has_to) s_fmt(&p, "{\"type\":\"pointerMove\",\"duration\":250,\"x\":0,\"y\":0,"
                                 "\"origin\":{\"%s\":\"%s\"}},", MAR_EK, h2);
            else       s_fmt(&p, "{\"type\":\"pointerMove\",\"duration\":250,\"x\":%ld,\"y\":%ld,"
                                 "\"origin\":\"pointer\"},", dx, dyv);
            s_cat(&p, "{\"type\":\"pointerUp\",\"button\":0}]}]}");
            if(!mar_cmd("WebDriver:PerformActions", p.p, NULL, 30000)){ *err = g_web_err; ok = 0; }
        }
    } else if(!strcmp(act, "move")){
        if(!WEB_HAS_EL){ *err = "move needs target or selector"; ok = 0; }
        else if(!WEB_GRAB(h)) ok = 0;
        else {
            s_fmt(&p, "{\"actions\":[{\"type\":\"pointer\",\"id\":\"m\","
                      "\"parameters\":{\"pointerType\":\"mouse\"},\"actions\":"
                      "[{\"type\":\"pointerMove\",\"duration\":80,\"x\":0,\"y\":0,"
                      "\"origin\":{\"%s\":\"%s\"}}]}]}", MAR_EK, h);
            if(!mar_cmd("WebDriver:PerformActions", p.p, NULL, 20000)){ *err = g_web_err; ok = 0; }
        }
    } else if(!strcmp(act, "back") || !strcmp(act, "forward") || !strcmp(act, "reload")){
        const char *c2 = !strcmp(act, "back") ? "WebDriver:Back"
                       : !strcmp(act, "forward") ? "WebDriver:Forward" : "WebDriver:Refresh";
        if(!mar_cmd(c2, "{}", NULL, 45000)){ *err = g_web_err; ok = 0; }
    } else if(!strcmp(act, "wait")){
        long ms = (long)a_num(a, "amount", 1500);
        if(ms < 0) ms = 0;
        if(ms > 20000) ms = 20000;
        pthread_mutex_unlock(&g_ff_mu);
        struct timespec ts = { ms/1000, (ms%1000)*1000000L };
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&g_ff_mu);
    } else {
        snprintf(msg, sizeof msg, "unknown action \"%s\"", act);
        *err = msg; ok = 0;
    }
    s_free(&p);
    #undef WEB_GRAB
    #undef WEB_HAS_EL

    if(ok){
        /* Let a click settle before reading: most pages navigate or re-render, and a digest
         * taken in the same millisecond describes the page that is about to be replaced. */
        if(strcmp(act, "wait")){
            struct timespec ts = { 0, 400*1000*1000 };
            pthread_mutex_unlock(&g_ff_mu); nanosleep(&ts, NULL); pthread_mutex_lock(&g_ff_mu);
        }
        ok = web_digest(o, err);
    }
    pthread_mutex_unlock(&g_ff_mu);
    return ok;
}

static int t_web_file(jval *a, Str *o, int *code, const char **err){
    (void)code;
    static __thread char msg[PATH_MAX + 128];
    const char *act = a_str(a, "action", NULL);
    if(!act){ *err = "action is required"; return 0; }
    const int idx = (int)a_num(a, "target", -1);

    if(!strcmp(act, "screenshot")){
        char abs[PATH_MAX];
        if(!h_resolve(a_str(a, "path", "screenshot.png"), 0, abs, err)) return 0;
        pthread_mutex_lock(&g_ff_mu);
        if(g_ff_fd < 0){ pthread_mutex_unlock(&g_ff_mu); *err = "no page is open"; return 0; }
        Str r = {0};
        const int ok = mar_cmd("WebDriver:TakeScreenshot",
                               "{\"full\":true,\"hash\":false,\"scroll\":true}", &r, 45000);
        pthread_mutex_unlock(&g_ff_mu);
        if(!ok){ s_free(&r); *err = g_web_err; return 0; }
        char *arena = NULL;
        jval *v = json_parse(r.p, &arena);
        jval *val = v ? json_get(v, "value") : NULL;
        int done = 0;
        if(val && val->t == J_STR){
            unsigned char *bin = (unsigned char*)malloc(strlen(val->str));
            if(bin){
                const size_t n = web_unb64(val->str, bin);
                FILE *f = fopen(abs, "wb");
                if(f){ done = fwrite(bin, 1, n, f) == n; fclose(f);
                       s_fmt(o, "%s: %zu bytes PNG\n", h_rel(abs), n); }
                free(bin);
            }
        }
        free(arena); s_free(&r);
        if(!done){ *err = "could not save the screenshot"; return 0; }
        return 1;
    }

    if(!strcmp(act, "upload")){
        char abs[PATH_MAX];
        if(!h_resolve(a_str(a, "path", NULL), 1, abs, err)) return 0;   /* the fence applies */
        if(idx < 0){ *err = "upload needs target — the [n] of the file input"; return 0; }
        pthread_mutex_lock(&g_ff_mu);
        if(g_ff_fd < 0){ pthread_mutex_unlock(&g_ff_mu); *err = "no page is open"; return 0; }
        char h[128];
        int ok = web_handle(idx, h, sizeof h, err);
        if(ok){
            Str p = {0};
            s_fmt(&p, "{\"id\":\"%s\",\"text\":", h);
            s_json(&p, abs, strlen(abs));
            s_cat(&p, "}");
            ok = mar_cmd("WebDriver:ElementSendKeys", p.p, NULL, 30000);
            if(!ok) *err = g_web_err;
            s_free(&p);
        }
        if(ok){ s_fmt(o, "%s an Element [%d] übergeben\n\n", h_rel(abs), idx);
                ok = web_digest(o, err); }
        pthread_mutex_unlock(&g_ff_mu);
        return ok;
    }

    if(!strcmp(act, "download")){
        const char *url = a_str(a, "url", NULL);
        char abs[PATH_MAX];
        if(!h_resolve(a_str(a, "path", NULL), 0, abs, err)) return 0;
        pthread_mutex_lock(&g_ff_mu);
        if(!web_start()){ pthread_mutex_unlock(&g_ff_mu); *err = g_web_err; return 0; }
        /* Note what is already there, then let the BROWSER fetch it — it has the cookies and
         * the session that got us to the page, which a bare HTTP client would not. */
        DIR *d = opendir(g_ff_dl);
        char before[512][256]; int nb = 0;
        if(d){ struct dirent *e;
               while((e = readdir(d)) && nb < 512) snprintf(before[nb++], 256, "%s", e->d_name);
               closedir(d); }
        int ok = 1;
        if(url && *url){
            Str p = {0};
            s_cat(&p, "{\"url\":");
            s_json(&p, url, strlen(url));
            s_cat(&p, "}");
            /* a download navigation does not "complete" as a page load; the error is expected */
            mar_cmd("WebDriver:Navigate", p.p, NULL, 20000);
            s_free(&p);
        } else if(idx >= 0){
            char h[128];
            if(!web_handle(idx, h, sizeof h, err)) ok = 0;
            else { Str p = {0}; s_fmt(&p, "{\"id\":\"%s\"}", h);
                   mar_cmd("WebDriver:ElementClick", p.p, NULL, 20000); s_free(&p); }
        } else { *err = "download needs url or target"; ok = 0; }

        char found[PATH_MAX] = "";
        for(int i = 0; ok && i < 240 && !found[0]; i++){          /* up to a minute */
            struct timespec ts = { 0, 250*1000*1000 }; nanosleep(&ts, NULL);
            DIR *d2 = opendir(g_ff_dl);
            if(!d2) break;
            struct dirent *e;
            while((e = readdir(d2))){
                if(e->d_name[0] == '.') continue;
                if(strstr(e->d_name, ".part")) continue;          /* still arriving */
                int seen = 0;
                for(int k = 0; k < nb; k++) if(!strcmp(before[k], e->d_name)) seen = 1;
                if(!seen){ snprintf(found, sizeof found, "%s/%s", g_ff_dl, e->d_name); break; }
            }
            closedir(d2);
        }
        pthread_mutex_unlock(&g_ff_mu);
        if(!ok) return 0;
        if(!found[0]){ *err = "nothing was downloaded within a minute"; return 0; }
        if(rename(found, abs) != 0){                              /* across filesystems: copy */
            FILE *in = fopen(found, "rb"), *out = fopen(abs, "wb");
            if(!in || !out){ if(in) fclose(in); if(out) fclose(out);
                             snprintf(msg, sizeof msg, "cannot move the download to %s", h_rel(abs));
                             *err = msg; return 0; }
            char b[65536]; size_t k;
            while((k = fread(b, 1, sizeof b, in)) > 0) fwrite(b, 1, k, out);
            fclose(in); fclose(out); unlink(found);
        }
        struct stat st;
        s_fmt(o, "%s: %lld bytes\n", h_rel(abs), stat(abs, &st) == 0 ? (long long)st.st_size : 0LL);
        return 1;
    }

    snprintf(msg, sizeof msg, "unknown action \"%s\"", act);
    *err = msg;
    return 0;
}

#endif /* QWEN_WEBTOOL_H */
