/* qwen35_cupti.cu — the real GPU timeline for one decode run, straight from CUPTI.
 *
 * WHY THIS EXISTS
 *   The speed plan's G1 gate asks one question: of the 40 ms a token takes, how much is the
 *   GPU actually executing? QWEN_PROFILE cannot answer it — it brackets each stage with a
 *   device sync, which inflates the total and erases the very bubbles it should expose.
 *   nsys is not installed here, but libcupti is, and that is the library nsys records with.
 *   So we record with it directly.
 *
 * WHAT IT MEASURES
 *   Every kernel and every memcpy gets (queued, submitted, start, end) in nanoseconds on the
 *   GPU clock. From that:
 *     busy   = union of all [start,end) — wall time with at least one op in flight
 *     sum    = Σ durations — exceeds busy exactly when streams overlap, which is how much
 *              the copy stream is actually hiding behind compute (G3)
 *     idle   = window - busy, the launch/sync bubble the plan calls the "Launch-Sockel" (G1)
 *   Aggregated per kernel name, and per memcpy kind and stream.
 *
 * Off unless QWEN_CUPTI=1. Enabling costs a few percent; it does not serialize (the
 * CONCURRENT_KERNEL activity is the non-serializing one).
 */
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cxxabi.h>

#define BUF_SIZE  (1024*1024)
#define BUF_ALIGN 8
#define MAXNAME   192
#define MAXKIND   256
#define MAXIVAL   (1u<<22)          /* 4M ops; ~64 MiB. A 100-token run makes ~120k. */

typedef struct { uint64_t s, e; int b; } Ival;
typedef struct {
    char     name[MAXNAME];
    uint64_t n, dur, bytes;
    uint64_t queue_lat, submit_lat;   /* Σ(submitted-queued), Σ(start-submitted) */
    uint64_t gap;                     /* Σ GPU idle immediately BEFORE this op ran */
    uint32_t stream;
    int      is_copy;
} Bucket;

static Bucket   g_b[MAXKIND];
static int      g_nb = 0;
static Ival    *g_iv = NULL;
static uint32_t g_niv = 0;
static uint64_t g_t_reset = 0, g_t_report = 0;
static int      g_on = 0, g_active = 0, g_overflow = 0;

extern "C" int q35cupti_on(void){ return g_on; }

/* ---- record sink ---- */

static void demangle(const char *in, char *out, size_t cap){
    int st = 0; char *d = abi::__cxa_demangle(in, NULL, NULL, &st);
    const char *s = (st == 0 && d) ? d : in;
    /* template args make every gemv look unique; cut at the first '<' or '(' */
    size_t i = 0; for(; s[i] && i+1 < cap && s[i] != '<' && s[i] != '('; i++) out[i] = s[i];
    out[i] = 0;
    free(d);
}

static Bucket *bucket(const char *name, uint32_t stream, int is_copy){
    for(int i = 0; i < g_nb; i++)
        if(g_b[i].stream == stream && g_b[i].is_copy == is_copy && !strcmp(g_b[i].name, name))
            return &g_b[i];
    if(g_nb >= MAXKIND) return &g_b[MAXKIND-1];
    Bucket *b = &g_b[g_nb++];
    memset(b, 0, sizeof *b);
    snprintf(b->name, MAXNAME, "%s", name);
    b->stream = stream; b->is_copy = is_copy;
    return b;
}

static void add_ival(uint64_t s, uint64_t e, int b){
    if(g_niv < MAXIVAL){ g_iv[g_niv].s = s; g_iv[g_niv].e = e; g_iv[g_niv].b = b; g_niv++; }
    else g_overflow = 1;
}

static const char *copyname(uint8_t k){
    switch(k){
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:  return "memcpy HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:  return "memcpy DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:  return "memcpy DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:  return "memcpy HtoH";
    default:                               return "memcpy other";
    }
}

static void handle(CUpti_Activity *rec){
    if(!g_active) return;
    if(rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL){
        CUpti_ActivityKernel12 *k = (CUpti_ActivityKernel12*)rec;
        if(k->start < g_t_reset) return;
        char nm[MAXNAME]; demangle(k->name ? k->name : "?", nm, sizeof nm);
        char key[MAXNAME];
        snprintf(key, sizeof key, "%s g=%d", nm, k->gridX * (k->gridY > 1 ? k->gridY : 1));
        Bucket *b = bucket(key, k->streamId, 0);
        b->n++; b->dur += k->end - k->start;
        if(k->queued && k->submitted >= k->queued)   b->queue_lat  += k->submitted - k->queued;
        if(k->submitted && k->start >= k->submitted) b->submit_lat += k->start - k->submitted;
        add_ival(k->start, k->end, (int)(b - g_b));
    } else if(rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY){
        CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6*)rec;
        if(m->start < g_t_reset) return;
        char key[MAXNAME];
        const unsigned long long kb = (unsigned long long)m->bytes >> 10;
        snprintf(key, sizeof key, "%s %s", copyname(m->copyKind),
                 kb == 0 ? "<1KiB" : kb < 16 ? "1-16KiB" : kb < 256 ? "16-256KiB" : ">=256KiB");
        Bucket *b = bucket(key, m->streamId, 1);
        b->n++; b->dur += m->end - m->start; b->bytes += m->bytes;
        add_ival(m->start, m->end, (int)(b - g_b));
    }
}

static void CUPTIAPI on_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords){
    uint8_t *p = (uint8_t*)malloc(BUF_SIZE + BUF_ALIGN);
    *buffer = (uint8_t*)(((uintptr_t)p + (BUF_ALIGN-1)) & ~(uintptr_t)(BUF_ALIGN-1));
    *size = BUF_SIZE; *maxNumRecords = 0;
    /* leak the unaligned base: CUPTI hands back the aligned pointer and the run is short */
}

static void CUPTIAPI on_complete(CUcontext, uint32_t, uint8_t *buffer, size_t, size_t validSize){
    CUpti_Activity *rec = NULL;
    if(validSize) while(cuptiActivityGetNextRecord(buffer, validSize, &rec) == CUPTI_SUCCESS) handle(rec);
}

/* ---- control ---- */

extern "C" void q35cupti_init(void){
    const char *e = getenv("QWEN_CUPTI");
    if(!e || !*e || !strcmp(e, "0")) return;
    g_iv = (Ival*)malloc(sizeof(Ival)*(size_t)MAXIVAL);
    if(!g_iv){ fprintf(stderr, "qwen35_cupti: no memory for the timeline\n"); return; }
    if(cuptiActivityRegisterCallbacks(on_request, on_complete) != CUPTI_SUCCESS){
        fprintf(stderr, "qwen35_cupti: cuptiActivityRegisterCallbacks failed\n"); return; }
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    g_on = 1;
    fprintf(stderr, "  cupti: GPU timeline recording armed\n");
}

extern "C" void q35cupti_reset(void){
    if(!g_on) return;
    cuptiActivityFlushAll(1);
    g_nb = 0; g_niv = 0; g_overflow = 0;
    memset(g_b, 0, sizeof g_b);
    cuptiGetTimestamp(&g_t_reset);
    g_active = 1;
}

static int ivcmp(const void *a, const void *b){
    const Ival *x = (const Ival*)a, *y = (const Ival*)b;
    return x->s < y->s ? -1 : x->s > y->s ? 1 : 0;
}
static int bcmp(const void *a, const void *b){
    const Bucket *x = (const Bucket*)a, *y = (const Bucket*)b;
    const uint64_t xc = x->dur + x->gap, yc = y->dur + y->gap;
    return xc < yc ? 1 : xc > yc ? -1 : 0;
}

extern "C" void q35cupti_report(int ntok){
    if(!g_on) return;
    cuptiGetTimestamp(&g_t_report);
    cuptiActivityFlushAll(1);
    g_active = 0;
    if(!g_niv){ fprintf(stderr, "  cupti: no activity recorded\n"); return; }

    const double win = (double)(g_t_report - g_t_reset) * 1e-9;

    /* union of every op interval = time the GPU had something in flight */
    qsort(g_iv, g_niv, sizeof(Ival), ivcmp);
    uint64_t busy = 0, cs = g_iv[0].s, ce = g_iv[0].e;
    for(uint32_t i = 1; i < g_niv; i++){
        if(g_iv[i].s > ce){ busy += ce - cs; cs = g_iv[i].s; ce = g_iv[i].e; }
        else if(g_iv[i].e > ce) ce = g_iv[i].e;
    }
    busy += ce - cs;

    for(uint32_t i = 1; i < g_niv; i++)
        if(g_iv[i].s > g_iv[i-1].e) g_b[g_iv[i].b].gap += g_iv[i].s - g_iv[i-1].e;

    uint64_t sum = 0, nops = 0, qlat = 0, slat = 0;
    for(int i = 0; i < g_nb; i++){ sum += g_b[i].dur; nops += g_b[i].n;
                                   qlat += g_b[i].queue_lat; slat += g_b[i].submit_lat; }

    const double busy_s = busy*1e-9, sum_s = sum*1e-9;
    fprintf(stderr, "\n  ── CUPTI GPU timeline%s ──\n", g_overflow ? " (TRUNCATED)" : "");
    fprintf(stderr, "  window %.3f s   ops %llu (%.0f/token)   tokens %d\n",
            win, (unsigned long long)nops, ntok ? (double)nops/ntok : 0.0, ntok);
    fprintf(stderr, "  GPU busy %.3f s = %.1f%% of window      idle/bubble %.3f s = %.1f%%\n",
            busy_s, 100*busy_s/win, win-busy_s, 100*(win-busy_s)/win);
    fprintf(stderr, "  Σ op durations %.3f s (%.2fx busy → %s)\n", sum_s, busy ? sum_s/busy_s : 0.0,
            sum > busy*102/100 ? "streams DO overlap" : "no meaningful overlap");
    if(ntok) fprintf(stderr, "  per token: %.2f ms wall | %.2f ms busy | %.2f ms idle\n",
                     1e3*win/ntok, 1e3*busy_s/ntok, 1e3*(win-busy_s)/ntok);
    fprintf(stderr, "  launch latency: Σ queued→submitted %.3f s, Σ submitted→start %.3f s\n",
            qlat*1e-9, slat*1e-9);

    qsort(g_b, g_nb, sizeof(Bucket), bcmp);
    fprintf(stderr, "\n  %-30s %8s %9s %9s %7s %9s %7s %7s\n",
            "kernel / copy", "count", "total ms", "µs/call", "%busy", "idle-before", "µs/gap", "GB/s");
    for(int i = 0; i < g_nb; i++){
        const Bucket *b = &g_b[i];
        const double ms = b->dur*1e-6;
        char gbs[16] = "";
        if(b->is_copy && b->dur) snprintf(gbs, sizeof gbs, "%.1f", b->bytes/(double)b->dur);
        fprintf(stderr, "  %-30s %8llu %9.2f %9.1f %6.1f%% %9.2f %7.1f %7s\n",
                b->name, (unsigned long long)b->n, ms,
                b->n ? 1e3*ms/b->n : 0.0, busy ? 100.0*b->dur/busy : 0.0,
                b->gap*1e-6, b->n ? 1e-3*b->gap/b->n : 0.0, gbs);
    }
    fprintf(stderr, "\n");
}
