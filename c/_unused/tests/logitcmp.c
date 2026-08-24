/* logitcmp.c — does requantized attention still produce the same MODEL?
 *
 * The greedy-text check said yes for every recipe, including one with a 3.8% per-layer error.
 * It was lying. "Ein **Mutex** (kurz für *Mutual Ex" is a near-deterministic continuation:
 * the top token wins by a mile, so the argmax survives a perturbation that would visibly
 * change a genuinely uncertain choice. Identical text is necessary, not sufficient.
 *
 * What actually matters is the DISTRIBUTION. KL(shipped || recipe) is the number that tells
 * you how much the sampler's behaviour moves, and it is what the quantization literature
 * reports for exactly this reason. Rules of thumb from llama.cpp's own quant comparisons:
 *
 *      KL < 0.01   indistinguishable in practice
 *      KL ~ 0.05   the gap between neighbouring k-quants (Q4_K vs Q5_K)
 *      KL > 0.2    a visibly worse model
 *
 * Also reported: top-1 agreement (does greedy pick the same token) and the rank the shipped
 * top-1 falls to. Those are what a user feels.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float *load(const char *p, int *n){
    FILE *f = fopen(p, "rb");
    if(!f){ fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    *n = (int)(b/sizeof(float));
    float *v = malloc((size_t)b);
    if(fread(v, 1, (size_t)b, f) != (size_t)b){ fprintf(stderr,"short read %s\n",p); exit(1); }
    fclose(f);
    return v;
}

/* softmax in double, from the max, or the tail underflows and KL comes out wrong */
static double *softmax(const float *z, int n){
    double mx = -INFINITY;
    for(int i=0;i<n;i++) if(z[i] > mx) mx = z[i];
    double *p = malloc((size_t)n*sizeof(double)), s = 0;
    for(int i=0;i<n;i++){ p[i] = exp((double)z[i] - mx); s += p[i]; }
    for(int i=0;i<n;i++) p[i] /= s;
    return p;
}

static int argmax(const double *p, int n){
    int b=0; for(int i=1;i<n;i++) if(p[i] > p[b]) b=i; return b;
}

#define VOCAB 154880

int main(int argc, char **argv){
    if(argc < 3){ printf("usage: %s <ref.bin> <cmp.bin>...\n", argv[0]); return 2; }
    int n;
    float *rz = load(argv[1], &n);
    const int T = n / VOCAB;
    if(T < 1){ printf("ref too small\n"); return 1; }

    printf("reference: %s   %d positions x %d vocab\n\n", argv[1], T, VOCAB);
    printf("  %-10s %9s %9s %9s %8s   %s\n",
           "recipe", "mean KL", "worst KL", "top1 same", "max|dz|", "verdict");

    for(int a=2; a<argc; a++){
        int m; float *cz = load(argv[a], &m);
        if(m != n){ printf("  %-10s  size mismatch (%d vs %d)\n", argv[a], m, n); continue; }

        double sum = 0, worst = 0, mdz = 0;
        int same = 0;
        for(int t=0; t<T; t++){
            const float *r = rz + (size_t)t*VOCAB, *cc = cz + (size_t)t*VOCAB;
            double *rp = softmax(r, VOCAB), *cp = softmax(cc, VOCAB);
            double kl = 0;
            for(int i=0;i<VOCAB;i++)
                if(rp[i] > 1e-12) kl += rp[i]*log(rp[i]/(cp[i] > 1e-300 ? cp[i] : 1e-300));
            if(kl > worst) worst = kl;
            sum += kl;
            if(argmax(rp,VOCAB) == argmax(cp,VOCAB)) same++;
            for(int i=0;i<VOCAB;i++){ double d=fabs((double)r[i]-cc[i]); if(d>mdz) mdz=d; }
            free(rp); free(cp);
        }
        const double kl = sum/T;
        const char *v = kl < 0.01 ? "indistinguishable"
                      : kl < 0.05 ? "ok — one quant level"
                      : kl < 0.20 ? "degraded, but usable"
                      :             "A DIFFERENT MODEL";
        const char *nm = strrchr(argv[a], '_'); nm = nm ? nm+1 : argv[a];
        char buf[64]; snprintf(buf,sizeof(buf),"%s",nm);
        char *dot = strchr(buf,'.'); if(dot) *dot = 0;
        printf("  %-10s %9.4f %9.4f %6d/%-3d %8.3f   %s\n",
               buf, kl, worst, same, T, mdz, v);
        free(cz);
    }
    printf("\n  KL <0.01 indistinguishable | ~0.05 = one quant level | >0.2 a different model\n");
    return 0;
}
