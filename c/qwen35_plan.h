#ifndef COLIBRI_QWEN35_PLAN_H
#define COLIBRI_QWEN35_PLAN_H
/* qwen35_plan.h — decide WHERE every byte lives, from the machine's real numbers.
 *
 * THE RULE THIS ENCODES
 *
 *   1. The disk must never idle while there is anything left to read.
 *   2. Whatever is left must saturate EITHER the RAM bus (CPU computing in place) OR the
 *      PCIe bus (GPU computing on streamed bytes) — and ideally both at once, because the
 *      two draw on different resources and their bandwidths ADD.
 *
 * The corollary, which is the actual placement policy: a byte should be computed on the
 * side of the bus it ALREADY sits on. A RAM-resident tensor pushed to the GPU pays the full
 * PCIe hop for nothing; a VRAM-resident tensor pulled back to the CPU pays it twice. So:
 *
 *   FFN, 9.65 GiB, read once per token   -> RAM, computed by the CPU. 64 gemv pairs of
 *                                           5120x17408. Nothing crosses the bus.
 *   GDN + attn weights, 4.11 GiB         -> VRAM, resident. Read every token, small enough
 *                                           to fit, and the GDN recurrence is a 48-step
 *                                           latency chain the CPU handles badly.
 *   output proj, 0.97 GiB                -> VRAM, resident. One gemv per token; resident
 *                                           costs zero PCIe forever.
 *   KV cache                             -> the ONLY thing that grows. It is therefore the
 *                                           only thing worth spending bus bandwidth on, and
 *                                           the only thing that can overflow to disk.
 *
 * WHY THE KV IS THE INTERESTING ONE, AND WHY IT DOES NOT GET TO LIVE IN RAM
 *
 * At 250k tokens and q8_0 the KV is 8.1 GiB. It would ALMOST fit in what is left of RAM
 * after the FFN — and planning around that is a trap. RAM is already committed: 9.65 GiB of
 * FFN that the CPU reads every single token, plus the embeddings, plus whatever the machine
 * was doing before this process started. Sizing the KV tier to "whatever happens to be free"
 * produces a plan that works on an idle box, swaps on a busy one, and cannot survive a
 * longer context or a second sequence. It also scales exactly nowhere: 500k does not fit,
 * ever.
 *
 * So the policy is the opposite. THE KV STREAMS FROM DISK BY DEFAULT. RAM holds only a
 * bounded hot window — the newest chunks, which are the ones still being written — and the
 * cold prefix, which is write-once and read-sequentially, lives where write-once
 * read-sequentially belongs.
 *
 * That is affordable because of what the access pattern is. Every generated token scans
 * positions 0..T of 16 layers IN ORDER, known a whole layer in advance. Measured on this
 * machine, 8.0 GiB off the NVMe with O_DIRECT and 8 readers in flight: 6.83 GB/s aggregate.
 * The drive, not the plan, is the limit — which is the point.
 *
 *   VRAM   the newest tokens — no bus traffic at all
 *   RAM    a bounded hot window (COLIBRI_KV_RAM_GB, default 1) — NOT "the rest"
 *   NVMe   everything else — O_DIRECT, 8 deep, read ahead of the compute
 *
 * THE OVERLAP THAT MAKES IT WORK
 *
 * Between two consecutive full-attention layers there are 3 GDN layers and 4 FFNs. The FFNs
 * are 4 x 154 MiB of CPU-side RAM traffic; at a measured 25-40 GB/s that is 16-25 ms of
 * compute during which the disk and the PCIe bus have nothing else to do. The KV for the
 * NEXT attention layer is prefetched into that window. The layer costs max(read, compute),
 * not the sum — the same trick moe_hetero.h plays with experts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "qwen35.h"

typedef struct {
    /* inputs */
    int64_t vram_total, vram_free;
    int64_t ram_total,  ram_avail;
    int64_t reserve;          /* RAM the plan must not touch (desktop, page cache, slack) */
    int     n_ctx;
    int     kv_fmt;

    /* weight placement */
    int64_t vram_weights;     /* gdn + attn + output */
    int64_t ram_weights;      /* ffn + token_embd + norms */
    int64_t gdn_state;        /* recurrent + conv state, VRAM */
    int64_t vram_work;        /* activations, staging, cublas slack */

    /* kv tiering */
    int64_t kv_per_token;
    int64_t kv_total;
    int64_t kv_vram, kv_ram, kv_disk;
    int64_t kv_ram_cap;       /* the policy cap, not "whatever was left over" */
    int     tok_vram, tok_ram, tok_disk;

    /* derived traffic per generated token */
    int64_t bytes_dma;
    int64_t bytes_cpu_ram;    /* CPU reads out of RAM */
    int64_t bytes_pcie;       /* host->device over PCIe */
    int64_t bytes_disk;       /* NVMe */

    int     gpu_mode;         /* 1 = gdn/attn/output resident in VRAM, 0 = everything on CPU */
    int     ok;
    char    why[512];
} Q35Plan;

/* MiB/GiB helpers kept local so the header stays standalone */
static double q35_gib(int64_t b){ return (double)b / (1024.0*1024.0*1024.0); }

static int64_t q35_ram_avail(void){ return q35_ram_avail_bytes(); }
static int64_t q35_ram_total(void){ return q35_meminfo_kb("MemTotal:"); }

/* Simon's rule, kept env-configurable: never plan into the last N GB of RAM. Pinned pages
 * cannot be reclaimed, so an over-optimistic plan does not swap, it OOM-kills the desktop. */
static int64_t q35_reserve(void){ return q35_reserve_bytes(); }

/* Build the plan. vram_free/ram_avail are passed in so this stays testable without a GPU. */
static void q35_plan(Q35Plan *P, const Q35Model *M, int n_ctx, int kv_fmt,
                     int64_t vram_free, int64_t ram_avail){
    memset(P, 0, sizeof *P);
    const Q35Cfg *c = &M->c;
    Q35Bytes B; q35_bytes(M, &B);

    P->n_ctx      = n_ctx;
    P->kv_fmt     = kv_fmt;
    P->vram_free  = vram_free;
    P->ram_total  = q35_ram_total();
    P->ram_avail  = ram_avail;
    P->reserve    = q35_reserve();

    P->vram_weights = B.gdn + B.attn + B.out;
    P->ram_weights  = B.ffn + B.embd + B.norms;
    P->gdn_state    = q35_gdn_state_bytes(c);

    /* Working set on the device: two KV staging chunks, the q/k/v and logits buffers, and
     * slack for cuBLAS workspaces. Sized generously — running out of VRAM mid-decode is a
     * hard failure, running 256 MB short of optimal is a rounding error. */
    P->vram_work = 512LL << 20;

    P->kv_per_token = q35_kv_bytes_per_token(c, kv_fmt);
    P->kv_total     = P->kv_per_token * (int64_t)n_ctx;

    int64_t vram_left = vram_free - P->vram_weights - P->gdn_state - P->vram_work;
    if(vram_left < 0) vram_left = 0;

    /* RAM available to the KV is a POLICY, not a leftover. The FFN and the reserve are
     * subtracted first because they are non-negotiable; then the KV hot window is capped, so
     * that a longer context spills further onto the disk instead of eating the machine. */
    int64_t ram_left = ram_avail - P->ram_weights - P->reserve;
    if(ram_left < 0) ram_left = 0;
    /* AUTO by default: take what is measurably free after the weights and the reserve, and
     * spill only the remainder. Not "assume it fits" (a plan that works on an idle box and
     * swaps on a busy one) and not "cap it low on principle" either (on a 32 GiB machine
     * that pushes 5 GiB onto the drive for no reason, and the drive is 5x slower than RAM).
     *
     * The measurement is MemAvailable, taken now, minus a reserve that is never planned into.
     * COLIBRI_KV_RAM_GB pins it if you want the disk path exercised deliberately. */
    const char *kve = getenv("COLIBRI_KV_RAM_GB");
    int64_t kv_ram_cap = kve ? (int64_t)(atof(kve)*(1024.0*1024.0*1024.0)) : ram_left;
    if(kv_ram_cap > ram_left) kv_ram_cap = ram_left;
    P->kv_ram_cap = kv_ram_cap;

    /* Newest tokens go closest to the compute. */
    P->kv_vram = P->kv_total < vram_left ? P->kv_total : vram_left;
    P->kv_vram -= P->kv_vram % P->kv_per_token;
    int64_t rest = P->kv_total - P->kv_vram;

    P->kv_ram = rest < kv_ram_cap ? rest : kv_ram_cap;
    P->kv_ram -= P->kv_ram % P->kv_per_token;
    P->kv_disk = P->kv_total - P->kv_vram - P->kv_ram;

    P->tok_vram = (int)(P->kv_vram / P->kv_per_token);
    P->tok_ram  = (int)(P->kv_ram  / P->kv_per_token);
    P->tok_disk = n_ctx - P->tok_vram - P->tok_ram;

    /* Per generated token at FULL context, WHICH BUS moves what.
     *
     * This depends on whether the GPU is actually taking gdn/attn/output. With a GPU those
     * weights are VRAM-resident (zero bus cost) and the KV crosses PCIe to reach them; with
     * no GPU the CPU reads all of it — weights AND KV — out of RAM, on one bus, serially.
     * Charging the KV to PCIe in the CPU-only case was flattering the estimate by nearly 3x:
     * it implied a concurrency that does not exist when there is only one bus. */
    P->gpu_mode = (P->vram_weights + P->gdn_state + P->vram_work <= vram_free);
    if(P->gpu_mode){
        /* THE HOST KV IS DRAM TRAFFIC TOO. The DMA engine reads it through the SAME memory
         * controller the FFN cores are streaming, so it ADDS to the CPU's DRAM time rather
         * than overlapping with it. This is the identical argument that rules out PCIe weight
         * streaming — it was applied there and forgotten here, and the result was a plan
         * claiming the token cost goes flat in context length. It does not: it gets flatter.
         * The PCIe figure below is a separate resource and only binds if it is the slower of
         * the two. */
        /* THE ENGINE RATE APPLIES ONLY TO BYTES THAT GO THROUGH THE ENGINE. 36 GB/s is what
         * the k-quant gemv achieves and it includes nibble decode, FMA and issue-port
         * pressure. The DMA does none of that — a pure sequential read, near the controller's
         * own rate. Charging its bytes at the engine rate overstated the token by ~50 ms.
         * One bandwidth number for every DRAM consumer is wrong in BOTH directions. */
        P->bytes_cpu_ram = B.ffn;                  /* through the engine */
        P->bytes_dma     = P->kv_ram;              /* same controller, pure sequential read */
        P->bytes_pcie    = P->kv_ram;              /* the same bytes, crossing the bus */
        P->bytes_disk    = P->kv_disk;
    } else {
        /* one bus for everything: weights + the whole resident KV */
        P->bytes_cpu_ram = B.ffn + B.gdn + B.attn + B.out + P->kv_ram + P->kv_vram;
        P->bytes_dma     = 0;
        P->bytes_pcie    = 0;
        P->bytes_disk    = P->kv_disk;
    }

    P->ok = 1;
    if(P->vram_weights + P->gdn_state + P->vram_work > vram_free){
        P->ok = 0;
        snprintf(P->why, sizeof P->why,
                 "VRAM too small: weights %.2f + state %.2f + work %.2f GiB > %.2f GiB free. "
                 "Move the GDN layers to RAM (COLIBRI_Q35_GDN=cpu) or use a bigger card.",
                 q35_gib(P->vram_weights), q35_gib(P->gdn_state),
                 q35_gib(P->vram_work), q35_gib(vram_free));
    } else if(P->ram_weights + P->reserve > ram_avail){
        P->ok = 0;
        snprintf(P->why, sizeof P->why,
                 "RAM too small: FFN+embd %.2f GiB + reserve %.2f GiB > %.2f GiB available. "
                 "Lower COLIBRI_RESERVE_GB or free memory.",
                 q35_gib(P->ram_weights), q35_gib(P->reserve), q35_gib(ram_avail));
    }
}

static void q35_plan_print(const Q35Plan *P, const Q35Model *M, FILE *o){
    const Q35Cfg *c = &M->c;
    Q35Bytes B; q35_bytes(M, &B);
    int n_attn = q35_n_attn_layers(c), n_gdn = c->n_layer - n_attn;

    fprintf(o, "\n  qwen3.5-27b  %d layers = %d gdn + %d attn (+%d mtp)   d_model %d  d_ff %d  vocab %d\n",
            c->n_layer, n_gdn, n_attn, c->n_layer_mtp, c->d_model, c->d_ff, c->vocab);
    fprintf(o, "  attn: %d heads / %d kv, head %d, rope %d of %d dims @ %.3g\n",
            c->n_head, c->n_head_kv, c->d_head, c->n_rot, c->d_head, (double)c->rope_base);
    fprintf(o, "  gdn:  %d k-heads / %d v-heads, state %dx%d, conv %d, inner %d\n\n",
            c->n_k_heads, c->n_v_heads, c->d_state, c->d_head_v, c->d_conv, c->d_inner);

    fprintf(o, "  %-26s %9s  %s\n", "tensor group", "GiB", "placement");
    fprintf(o, "  %-26s %9.3f  RAM   -> CPU computes it in place (no PCIe)\n", "ffn (dense, 64)", q35_gib(B.ffn));
    fprintf(o, "  %-26s %9.3f  VRAM  -> resident\n", "gdn/ssm (48)",  q35_gib(B.gdn));
    fprintf(o, "  %-26s %9.3f  VRAM  -> resident\n", "attn (16)",     q35_gib(B.attn));
    fprintf(o, "  %-26s %9.3f  VRAM  -> resident\n", "output proj",   q35_gib(B.out));
    fprintf(o, "  %-26s %9.3f  RAM   -> gather, one row per token\n", "token_embd", q35_gib(B.embd));
    fprintf(o, "  %-26s %9.3f  VRAM  -> fixed, does NOT grow with ctx\n", "gdn recurrent state", q35_gib(P->gdn_state));
    fprintf(o, "  %-26s %9.3f  (skipped unless speculative decode)\n", "mtp/nextn", q35_gib(B.mtp));
    fprintf(o, "  %-26s %9.3f\n\n", "TOTAL weights", q35_gib(B.total));

    fprintf(o, "  KV cache, %s, %d ctx:  %.1f KiB/token x %d = %.3f GiB\n",
            q35_kv_fmt_name(P->kv_fmt), P->n_ctx,
            P->kv_per_token / 1024.0, P->n_ctx, q35_gib(P->kv_total));
    fprintf(o, "    VRAM  %8.3f GiB  %8d tokens   (no bus traffic)\n", q35_gib(P->kv_vram), P->tok_vram);
    fprintf(o, "    RAM   %8.3f GiB  %8d tokens   (%s, room for %.2f GiB)\n",
            q35_gib(P->kv_ram), P->tok_ram,
            getenv("COLIBRI_KV_RAM_GB") ? "pinned by COLIBRI_KV_RAM_GB" : "auto, from MemAvailable",
            q35_gib(P->kv_ram_cap));
    if(P->kv_disk > 0)
        fprintf(o, "    NVMe  %8.3f GiB  %8d tokens   (O_DIRECT, 8 deep, read ahead)\n\n",
                q35_gib(P->kv_disk), P->tok_disk);
    else
        fprintf(o, "    NVMe      none                       (whole cache fits above; the disk\n"
                   "                                          tier stays the overflow valve)\n\n");

    fprintf(o, "  budget:  VRAM %.2f / %.2f GiB used    RAM %.2f used, %.2f avail, %.2f reserved\n",
            q35_gib(P->vram_weights + P->gdn_state + P->vram_work + P->kv_vram), q35_gib(P->vram_free),
            q35_gib(P->ram_weights + P->kv_ram), q35_gib(P->ram_avail), q35_gib(P->reserve));

    /* Measured rates on this machine, so the printed limit is an observation and not a
     * hope: NVMe 6.83 GB/s (O_DIRECT, 8 deep, kvtier_test), CPU->RAM 30 GB/s for k-quant
     * gemv (moe_cpu measured 21 GB/s on lattice quants; Q4_K/Q6_K decode cheaper),
     * PCIe 5.0 x16 ~50 GB/s (hetero_test). */
    /* R_RAM is the MEASURED aggregate of the k-quant gemv path on this machine (qwen35_fwd_test:
     * 15.39 GiB/token in 0.43 s = 36 GB/s), not the DDR5 peak. Quoting the peak here would
     * overstate every projection by 2-3x. */
    /* R_PCIE MEASURED, not assumed. nvidia-smi reports width.max=16, but that is the GPU's
     * capability — sysfs current_link_width stays 8 even under load, so the board is wired x8.
     * PCIe 5.0 x8 is 31.5 GB/s theoretical, ~25 realistic. The 50 GB/s this used to carry was
     * an x16 figure, about 2x optimistic.
     *
     * And the idle reading is Gen1 (2.5 GT/s) — ASPM downtrain, NOT an operating state; it
     * trains to Gen5 within a second of the GPU being touched. Reading link state on an idle
     * GPU and planning from it is the mistake this comment exists to prevent. */
    const double R_NVME = 6.83e9, R_RAM = 36.0e9, R_PCIE = 25.0e9, R_DMA = 50.0e9;
    /* the two DRAM terms ADD (shared controller) but at their own rates. t_floor is the
     * unreachable bound if the controller were shared perfectly — a measurement below the
     * sum stops there. */
    const double t_cpu   = P->bytes_cpu_ram / R_RAM + P->bytes_dma / R_DMA;
    const double t_floor = (P->bytes_cpu_ram + P->bytes_dma) / 57.5e9;
    const double t_pci = P->bytes_pcie / R_PCIE;
    const double t_dsk = P->bytes_disk / R_NVME;
    const double t_tok = t_cpu > t_pci ? (t_cpu > t_dsk ? t_cpu : t_dsk)
                                       : (t_pci > t_dsk ? t_pci : t_dsk);
    fprintf(o, "\n  per generated token at full ctx (%s):\n",
            P->gpu_mode ? "GPU split: CPU line already INCLUDES the DMA's KV read from DRAM"
                        : "CPU-ONLY: one bus, so these ADD rather than overlap");
    fprintf(o, "    DRAM %6.2f GiB engine@36 + %5.2f GiB DMA@50   ~%5.0f ms (floor %.0f)\n",
            q35_gib(P->bytes_cpu_ram), q35_gib(P->bytes_dma), t_cpu*1e3, t_floor*1e3);
    fprintf(o, "    PCIe moves %6.2f GiB host->dev  (kv)       ~%5.0f ms @ 25 GB/s (5.0 x8)\n",  q35_gib(P->bytes_pcie), t_pci*1e3);
    fprintf(o, "    NVMe reads %6.2f GiB            (kv)       ~%5.0f ms @ 6.8 GB/s\n", q35_gib(P->bytes_disk), t_dsk*1e3);
    if(P->gpu_mode)
        fprintf(o, "    token = max() = %.0f ms -> %.2f tok/s, bound by %s\n",
                t_tok*1e3, 1.0/t_tok,
                t_tok == t_dsk ? "the DISK (saturated, as intended)"
                               : (t_tok == t_cpu ? "the CPU/RAM side" : "PCIe"));
    else {
        const double t_sum = t_cpu + t_dsk;
        fprintf(o, "    token = sum() = %.0f ms -> %.2f tok/s (no second bus to hide behind)\n",
                t_sum*1e3, 1.0/t_sum);
    }

    if(!P->ok) fprintf(o, "\n  PLAN NOT VIABLE: %s\n", P->why);
    fprintf(o, "\n");
}

#endif
