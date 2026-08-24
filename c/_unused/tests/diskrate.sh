#!/bin/bash
# diskrate.sh — what the drive ACTUALLY does during decode, not at its peak.
#
# I told Simon "3.4 GB/s" from the tail samples of one run. That was the peak, not the mean,
# and he was right to push back. This separates the phases, because they are nothing alike:
#
#   load      — mmap headers, requantize attention, upload to VRAM. CPU-bound, barely reads.
#   prefill   — every layer's experts, once per prompt token. Reads hard.
#   decode    — the number that matters. One token at a time.
#
# It samples /proc/<pid>/io, which counts bytes this process actually pulled from the block
# layer (read_bytes excludes page-cache hits), and marks the phase boundaries from the
# engine's own stderr so the mean is taken over decode ALONE.
set -u
MODEL=${1:?usage: diskrate.sh <shard1.gguf> [prompt] [n_tokens]}
PROMPT=${2:-"Erkläre den Unterschied zwischen einem Mutex und einem Semaphor."}
NTOK=${3:-16}

LOG=$(mktemp); IO=$(mktemp)
ATTN_Q=${ATTN_Q:-shipped} /tmp/chat "$MODEL" "$PROMPT" "$NTOK" >"$LOG" 2>&1 &
PID=$!

# find the engine's pid (it is the child we just started)
while ! [ -r /proc/$PID/io ]; do sleep 0.2; kill -0 $PID 2>/dev/null || exit 1; done

# sample every 0.5 s: cumulative bytes + a marker of which phase the engine says it is in
while kill -0 $PID 2>/dev/null; do
    B=$(awk '/^read_bytes/{print $2}' /proc/$PID/io 2>/dev/null)
    P=load
    grep -q "^prompt (" "$LOG" 2>/dev/null && P=prefill
    grep -q "^prefill .* tokens in" "$LOG" 2>/dev/null && P=decode
    [ -n "${B:-}" ] && echo "$(date +%s.%N) $B $P" >>"$IO"
    sleep 0.5
done
wait $PID

echo
grep -E "^prefill|^decode |experts:|cache |\[mem\]" "$LOG"
echo
awk '
  { t=$1; b=$2; p=$3
    if (pt != "" && p == pp && b >= pb) {
      dt = t - pt; db = b - pb
      if (dt > 0) { sum[p] += db; time[p] += dt
                    r = db/dt/1048576; if (r > peak[p]) peak[p] = r }
    }
    pt=t; pb=b; pp=p
  }
  END {
    printf "  %-9s %12s %10s %12s %10s\n", "phase", "gelesen GB", "dauer s", "SCHNITT", "Spitze"
    split("load prefill decode", ph, " ")
    for (i=1; i<=3; i++) { p = ph[i]
      if (time[p] > 0)
        printf "  %-9s %11.2f %10.1f %9.0f MiB/s %6.0f MiB/s\n",
               p, sum[p]/1e9, time[p], sum[p]/time[p]/1048576, peak[p]
    }
    print ""
    print "  Das Laufwerk schafft 5.8 GB/s = 5530 MiB/s."
  }' "$IO"
rm -f "$LOG" "$IO"
