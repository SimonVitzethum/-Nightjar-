#!/usr/bin/env bash

# One model root for every engine in the runtime: $NIGHTJAR_MODELS, else /models, else
# $HOME/Models. Overriding MODELS_DIR moves all of them at once.
MODELS_DIR="${NIGHTJAR_MODELS:-$([ -d /models ] && echo /models || echo "$HOME/Models")}"
# Start (or restart) the engine server, waiting for the previous instance to actually let go.
# A CUDA context takes seconds to tear down and keeps both the port and the VRAM until it
# does, so a fixed sleep races it — poll instead.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-"$MODELS_DIR"/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf}"
# The switchable registry: Ornith (default) and the dense Qwen, selectable from the web UI's
# model dropdown. MODEL stays the Qwen path for back-compat; override ORNITH/DEFMODEL to taste.
ORNITH="${ORNITH:-"$MODELS_DIR"/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-Abliterated-Q4_K_M.gguf}"
DEFMODEL="${DEFMODEL:-ornith}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"      # the context still grows past this; see qwen35_server.c
ENGINE_HOME="${ENGINE_HOME:-$HOME/.nightjar}"
LOG="${LOG:-$ENGINE_HOME/logs/server.log}"
KV_SPILL="${KV_SPILL:-$ENGINE_HOME/kvspill}"

# Kill the agent's own Firefox, and NOTHING ELSE that is a Firefox.
#
# Deliberately not `pkill -f -- --profile .../browser/profile`: -f matches any command line
# CONTAINING the pattern, which includes the shell running this script. That has already cost
# this project two self-inflicted kills. pgrep -x matches the process NAME, so the shell is
# never a candidate, and the profile path is then checked from /proc.
stop_browser() {
    for p in $(pgrep -x firefox 2>/dev/null); do
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "$ENGINE_HOME/browser/profile"; then
            kill -9 "$p" 2>/dev/null
        fi
    done
    rm -f "$ENGINE_HOME/browser/firefox.pid" 2>/dev/null
}

if [ "${1:-}" = "stop" ]; then
    pkill -x qwen35_server 2>/dev/null && echo "stopped" || echo "not running"
    stop_browser
    exit 0
fi

pkill -x qwen35_server 2>/dev/null   # -x: match the process NAME, not any command line containing it
# A Firefox left over from a crashed server holds the listening socket it inherited, and then
# nothing named qwen35_server is running while the port is still bound.
stop_browser

# Wait for the PROCESS to be gone, not for the port to be free. The port is released before
# the CUDA context is, and this is not a theoretical gap: starting on a released port while
# the old context still held VRAM produced
#     qwen35_cuda: out of VRAM uploading output.weight (994.6 MiB): cudaMalloc: out of memory
#     gpu unavailable — CPU only
# and the run went on for ten minutes at a third of the speed with nothing but that one line
# in the log to say why. Process exit is the event that frees the VRAM, so wait for that.
for i in $(seq 1 120); do
    pgrep -x qwen35_server >/dev/null 2>&1 || break
    sleep 0.5
done
if pgrep -x qwen35_server >/dev/null 2>&1; then
    echo "an old qwen35_server will not die" >&2; exit 1
fi
# Belt and braces: the driver can lag the exit. Poll the card until it is actually empty.
if command -v nvidia-smi >/dev/null 2>&1; then
    for i in $(seq 1 40); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -z "${used:-}" ] && break
        [ "$used" -lt 900 ] && break
        sleep 0.5
    done
fi
for i in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || break
    sleep 0.5
done
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is still held by something else" >&2; exit 1
fi

mkdir -p "$KV_SPILL" "$(dirname "$LOG")"
: > "$LOG"
# MEMMAX puts the engine in its own memory cgroup. Without one, the engine is simply the
# biggest anonymous allocation on the box, so when memory runs out the kernel's OOM killer
# picks it — or, just as often, something else entirely that had nothing to do with it. With
# one, the limit is enforced against THIS process and the kill lands where it belongs.
# MemorySwapMax=0 because the weights are pinned anyway: swapping the rest buys nothing and
# turns a fast failure into ten minutes of thrashing first.
#   MEMMAX=20G make serve
# NO SWAP. The engine runs in its own systemd scope with MemorySwapMax=0, so its pages can
# never be pushed to disk — a swapped-out weight is not a slow weight, it is a stalled token,
# and this box was measured with 4.8 GiB already in swap while the model sat in the page cache.
# The page-cache mmap is reclaimable (not swap), so this does not fight residency; it only stops
# the anon weights and the KV tier from swapping. MemoryMax is added on top only if asked for.
# (System-wide swap is the user's to disable with `sudo swapoff -a`; this scope covers the
# engine without root.)
LAUNCH=""
if command -v systemd-run >/dev/null 2>&1; then
    SWAP_CAP="-p MemorySwapMax=0"
    [ -n "${MEMMAX:-}" ] && SWAP_CAP="-p MemoryMax=$MEMMAX $SWAP_CAP"
    LAUNCH="systemd-run --user --scope --quiet --collect $SWAP_CAP --"
    echo "  no swap for the engine (own cgroup, MemorySwapMax=0)${MEMMAX:+, capped at $MEMMAX}"
else
    echo "  note: systemd-run not found — cannot fence swap; consider 'sudo swapoff -a'" >&2
fi
setsid $LAUNCH env QWEN_KV_SPILL="$KV_SPILL" QWEN_RESERVE_GB="${RESERVE:-2}" \
    "$HERE/qwen35_server" --model "ornith:$ORNITH" --model "qwen3.5-27b:$MODEL" \
    --default "$DEFMODEL" --port "$PORT" --ctx "$CTX" "$@" \
    >> "$LOG" 2>&1 < /dev/null &

echo -n "loading"
for i in $(seq 1 400); do   # a cold page cache makes residency take minutes, not seconds
    sleep 2
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo " ready after $((i*2))s"
        grep -E "weights|streamed|ready|web UI|API |tools " "$LOG"
        # A silent fallback to CPU costs a factor of three and is one line deep in the log.
        if grep -q "gpu unavailable" "$LOG"; then
            echo
            echo "  !! RUNNING WITHOUT THE GPU — everything below is 2-3x slower than it should be:" >&2
            grep -E "out of VRAM|gpu unavailable|refusing to make" "$LOG" | sed "s/^/     /" >&2
        fi
        exit 0
    fi
    grep -q "bind .* failed" "$LOG" && { echo; tail -3 "$LOG"; exit 1; }
    # A refusal is a decision, not a slow start: say so now instead of polling health for
    # thirteen minutes against a process that already exited on purpose.
    grep -q "REFUSING TO START" "$LOG" && { echo; sed -n "/REFUSING TO START/,\$p" "$LOG"; exit 1; }
    echo -n "."
done
echo " timed out"; tail -8 "$LOG"; exit 1
