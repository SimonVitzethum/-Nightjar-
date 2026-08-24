#!/usr/bin/env bash
# Start (or restart) the engine server, waiting for the previous instance to actually let go.
# A CUDA context takes seconds to tear down and keeps both the port and the VRAM until it
# does, so a fixed sleep races it — poll instead.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"      # the context still grows past this; see qwen35_server.c
ENGINE_HOME="${ENGINE_HOME:-$HOME/QwenEngine}"
LOG="${LOG:-$ENGINE_HOME/logs/server.log}"
KV_SPILL="${KV_SPILL:-$ENGINE_HOME/kvspill}"

pkill -x qwen35_server 2>/dev/null   # -x: match the process NAME, not any command line containing it

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
setsid env QWEN_KV_SPILL="$KV_SPILL" QWEN_RESERVE_GB="${RESERVE:-4}" \
    "$HERE/qwen35_server" "$MODEL" --port "$PORT" --ctx "$CTX" "$@" \
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
    echo -n "."
done
echo " timed out"; tail -8 "$LOG"; exit 1
