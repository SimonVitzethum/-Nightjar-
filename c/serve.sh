#!/usr/bin/env bash
# Start (or restart) the engine server, waiting for the previous instance to actually let go.
# A CUDA context takes seconds to tear down and keeps both the port and the VRAM until it
# does, so a fixed sleep races it — poll instead.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"      # the context still grows past this; see qwen35_server.c
LOG="${LOG:-$HERE/server.log}"

pkill -x qwen35_server 2>/dev/null   # -x: match the process NAME, not any command line containing it
for i in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || break
    sleep 0.5
done
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is still held by something else" >&2; exit 1
fi

mkdir -p "$HERE/../../kvspill"
: > "$LOG"
setsid env COLIBRI_KV_SPILL="$HERE/../../kvspill" COLIBRI_RESERVE_GB="${RESERVE:-4}" \
    "$HERE/qwen35_server" "$MODEL" --port "$PORT" --ctx "$CTX" "$@" \
    >> "$LOG" 2>&1 < /dev/null &

echo -n "loading"
for i in $(seq 1 400); do   # a cold page cache makes residency take minutes, not seconds
    sleep 2
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo " ready after $((i*2))s"
        grep -E "weights|streamed|ready|web UI|API " "$LOG"
        exit 0
    fi
    grep -q "bind .* failed" "$LOG" && { echo; tail -3 "$LOG"; exit 1; }
    echo -n "."
done
echo " timed out"; tail -8 "$LOG"; exit 1
