# Qwen3.5 (arch "qwen35") engine for colibrì — CPU + CUDA.
#
# WHERE THE WORK ACTUALLY HAPPENS
#
#   gdn / attn / output   5.08 GiB   VRAM resident, read at ~254 GB/s
#   ffn                   9.65 GiB   RAM, and split PER LAYER between the CPU computing in
#                                    place at ~38 GB/s and the GPU pulling 60% of the rows
#                                    over PCIe at ~27.7 — AT THE SAME TIME. The two share one
#                                    DRAM controller whose ceiling measures 64.8 GB/s, and
#                                    that is what the engine now runs against.
#   token_embd            0.67 GiB   RAM, one row per token
#
# Targets:
#   make            build everything
#   make test       the fast checks (no model needed for some)
#   make gates      the checks that decide correctness, against the real model
#   make run        an interactive chat in the terminal
#   make serve      the web UI + OpenAI-compatible API on :8080 (this is what opencode uses)
#   make tools      just the agent tool layer: policy gate, confinement, output budget
#   make bench      bandwidth benchmarks
#
ifeq ($(OS),Windows_NT)
IS_WIN  := 1
else
UNAME_S := $(shell uname -s)
endif

CC      ?= gcc
NVCC    ?= nvcc
ARCH    ?= native
# sm_120 = Blackwell (RTX 50xx). Override for another card; the kernels use nothing newer
# than shuffles and fp16 conversion, so anything from sm_70 up compiles.
CUDA_ARCH ?= sm_120
CUDA_HOME ?= /opt/cuda

CFLAGS  ?= -O3 -march=$(ARCH) -fopenmp -Wall -Wextra \
           -Wno-unused-parameter -Wno-unused-function -Wno-misleading-indentation
LDFLAGS ?= -lm -fopenmp -lpthread
CUFLAGS ?= -O3 -arch=$(CUDA_ARCH)
CULIBS  ?= -L$(CUDA_HOME)/lib64 -lcudart -lstdc++

HAVE_CUDA := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)

MODEL ?= /home/simon/Models/Qwen3.8-27B/Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf
# Everything the engine owns at runtime lives under one directory in $HOME, not inside this
# checkout: projects the agent works in, the KV spill tier, logs. A coding agent whose
# workspace is the directory holding its own source is a footgun waiting for the first
# "tidy up this repo".
#
# The KV spill tier MUST NOT point at tmpfs — that would put the "disk" tier back in the RAM it
# exists to free. $HOME is on disk; kv_tier.h warns if it ever is not.
ENGINE_HOME ?= $(HOME)/QwenEngine
KV_SPILL    ?= $(ENGINE_HOME)/kvspill

HDRS = qwen35.h qwen35_plan.h qwen35_cpu.h qwen35_batch.h qwen35_mtp.h kv_tier.h kv_tier_simd.h \
       kquant_simd.h kquant.h gguf.h iq_grids.h tok.h tok_gguf.h tok_unicode.h \
       json.h strbuf.h harness.h
CUHDRS = qwen35_cuda.h qwen35_hetero.h qwen35_cu_spec.h

CPU_TESTS = tests/qwen35_test tests/qwen35_fwd_test tests/kvtier_test tests/kquant_simd_test \
            tests/qwen35_batch_test tests/qwen35_mtp_test tests/qwen35_sparsity_test \
            tests/harness_test
CU_TESTS  = tests/qwen35_cuda_test tests/qwen35_hetero_test tests/qwen35_cu_fwd_test \
            tests/qwen35_cu_spec_test tests/qwen35_cu_split_test

ifeq ($(HAVE_CUDA),1)
ALL = qwen35_cuda.o $(CPU_TESTS) $(CU_TESTS) qwen35_run qwen35_server
else
ALL = $(CPU_TESTS) qwen35_run qwen35_server
endif

all: $(ALL)

qwen35_cuda.o: qwen35_cuda.cu qwen35_cuda.h
	$(NVCC) $(CUFLAGS) -I. -c $< -o $@

# CPU-only tests link nothing from CUDA.
$(CPU_TESTS): tests/%: tests/%.c $(HDRS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

$(CU_TESTS): tests/%: tests/%.c $(HDRS) $(CUHDRS) qwen35_cuda.o
	$(CC) $(CFLAGS) -I. -o $@ $< qwen35_cuda.o $(LDFLAGS) $(CULIBS)

ifeq ($(HAVE_CUDA),1)
qwen35_run: qwen35_run.c $(HDRS) $(CUHDRS) qwen35_cuda.o
	$(CC) $(CFLAGS) -I. -o $@ $< qwen35_cuda.o $(LDFLAGS) $(CULIBS)
qwen35_server: qwen35_server.c $(HDRS) $(CUHDRS) qwen35_cuda.o
	$(CC) $(CFLAGS) -I. -o $@ $< qwen35_cuda.o $(LDFLAGS) $(CULIBS)
else
qwen35_run: qwen35_run.c $(HDRS)
	$(CC) $(CFLAGS) -DQWEN_NO_CUDA -I. -o $@ $< $(LDFLAGS)
qwen35_server: qwen35_server.c $(HDRS)
	$(CC) $(CFLAGS) -DQWEN_NO_CUDA -I. -o $@ $< $(LDFLAGS)
endif

# Fast checks. kquant_simd needs no model; the others read geometry only.
test: all
	@echo "=== geometry + placement plan ===" && ./tests/qwen35_test "$(MODEL)"
	@echo "=== k-quant SIMD kernels ==="      && ./tests/kquant_simd_test
	@echo "=== agent tools: gate, confinement, budget ===" && ./tests/harness_test | tail -3
	@echo "=== KV tier, 64k ctx ==="          && mkdir -p "$(KV_SPILL)" && \
	    ./tests/kvtier_test 65536 0.3 4096 "$(KV_SPILL)" 8

# THE GATES. These are what decide whether the engine is correct.
#  1. every device kernel vs its CPU counterpart, scored against the POPULATION RMS
#  2. every layer kind vs the CPU reference, including the recurrent state itself
#  3. the whole model, CPU and GPU, must both print 11751 ' Paris' at p ~ 0.606
#  4. a batch must reproduce sequential decode at EVERY batch width the engine emits,
#     not just the one speculation happens to use
#  5. greedy speculation must reproduce greedy decode TOKEN FOR TOKEN — otherwise the
#     rollback is wrong
gates: all
	@mkdir -p "$(KV_SPILL)"
	@echo "=== 1. device kernels ==="   && ./tests/qwen35_cuda_test "$(MODEL)"
	@echo "=== 2. layers ==="           && ./tests/qwen35_hetero_test "$(MODEL)"
	@echo "=== 3a. cpu forward ==="     && ./tests/qwen35_fwd_test "$(MODEL)" 760,6511,314,9338,369 | tail -14
	@echo "=== 3b. gpu forward ==="     && ./tests/qwen35_cu_fwd_test "$(MODEL)" 760,6511,314,9338,369 | tail -14
	@echo "=== 4. batch width, every S ===" && ./tests/qwen35_cu_split_test "$(MODEL)"
	@echo "=== 5. batch + speculation ===" && ./tests/qwen35_cu_spec_test "$(MODEL)" 24

# Interactive. qwen35_run sets the OpenMP environment itself and re-execs once — measured
# worth 20%, and libgomp reads those variables before main() could set them.
run: qwen35_run
	@mkdir -p "$(KV_SPILL)"
	QWEN_KV_SPILL="$(KV_SPILL)" ./qwen35_run "$(MODEL)" $(ARGS)

# The HTTP server: a web chat UI at / and an OpenAI-compatible API at /v1. serve.sh waits for
# a previous instance to actually release the port and the CUDA context — a fixed sleep races
# the teardown and the new process then dies on "address already in use" with its VRAM already
# half-claimed by the old one.
# PROJECTS holds one subdirectory per project; the web UI picks one by name and the agent is
# confined to it. Nothing outside the chosen project is reachable.
serve: qwen35_server webui.html
	@mkdir -p "$(PROJECTS)" "$(ENGINE_HOME)/logs"
	MODEL="$(MODEL)" PORT=$(PORT) CTX=$(CTX) LOG="$(ENGINE_HOME)/logs/server.log" \
	    KV_SPILL="$(KV_SPILL)" ./serve.sh --projects "$(PROJECTS)" --tools $(TOOLS)

# -x matches the process NAME. The -f form matches any command line CONTAINING
# "qwen35_server", which includes the shell running this rule.
stop:
	@pkill -x qwen35_server && echo "stopped" || echo "not running"

tools: tests/harness_test
	./tests/harness_test

bench: tests/kvtier_test tests/kquant_simd_test
	@mkdir -p "$(KV_SPILL)"
	./tests/kquant_simd_test
	./tests/kvtier_test 250000 1.0 4096 "$(KV_SPILL)" 8

fwd: tests/qwen35_fwd_test
	OMP_NUM_THREADS=16 ./tests/qwen35_fwd_test "$(MODEL)" 760,6511,314,9338,369

batch: tests/qwen35_batch_test
	QWEN_RESERVE_GB=4 OMP_NUM_THREADS=16 ./tests/qwen35_batch_test "$(MODEL)" 24 8

mtp: tests/qwen35_mtp_test
	QWEN_RESERVE_GB=4 OMP_NUM_THREADS=16 ./tests/qwen35_mtp_test "$(MODEL)" 24

clean:
	rm -f $(CPU_TESTS) $(CU_TESTS) qwen35_cuda.o qwen35_run qwen35_server

PORT      ?= 8080
CTX       ?= 8192
PROJECTS  ?= $(ENGINE_HOME)/projects
TOOLS     ?= workspace

.PHONY: all test gates run serve stop tools bench fwd batch mtp clean
