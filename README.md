# Qwen3.5-27B, on one laptop

A dense 27B model with a hybrid Gated-DeltaNet/attention architecture, running in C on an
8 GB laptop GPU and a 10-core CPU **at the same time** — plus an agent harness with shell,
file and search tools, and a web UI to drive it.

No Python at runtime, no BLAS, no inference framework. A GGUF reader, k-quant kernels, CUDA
kernels, an HTTP server and a tool layer.

**Why it is shaped the way it is.** 48 of the 64 layers are Gated DeltaNet — a recurrent
linear-attention block whose state is constant in context length — and the model is dense, so
there is nothing to stream selectively. What there is instead is a memory hierarchy with two
independent buses: 5.1 GiB of weights fit in VRAM and are read at 254 GB/s, the 9.65 GiB of
FFN does not and lives in RAM. The engine splits **every** FFN layer between the CPU computing
in place and the GPU pulling its share over PCIe, and runs both halves concurrently. The DRAM
controller tops out at 64.8 GB/s; the split reaches 59.

**Measured on an RTX 5070 Laptop (8 GB) + 10-core CPU + 31 GB RAM, Q4_K_M:**

| | tok/s |
|---|---|
| CPU only (where this started) | 2.33 |
| + gdn/attn/output resident in VRAM | 3.49 |
| + FFN split, CPU in place and GPU over PCIe **at the same time** | 4.84 |
| + MTP speculative decode | **6.1 – 6.7** |
| prefill, batched | **10.4** |
| prefix-cache hit (1349-token prompt) | 129 s → **32.5 s** |

The full design record — including everything that was measured and **rejected**, which is most
of it — is in [`Plan.md`](../Plan.md).

### Build

```bash
make                 # from the repo root, the workspace root, or c/ — all forward to c/
make -j8             # faster
```

Every target below works from any of those three directories. `make help` at the workspace
root lists them.

Without `nvcc` on PATH the same sources build a CPU-only engine (`-DQWEN_NO_CUDA`); nothing
else changes. Override the GPU target if you are not on Blackwell:

```bash
make CUDA_ARCH=sm_86        # RTX 30xx      (sm_89 = 40xx, sm_120 = 50xx, the default)
make CUDA_HOME=/usr/local/cuda
```

### The server: web UI + OpenAI-compatible API

```bash
make serve                              # http://127.0.0.1:8080/
make serve PORT=9000                    # another port
make serve CTX=16384                    # larger FIRST allocation — see below
make serve MODEL=/path/to/other.gguf    # another model — see below
make stop                               # stop it
make serve                              # restarting is just serve again; it waits for the
                                        # old instance to release the port AND the CUDA context
```

`make serve` calls `./serve.sh`, which you can also run directly with the same variables:

```bash
MODEL=/models/qwen35-27b-Q4_K_M.gguf PORT=8080 CTX=8192 ./serve.sh
```

Startup takes 45 s to a few minutes: 5.1 GiB of weights go to VRAM, 9.3 GiB become resident in
RAM and get pinned for DMA. `serve.sh` polls until `/health` answers and prints the placement
it chose. Logs go to `c/server.log`.

**`--ctx` is the first allocation, not a limit.** The context grows on demand and stops only
when `MemAvailable` reaches the floor — 2 GiB by default, `QWEN_MEM_FLOOR_GB` to change it.
At that point the store says so and keeps the context where it is, rather than allocating into
the floor. What `--ctx` actually influences is how much KV stays in RAM instead of the spill
tier, because that hot arena is sized once at open; the server reports both:

```
context: 8192 initial, UNBOUNDED — grows on demand and stops only when
         MemAvailable reaches the 2.00 GiB floor. Hot KV tier 3.17 GiB (97707 tokens),
         the rest spills to ../../kvspill
```

The hot tier is sized from the memory that will be spare **after** residency, not from
MemAvailable as measured before it — the difference is 9.3 GiB, and getting it wrong is how
this process was OOM-killed once.

For anything beyond localhost, bind explicitly **and set a key** — without one the endpoint is
open:

```bash
./qwen35_server "$MODEL" --host 0.0.0.0 --port 8080 --api-key "$(openssl rand -hex 16)"
```

| | |
|---|---|
| `GET /` | the chat UI |
| `GET /v1/models` | model list |
| `POST /v1/chat/completions` | OpenAI-compatible, streaming (SSE) and not, with tool calls |
| `GET /v1/tools` | tool schemas, the policy for each, and the agent system prompt |
| `POST /v1/tools/exec` | run one tool: `{name, arguments, approved}` |
| `GET /health` | device, context, VRAM, cache state |

```bash
curl localhost:8080/health
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream":true}'
```

### The agent harness: shell, files, tool calls

The web UI is an agent, not just a chat window. It can read and search the workspace, edit
files, and run commands — and everything it does goes through one registry in
[`c/harness.h`](c/harness.h) with one gate in front of it.

```bash
make serve                                   # agent on, workspace-write, projects in $HOME
make serve TOOLS=ro                          # read and search only; nothing that writes
make serve TOOLS=off                         # a plain chat server, no tools offered at all
make serve PROJECTS=/other/place             # another projects root
make serve ARGS='--workspace /one/repo'      # one fixed workspace instead of projects
make tools                                   # test the tool layer alone — no model needed
```

#### Projects

One engine, many workspaces. Everything the engine owns at runtime lives under
`~/QwenEngine`, outside any checkout:

```
~/QwenEngine/
├── projects/<name>/     one directory per project — the agent is confined to the chosen one
├── kvspill/             the cold KV tier
├── logs/server.log
└── models/
```

The web UI has a project picker in the header and a `+` to make a new one. The agent is
confined to the selected project and cannot see its siblings.

The client picks a project by **name**, never by path, and the name has to match
`[A-Za-z0-9._-]` — anything else is refused rather than sanitized, because sanitizing produces
a *different* name that still resolves somewhere, which is how a cleaner becomes a traversal.
The resolved directory is then checked against the projects root, so a project that is a
symlink to `/etc` is refused too. Each of those is a test in
[`c/tests/harness_test.c`](c/tests/harness_test.c).

The active workspace is **per request and therefore per thread**: two tool calls can be in
flight at once, and a global would let one request move the fence under the other.

| tool | class | what it is for |
|---|---|---|
| `bash` | exec | build, test, git, anything the rest does not cover |
| `read_file` | read | numbered lines, with `offset`/`limit` to page a long file |
| `write_file` | write | a new file, or a full rewrite |
| `edit_file` | write | exact-string replacement; **refuses** an ambiguous match |
| `list_dir` | read | one directory |
| `glob` | read | find files by name pattern, recursively |
| `grep` | read | POSIX regex over file contents → `path:line:text` |

#### The terminal

Commands run on a **PTY**, not a pipe, and the panel shows that terminal live — the transcript
is polled from the server while the command is still running, and the input line writes back
down the same descriptor.

The PTY is not a nicety. On a pipe `sudo` says *"no tty present and no askpass program
specified"* and exits, because it will not read a password from something that is not a
terminal. Give it a terminal and it prompts:

```
$ sudo -k id -un
[sudo] Passwort für simon:
```

The panel recognises that prompt, masks the input box, and posts what you type to
`POST /v1/term/input` with `secret: true`.

**The password does not reach the model, and does not depend on the command to keep it out.**
sudo turns off terminal echo while it reads, so on a PTY it never comes back — but that is
sudo's behaviour, not a property of this design, and the first live test proved the difference:
`stty -echo; read p` in bash echoes anyway, because bash's `read` **re-enables** echo unless
given `-s`. So the harness strips the bytes it was told are secret out of the output stream
itself, incrementally, whatever the command does with termios. The transcript gets
`[Eingabe verborgen]` in their place — you should know something was typed, only not what.

```
GET  /v1/term?since=N     the transcript from an absolute offset
POST /v1/term/input       {data, secret}   → written to the PTY, logged nowhere
POST /v1/term/signal      {sig:"int"|"kill"}
```

The command timeout is an **idle** timeout: output or input resets it. A build that prints for
five minutes is not hung, and without this rule "type the sudo password" would race a
120-second kill.

#### Outside the project: consent, not refusal

The first version hard-refused every path outside the workspace. That is the right default and
the wrong only-option — a coding agent is regularly asked about a file one directory over, and
"refused" with no way to say yes makes the human do it by hand, which is not more secure, just
more tedious, and it trains them to run everything with `--tools full`.

So an out-of-tree path is now a **question**:

```
decision=ask  kind=path  inside_home=true   /home/me/QwenEngine/projects/other
              leaves the project, but stays inside /home/me/QwenEngine
decision=ask  kind=path  inside_home=false  /etc/hostname
              leaves /home/me/QwenEngine entirely
```

No password — the user asked for consent, not authentication, and a password here would be
theatre: the process already runs as them. What matters is that the **resolved** path is what
is shown, and that saying yes binds to *that* path. Consent to `/etc/hostname` is not consent
to `/etc/hosts`; the next one asks again. And it never upgrades the policy class — agreeing to
a path does not buy a write in `--tools ro`.

#### The browser

Three tools drive a **headless Firefox** — filling forms, clicking, dragging, uploading,
downloading, screenshotting.

```
web_open  {url}
web_do    {action: click|type|select|press|scroll|drag|move|back|forward|reload|wait,
           target|selector, text, enter, to|to_selector, dx, dy, x, y, amount}
web_file  {action: screenshot|upload|download, path, target, url}
```

**Marionette, not Selenium or Playwright.** Marionette is Firefox's own remote protocol and it
is already in the browser you have: length-prefixed JSON over TCP, `<bytes>:<json>`. No driver
binary to version-match, no Node or Python in the runtime of an engine that has neither. The
client is [`c/webtool.h`](c/webtool.h), speaking it with the socket code and JSON parser that
were already here. It runs on a **private profile with `--no-remote` on a non-default port** —
without all three it attaches to the Firefox you already have open and starts driving your tabs.

**Three tools and not fifteen, and that is a measurement.** Every schema is prefilled on every
turn (§12.3): fifteen browser verbs would be ~2 KB ≈ 550 tokens ≈ most of a minute added to
every request in the session, used or not. So the verbs are an `action` enum on one tool — a
worse API in the abstract and the right one here. `--web off` removes them from the prompt
entirely.

**The page comes back as a numbered index, not as HTML.** A login page is 200 KB of markup
around four interesting elements, which at a 4 KiB budget the model would never reach. After
every action the page is digested into visible text plus a numbered list of what can be
operated:

```
Live-Probe
file:///…/probe.html

Formular

Bedienbare Elemente:
[0] text "Name" =Simon
[1] select "blau" {rot | blau}
[2] button "Los"
```

The model addresses those numbers, and the same call that changed the page renumbers it —
an index that survives a navigation clicks the wrong button.

**Typing is real key events** (`WebDriver:ElementSendKeys`), not `el.value = x`. Sites built on
React ignore a value that was assigned rather than typed, and the form then submits empty.
**Dragging is a real press-move-release** (`WebDriver:PerformActions`), which is the only thing
a slider, a canvas or an HTML5 drag target reacts to.

**And what the numbering cannot see, a selector can.** A slider is often a bare `div` with a
handler attached from script — no role, no semantics, nothing to number. So `web_do` also takes
a CSS `selector`, and drag additionally takes viewport `x`/`y`. That gap was found by the gate,
not by reasoning: the first version could fill every form on the test page and could not move
the slider on it.

Uploads and downloads keep the workspace fence: the local side of an upload goes through
`h_resolve` like any other path, so uploading `/etc/passwd` asks for consent first. Downloads
are fetched **by the browser** — it has the cookies and the session that got to the page — and
then moved into the workspace.

Screenshots are written into the project as PNG and the file view renders them
(`GET /v1/fs/raw`); a screenshot you cannot look at is a log line.

The browser tools are class `net`, which **asks** in `workspace` mode like the shell does: the
network is the one direction the workspace fence does not cover, since a URL can carry the
contents of a file out. Firefox starts on first use, not at boot — it costs half a gigabyte and
several seconds, and most sessions never touch it.

#### The four modes

| `--tools` | read | write | shell |
|---|---|---|---|
| `off` | — | — | — |
| `ro` | yes | denied | denied |
| `workspace` *(default)* | yes | yes, confined | **asks** |
| `full` | yes | anywhere | yes, unasked |

Three properties are worth knowing, because they are the reason the thing is safe enough to
leave running:

**Confinement is resolved, not textual.** Every path argument goes through `realpath()` before
it is compared to the workspace root. `../../etc/passwd`, an absolute `/etc/shadow`, and a
symlink pointing out of the tree all land outside and are refused — a `strncmp` on the argument
as typed lets all three through. A refusal is reported as `"decision":"deny"`, distinct from a
tool that merely failed.

**An `ask` with nobody to ask is a refusal.** `POST /v1/tools/exec` returns
`{"decision":"ask"}` and runs nothing; the caller has to come back with `"approved":true`. A
script that ignores the question gets nothing done, which is the correct outcome — there is no
timeout that eventually says yes.

**It will not start in a configuration that hands the machine to the network.** Tools enabled
on a non-loopback address with no `--api-key` is refused at startup, not warned about:

```
refusing to start: --tools workspace on 0.0.0.0 with no --api-key.
Anything that can reach 0.0.0.0:8080 could run commands here. Pick one:
  --api-key KEY   require a key, or
  --tools off     serve chat only, or
  --host 127.0.0.1
```

#### Output is a budget

This engine prefills at about 10 tok/s, so a tool that returns 400 KiB of build log costs the
next turn two minutes before the model says a word. Every result is capped at 16 KiB
(`--tool-output`) — and for commands it is kept as **head plus a ring of the tail**, not as the
first 16 KiB. That direction matters: a build prints warnings for a minute and its *errors* in
the last twenty lines. Keeping the head only hands the model a log that ends before anything
went wrong, and it then reports success.

Commands are killed at `--tool-timeout` (120 s default) and killed as a process *group*, so a
shell that backgrounded something does not leave it behind.

#### What it costs per turn

The schemas and the agent system prompt are prefilled on every turn: 3.6 KB of JSON plus 673
characters, about 1.2k tokens. They are also the **first** bytes of the prompt and never
change, so after the first turn the prefix cache serves all of them. That is why the UI never
rewrites history — appending keeps the cache; editing an earlier message throws it away.

```bash
curl localhost:8080/v1/tools                    # the schemas, the policy, the system prompt
curl localhost:8080/v1/tools/exec -H 'Content-Type: application/json' \
  -d '{"name":"grep","arguments":{"pattern":"h_policy","glob":"*.h"}}'
curl localhost:8080/v1/tools/exec -H 'Content-Type: application/json' \
  -d '{"name":"bash","arguments":{"command":"git log --oneline -3"},"approved":true}'
```

#### The file view

The panel on the left is the project tree, expanded lazily — a real repository is thousands of
entries and the point of the panel is to find one file. A file the agent writes or edits gets a
green dot in the tree and an **öffnen** button on its tool card, which opens it in the viewer on
the right at the line that changed. Line numbers, syntax colouring, and a **Datei / Trajectory**
tab pair sharing the panel.

```
GET /v1/fs/list?project=NAME&path=REL     directory entries
GET /v1/fs/read?project=NAME&path=REL     one file, up to 2 MiB
```

These are shaped for a person, not for the model: raw bytes, no line numbers, no 4 KiB budget.
But they are **not a second way into the filesystem** — they go through the same `h_resolve`,
the same `realpath`, the same workspace fence and the same per-thread project as the tools. A
second reader with its own idea of what "inside" means is exactly the second door this design
does not have.

The UI also shows a card per tool call — the command or the exact two lines an edit swaps, an
approve/deny row when one is needed, and the output collapsed underneath. The **⏱** tab shows
per-step prefill and decode time, tokens, and how many of them came from the cache.

opencode does not use any of this: it brings its own tools and only needs
`/v1/chat/completions`. The two can share one server.

### Terminal chat

```bash
make run                                        # interactive
make run ARGS='-p "Explain gated delta nets." -n 200'
./qwen35_run "$MODEL" -p "..." --temp 0.7 --top-p 0.95 --no-think --stats
./qwen35_run "$MODEL" --cpu                     # no GPU at all
./qwen35_run --help
```

### Other models

The engine reads its whole geometry from GGUF metadata, so **any Qwen3.5 GGUF works** —
different sizes, different quantizations, different context settings. It refuses anything else
by design and says so:

```
qwen35: architecture is "glm-dsa", not qwen35
```

(that is the GLM-5.2 file from the other engine, refused by the qwen35 loader.)

The check is deliberate: this engine implements Gated DeltaNet, partial NeoX rope over 64 of
256 head dims, and an interleaved q/gate projection. Loading a Llama or Qwen2 file with those
assumptions would not crash — it would produce fluent, wrong text.

```bash
make serve MODEL=/models/Qwen3.5-27B-Q4_K_M.gguf
make serve MODEL=/models/Qwen3.5-27B-Q6_K.gguf     # needs more VRAM+RAM; the planner adapts
MODEL=/models/other-qwen35.gguf make gates          # check it before trusting it
```

Quantization support on the GPU path is **F32, F16, Q8_0, Q4_K, Q6_K**. A tensor in any other
type is refused at upload with the tensor name and its type, rather than silently falling back
to something slower. The CPU path additionally decodes Q2_K/Q3_K/Q5_K/IQ2_XS/IQ3_XXS/IQ4_XS.

### Tests, and which of them decide correctness

```bash
make test      # fast: geometry, SIMD kernels vs a double reference, the KV tier
make gates     # the ones that decide whether the engine is right — needs the model
make bench     # bandwidth benchmarks, 250k-token KV
```

`make gates` runs five checks in order, and every one of them exists because something got
through:

1. every device kernel against its CPU counterpart, scored against the **population RMS** and
   never per row — per-row relative error has produced three false negatives here;
2. every layer kind against the CPU reference, including the recurrent state, run twice so a
   wrong conv ring cannot hide;
3. the whole model, CPU and GPU, must both print `11751 ' Paris'` at p ≈ 0.606;
4. a batch must reproduce sequential decode at **every** batch width the engine emits;
5. greedy speculation must reproduce greedy decode **token for token**.

### opencode

`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "provider": {
    "qwenengine": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" },
      "models": { "qwen3.5-27b": { "tool_call": true, "reasoning": true,
                                   "limit": { "context": 32768, "output": 8192 } } }
    }
  },
  "model": "qwenengine/qwen3.5-27b"
}
```

One step is needed once, and opencode hangs silently without it — it does not fetch the
provider itself:

```bash
npm install --prefix ~/.config/opencode @ai-sdk/openai-compatible
opencode models | grep qwenengine      # confirms it resolves
opencode run "Read hello.py and say whether add() is correct."
```

Tool calling works: the model emits its own XML form and the server converts it to OpenAI
`tool_calls`, with types preserved. **The honest limitation is prefill**: opencode's agent
prompt is ~7700 tokens, so the first command of a session costs minutes. Later turns in the
same session are cheap because the prefix cache reuses the KV.

### Knobs

| Variable | Default | |
|---|---|---|
| `QWEN_CUDA_STREAM` | `0.60` | share of each non-resident FFN layer the GPU pulls over PCIe while **decoding** |
| `QWEN_CUDA_STREAM_PREFILL` | `0.85` | the same for **batch** work — higher, because a batch reuses each weight S times |
| `QWEN_CUDA_KV_TOK` | `8192` | cap on the VRAM KV window. Larger only pays past ~24k context; below that, resident FFN layers are worth more |
| `QWEN_CUDA_FFN_GB` | auto | cap on resident FFN layers |
| `QWEN_CACHE_GB` | `1.5` | prefix-checkpoint budget. `0` disables |
| `QWEN_CACHE_SPACING` | `512` | tokens between checkpoints — the worst-case recompute |
| `QWEN_KV_SPILL` | `../../kvspill` | cold KV tier. **Never point it at tmpfs** — that puts the "disk" tier back in the RAM it exists to free |
| `QWEN_RESERVE_GB` | `4` | RAM the engine never touches |
| `QWEN_KERNEL_LOG` | off | prints which device each component actually ran on |

`qwen35_run` and `qwen35_server` set `OMP_WAIT_POLICY`, `OMP_PROC_BIND`, `OMP_PLACES` and
`OMP_NUM_THREADS` themselves and re-exec once — libgomp reads those before `main()` could, and
they are worth 20% here.

## Repo layout

```
c/
├── qwen35.h              model, geometry, residency
├── qwen35_cpu.h          the CPU forward pass
├── qwen35_cuda.{h,cu}    every device kernel
├── qwen35_hetero.h       the GPU/CPU scheduler and the split FFN
├── qwen35_cu_spec.h      batched forward + MTP speculation
├── qwen35_cache.h        prefix checkpoints
├── kv_tier.h             the tiered KV store (RAM, then disk)
├── kquant.h, kquant_simd.h   k-quant decode, scalar and AVX2
├── harness.h             the agent tool layer: registry, gate, dispatch
├── strbuf.h, json.h, gguf.h, tok*.h   support
├── qwen35_run.c          terminal client
├── qwen35_server.c       HTTP server: web UI + OpenAI API + tools
├── webui.html            the web UI and the agent loop
└── tests/                every gate, each runnable on its own
```

`c/_unused/` holds an unrelated earlier engine for a 744B MoE model. Nothing in the Qwen3.5
build reaches it; it has its own makefile if you want it.

## License

Apache 2.0.
