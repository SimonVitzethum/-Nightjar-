# Colibrì · Web UI und OpenAI-kompatible API

Ein Prozess, zwei Zugänge: eine Chat-Oberfläche im Browser und dieselbe Engine als
OpenAI-kompatibler Endpunkt, an den sich opencode (oder jeder andere Client) hängt.

## Starten

```bash
cd colibri/c
make serve                      # :8080, 32k Kontext
make serve PORT=9000 CTX=8192   # andere Werte
make stop
```

Der erste Start dauert ~45 s: 5,1 GiB Gewichte gehen in den VRAM, 9,3 GiB werden im RAM
resident gemacht und für DMA gepinnt.

| | |
|---|---|
| Web UI | <http://127.0.0.1:8080/> |
| API | `http://127.0.0.1:8080/v1` |
| Status | `curl localhost:8080/health` |

Nach außen öffnen mit `--host 0.0.0.0`, dann bitte auch `--api-key <schlüssel>` setzen —
ohne den ist der Endpunkt offen.

## opencode

Die Konfiguration liegt bereits in `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "provider": {
    "colibri": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" },
      "models": { "qwen3.5-27b": { "tool_call": true, "reasoning": true,
                                   "limit": { "context": 32768, "output": 8192 } } }
    }
  },
  "model": "colibri/qwen3.5-27b"
}
```

Einmalig nötig, weil opencode den Provider nicht selbst nachlädt:

```bash
npm install --prefix ~/.config/opencode @ai-sdk/openai-compatible
```

Dann `opencode` starten, oder direkt `opencode run "..."`. `opencode models | grep colibri`
zeigt, ob der Provider erkannt wird.

**Tool-Calling funktioniert.** Das Modell schreibt Funktionsaufrufe in einem eigenen
XML-Format (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`);
der Server parst das in OpenAI-`tool_calls` um, inklusive Typen — `"Berlin"` bleibt String,
`3` wird Integer.

## Was langsam ist, und warum

**Prefill kostet ~10 tok/s.** opencodes Agenten-Prompt ist etwa 7700 Token groß, das erste
Kommando einer Sitzung braucht also rund 13 Minuten. Danach greift der Prefix-Cache: der
Server merkt sich die Tokenfolge, die im KV-Cache steht, und prefillt bei der nächsten Anfrage
nur die Differenz. Im Log steht, wie viel wiederverwendet wurde:

```
  566 prompt (0 cached) in 57.11s = 9.9 tok/s | 129 gen in 22.43s = 5.75 tok/s | accept 88%
   42 prompt (22 cached) in ...
```

Die Prüfpunkte sind **selbsttragend**: jeder hält den rekurrenten Zustand *und* den
Trunk-KV seines eigenen Präfixes und wird gegen seine eigene Tokenfolge geprüft, nicht gegen
das, was zuletzt lief. Deshalb überlebt ein langer Agenten-Prompt den kurzen Title-Request,
den opencode dazwischenschiebt — vorher hat der alles entwertet.

Fällt der Cache trotzdem aus, wird der Zustand komplett neu aufgebaut — ein teilweises
Zurückspulen gibt es nicht. Das ist keine Bequemlichkeit: 48 der 64 Schichten sind Gated DeltaNet, deren
Zustand bei jedem Schritt gedämpft und rang-1-aktualisiert wird. Diese Operation ist nicht
invertierbar. Der Attention-KV allein ließe sich abschneiden, der rekurrente Teil nicht.

Für interaktive Nutzung ist die Oberfläche deshalb angenehmer als ein Agent: kurze Prompts,
sofortige Antwort. Für opencode lohnt es sich, eine Sitzung offen zu lassen statt viele neue
zu starten.

**Eine Anfrage zur Zeit.** Eine Engine, ein rekurrenter Zustand. Parallele Generierung bräuchte
einen zweiten 146-MiB-Zustand und ein zweites KV-Fenster; dafür ist auf dieser Karte kein Platz.

## Stellschrauben

| Variable | Default | Wirkung |
|---|---|---|
| `COLIBRI_CUDA_STREAM` | `0.60` | Anteil jeder nicht-residenten FFN-Schicht, den die GPU beim **Decode** über PCIe zieht. |
| `COLIBRI_CUDA_STREAM_PREFILL` | `0.85` | Dasselbe für **Batch-Arbeit**. Höher, weil ein Batch jedes Gewicht S-mal benutzt: die CPU-Seite amortisiert ihre Dekodierkosten, die Buskosten pro Byte bleiben gleich. Nicht 1,0 — die GPU-Seite ist eine DMA und bei f=1 dauert ein Batch 400 ms statt 339. |
| `COLIBRI_CACHE_GB` | `3.0` | Speicher für Präfix-Prüfpunkte. Jeder hält den rekurrenten Zustand plus den Trunk-KV seines Präfixes und ist damit unabhängig von allem, was dazwischen lief. |
| `COLIBRI_CACHE_SPACING` | `512` | Abstand der Prüfpunkte in Token. Bestimmt, wie viel im schlimmsten Fall neu gerechnet wird. |
| `COLIBRI_CUDA_KV_TOK` | `8192` | Obergrenze des KV-Fensters im VRAM. Größer lohnt erst jenseits ~24k Kontext — davor sind residente FFN-Schichten mehr wert (§7.8 im Plan.md). |
| `COLIBRI_CUDA_FFN_GB` | auto | Deckel für residente FFN-Schichten. |
| `COLIBRI_KV_SPILL` | `../../kvspill` | Auslagerungspfad des kalten KV-Tiers. **Nicht auf tmpfs zeigen lassen** — das legt den „Disk"-Tier zurück in genau den RAM, den er freimachen soll. |
| `COLIBRI_RESERVE_GB` | `4` | RAM, den die Engine nie anfasst. |
| `COLIBRI_KERNEL_LOG` | aus | Meldet pro Komponente, auf welchem Gerät sie tatsächlich lief. |

## Endpunkte

| Methode | Pfad | |
|---|---|---|
| `GET` | `/` | die Chat-Oberfläche |
| `GET` | `/v1/models` | Modellliste |
| `POST` | `/v1/chat/completions` | streamend (SSE) und nicht-streamend |
| `GET` | `/health` | Gerät, Kontext, VRAM |

Unterstützt werden `messages`, `tools`, `stream`, `temperature`, `top_p`, `top_k`,
`max_tokens`, `seed` und `enable_thinking`. Der Gedankengang kommt als `reasoning_content`
zurück — getrennt vom `content`, damit ihn ein Client anzeigen oder ignorieren kann.
