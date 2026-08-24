# QwenEngine · Web UI, Agent und OpenAI-kompatible API

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

Einmalig nötig, weil opencode den Provider nicht selbst nachlädt:

```bash
npm install --prefix ~/.config/opencode @ai-sdk/openai-compatible
```

Dann `opencode` starten, oder direkt `opencode run "..."`. `opencode models | grep qwenengine`
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

## Der Agent: Shell, Dateien, Werkzeuge

Die Weboberfläche ist ein Agent, kein reines Chatfenster. Sie kann suchen, lesen, Dateien
ändern und Befehle ausführen. Alles davon läuft über **eine** Registry in `harness.h` und
**einen** Riegel davor — die Seite selbst kann kein Werkzeug erfinden und keins am Riegel
vorbeiführen; sie schickt nur einen Namen und ein Argumentobjekt an `/v1/tools/exec`.

```bash
make serve                              # Agent an, workspace-write, Projekte in $HOME
make serve TOOLS=ro                     # nur lesen und suchen
make serve TOOLS=off                    # reiner Chatserver
make serve PROJECTS=/anderer/ort        # andere Projektwurzel
make serve ARGS='--workspace /ein/repo' # ein fester Arbeitsbereich statt Projekte
make tools                              # nur die Werkzeugschicht testen, ohne Modell
```

**Projekte.** Eine Engine, viele Arbeitsbereiche. Alles, was die Engine zur Laufzeit besitzt,
liegt unter `~/QwenEngine` — außerhalb jedes Checkouts:

```
~/QwenEngine/
├── projects/<name>/     ein Verzeichnis pro Projekt
├── kvspill/             der kalte KV-Tier
├── logs/server.log
└── models/
```

Die Oberfläche hat oben eine Projektauswahl und ein `+` für ein neues. Der Agent ist auf das
gewählte Projekt eingegrenzt und sieht seine Nachbarn nicht. Der Client wählt per **Name**, nie
per Pfad, und der Name muss `[A-Za-z0-9._-]` entsprechen — alles andere wird abgelehnt, nicht
bereinigt: Bereinigen erzeugt einen *anderen* Namen, der trotzdem irgendwohin auflöst, und
genau so wird aus einem Filter ein Traversal.

| `--tools` | lesen | schreiben | Shell |
|---|---|---|---|
| `off` | — | — | — |
| `ro` | ja | verweigert | verweigert |
| `workspace` *(Default)* | ja | ja, eingegrenzt | **fragt** |
| `full` | ja | überall | ja, ungefragt |

Drei Eigenschaften, und alle drei sind Absichten, keine Nebenwirkungen:

**Die Eingrenzung wird aufgelöst, nicht verglichen.** Jedes Pfadargument geht durch
`realpath()`, bevor es gegen die Wurzel geprüft wird. `../../etc/passwd`, ein absolutes
`/etc/shadow` und ein Symlink aus dem Baum heraus landen alle außerhalb und werden abgelehnt —
ein `strncmp` auf das Argument, wie es getippt wurde, lässt alle drei durch.

**Ein „fragen" ohne jemanden zum Fragen ist eine Ablehnung.** `/v1/tools/exec` antwortet
`{"decision":"ask"}` und führt nichts aus. Wer die Frage ignoriert, kommt nicht weiter — es gibt
keinen Timeout, der irgendwann ja sagt.

**Der Server startet nicht in einer Konfiguration, die die Maschine ans Netz gibt.** Werkzeuge
auf einer Nicht-Loopback-Adresse ohne `--api-key` werden beim Start abgelehnt, nicht bewarnt.

**Ausgabe ist ein Budget.** 16 KiB pro Ergebnis (`--tool-output`), und bei Befehlen als *Kopf
plus Ringpuffer des Endes* — nicht als die ersten 16 KiB. Die Richtung ist der Punkt: ein Build
schreibt eine Minute Warnungen und seine **Fehler** in die letzten zwanzig Zeilen. Nur den Kopf
zu behalten heißt, dem Modell ein Log zu geben, das endet, bevor etwas schiefging — und es
meldet dann Erfolg. Befehle werden nach `--tool-timeout` (120 s) als Prozess**gruppe** getötet,
damit eine Shell nichts Hinterlassenes zurücklässt.

**Was es pro Zug kostet:** 3,6 KB Schemata plus 673 Zeichen Systemprompt, etwa 1,2k Token. Sie
stehen ganz vorne im Prompt und ändern sich nie, also bedient der Präfix-Cache sie ab dem
zweiten Zug vollständig. Genau darum schreibt die Oberfläche die Historie nie um: Anhängen
behält den Cache, das Ändern einer früheren Nachricht wirft ihn weg.

**Datei-Ansicht.** Links der Projektbaum, lazy aufgeklappt. Was der Agent schreibt oder ändert,
bekommt einen grünen Punkt im Baum und einen **öffnen**-Knopf auf der Werkzeugkarte — der
Betrachter rechts springt dann auf die geänderte Zeile. Zeilennummern, Syntaxfarben, und die
Tabs **Datei / Trajectory** teilen sich das Panel.

```
GET /v1/fs/list?project=NAME&path=REL     Verzeichnisinhalt
GET /v1/fs/read?project=NAME&path=REL     eine Datei, bis 2 MiB
```

Diese Endpunkte sind für Menschen geformt, nicht fürs Modell: rohe Bytes, keine Zeilennummern,
kein 4-KiB-Budget. Aber sie sind **kein zweiter Weg ins Dateisystem** — sie gehen durch dasselbe
`h_resolve`, dasselbe `realpath`, denselben Zaun und dasselbe thread-lokale Projekt wie die
Werkzeuge.

Oben rechts stehen dauerhaft **Cache-Trefferquote** und **Token/s**; das ⏱-Panel zeigt pro
Schritt Prefill- und Decodezeit, Tokenzahlen und wie viele davon aus dem Cache kamen.

opencode benutzt davon nichts — es bringt eigene Werkzeuge mit und braucht nur
`/v1/chat/completions`. Beides kann an einem Server hängen.

## Stellschrauben

| Variable | Default | Wirkung |
|---|---|---|
| `QWEN_CUDA_STREAM` | `0.60` | Anteil jeder nicht-residenten FFN-Schicht, den die GPU beim **Decode** über PCIe zieht. |
| `QWEN_CUDA_STREAM_PREFILL` | `0.85` | Dasselbe für **Batch-Arbeit**. Höher, weil ein Batch jedes Gewicht S-mal benutzt: die CPU-Seite amortisiert ihre Dekodierkosten, die Buskosten pro Byte bleiben gleich. Nicht 1,0 — die GPU-Seite ist eine DMA und bei f=1 dauert ein Batch 400 ms statt 339. |
| `QWEN_CACHE_GB` | `3.0` | Speicher für Präfix-Prüfpunkte. Jeder hält den rekurrenten Zustand plus den Trunk-KV seines Präfixes und ist damit unabhängig von allem, was dazwischen lief. |
| `QWEN_CACHE_SPACING` | `512` | Abstand der Prüfpunkte in Token. Bestimmt, wie viel im schlimmsten Fall neu gerechnet wird. |
| `QWEN_CUDA_KV_TOK` | `8192` | Obergrenze des KV-Fensters im VRAM. Größer lohnt erst jenseits ~24k Kontext — davor sind residente FFN-Schichten mehr wert (§7.8 im Plan.md). |
| `QWEN_CUDA_FFN_GB` | auto | Deckel für residente FFN-Schichten. |
| `QWEN_KV_SPILL` | `../../kvspill` | Auslagerungspfad des kalten KV-Tiers. **Nicht auf tmpfs zeigen lassen** — das legt den „Disk"-Tier zurück in genau den RAM, den er freimachen soll. |
| `QWEN_RESERVE_GB` | `4` | RAM, den die Engine nie anfasst. |
| `QWEN_KERNEL_LOG` | aus | Meldet pro Komponente, auf welchem Gerät sie tatsächlich lief. |
| `--tools MODE` | `workspace` | Was der Agent darf: `off`, `ro`, `workspace`, `full`. |
| `--workspace DIR` | `.` | Wurzel, in der der Agent arbeitet. |
| `--tool-output N` | `16384` | Bytes Werkzeugausgabe, die das Modell sieht. |
| `--tool-timeout MS` | `120000` | Wann ein Befehl getötet wird. |

## Endpunkte

| Methode | Pfad | |
|---|---|---|
| `GET` | `/v1/tools` | Schemata, Policy je Werkzeug, Systemprompt des Agenten |
| `POST` | `/v1/tools/exec` | Ein Werkzeug ausführen: `{name, arguments, approved}` |
| `GET` | `/` | die Chat-Oberfläche |
| `GET` | `/v1/models` | Modellliste |
| `POST` | `/v1/chat/completions` | streamend (SSE) und nicht-streamend |
| `GET` | `/health` | Gerät, Kontext, VRAM |

Unterstützt werden `messages`, `tools`, `stream`, `temperature`, `top_p`, `top_k`,
`max_tokens`, `seed` und `enable_thinking`. Der Gedankengang kommt als `reasoning_content`
zurück — getrennt vom `content`, damit ihn ein Client anzeigen oder ignorieren kann.
