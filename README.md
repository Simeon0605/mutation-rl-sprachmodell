#  Mutation-RL-Sprachmodell

Ein Sprachmodell mit **1.014.400 Parametern** (auf dem Branch [`10m-parameter`](../../tree/10m-parameter) gibt es eine 10M-Variante), das aus ~950 KB echtem Deutsch (Goethe, Nietzsche) **nur durch Mutation und Belohnung** Deutsch lernt.

Kein PyTorch, kein TensorFlow, kein klassisches supervised Training — **reines numpy**, eigene Transformer-Implementierung mit handgeschriebenem Backprop (per numerischem Gradienten-Check verifiziert).

```
Du: Es war einmal
Modell: e send dender it d stese silin war undenden der d den st ser inden in der an ...
```

Das Modell hat deutsche Wortformen, Morphologie und Frequenzen gelernt (`der`, `den`, `und`, `war`, `sein`, `-enden`, `-er`, `-in`) — bei 30 Minuten CPU-Training sind viele Wörter noch „Pseudo-Deutsch". Länger trainieren macht es besser (siehe unten).

## Wie es lernt (nur Belohnung, keine Labels)

Pro Generation drei Signale:

1. **🧬 Mutation (Evolutionstrategie):** Es werden 8 Paare „antithetischer" Gewichts-Mutanten erzeugt (`θ ± σ·ε`). Jeder Mutant wird auf echtem Deutsch bewertet (Fitness = −Cross-Entropy). Die besten Mutanten vererben (zentrierte Ranks + Momentum — OpenAI-ES-Update). Das ist reines Trial-and-Error auf Ebene der Gewichte.
2. **🎯 REINFORCE (Trial-and-Error):** Das Modell *samplet* an jeder Position ein Zeichen als Aktion. Entspricht die Aktion dem echten deutschen Zeichen: **+1 Reward** (Aktion wird verstärkt), sonst **−1 Reward** (Aktion wird gezielt abgeschwächt). EMA-Baseline, Adam-Update.
3. **🎓 Expert-Verstärkung:** Das echte deutsche Zeichen an jeder Position gilt als belohnte „Expertenaktion" (Policy-Gradient auf KL(Deutsch ‖ Politik)).

**Warum Signal 3?** Reines 0/1-REINFORCE (nur Signal 2) **kollabiert** — nachweisbar im ersten Lauf in `train_log.txt`: Die Trefferquote stieg auf 24 %, aber die Policy wurde deterministisch auf das häufigste Zeichen (`„e e e e e e ..."`), val_ce explodierte auf 7,7. Das ist die optimale Strategie für „Treffer zählen", aber sinnloses Deutsch. Signal 3 richtet das Belohnungsoptimum auf die *kalibrierte* Sprachverteilung aus (der Kollaps ist ein Feature-Verhalten spärlicher Belohnungen, kein Bug).

## Befehle

Voraussetzung: Python 3.10+ mit numpy (auf Windows: `py` statt `python`).

```powershell
# 1) Sofort chatten (fertiges 1M-Modell liegt bei: models/best_1m.npz)
py chat_de.py                          # interaktiv, :exit beendet
py chat_de.py --prompt "Hallo"        # einmalige Antwort
py chat_de.py --temp 0.5 --topk 5     # konservativer samplen (weniger Phantasie)
py chat_de.py --chars 300             # längere Antworten

# 2) Von Null reproduzieren (Korpus ist im Repo enthalten, Download optional)
py prepare_corpus.py                   # Vokabular + Token-IDs erzeugen (~5 s)
py train_de.py --minutes 15            # 15 Minuten Mutation + RL
py chat_de.py                          # mit dem frisch trainierten Modell chatten

# 3) Weitertrainieren / größer trainieren
py train_de.py --minutes 60 --resume  # ab Checkpoint weiter
py train_de.py --minutes 15 --expert 0  # Achtung: reines 0/1-RL -> kollabiert (zum Ausprobieren!)
```

Eigene Hyperparameter: `--pairs` (Mutantenpaare), `--sigma` (Mutationsstärke), `--rl-steps`, `--rl-lr`, `--batch`, `--sample-every`. Siehe `py train_de.py --help`.

## Ergebnisse (1M-Modell, 2 × 15 min CPU)

| Lauf | val_ce | Top-1-Treffer | Probetext |
|---|---|---|---|
| Start (zufällig) | 4,25 | ~1,5 % | `KVecLCEEc?gCröE_pN:,.öwL...` |
| Reines 0/1-RL ❌ | 7,70 | (kollabiert) | `e e e e e e e e e e e e` |
| **Mit Expert-Signal ✅** | **2,56** | **28 %** | `Die westen geiest den Hender ... ich els` |

Verlauf: `train_log.txt` · Textproben alle 20 Generationen: `samples.txt`

## Architektur

| | |
|---|---|
| Typ | Char-Level-Transformer (pre-LN), eigene numpy-Implementierung |
| Parameter | **1.014.400** (d_model=128, 5 Layer, 4 Köpfe, FFN=512) |
| Kontext | 64 Zeichen |
| Vokabular | 66 Zeichen + `<unk>` (äöüß inklusive) |
| Korpus | Goethes *Faust* (I & II), *Faust I*, Nietzsches *Also sprach Zarathustra* (~958.000 Zeichen, Project Gutenberg, gemeinfrei) |

## Troubleshooting

- **`python` nicht gefunden (Windows):** Microsoft-Store-Alias deaktivieren oder `py` verwenden.
- **Training extrem langsam (>10 s/Generation statt ~2 s/Eval):** OpenBLAS startet auf Windows zu viele Threads für kleine Matrizen. Die Skripte setzen `OMP_NUM_THREADS=1` selbst — manuell: `$env:OPENBLAS_NUM_THREADS=1`.
- **Umlaute werden falsch angezeigt:** PowerShell: `chcp 65001`.
- **Gradienten-Check:** `py model_np.py` prüft das Backprop numerisch (max. rel. Fehler < 1e-5).

## Projektstruktur

| Datei | Zweck |
|---|---|
| `model_np.py` | Transformer, Backprop, Sampling, Gradienten-Selbsttest |
| `train_de.py` | Training: Mutation (ES) + REINFORCE, Checkpoints, Logs |
| `chat_de.py` | Chat-Frontend |
| `prepare_corpus.py` | Gutenberg-Texte → Vokabular + Token-IDs |
| `bench.py` | Performance-Messung (Forward/Backward) |
| `models/best_1m.npz` | fertiges 1M-Modell (val_ce 2,56) |
| `train_log.txt` / `samples.txt` | Trainingsgeschichte inkl. Kollaps-Demo |

## Branch: 10M-Variante

Auf dem Branch [`10m-parameter`](../../tree/10m-parameter) läuft die Skalierung auf **9.917.440 Parameter** (d_model=320, 8 Layer, FFN=1280). Training dauert dort ~10× länger (~2 min/Generation auf 12 Kernen); die Checkpoints speichern ihre Konfiguration selbst, `chat_de.py` erkennt sie automatisch.
