# Baut den deutschen Trainingskorpus aus ALLEN pg*.txt-Dateien (Project Gutenberg,
# gemeinfrei). Originale: 17.-20. Jahrhundert - alles VOR 2025.
# Das Vokabular wird eingefroren (vorhandene vocab.json wird wiederverwendet),
# damit alte Checkpoints wie models/best_1m.npz kompatibel bleiben.
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def strip_gutenberg(text: str) -> str:
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        nl = text.find("\n", start)
        text = text[nl + 1:]
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if end != -1:
        text = text[:end]
    return text


def normalize(text: str) -> str:
    repl = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
        "\r": "",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n *\n+", "\n\n", text)
    return text.strip()


books = sorted(glob.glob("pg*.txt"))
assert books, "Keine pg*.txt gefunden."
print(f"{len(books)} Buecher gefunden")

parts = []
for f in books:
    body = normalize(strip_gutenberg(open(f, encoding="utf-8-sig", errors="ignore").read()))
    unk_anteil = 1.0 - sum(ch in "abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIKLMNOPRSTUVWXYZ" for ch in body.lower()) / max(len(body), 1)
    print(f"  {f:16s} {len(body):8d} Zeichen")
    parts.append(body)

text = "\n\n".join(parts)

# Vokabular: einfrieren wenn vorhanden, sonst neu bauen (Schwellwert: >= 100x)
if os.path.exists("vocab.json"):
    vocab = json.load(open("vocab.json", encoding="utf-8"))["vocab"]
    print(f"Vokabular eingefroren: {len(vocab)} Zeichen (bestehende Checkpoints bleiben kompatibel)")
else:
    cnt = Counter(text)
    vocab = sorted(c for c, n in cnt.items() if n >= 100 and c != "\r")
    with open("vocab.json", "w", encoding="utf-8") as fh:
        json.dump({"vocab": vocab}, fh, ensure_ascii=False)
    print(f"Vokabular neu gebaut: {len(vocab)} Zeichen")

stoi = {c: i for i, c in enumerate(vocab)}
unk = len(vocab)
ids = np.array([stoi.get(c, unk) for c in text], dtype=np.int16)
np.save("corpus_ids.npy", ids)
with open("corpus_de.txt", "w", encoding="utf-8") as fh:
    fh.write(text)

unk_rate = float((ids == unk).mean()) * 100
print(f"\nKorpus gesamt: {len(text):,} Zeichen | <unk>-Anteil: {unk_rate:.2f}%")
print(f"Vokabular: {len(vocab)} Zeichen + 1 <unk> = {len(vocab) + 1}")
