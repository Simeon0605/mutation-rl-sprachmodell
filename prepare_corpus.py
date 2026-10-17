# Baut den deutschen Trainingskorpus: Gutenberg-Boilerplate entfernen,
# normalisieren, Vokabular bauen, als Token-IDs speichern.
import json
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BOOKS = ["pg2228.txt", "pg2229.txt", "pg7205.txt"]  # Faust, Verwandlung, Zarathustra


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


parts = []
for f in BOOKS:
    raw = open(f, encoding="utf-8-sig", errors="ignore").read()
    body = normalize(strip_gutenberg(raw))
    print(f"{f}: {len(body)} Zeichen | Anfang: {body[:80]!r}")
    parts.append(body)

text = "\n\n".join(parts)

# Vokabular: alle Zeichen mit >= 100 Vorkommen
cnt = Counter(text)
chars = sorted(c for c, n in cnt.items() if n >= 100 and c != "\r")
vocab = chars
stoi = {c: i for i, c in enumerate(vocab)}
unk = len(vocab)

ids = np.array([stoi.get(c, unk) for c in text], dtype=np.int16)
np.save("corpus_ids.npy", ids)
with open("vocab.json", "w", encoding="utf-8") as fh:
    json.dump({"vocab": vocab}, fh, ensure_ascii=False)

with open("corpus_de.txt", "w", encoding="utf-8") as fh:
    fh.write(text)

print(f"\nKorpus gesamt: {len(text)} Zeichen")
print(f"Vokabular: {len(vocab)} Zeichen + 1 <unk> = {len(vocab) + 1}")
print("Vokabular:", "".join(vocab))
