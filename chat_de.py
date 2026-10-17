# Chat mit dem trainierten Modell: Du gibst deutschen Text ein (als Kontext),
# das Modell sampelt die Fortsetzung - gelernt nur durch Mutation + Reinforcement.
#
#   py chat_de.py                      interaktiv
#   py chat_de.py --prompt "Hallo"     einmalige Antwort
#   py chat_de.py --temp 0.6           konservativer
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import model_np as M

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", type=str, default=None, help="einmalig antworten und beenden")
ap.add_argument("--temp", type=float, default=0.75)
ap.add_argument("--topk", type=int, default=15)
ap.add_argument("--chars", type=int, default=140)
args = ap.parse_args()

# 1) frisch trainierter Checkpoint, 2) letzter Stand, 3) fertiges 10M-Modell aus dem Repo
for cand in ["ckpt_best.npz", "ckpt.npz", "models/best_10m.npz", "models/best_1m.npz"]:
    if os.path.exists(cand):
        ckpt = cand
        break
else:
    sys.exit("Kein Checkpoint gefunden. Erst py train_de.py laufen lassen.")
z = np.load(ckpt)
theta = z["theta"].astype(np.float32)
# Konfiguration steht seit der 10M-Version im Checkpoint selbst;
# aeltere 1M-Checkpoints (z.B. models/best_1m.npz) nutzen den Fallback.
if "cfg" in z.files:
    CFG = json.loads(str(z["cfg"]))
else:
    CFG = dict(d=128, L=5, T=64, H=4, ff=512)
vocab = json.load(open("vocab.json", encoding="utf-8"))["vocab"]
stoi = {c: i for i, c in enumerate(vocab)}
itos = vocab + [""]
V = len(vocab) + 1
P = M.from_flat(theta, V, CFG)
rng = np.random.default_rng()

print(f"[Modell geladen: {ckpt} | d={CFG['d']}, L={CFG['L']}]", flush=True)


def antwort(text: str) -> str:
    ids = [stoi[c] for c in text if c in stoi][-CFG["T"]:]
    if not ids:
        ids = [stoi[" "], stoi["D"], stoi["i"], stoi["e"], stoi[" "]]
    out = M.generate(P, CFG, ids, args.chars, temp=args.temp, topk=args.topk, rng=rng)[len(ids):]
    txt = "".join(itos[i] for i in out)
    txt = txt.replace("\r", " ").replace("\n", " ").strip()
    # an letztem Satzende kappen, damit es nach einer Antwort aussieht
    for ch in (".", "!", "?"):
        pos = txt.rfind(ch)
        if pos != -1:
            txt = txt[: pos + 1]
            break
    return txt if txt else "(...)"


if args.prompt is not None:
    print(antwort(args.prompt))
    sys.exit(0)

print("Chat mit dem Modell (nur Mutation+RL trainiert). Befehle: :temp 0.7, :chars 200, :exit")
while True:
    try:
        q = input("Du: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nTschüss!")
        break
    if not q:
        continue
    if q == ":exit":
        print("Tschüss!")
        break
    if q.startswith(":temp"):
        args.temp = float(q.split()[1]); print(f"temp={args.temp}"); continue
    if q.startswith(":chars"):
        args.chars = int(q.split()[1]); print(f"chars={args.chars}"); continue
    print("Modell:", antwort(q), flush=True)
