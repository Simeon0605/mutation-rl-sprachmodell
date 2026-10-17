# Training NUR durch Belohnung:
#   1) Mutation/Evolution: Gewichte werden per Gaussian-Rauschen mutiert (antithetisch),
#      Fitness = -CrossEntropy auf echtem Deutsch; beste Mutanten vererben (OpenAI-ES-Update).
#   2) Reinforcement: REINFORCE — das Modell samplet Zeichen als Aktionen:
#        +1 Reward wenn die Aktion dem echten Deutschen entspricht, sonst -1
#        (falsche Aktionen werden gezielt abgeschwaecht).
#     Plus Expert-Reinforcement: das echte deutsche Zeichen an jeder Position gilt als
#      belohnte Expertenaktion (Policy-Gradient auf KL(Deutsch||Politik)).
#      Reines 0/1-RL kollabiert sonst auf "e e e e" (beweisbar im alten train_log).
# Kein klassisches supervised Gradient-Descent auf Labels.
import os

# WICHTIG: muss vor dem numpy-Import stehen. 12 BLAS-Threads auf kleinen Matrizen
# sind auf Windows ~40x LANGSAMER als ein Thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time

import numpy as np

import model_np as M

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=15.0)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--pairs", type=int, default=6, help="ES: Paare antithetischer Mutanten pro Generation")
ap.add_argument("--sigma", type=float, default=0.015, help="ES: Mutationsstaerke")
ap.add_argument("--es-lr", type=float, default=0.001)
ap.add_argument("--rl-steps", type=int, default=6, help="REINFORCE-Schritte pro Generation")
ap.add_argument("--rl-lr", type=float, default=8e-4)
ap.add_argument("--expert", type=float, default=1.0, help="Gewicht der Expert-Aktion (0 = reines 0/1-RL)")
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--sample-every", type=int, default=20)
args = ap.parse_args()

CFG = dict(d=320, L=8, T=64, H=4, ff=1280)  # 10M-Variante: 9.917.440 Parameter

data = np.load("corpus_ids.npy").astype(np.int64)
vocab = json.load(open("vocab.json", encoding="utf-8"))["vocab"]
V = len(vocab) + 1
N = M.count_params(V, CFG)
print(f"Vokabular: {V} | Parameter gesamt: {N:,}")
assert 9_500_000 < N < 10_500_000, "Parameterzahl soll ~10M sein"

rng = np.random.default_rng(1234)


def get_batch(B):
    starts = rng.integers(0, len(data) - CFG["T"] - 1, size=B)
    win = data[starts[:, None] + np.arange(CFG["T"] + 1)[None, :]]
    return win[:, :-1], win[:, 1:]


# Validierungsbatch aus dem Korpus-Ende (feste Referenz)
vstart = len(data) - 10_000
vi = rng.integers(vstart, len(data) - CFG["T"] - 1, size=args.batch)
vwin = data[vi[:, None] + np.arange(CFG["T"] + 1)[None, :]]
val_x, val_y = vwin[:, :-1], vwin[:, 1:]

# Startwerte
if args.resume:
    theta = np.load("ckpt.npz")["theta"].astype(np.float32)
    print("Checkpoint geladen (resume).")
else:
    theta = M.to_flat(M.init_params(V, CFG, seed=7), V, CFG)

# Adam-Zustand fuer REINFORCE
adam_m = np.zeros(N, dtype=np.float32)
adam_v = np.zeros(N, dtype=np.float32)
adam_t = 0
# Momentum fuer ES
es_mom = np.zeros(N, dtype=np.float32)

BASE_LOG = "train_log.txt"
log = open(BASE_LOG, "a", encoding="utf-8")
BASE_RL = 0.0  # EMA-Baseline fuer REINFORCE
t_start = time.time()
last_ckpt = time.time()
gen = 0
best_ce = float("inf")


def save_ckpt():
    np.savez("ckpt.npz", theta=theta, cfg=np.array(json.dumps(CFG)))
    print(f"[ckpt] gespeichert bei {time.time()-t_start:.0f}s", flush=True)


def save_best(val_ce):
    global best_ce
    if val_ce < best_ce:
        best_ce = val_ce
        np.savez("ckpt_best.npz", theta=theta, cfg=np.array(json.dumps(CFG)))
        print(f"[best] neue beste val_ce {val_ce:.4f} -> ckpt_best.npz", flush=True)


def write_sample():
    P = M.from_flat(theta, V, CFG)
    ids = M.generate(P, CFG, [stoi_chars.get(c, V - 1) for c in "Die "],
                     90, temp=0.8, topk=20, rng=rng)
    itos = vocab + [""]
    txt = "".join(itos[i] for i in ids).replace("\n", " ")
    with open("samples.txt", "a", encoding="utf-8") as fh:
        fh.write(f"--- Generation {gen} (val_ce {last_ce:.3f}) ---\n{txt}\n")
    print(f"[probe] {txt!r}")


stoi_chars = {c: i for i, c in enumerate(vocab)}
last_ce = float("nan")
print(f"Start-Loss (zufaellig): {M.ce_loss(M.forward(M.from_flat(theta, V, CFG), val_x, CFG)[0], val_y):.3f} "
      f"(ln({V}) = {np.log(V):.3f} waere Raten)")
budget = args.minutes * 60

try:
    while time.time() - t_start < budget:
        gen_t0 = time.time()
        gen += 1

        # ---------- 1) MUTATION / EVOLUTIONSTRATEGIE ----------
        bx, by = get_batch(args.batch)
        fits = np.empty(2 * args.pairs, dtype=np.float64)
        for i in range(args.pairs):
            eps = np.random.default_rng([777, gen, i]).standard_normal(N).astype(np.float32)
            for s, off in ((1.0, 0), (-1.0, 1)):
                P = M.from_flat(theta + s * args.sigma * eps, V, CFG)
                fits[2 * i + off] = -M.ce_loss(M.forward(P, bx, CFG)[0], by)

        # zentrierte Ranks: beste Mutante +0.5, schlechteste -0.5
        order = np.argsort(fits)
        u = np.empty_like(fits)
        u[order] = np.arange(2 * args.pairs) / (2 * args.pairs - 1) - 0.5

        acc = np.zeros(N, dtype=np.float32)
        for i in range(args.pairs):
            eps = np.random.default_rng([777, gen, i]).standard_normal(N).astype(np.float32)
            acc += u[2 * i] * eps          # +Variante
            acc -= u[2 * i + 1] * eps      # -Variante
        g_es = acc * (args.es_lr / (args.pairs * args.sigma))
        es_mom = 0.5 * es_mom + g_es
        theta = theta + es_mom

        # ---------- 2) REINFORCEMENT LEARNING ----------
        accs, p1s = [], []
        BT = args.batch * CFG["T"]
        for _ in range(args.rl_steps):
            bx2, by2 = get_batch(args.batch)
            P = M.from_flat(theta, V, CFG)
            logits, cache = M.forward(P, bx2, CFG, need_cache=True)
            p = M.softmax(logits)

            # a) Expert-Reinforcement: das echte deutsche Zeichen ist die belohnte Aktion
            onehot_y = np.eye(V, dtype=np.float32)[by2]
            dlog = (p - onehot_y) * (args.expert / BT)

            # b) Trial-and-Error: eigene gesampelte Aktionen bekommen +/-1 Reward
            a = M.sample_actions(p, rng)
            r = np.where(a == by2, 1.0, -1.0).astype(np.float32)
            BASE_RL = 0.9 * BASE_RL + 0.1 * float(r.mean())
            adv = r - BASE_RL
            dlog += M.rl_dlogits(logits, a, adv)

            p1s.append(float((logits.argmax(-1) == by2).mean()))
            gflat = M.to_flat(M.backward(cache, dlog, P, CFG), V, CFG)
            nrm = float(np.linalg.norm(gflat))
            if nrm > 5.0:
                gflat *= 5.0 / nrm
            adam_t += 1
            adam_m = 0.9 * adam_m + 0.1 * gflat
            adam_v = 0.999 * adam_v + 0.001 * (gflat * gflat)
            mh = adam_m / (1 - 0.9 ** adam_t)
            vh = adam_v / (1 - 0.999 ** adam_t)
            theta = theta - args.rl_lr * mh / (np.sqrt(vh) + 1e-8)
            accs.append((float(r.mean()) + 1.0) / 2.0)

        # ---------- Logging / Checkpoints ----------
        val_ce = M.ce_loss(M.forward(M.from_flat(theta, V, CFG), val_x, CFG)[0], val_y)
        last_ce = val_ce
        save_best(val_ce)
        dt = time.time() - gen_t0
        line = (f"gen {gen:4d} | val_ce {val_ce:.4f} | top1 {np.mean(p1s)*100:5.2f}% "
                f"| rl_treffer {np.mean(accs)*100:5.2f}% "
                f"| es_fit_std {fits.std():.4f} | {dt:.2f}s/gen "
                f"| {time.time()-t_start:.0f}s")
        print(line)
        log.write(line + "\n")
        log.flush()

        if gen % args.sample_every == 0:
            write_sample()
        if time.time() - last_ckpt > 120:
            save_ckpt()
            last_ckpt = time.time()

except KeyboardInterrupt:
    print("Abbruch durch Nutzer.")
finally:
    save_ckpt()
    write_sample()
    log.close()
    print(f"Fertig: {gen} Generationen in {time.time()-t_start:.0f}s. Checkpoint: ckpt.npz")
