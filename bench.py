import time
import numpy as np
import model_np as M
import json

vocab = json.load(open("vocab.json", encoding="utf-8"))["vocab"]
V = len(vocab) + 1
CFG = dict(d=128, L=5, T=64, H=4, ff=512)
N = M.count_params(V, CFG)
print(f"N={N:,}", flush=True)

data = np.load("corpus_ids.npy").astype(np.int64)
rng = np.random.default_rng(0)
theta = M.to_flat(M.init_params(V, CFG, seed=7), V, CFG)


def batch(B=32):
    s = rng.integers(0, len(data) - 65, size=B)
    w = data[s[:, None] + np.arange(65)[None, :]]
    return w[:, :-1], w[:, 1:]


x, y = batch()

t0 = time.perf_counter()
P = M.from_flat(theta, V, CFG)
logits, cache = M.forward(P, x, CFG, need_cache=True)
t1 = time.perf_counter()
print(f"forward+cache: {t1-t0:.3f}s", flush=True)

loss, dlog = M.ce_loss_and_dlogits(logits, y)
g = M.backward(cache, dlog, P, CFG)
t2 = time.perf_counter()
print(f"backward:      {t2-t1:.3f}s", flush=True)

t0 = time.perf_counter()
eps = np.random.default_rng([1, 2, 3]).standard_normal(N).astype(np.float32)
print(f"eps 1M:        {time.perf_counter()-t0:.3f}s", flush=True)

t0 = time.perf_counter()
for _ in range(5):
    M.from_flat(theta + 0.05 * eps, V, CFG)
print(f"from_flat:     {(time.perf_counter()-t0)/5*1000:.1f}ms", flush=True)

t0 = time.perf_counter()
for _ in range(3):
    M.ce_loss(M.forward(P, x, CFG)[0], y)
print(f"ES-Eval:       {(time.perf_counter()-t0)/3:.3f}s", flush=True)
