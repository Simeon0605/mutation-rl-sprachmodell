# Kleiner Char-Level-Transformer in purem numpy.
# Forward + manuelles Backprop (kein Autograd), ~1,0 M Parameter.
import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def gelu_grad(x):
    u = 0.7978845608 * (x + 0.044715 * x * x * x)
    t = np.tanh(u)
    du = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


def ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd)


def ln_backward(dy, xhat, rstd, g):
    dg = (dy * xhat).sum(axis=(0, 1))
    db = dy.sum(axis=(0, 1))
    dxhat = dy * g
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xhat).mean(-1, keepdims=True)
    return (dxhat - m1 - xhat * m2) * rstd, dg, db


def causal_mask(T, dtype=np.float32):
    m = np.zeros((T, T), dtype=dtype)
    iu = np.triu_indices(T, 1)
    m[iu] = -1e9
    return m


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------- Parameter-Layout ----------------

def param_spec(V, cfg):
    d, L, T, ff = cfg["d"], cfg["L"], cfg["T"], cfg["ff"]
    spec = [("wte", (V, d)), ("wpe", (T, d)),
            ("lnf_g", (d,)), ("lnf_b", (d,)), ("whead", (d, V))]
    for l in range(L):
        spec += [
            (f"ln1g{l}", (d,)), (f"ln1b{l}", (d,)),
            (f"wq{l}", (d, d)), (f"wk{l}", (d, d)),
            (f"wv{l}", (d, d)), (f"wo{l}", (d, d)),
            (f"ln2g{l}", (d,)), (f"ln2b{l}", (d,)),
            (f"w1{l}", (d, ff)), (f"b1{l}", (ff,)),
            (f"w2{l}", (ff, d)), (f"b2{l}", (d,)),
        ]
    return spec


def count_params(V, cfg):
    return sum(int(np.prod(s)) for _, s in param_spec(V, cfg))


def init_params(V, cfg, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    d, L = cfg["d"], cfg["L"]
    res_scale = 1.0 / np.sqrt(2.0 * L)
    P = {}
    for name, shape in param_spec(V, cfg):
        if name == "wte":
            P[name] = (rng.normal(0, 0.08, shape)).astype(dtype)
        elif name == "wpe":
            P[name] = (rng.normal(0, 0.02, shape)).astype(dtype)
        elif name == "whead":
            P[name] = (rng.normal(0, 0.02, shape)).astype(dtype)
        elif name.startswith(("wq", "wk", "wv", "w1")):
            P[name] = (rng.normal(0, 0.06, shape)).astype(dtype)
        elif name.startswith(("wo", "w2")):
            P[name] = (rng.normal(0, 0.06 * res_scale, shape)).astype(dtype)
        elif name.startswith("ln") and name.endswith("g"):
            P[name] = np.ones(shape, dtype=dtype)
        else:  # Biases und ln_b
            P[name] = np.zeros(shape, dtype=dtype)
    return P


_ORDER_CACHE = {}


def _order(V, cfg):
    key = (V, cfg["d"], cfg["L"], cfg["T"], cfg["ff"])
    if key not in _ORDER_CACHE:
        _ORDER_CACHE[key] = param_spec(V, cfg)
    return _ORDER_CACHE[key]


def to_flat(P, V, cfg):
    return np.concatenate([P[k].ravel() for k, _ in _order(V, cfg)]).astype(
        next(iter(P.values())).dtype, copy=False)


def from_flat(theta, V, cfg, dtype=np.float32):
    P = {}
    off = 0
    for k, shape in _order(V, cfg):
        n = int(np.prod(shape))
        P[k] = theta[off:off + n].reshape(shape).astype(dtype)
        off += n
    return P


# ---------------- Forward ----------------

def forward(P, idx, cfg, need_cache=False):
    d, L, H, ff = cfg["d"], cfg["L"], cfg["H"], cfg["ff"]
    hd = d // H
    B, T = idx.shape
    x = P["wte"][idx] + P["wpe"][:T][None, :, :]
    mask = causal_mask(T, dtype=x.dtype)
    layers = []
    for l in range(L):
        h, ln1c = ln_forward(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
        q = h @ P[f"wq{l}"]
        k = h @ P[f"wk{l}"]
        v = h @ P[f"wv{l}"]
        qh = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        kh = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        sc = (qh @ kh.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd)) + mask[None, None]
        p = softmax(sc)
        ah = p @ vh
        aflat = ah.transpose(0, 2, 1, 3).reshape(B, T, d)
        a = aflat @ P[f"wo{l}"]
        x2 = x + a
        h2, ln2c = ln_forward(x2, P[f"ln2g{l}"], P[f"ln2b{l}"])
        pre = h2 @ P[f"w1{l}"] + P[f"b1{l}"]
        act = gelu(pre)
        m = act @ P[f"w2{l}"] + P[f"b2{l}"]
        if need_cache:
            layers.append((l, h, ln1c, qh, kh, vh, p, aflat, h2, ln2c, pre, act))
        x = x2 + m
    xf, lnfc = ln_forward(x, P["lnf_g"], P["lnf_b"])
    logits = xf @ P["whead"]
    cache = None
    if need_cache:
        cache = {"idx": idx, "layers": layers, "xf": xf, "lnfc": lnfc}
    return logits, cache


# ---------------- Backward ----------------

def backward(cache, dlogits, P, cfg):
    d, L, H, ff = cfg["d"], cfg["L"], cfg["H"], cfg["ff"]
    hd = d // H
    B, T = cache["idx"].shape
    V = P["whead"].shape[1]
    grads = {}

    def zeros_like(name):
        if name not in grads:
            grads[name] = np.zeros_like(P[name])
        return grads[name]

    xf = cache["xf"]
    dx = dlogits @ P["whead"].T                      # (B,T,d)
    zeros_like("whead")[:] += xf.reshape(-1, d).T @ dlogits.reshape(-1, V)
    dxi, dg, db = ln_backward(dx, *cache["lnfc"], P["lnf_g"])
    zeros_like("lnf_g")[:] += dg
    zeros_like("lnf_b")[:] += db
    dx = dxi

    for (l, h, ln1c, qh, kh, vh, p, aflat, h2, ln2c, pre, act) in reversed(cache["layers"]):
        # x_out = x2 + m  -> 两部分
        dx2 = dx.copy()          # direkt auf x2 (Residual)
        dm = dx                  # in den MLP-Ausgang

        # --- MLP ---
        zeros_like(f"w2{l}")[:] += act.reshape(-1, ff).T @ dm.reshape(-1, d)
        zeros_like(f"b2{l}")[:] += dm.sum(axis=(0, 1))
        dact = dm @ P[f"w2{l}"].T
        dpre = dact * gelu_grad(pre)
        zeros_like(f"w1{l}")[:] += h2.reshape(-1, d).T @ dpre.reshape(-1, ff)
        zeros_like(f"b1{l}")[:] += dpre.sum(axis=(0, 1))
        dh2 = dpre @ P[f"w1{l}"].T
        dx2b, dg2, db2 = ln_backward(dh2, *ln2c, P[f"ln2g{l}"])
        zeros_like(f"ln2g{l}")[:] += dg2
        zeros_like(f"ln2b{l}")[:] += db2
        dx2 += dx2b

        # --- Attention: x2 = x_in + aflat @ wo ---
        da = dx2
        zeros_like(f"wo{l}")[:] += aflat.reshape(-1, d).T @ da.reshape(-1, d)
        daflat = da @ P[f"wo{l}"].T
        dah = daflat.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        dvh = p.transpose(0, 1, 3, 2) @ dah
        dp = dah @ vh.transpose(0, 1, 3, 2)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True))
        ds *= 1.0 / np.sqrt(hd)
        dqh = ds @ kh
        dkh = ds.transpose(0, 1, 3, 2) @ qh
        dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, d)
        dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, d)
        dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, d)
        dh = dq @ P[f"wq{l}"].T + dk @ P[f"wk{l}"].T + dv @ P[f"wv{l}"].T
        hr = h.reshape(-1, d)
        zeros_like(f"wq{l}")[:] += hr.T @ dq.reshape(-1, d)
        zeros_like(f"wk{l}")[:] += hr.T @ dk.reshape(-1, d)
        zeros_like(f"wv{l}")[:] += hr.T @ dv.reshape(-1, d)
        dhb, dg1, db1 = ln_backward(dh, *ln1c, P[f"ln1g{l}"])
        zeros_like(f"ln1g{l}")[:] += dg1
        zeros_like(f"ln1b{l}")[:] += db1
        dx = dx2 + dhb           # Gradient auf Eingabe dieser Schicht

    # --- Embeddings ---
    idx = cache["idx"]
    dwte = np.zeros_like(P["wte"])
    np.add.at(dwte, idx.ravel(), dx.reshape(-1, d))
    zeros_like("wte")[:] += dwte
    zeros_like("wpe")[:T] += dx.sum(axis=0)
    return grads


# ---------------- Loss / RL-Helfer ----------------

def _gather_vocab(p, y):
    """p (B,T,V), y (B,T) -> p[b, t, y[b,t]] mit Werten (B,T)."""
    B, T, V = p.shape
    return p[np.arange(B)[:, None], np.arange(T)[None, :], y]


def ce_loss(logits, y):
    z = logits - logits.max(-1, keepdims=True)
    logZ = np.log(np.exp(z).sum(-1))
    return float(-(_gather_vocab(z, y) - logZ).mean())


def ce_loss_and_dlogits(logits, y):
    B, T, V = logits.shape
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(-1, keepdims=True)
    loss = float(-np.log(_gather_vocab(p, y) + 1e-12).mean())
    d = p.copy()
    d[np.arange(B)[:, None], np.arange(T)[None, :], y] -= 1.0
    d /= (B * T)
    return loss, d


def sample_actions(p, rng):
    """Zieht für jede Position eine Aktion aus der Politik p."""
    B, T, V = p.shape
    cum = np.cumsum(p.reshape(-1, V), axis=1)
    r = rng.random((cum.shape[0], 1))
    a = (cum < r).sum(axis=1).clip(0, V - 1)
    return a.reshape(B, T)


def rl_dlogits(logits, a, adv):
    """REINFORCE: L = -1/(B*T) * sum(adv * log pi(a)). adv > 0 -> Aktion verstärken."""
    B, T, V = logits.shape
    d = softmax(logits)
    d[np.arange(B)[:, None], np.arange(T)[None, :], a] -= 1.0
    d *= (adv / (B * T))[:, :, None]
    return d


def generate(P, cfg, ids, n_new, temp=0.8, topk=20, rng=None):
    rng = rng or np.random.default_rng()
    ids = list(ids)
    for _ in range(n_new):
        ctx = np.array(ids[-cfg["T"]:])[None, :]
        logits = forward(P, ctx, cfg)[0][0, -1].astype(np.float64)
        logits /= max(temp, 1e-3)
        if topk and topk < logits.shape[0]:
            th = np.partition(logits, -topk)[-topk]
            logits[logits < th] = -1e9
        p = softmax(logits)
        cum = np.cumsum(p)
        a = int((cum < rng.random()).sum())
        ids.append(min(a, len(p) - 1))
    return ids


# ---------------- Gradienten-Selbsttest ----------------

if __name__ == "__main__":
    sys_cfg = dict(d=16, L=2, T=8, H=2, ff=32)
    V = 11
    rng = np.random.default_rng(0)
    P = init_params(V, sys_cfg, seed=1, dtype=np.float64)
    x = rng.integers(0, V, (2, sys_cfg["T"]))
    y = rng.integers(0, V, (2, sys_cfg["T"]))

    theta0 = to_flat(P, V, sys_cfg).astype(np.float64)

    def loss_of(theta):
        return ce_loss(forward(from_flat(theta, V, sys_cfg, dtype=np.float64), x, sys_cfg)[0], y)

    logits, cache = forward(P, x, sys_cfg, need_cache=True)
    l0, dlog = ce_loss_and_dlogits(logits, y)
    gflat = to_flat(backward(cache, dlog, P, sys_cfg), V, sys_cfg).astype(np.float64)

    h = 1e-5
    worst, idxs = 0.0, rng.choice(len(theta0), 30, replace=False)
    for i in idxs:
        tp = theta0.copy(); tp[i] += h
        tm = theta0.copy(); tm[i] -= h
        num = (loss_of(tp) - loss_of(tm)) / (2 * h)
        ana = gflat[i]
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        worst = max(worst, rel)
    print(f"Parameter gesamt (kleine Test-Konfig): {len(theta0)}")
    print(f"Gradienten-Check: max. relativer Fehler = {worst:.2e}  "
          f"({'OK' if worst < 1e-5 else 'FEHLER!'})")
