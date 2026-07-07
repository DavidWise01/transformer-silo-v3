#!/usr/bin/env python3
"""Verify-first self-test. The honesty of v3 rests on the training being REAL and
the reported finding being what actually happens -- so this proves both, with no
network:
(1) GRADIENT CHECK -- analytic gradients match numerical to < 1e-5 (the training
    is genuinely doing gradient descent, not theatre);
(2) determinism -- same seed, same result;
(3) it LEARNS -- a quick train beats chance on the order-insensitive task;
(4) the honest FINDING holds -- on the order-SENSITIVE task the silo (an unordered
    set of intents) lands far closer to chance than the plain model. v3 does not
    hide that the silo loses there.
"""
from __future__ import annotations
import numpy as np
from model import init_params, loss_and_grad, forward, softmax
from tasks import make_world, dataset, D, G
from train import train_arm

fails = 0
def check(cond, msg):
    global fails
    print(("ok  · " if cond else "FAIL· ") + msg)
    fails += 0 if cond else 1


# 1. GRADIENT CHECK -- the anchor: the backprop is correct, so training is real.
rng = np.random.default_rng(3)
d, h, C, n = 6, 10, 4, 5
P = init_params(d, h, C, seed=1)
X = rng.standard_normal((n, d)); y = 2; w = rng.random(n) + 0.3
loss, g, _ = loss_and_grad(P, X, y, w)
def nloss(Pp):
    lg, _ = forward(Pp, X, w); pr = softmax(lg); return -np.log(pr[y] + 1e-12)
eps = 1e-6; maxrel = 0.0
for name in P:
    Wm = P[name]; num = np.zeros_like(Wm); it = np.nditer(Wm, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index; o = Wm[i]
        Wm[i] = o + eps; lp = nloss(P); Wm[i] = o - eps; lm = nloss(P); Wm[i] = o
        num[i] = (lp - lm) / (2 * eps); it.iternext()
    rel = np.abs(num - g[name]).max() / (np.abs(num).max() + np.abs(g[name]).max() + 1e-12)
    maxrel = max(maxrel, rel)
check(maxrel < 1e-5, f"gradient check passes (max rel err {maxrel:.1e}) -- the training is real")

# quick-train helper (small + few epochs, for a fast but genuine signal)
world = make_world(seed=0)

def quick(task, arm, seed=0, epochs=12):
    data = dataset(world, task, n_train=300, n_test=200, seed=100)
    _, acc, _ = train_arm(world, data, arm, seed=seed, record=False, epochs=epochs)
    return acc

# 2. Determinism.
check(quick("plurality", "plain") == quick("plurality", "plain"), "training is deterministic (same seed -> same accuracy)")

# 3. It LEARNS: quick train clears chance on the order-insensitive task.
chance = 1.0 / G
pp = quick("plurality", "plain"); ps = quick("plurality", "silo")
check(pp > chance + 0.2, f"plain learns plurality well above chance ({pp:.2f} vs {chance:.2f})")
check(ps > chance + 0.2, f"silo learns plurality well above chance ({ps:.2f} vs {chance:.2f})")

# 4. The honest FINDING: on the ORDER task the silo lands far nearer chance than a
#    trained plain model -- the silo, being an unordered set of intents, cannot
#    read position. (Given enough epochs for plain to actually learn the order.)
fp = quick("first", "plain", epochs=35); fs = quick("first", "silo", epochs=35)
check(fp - fs > 0.15, f"on the order task the silo loses to plain by a clear margin ({fp:.2f} vs {fs:.2f})")
check(fs < fp, "v3 reports the silo LOSING where order matters (not hidden)")

print("\n" + ("SOME CHECKS FAILED" if fails else "all transformer-silo-v3 checks passed"))
raise SystemExit(1 if fails else 0)
