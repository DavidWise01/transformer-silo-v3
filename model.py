#!/usr/bin/env python3
"""A tiny, REAL, trainable transformer classifier (numpy) with hand-written
forward + backward -- the same model for both arms of the v3 comparison.

One single-head self-attention block (no causal mask -- it classifies a set/bag)
+ a 2-layer MLP, both with residuals, then mean-pool -> linear -> softmax over C
classes. Token and positional embeddings are FIXED (frozen); the trainable
parameters are the attention block, the MLP, and the classifier head, so both the
plain arm and the silo arm have identical capacity and differ only in their INPUT
(N token+position vectors vs K clustered intent vectors).

The gradients are hand-derived and gradient-checked in selftest.py -- that check
is the honesty anchor of v3: it proves the training is real.
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def init_params(d, h, C, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.2
    def w(*shape): return rng.standard_normal(shape) * s
    return {
        "Wq": w(d, d), "Wk": w(d, d), "Wv": w(d, d), "Wo": w(d, d),
        "W1": w(d, h), "b1": np.zeros(h), "W2": w(h, d), "b2": np.zeros(d),
        "Wc": w(d, C), "bc": np.zeros(C),
    }


def forward(P, X, w=None):
    """X: (n, d) input vectors for ONE example. w: (n,) pooling weights (uniform
    for the plain arm; cluster sizes for the silo arm, so a size-weighted mean
    over centroids equals a plain mean over the tokens they summarise). Returns
    (logits, cache)."""
    n, d = X.shape
    if w is None:
        w = np.ones(n)
    wn = w / w.sum()
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]
    S = (Q @ K.T) / np.sqrt(d)
    A = softmax(S, axis=1)
    Ctx = A @ V
    Attn = Ctx @ P["Wo"]
    Z1 = X + Attn
    Hpre = Z1 @ P["W1"] + P["b1"]
    H = np.maximum(0.0, Hpre)
    M = H @ P["W2"] + P["b2"]
    Z2 = Z1 + M
    p = (Z2 * wn[:, None]).sum(axis=0)                  # weighted mean-pool
    logits = p @ P["Wc"] + P["bc"]
    cache = dict(X=X, Q=Q, K=K, V=V, S=S, A=A, Ctx=Ctx, Z1=Z1, Hpre=Hpre, H=H, Z2=Z2, p=p, d=d, wn=wn)
    return logits, cache


def loss_and_grad(P, X, y, w=None):
    """Cross-entropy loss for one example + full analytic gradient dict."""
    logits, c = forward(P, X, w)
    probs = softmax(logits, axis=-1)
    loss = -np.log(probs[y] + 1e-12)

    g = {}
    dlogits = probs.copy(); dlogits[y] -= 1.0          # (C,)
    g["Wc"] = np.outer(c["p"], dlogits)
    g["bc"] = dlogits
    dp = P["Wc"] @ dlogits                              # (d,)
    dZ2 = np.outer(c["wn"], dp)                         # weighted-pool backward

    # MLP block:  Z2 = Z1 + M ; M = H W2 + b2 ; H = relu(Hpre) ; Hpre = Z1 W1 + b1
    dZ1 = dZ2.copy()
    dM = dZ2
    g["W2"] = c["H"].T @ dM
    g["b2"] = dM.sum(axis=0)
    dH = dM @ P["W2"].T
    dHpre = dH * (c["Hpre"] > 0)
    g["W1"] = c["Z1"].T @ dHpre
    g["b1"] = dHpre.sum(axis=0)
    dZ1 += dHpre @ P["W1"].T

    # Attention block: Z1 = X + Attn ; Attn = Ctx Wo ; Ctx = A V ; A = softmax(S)
    dAttn = dZ1                                         # (dX from residual added later)
    g["Wo"] = c["Ctx"].T @ dAttn
    dCtx = dAttn @ P["Wo"].T
    dA = dCtx @ c["V"].T                                # (n,n)
    dV = c["A"].T @ dCtx
    # softmax (row-wise) backward
    dS = c["A"] * (dA - (dA * c["A"]).sum(axis=1, keepdims=True))
    dS /= np.sqrt(c["d"])
    dQ = dS @ c["K"]
    dK = dS.T @ c["Q"]
    g["Wq"] = c["X"].T @ dQ
    g["Wk"] = c["X"].T @ dK
    g["Wv"] = c["X"].T @ dV
    # input X gradient (not needed for training frozen embeddings, but kept exact)
    # dX = dZ1(from residual) + through Q,K,V
    return loss, g, (probs, logits)


def predict(P, X):
    logits, _ = forward(P, X)
    return int(np.argmax(logits))
