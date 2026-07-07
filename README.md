# transformer-silo v3 — trained it, tested it, reported it straight

v1 built the silo. v2 measured the compression trade-off on *untrained* toys (the
elbow method + rate–distortion). v3 is the real thing David asked for: **train
both arms from scratch, test on held-out data, and report what actually happens —
including where the silo loses.**

## The setup (honest and standard)

- **Two arms, identical capacity.** A tiny transformer classifier (one self-attention
  block + MLP + weighted mean-pool + linear head), trained with Adam. The **plain**
  arm attends over all N tokens (with positional embeddings); the **silo** arm runs
  the *same* model over K k-means intent centroids (weighted by cluster size, so a
  size-weighted mean over centroids equals a plain mean over the tokens). They differ
  only in their input — N tokens vs K intents — so it is a fair, matched comparison.
- **Two standard synthetic probes** (the kind used for architecture ablations):
  - **PLURALITY** — label = the most frequent group in the bag. *Order-insensitive.*
  - **FIRST** — label = the group of the first token. *Order-sensitive.*
- **The training is real, and provably so.** `selftest.py` **gradient-checks** the
  hand-written backprop (analytic vs numerical, < 1e-5). That check is v3's honesty
  anchor — the model genuinely learns by gradient descent, it isn't theatre.

## What happened (seed 0, 800 train / 400 test, held out)

| task | chance | plain (N tokens) | silo (K intents) | attention |
|------|--------|------------------|------------------|-----------|
| **plurality** (order-insensitive) | 25% | 95.3% | **99.5%** | **4× cheaper** |
| **first** (order-sensitive) | 25% | **85.0%** | 42.5% | 4× cheaper |

**Read it straight:** the silo **wins** on the set task — cheaper *and* more accurate,
because the task's latent structure genuinely *is* clusters and the silo's bias fits
it. It **collapses** on the order task — clustering into an unordered set of intents
throws position away, so it can only ride the mode (which weakly predicts the first
token), while the plain model reads position and pulls ahead. The compression helps
where order doesn't matter and hurts where it does.

## The honest caveats (which is the whole point)

- The plurality win is **k-means' home turf** — the task was built from latent groups.
  Don't read it as "the silo is better"; read it as *"when your context really is a
  few intents, clustering to them is a cheap, strong prior."* A task with no cluster
  structure would erase that edge.
- These are **synthetic probes and tiny models** (D=6, one head), not a real-world
  benchmark. The point is a **clean, controlled, honestly-reported ablation**, not a
  leaderboard.
- It is one seed's story, reproducible with `python train.py`.

## Verify first

```bash
python selftest.py     # gradient check + determinism + it learns + the silo LOSES on order
python train.py        # retrain both arms on both tasks, rewrite results.json
```

## Files

| File | Role |
|------|------|
| `model.py` | the tiny transformer classifier: forward + **hand-written, gradient-checked** backward |
| `tasks.py` | the embedding world, the k-means silo front-end, the two probes |
| `train.py` | Adam training of both arms on both tasks → `results.json` |
| `selftest.py` | the gradient check + the honest finding |
| `results.json` | the trained results the page reports |
| `index.html` | the results page — the table, the learning curves, the verdict |

The trilogy: [v1](https://davidwise01.github.io/transformer-silo/) (build) ·
[v2](https://davidwise01.github.io/transformer-silo-v2/) (measure) · v3 (train + test).

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0
