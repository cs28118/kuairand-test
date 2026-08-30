import argparse
import time
import numpy as np

from data import load, encode
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class PairwiseFM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-5, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.l2 = l2

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    def pair_step(self, X_pos, X_neg):
        pos = self.logits(X_pos)
        neg = self.logits(X_neg)
        diff = pos - neg

        # BPR loss: -log(sigmoid(pos - neg))
        g = (sigmoid(-diff) / len(diff)).astype(np.float32)

        gV_pos = np.zeros_like(self.V)
        gV_neg = np.zeros_like(self.V)
        gW_pos = np.zeros_like(self.W)
        gW_neg = np.zeros_like(self.W)

        E_pos = self.V[X_pos]
        S_pos = E_pos.sum(1)
        E_neg = self.V[X_neg]
        S_neg = E_neg.sum(1)

        np.add.at(gW_pos, X_pos, -g[:, None])
        np.add.at(gW_neg, X_neg, g[:, None])

        np.add.at(
            gV_pos,
            X_pos,
            -g[:, None, None] * (S_pos[:, None, :] - E_pos),
        )
        np.add.at(
            gV_neg,
            X_neg,
            g[:, None, None] * (S_neg[:, None, :] - E_neg),
        )

        self.V -= self.lr * (gV_pos + gV_neg + self.l2 * self.V)
        self.W -= self.lr * (gW_pos + gW_neg + self.l2 * self.W)

        return float(np.mean(np.logaddexp(0, -diff)))

    def predict(self, X, bs=200_000):
        return np.concatenate([
            self.logits(X[i:i + bs])
            for i in range(0, len(X), bs)
        ])


def make_pairs(X, y, users, rng, n_pairs=None):
    by_user = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, [[], []])
        by_user[u][int(y[i])].append(i)

    pos, neg = [], []
    for p, n in by_user.values():
        if p and n:
            count = min(len(p), len(n))
            pos.extend(rng.choice(p, count, replace=True))
            neg.extend(rng.choice(n, count, replace=True))

    if not pos:
        raise RuntimeError("没有找到正负样本对")

    pos = np.asarray(pos, dtype=np.int64)
    neg = np.asarray(neg, dtype=np.int64)

    order = rng.permutation(len(pos))
    if n_pairs is not None:
        order = order[:n_pairs]

    return X[pos[order]], X[neg[order]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()

    print(f"loading {args.data_dir} ...")
    splits = load(args.data_dir)
    enc, dim = encode(splits)

    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    model = PairwiseFM(dim, lr=args.lr, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    best = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        Xp, Xn = make_pairs(Xtr, ytr, utr, rng)
        loss = model.pair_step(Xp, Xn)

        valid = evaluate(uva, yva, model.predict(Xva))
        print(
            f"epoch {epoch:2d} | loss {loss:.4f} | "
            f"valid GAUC {valid['GAUC']:.4f} "
            f"nDCG@5 {valid['nDCG@5']:.4f} "
            f"primary {valid['primary']:.4f} | "
            f"{time.time() - t0:.1f}s"
        )

        if valid["primary"] > best + 1e-5:
            best = valid["primary"]
            bad = 0
            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
            )
        else:
            bad += 1
            if bad >= 4:
                print(f"early stop at epoch {epoch}")
                break

    model.V, model.W, model.b = best_state

    valid = evaluate(uva, yva, model.predict(Xva))
    test = evaluate(ute, yte, model.predict(Xte))

    print("\n=== pairwise FM ===")
    print(
        f"valid  GAUC {valid['GAUC']:.4f} | "
        f"nDCG@5 {valid['nDCG@5']:.4f} | "
        f"primary {valid['primary']:.4f}"
    )
    print(
        f"test   GAUC {test['GAUC']:.4f} | "
        f"nDCG@5 {test['nDCG@5']:.4f} | "
        f"primary {test['primary']:.4f}"
    )


if __name__ == "__main__":
    main()
