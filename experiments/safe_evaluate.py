"""Development-only copy of the scoring algorithm for the Docker candidate.

The official evaluate.py is intentionally not mounted into the container.
The host still fingerprints and owns the official evaluator.
"""
import collections
import math


def auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(label for _, label in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def ndcg_at_k(labels, k):
    discounts = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** label) - 1) / discounts[i] for i, label in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** label) - 1) / discounts[i] for i, label in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate(user_ids, labels, scores, k=5):
    by_user = collections.defaultdict(list)
    for user, label, score in zip(user_ids, labels, scores):
        by_user[user].append((score, label))
    gnum = gden = 0.0
    ndcgs = []
    for rows in by_user.values():
        rows.sort(key=lambda item: -item[0])
        ordered_labels = [label for _, label in rows]
        positives = sum(ordered_labels)
        if 0 < positives < len(ordered_labels):
            gnum += positives * auc(ordered_labels, [score for score, _ in rows])
            gden += positives
        ndcgs.append(ndcg_at_k(ordered_labels, k))
    gauc = gnum / gden if gden else 0.5
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    return {"GAUC": gauc, f"nDCG@{k}": ndcg, "primary": (gauc + ndcg) / 2.0,
            "users": len(by_user), "rows": len(labels)}
