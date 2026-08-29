# 2.8 Appendix A. A Primer on Recommender Systems

> This appendix gives participants without a recommender-systems background just enough to get started. It is a concept map plus an annotated reading list — not a textbook. Use it to understand the KuaiRand benchmarks and to know what to look up when you get stuck.

## A.1 The Big Picture: The Recommendation Pipeline

A modern industrial recommender does not score every item directly. It runs a funnel of stages, each narrowing the candidate set:

```
Recall → Pre-ranking → Ranking → Re-ranking
millions   thousands      hundreds   final list
```

* **Recall / Retrieval**: cheaply retrieve a few thousand candidates from millions.
* **Pre-ranking**: a lightweight model trims the candidates further.
* **Ranking**: a heavy, accurate model scores each candidate. **This challenge mostly lives here.**
* **Reranking**: adjust the final ordering for diversity, business rules, and so on.

> For this competition you mainly need the **ranking** stage. The KuaiRand benchmarks are ranking/prediction tasks, not full end-to-end pipelines.

## A.2 Core Tasks: CTR and the Feedback Funnel

Most industrial ranking is framed as predicting the probability of user feedback:

* **CTR (Click-Through Rate)** — `P(click | impression)`. The user saw the item; will they click?
* **CVR (Conversion Rate)** — `P(conversion | click)`. The user clicked; will they convert (buy)? E-commerce background only; not a task in this challenge.
* **The funnel**: `impression -> click -> deeper engagement` (in e-commerce, `-> conversion`). Because these stages are linked, two well-known problems arise:
    * **Sample selection bias**: the post-click signal is only observed on *clicked* items, yet must be predicted for *all* impressions.
    * **Data sparsity**: post-click signals such as `long_view` or `like` are far rarer than clicks.

> **KuaiRand** has no purchase label, so CVR itself is never scored here. The funnel framing above is general background — note that in KuaiRand the scored label `long_view` is logged on *every* impression, not only on clicked ones, so classic sample selection bias does not apply directly to this challenge's task. Data sparsity still does, and the multi-feedback structure (`click`, `like`, `follow`, `play_time` ...) makes ESMM-style multi-task modelling — see A.3 — a legitimate way to exploit the other signals as auxiliary tasks.

## A.3 Multi-Task & Multi-Feedback Learning

Real users produce many signals (click, like, follow, comment, watch-time, and so on). Predicting them jointly — rather than training a separate model per signal — shares representations and tends to improve every task.

* Why it matters here: **KuaiRand** provides **12 feedback signals**, so a multi-task model can learn from several of them jointly even though only `long_view` is scored.
* The key idea is to balance *shared* parameters (which transfer useful knowledge across tasks) against *task-specific* parameters (which prevent conflicting tasks from hurting one another — the "seesaw" problem).

## A.4 Evaluation Metrics

| Metric | Intuition | Used for |
| :--- | :--- | :--- |
| **AUC** | Probability that a random positive is ranked above a random negative. Threshold-free and robust to class imbalance. | **Scored in this challenge** as **GAUC** — per-user AUC averaged with each user's positive count as the weight; users whose impressions are all-positive or all-negative are excluded. |
| **NDCG** | Quality of a *ranked list*, rewarding relevant items near the top (with a position discount). | **Scored in this challenge** as **nDCG@5**. Users with no positive label score 0 and are included in the average. |
| **Recall** | Fraction of all relevant items that appear in the returned list. | Retrieval / coverage tasks — **not scored here.** Each user has only ~5 logged impressions in the evaluation split, so Recall@50 is 0.999+ for every model, including random scoring. |

> **Offline vs. online**: a higher offline metric does not always mean better real-world performance (because of distribution shift and feedback loops). This competition is evaluated offline, but it is worth knowing the gap exists.

## A.5 Feature Engineering Basics

* **ID features**: user ID, item ID, category ID — high-cardinality discrete features.
* **Embedding**: map each discrete ID to a learnable dense vector. This is the foundation of all deep recommenders.
* **Feature crossing**: combine features (e.g. user × category) to capture interactions. Models such as FM and DeepFM automate this.

## A.6 Annotated Reading List

[Hints: If you find reading the following material challenging or find you have missing backgrounds, you can use ChatGPT / Claude / ... to explain it to you.]

The goal here is only to understand **how a recommender system is structured** — the recall → ranking → re-ranking pipeline — and where the ranking stage (which this challenge targets) sits within it. You do **not** need to read a whole course; the introductory overview is enough. **Read just one of the following:**

* Google, *Recommendation Systems* (Machine Learning Crash Course), the **Overview** section — `https://developers.google.com/machine-learning/recommendation` A short, official overview of the pipeline. Note: Google calls the ranking stage **"scoring"** — this is the same thing as **ranking**, and it is the part this challenge focuses on.
* Wang Shusen, *Recommender Systems*, **Chapter 1 (Overview)** — `https://github.com/wangshusen/RecommenderSystem` The most beginner-friendly Chinese resource; the first chapter alone gives the full architecture.