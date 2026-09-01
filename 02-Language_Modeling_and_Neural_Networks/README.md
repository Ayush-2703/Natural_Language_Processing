<div align="center">

![Phase 2: Language Modeling and Neural Networks](https://capsule-render.vercel.app/api?type=waving&color=0:18181B,100:3F3F46&height=220&section=header&text=Phase%202%20%C2%B7%20Language%20Modeling%20and%20Neural%20Networks&fontSize=34&fontColor=FFFFFF&fontAlignY=38&animation=fadeIn&desc=From%20counting%20bigrams%20to%20the%20first%20neural%20language%20model&descSize=16&descAlignY=62)

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

*[`Natural_Language_Processing`](../README.md) — a topic-wise, theory-to-implementation NLP curriculum*

</div>

---

> **🚧 Status: scaffolding only.** The three topic folders and this module README exist, but the theory write-ups, `implementation.py` files, `explanation.md` walkthroughs, and diagrams have not been added yet. Everything below describes the **planned** scope and structure for this phase — it deliberately does not claim results, datasets-in-hand, or images that don't exist yet. See [Build Progress](#build-progress) for exactly what's outstanding.

---

## Table of Contents

- [Overview](#overview)
- [The Arc of This Phase](#the-arc-of-this-phase)
- [Topics at a Glance](#topics-at-a-glance)
- [Planned Folder Structure](#planned-folder-structure)
- [Datasets Planned for This Phase](#datasets-planned-for-this-phase)
- [Build Progress](#build-progress)
- [Getting Started](#getting-started)
- [Key References](#key-references)
- [Navigate](#navigate)
- [License](#license)
- [Author](#author)

---

## Overview

Phase 1 established *what* a word vector is and how to evaluate one. Phase 2 asks a different question: how do you model **sequences** of words — specifically, how likely is a given sentence, or a given next word, under some model of the language? This is the oldest problem in the field, and it's the problem Bengio et al.'s 2003 neural probabilistic language model (already referenced throughout Phase 1) was originally built to solve — this phase is where that motivation gets unpacked properly, starting from the classical, count-based approach it was reacting against.

The throughline: start with the simplest possible statistical model of language (counting), find its sharpest failure mode (data sparsity — most bigrams a test sentence needs were never seen in training), and then show that a small neural network sidesteps that failure mode entirely by generalizing through shared, dense parameters instead of memorizing discrete counts.

## The Arc of This Phase

**2.1** lays the conceptual groundwork: what a language model is (a probability distribution over sequences), why it's usually factored autoregressively via the chain rule, and where neural networks enter the picture as an alternative to counting.

**2.2** gets concrete with the simplest non-trivial case — **bigrams** — building a frequency-based bigram language model from a real corpus, running into data sparsity directly (bigrams that appear in test data but never in training get zero probability), and covering the classical fixes: smoothing (Laplace/add-k, Good-Turing) and backoff/interpolation.

**2.3** replaces the count table with a small neural network that predicts the next word from the previous one — the same shift in kind (from memorized statistics to learned, generalizing parameters) that Phase 1 traced from one-hot vectors to Word2Vec, now applied to sequence prediction instead of static representation. This is also the most direct, hands-on echo of Bengio et al. (2003) in the whole repository, at the smallest scale that still makes the idea concrete.

## Topics at a Glance

| # | Topic | Folder | Planned to cover |
|---|-------|--------|--------------------|
| 2.1 | Introduction to Language Modeling and Neural Networks | [`01-Introduction_to_Language_Modeling`](01-Introduction_to_Language_Modeling) | What a language model is, chain-rule factorization of sentence probability, perplexity as an evaluation metric, and why neural approaches were introduced |
| 2.2 | Bigrams and Language Constructs | [`02-Bigrams_and_Language_Constructs`](02-Bigrams_and_Language_Constructs) | A frequency-based bigram model built from a real corpus, the sparsity problem it runs into, and classical smoothing/backoff fixes |
| 2.3 | Neural Network Bigram Model | [`03-Neural_Network_Bigram_Model`](03-Neural_Network_Bigram_Model) | A small feedforward network predicting the next word from the previous one — the neural counterpart to 2.2, and a direct, small-scale echo of Bengio et al. (2003) |

This mirrors the [lab-practicals mapping](../README.md#lab-practicals-mapping) already listed in the root README: Lab 6 (bigram frequency language model) → Topic 2.2, Lab 7 (neural bigram language model) → Topic 2.3.

## Planned Folder Structure

Once populated, each topic will follow the same four-part pattern as Phase 1 — a theory `README.md`, a runnable `implementation.py`, a line-by-line `explanation.md`, and an `Image/` folder of diagrams and generated plots:

```
02-Language_Modeling_and_Neural_Networks/
├── README.md                                    (this file)
│
├── 01-Introduction_to_Language_Modeling/
│   └── README.md            — currently empty; theory to be added
│
├── 02-Bigrams_and_Language_Constructs/
│   └── README.md            — currently empty; theory, implementation.py,
│                               explanation.md, and Image/ to be added
│
└── 03-Neural_Network_Bigram_Model/
    └── README.md            — currently empty; theory, implementation.py,
                                 explanation.md, and Image/ to be added
```

## Datasets Planned for This Phase

Per the [root README's dataset table](../README.md#datasets-used), this phase is scoped to use the **Brown** and/or **Reuters** corpora via `nltk.corpus` — both are already covered by the repo-wide `nltk.download` step in [Getting Started](#getting-started) below, so no additional setup will be needed once implementation begins.

## Build Progress

- ✅ 2.1 — Introduction to Language Modeling: theory
- ✅ 2.2 — Bigrams and Language Constructs: theory, implementation, explanation, diagrams
- ✅ 2.3 — Neural Network Bigram Model: theory, implementation, explanation, diagrams

## Getting Started

Repo-wide setup (shared with every phase):

```bash
git clone https://github.com/Ayush-2703/Natural_Language_Processing.git
cd Natural_Language_Processing

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt', 'brown', 'reuters']]"

cd 02-Language_Modeling_and_Neural_Networks
```

There's no `implementation.py` to run in this phase yet — check the [Build Progress](#build-progress) checklist above for what's still outstanding.

## Key References

These anchor the topics planned above; each topic's own `README.md` will cite the specific subset relevant to it once written.

1. Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal. (The original information-theoretic framing of language as a stochastic process — the root of "language modeling" itself.)
2. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft), Chapter on N-gram Language Models. (Standard modern reference for n-gram LMs, smoothing, and perplexity.)
3. Katz, S. M. (1987). *Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer.* IEEE Transactions on Acoustics, Speech, and Signal Processing. (Backoff.)
4. Chen, S. F., & Goodman, J. (1999). *An Empirical Study of Smoothing Techniques for Language Modeling.* Computer Speech & Language. (The standard comparative survey of add-k, Good-Turing, Katz backoff, and Kneser-Ney smoothing.)
5. Kneser, R., & Ney, H. (1995). *Improved Backing-off for M-gram Language Modeling.* ICASSP. (Kneser-Ney smoothing.)
6. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR, 3, 1137–1155. (Already the anchor reference for [Phase 1](../01-Word_Vectors_and_Basics) — Topic 2.3 is this phase's direct, hands-on engagement with it.)

## Navigate

⬅ [Phase 1 — Word Vectors and Basics](../01-Word_Vectors_and_Basics) · [Repository root](../README.md) · ➡ [Phase 3 — Embeddings and Matrix Factorization](../03-Embeddings_and_Matrix_Factorization)

---


<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:18181B,100:3F3F46&height=100&section=footer" width="100%"/>

</div>
