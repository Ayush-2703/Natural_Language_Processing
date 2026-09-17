<div align="center">

![Phase 4: Sequence Tagging and Recurrent Networks](https://capsule-render.vercel.app/api?type=waving&color=0:18181B,100:3F3F46&height=220&section=header&text=Phase%204%20%C2%B7%20Sequence%20Tagging%20and%20Recurrent%20Networks&fontSize=32&fontColor=FFFFFF&fontAlignY=38&animation=fadeIn&desc=From%20one%20label%20per%20sentence%20to%20one%20label%20per%20word&descSize=16&descAlignY=62)

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

*[`Natural_Language_Processing`](../README.md) — a topic-wise, theory-to-implementation*

</div>

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

Every phase so far has assigned **one** label to a whole unit of text: Phase 1 classified an entire review as positive or negative, Phase 2 scored an entire sentence's probability. Phase 4 changes the unit of prediction itself — the task becomes assigning a label to **every token** in a sequence, with each token's correct label depending on its neighbours. Part-of-speech tagging is the cleanest version of this problem (`"book"` is a verb after "to" and a noun after "the"), and named entity recognition is a harder variant of the same idea, where the labels themselves span multiple tokens rather than describing one word in isolation.

The planned arc mirrors a pattern the repository has now run twice — Phase 2 went from counting bigrams to a small neural language model, Phase 3 went from predictive Word2Vec to matrix factorization and back — and runs it a third time at the sequence-labeling level: start with a classical, well-understood generative model with a clean decoding algorithm (Hidden Markov Models and Viterbi), then replace it with a recurrent neural network that drops the model's independence assumptions in exchange for learning representations directly from data.

## The Arc of This Phase

**4.1** lays the conceptual groundwork: what a tagset is (Penn Treebank's POS tags, as the planned dataset below), why POS tagging is genuinely ambiguous rather than a lookup problem (the same word takes different tags in different contexts), and how tagging accuracy is measured.

**4.2** implements the classical answer: a **Hidden Markov Model**, where tags are hidden states, words are observed emissions, and the **Viterbi algorithm** finds the single most probable tag sequence for a sentence via dynamic programming rather than an intractable search over every possible tagging.

**4.3** replaces the HMM's Markov assumption (a tag depends only on the previous tag) with a **recurrent neural network** that can, in principle, condition on the entire preceding sequence through its hidden state — the same shift in kind from Phase 2's neural bigram model, now applied to per-token labeling instead of next-word prediction.

**4.4** extends the RNN tagger from POS tags to **named entities** (people, organizations, locations), introducing the added wrinkle that entities can span multiple tokens — which is what tagging schemes like BIO (Begin/Inside/Outside) exist to handle — and evaluating with metrics suited to that spanning structure rather than plain per-token accuracy.

## Topics at a Glance

| # | Topic | Folder | Planned to cover |
|---|-------|--------|--------------------|
| 4.1 | Introduction to Parts-of-Speech (POS) Tagging | [`01-Introduction_to_POS_Tagging`](01-Introduction_to_POS_Tagging) | Tagsets, why tagging is ambiguous rather than lookup-based, and accuracy as an evaluation metric |
| 4.2 | Hidden Markov Models (HMM) for POS Tagging | [`02-Hidden_Markov_Models_for_POS_Tagging`](02-Hidden_Markov_Models_for_POS_Tagging) | Transition/emission probabilities estimated from a tagged corpus, and Viterbi decoding for exact MAP inference |
| 4.3 | Neural Networks and RNNs applied to POS Tagging | [`03-RNNs_for_POS_Tagging`](03-RNNs_for_POS_Tagging) | An RNN sequence tagger replacing the HMM's Markov assumption with a learned hidden state |
| 4.4 | NER Basics and NER utilizing RNNs | [`04-Named_Entity_Recognition_with_RNNs`](04-Named_Entity_Recognition_with_RNNs) | BIO tagging, an RNN-based NER model, and entity-level (not just token-level) evaluation |

This mirrors the [lab-practicals mapping](../README.md#lab-practicals-mapping) already listed in the root README: Lab 12 (HMM POS tagger with Viterbi decoding) → Topic 4.2, Lab 13 (RNN-based POS tagger) → Topic 4.3, Lab 14 (RNN-based NER tagger) → Topic 4.4.

## Planned Folder Structure

Once populated, each topic will follow the same four-part pattern as Phases 1 and 3 — a theory `README.md`, a runnable `implementation.py`, a line-by-line `explanation.md`, and an `Image/` folder of diagrams and generated plots:

```
04-Sequence_Tagging_and_Recurrent_Networks/
├── README.md                                    (this file)
│
├── 01-Introduction_to_POS_Tagging/
│   └── README.md            — currently empty; theory to be added
│
├── 02-Hidden_Markov_Models_for_POS_Tagging/
│   └── README.md            — currently empty; theory, implementation.py,
│                               explanation.md, and Image/ to be added
│
├── 03-RNNs_for_POS_Tagging/
│   └── README.md            — currently empty; theory, implementation.py,
│                               explanation.md, and Image/ to be added
│
└── 04-Named_Entity_Recognition_with_RNNs/
    └── README.md            — currently empty; theory, implementation.py,
                                 explanation.md, and Image/ to be added
```

## Datasets Planned for This Phase

Per the [root README's dataset table](../README.md#datasets-used), this phase is scoped to use a **Penn Treebank sample** for POS tagging (via `nltk.corpus.treebank`) and **CoNLL-2003** for NER (via Hugging Face `datasets`). The Treebank download is already covered by the repo-wide `nltk.download` step in [Getting Started](#getting-started) below; CoNLL-2003 will be pulled on first run once Topic 4.4's implementation exists.

## Build Progress

- [ ] 4.1 — Introduction to POS Tagging: theory
- [ ] 4.2 — Hidden Markov Models for POS Tagging: theory, implementation, explanation, diagrams
- [ ] 4.3 — RNNs for POS Tagging: theory, implementation, explanation, diagrams
- [ ] 4.4 — Named Entity Recognition with RNNs: theory, implementation, explanation, diagrams

## Getting Started

Repo-wide setup (shared with every phase):

```bash
git clone https://github.com/Ayush-2703/Natural_Language_Processing.git
cd Natural_Language_Processing

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt', 'averaged_perceptron_tagger', 'treebank']]"

cd 04-Sequence_Tagging_and_Recurrent_Networks
```

There's no `implementation.py` to run in this phase yet — check the [Build Progress](#build-progress) checklist above for what's still outstanding.

## Key References

These anchor the topics planned above; each topic's own `README.md` will cite the specific subset relevant to it once written.

1. Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* Proceedings of the IEEE. (Already the root README's anchor reference for HMMs — the Viterbi algorithm derivation Topic 4.2 will implement is here.)
2. Viterbi, A. J. (1967). *Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm.* IEEE Transactions on Information Theory. (The original dynamic-programming decoding algorithm.)
3. Brants, T. (2000). *TnT: A Statistical Part-of-Speech Tagger.* ANLP. (A widely-cited, practically-tuned HMM POS tagger.)
4. Marcus, M. P., Santorini, B., & Marcinkiewicz, M. A. (1993). *Building a Large Annotated Corpus of English: The Penn Treebank.* Computational Linguistics. (The Treebank and its POS tagset, planned as this phase's dataset.)
5. Elman, J. L. (1990). *Finding Structure in Time.* Cognitive Science. (The foundational simple-recurrent-network architecture underlying Topic 4.3.)
6. Graves, A., & Schmidhuber, J. (2005). *Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures.* Neural Networks. (Bidirectional recurrent tagging, relevant to both 4.3 and 4.4.)
7. Tjong Kim Sang, E. F., & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.* CoNLL. (The CoNLL-2003 dataset and task definition planned for Topic 4.4.)
8. Huang, Z., Xu, W., & Yu, K. (2015). *Bidirectional LSTM-CRF Models for Sequence Tagging.* arXiv:1508.01991. (A standard modern architecture for the RNN-based NER Topic 4.4 plans to build toward.)
9. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). *Neural Architectures for Named Entity Recognition.* NAACL. (The paper that popularized neural, feature-free NER — relevant context for Topic 4.4.)

## Navigate

⬅ [Phase 3 — Embeddings and Matrix Factorization](../03-Embeddings_and_Matrix_Factorization) · [Repository root](../README.md) · ➡ [Phase 5 — Recursive Neural Networks and Tree Structures](../05-Recursive_Neural_Networks_and_Tree_Structures)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:18181B,100:3F3F46&height=100&section=footer" width="100%"/>

</div>
