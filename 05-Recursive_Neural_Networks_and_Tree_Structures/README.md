<div align="center">

![Phase 5: Recursive Neural Networks and Tree Structures](https://capsule-render.vercel.app/api?type=waving&color=0:F0FDF4,100:BBF7D0&height=220&section=header&text=Phase%205%20%C2%B7%20Recursive%20Neural%20Networks%20and%20Tree%20Structures&fontSize=28&fontColor=14532D&fontAlignY=38&animation=fadeIn&desc=Composing%20meaning%20bottom-up%20over%20a%20parse%20tree%2C%20not%20left-to-right&descSize=16&descAlignY=62)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/status-planned%20%2F%20not%20yet%20implemented-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

*Module V of [`Natural_Language_Processing`](../README.md) — a topic-wise, theory-to-implementation NLP curriculum*

</div>

---

> **🚧 Status: scaffolding only.** The five topic folders and this module README exist, but the theory write-ups, `implementation.py` files, `explanation.md` walkthroughs, and diagrams have not been added yet. Everything below describes the **planned** scope and structure for this phase — it deliberately does not claim results, datasets-in-hand, or images that don't exist yet. See [Build Progress](#build-progress) for exactly what's outstanding.

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

Every recurrent model built so far in this repository — Phase 2's neural bigram model, Phase 4's RNN taggers — processes a sentence the same way: strictly left to right, one token after another. Phase 5 challenges that assumption directly. A sentence isn't just a sequence, it has *syntactic structure* — `"the movie was not good"` is negated at the phrase level (`"not good"`), not just token by token — and a **recursive** neural network is built to exploit exactly that structure, composing a vector for each node of a sentence's parse tree from its children, bottom-up, until a single vector represents the whole sentence at the root.

This is also the phase that closes the loop on the classical-vs-neural pattern this repository has run since Phase 2: where Phases 2–4 replaced sequential, hand-engineered models (n-grams, HMMs) with sequential neural ones (RNNs), Phase 5 asks whether the sequential assumption itself was ever necessary, and builds the tree-structured alternative pioneered by Socher et al.

## The Arc of This Phase

**5.1** motivates the shift: why linear, left-to-right composition (an RNN's hidden state) can miss structure that's obvious from a parse tree, and introduces the recursive neural network as a composition function applied bottom-up over tree nodes rather than a chain-structured cell applied left to right.

**5.2** covers how a sentence becomes a tree a network can actually consume — in practice, this means working with the Stanford Sentiment Treebank's already-parsed, binarized constituency trees (each labeled with sentiment at every node, not just the root) rather than running a parser from scratch, and describes the data format Topics 5.3–5.5 will build on.

**5.3** implements the Tree Neural Network itself: a single shared composition function, applied recursively — combine two children's vectors into their parent's vector, repeat up the tree — that turns a variable-shaped parse tree into one fixed-size vector at the root.

**5.4** addresses a genuinely open question once you have tree-structured representations: how to convert or relate them back to sequences, for comparison with — or integration into — the sequence models built in Phase 4, in the spirit of linearizing parse-tree structure into a form a sequence model can consume or predict.

**5.5** upgrades the plain Tree Neural Network to a **Recursive Neural Tensor Network** — adding a bilinear tensor term so a parent's representation can capture multiplicative interactions between its two children, not just their (additive) sum — and applies it to sentiment classification on the Sentiment Treebank, the setting Socher et al. (2013) introduced it for.

## Topics at a Glance

| # | Topic | Folder | Planned to cover |
|---|-------|--------|--------------------|
| 5.1 | Introduction to Recursive Neural Networks | [`01-Introduction_to_Recursive_Neural_Networks`](01-Introduction_to_Recursive_Neural_Networks) | Why tree-structured composition differs from an RNN's chain-structured composition, and where it helps |
| 5.2 | Parsing Sentences as Trees and Data Description for RNNs | [`02-Parsing_Sentences_as_Trees`](02-Parsing_Sentences_as_Trees) | Reading the Sentiment Treebank's pre-parsed constituency trees and their per-node sentiment labels |
| 5.3 | Tree Neural Network (TNN) utilizing Recursion | [`03-Tree_Neural_Networks_TNN`](03-Tree_Neural_Networks_TNN) | A shared recursive composition function applied bottom-up over parse trees |
| 5.4 | Converting Trees to Sequences | [`04-Converting_Trees_to_Sequences`](04-Converting_Trees_to_Sequences) | Linearizing tree structure into sequence form, connecting back to Phase 4's sequence models |
| 5.5 | Recursive Neural Tensor Networks (RNTN) | [`05-Recursive_Neural_Tensor_Networks_RNTN`](05-Recursive_Neural_Tensor_Networks_RNTN) | A bilinear-tensor composition function, trained for sentiment classification on the Sentiment Treebank |

This mirrors the [lab-practicals mapping](../README.md#lab-practicals-mapping) already listed in the root README: Lab 15 (Recursive Tree Neural Network over parse trees) → Topic 5.3, Lab 16 (RNTN for sentiment on the Stanford Sentiment Treebank) → Topic 5.5.

## Planned Folder Structure

Once populated, each topic will follow the same four-part pattern as Phases 1 and 3 — a theory `README.md`, a runnable `implementation.py`, a line-by-line `explanation.md`, and an `Image/` folder of diagrams and generated plots:

```
05-Recursive_Neural_Networks_and_Tree_Structures/
├── README.md                                    (this file)
│
├── 01-Introduction_to_Recursive_Neural_Networks/
│   └── README.md            — currently empty; theory to be added
│
├── 02-Parsing_Sentences_as_Trees/
│   └── README.md            — currently empty; theory, implementation.py,
│                               explanation.md, and Image/ to be added
│
├── 03-Tree_Neural_Networks_TNN/
│   └── README.md            — currently empty; theory, implementation.py,
│                               explanation.md, and Image/ to be added
│
├── 04-Converting_Trees_to_Sequences/
│   └── README.md            — currently empty; theory, implementation.py,
│                               explanation.md, and Image/ to be added
│
└── 05-Recursive_Neural_Tensor_Networks_RNTN/
    └── README.md            — currently empty; theory, implementation.py,
                                 explanation.md, and Image/ to be added
```

## Datasets Planned for This Phase

Per the [root README's dataset table](../README.md#datasets-used), this phase is scoped to use the **Stanford Sentiment Treebank (SST)** via Hugging Face `datasets` — the original dataset Socher et al. (2013) introduced the RNTN for, which conveniently ships pre-parsed into binary constituency trees with a sentiment label at every node, not just the sentence root. That's exactly the tree-structured format Topics 5.2–5.5 are planned to consume, so no separate parsing step is expected to be needed.

## Build Progress

- [ ] 5.1 — Introduction to Recursive Neural Networks: theory
- [ ] 5.2 — Parsing Sentences as Trees and Data Description for RNNs: theory, implementation, explanation, diagrams
- [ ] 5.3 — Tree Neural Networks (TNN): theory, implementation, explanation, diagrams
- [ ] 5.4 — Converting Trees to Sequences: theory, implementation, explanation, diagrams
- [ ] 5.5 — Recursive Neural Tensor Networks (RNTN): theory, implementation, explanation, diagrams

## Getting Started

Repo-wide setup (shared with every phase):

```bash
git clone https://github.com/Ayush-2703/Natural_Language_Processing.git
cd Natural_Language_Processing

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The Stanford Sentiment Treebank is expected to be pulled via Hugging Face `datasets` on first run once Topic 5.2's implementation exists (`pip install datasets`, already covered if it's listed in `requirements.txt` once this phase is built out). There's no `implementation.py` to run in this phase yet — check the [Build Progress](#build-progress) checklist above for what's still outstanding.

## Key References

These anchor the topics planned above; each topic's own `README.md` will cite the specific subset relevant to it once written.

1. Goller, C., & Küchler, A. (1996). *Learning Task-Dependent Distributed Representations by Backpropagation Through Structure.* ICNN. (The original recursive-neural-network / backprop-through-structure formulation underlying Topic 5.1.)
2. Socher, R., Lin, C. C., Ng, A., & Manning, C. (2011). *Parsing Natural Scenes and Natural Language with Recursive Neural Networks.* ICML. (Already the root README's anchor reference for this phase — the Tree Neural Network architecture Topic 5.3 plans to implement.)
3. Vinyals, O., Kaiser, Ł., Koo, T., Petrov, S., Sutskever, I., & Hinton, G. (2015). *Grammar as a Foreign Language.* NeurIPS. (Linearizing parse trees into sequences for a seq2seq model — directly relevant to Topic 5.4's planned scope.)
4. Tai, K. S., Socher, R., & Manning, C. D. (2015). *Improved Semantic Representations from Tree-Structured Long Short-Term Memory Networks.* ACL. (Tree-LSTM — a natural extension once Topics 5.3–5.4 establish tree-structured composition and tree-to-sequence conversion.)
5. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP. (Already the root README's anchor reference for this phase — the RNTN architecture and the Sentiment Treebank dataset, both planned for Topic 5.5.)

## Navigate

⬅ [Phase 4 — Sequence Tagging and Recurrent Networks](../04-Sequence_Tagging_and_Recurrent_Networks) · [Repository root](../README.md)

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](../LICENSE) for details.

---

## 👤 Author

<div align="center">

### Ayush Kumar Singh

*Researcher in Adversarial ML, Geospatial AI, and LLM/NLP Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Ayush%20Kumar%20Singh-181717?style=for-the-badge&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush%20Kumar%20Singh-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ayushsingh2703)
[![Email](https://img.shields.io/badge/Email-Ayush%20Kumar%20Singh-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ab49ayush@gmail.com)

</div>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:F0FDF4,100:BBF7D0&height=100&section=footer" width="100%"/>

</div>
