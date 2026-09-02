# 3.1 — Introduction to Word Embedding (CBOW and Skip-Gram)

## 1. From a neural language model to Word2Vec

Topic 2.3 built a neural bigram language model: predict the next word from one word of context, scored with a full softmax over the vocabulary. **Word2Vec** (Mikolov et al., 2013a) is best understood as a deliberate simplification of that same family of model, optimised purely for producing good *word vectors* rather than a good *language model*. Three changes mark the shift: the hidden `tanh` layer is dropped entirely (the projection from embedding to output is now a single linear map — Word2Vec is, architecturally, "shallower" than Bengio et al.'s 2003 model), the context is no longer restricted to "the previous word" but can look both backward and forward within a window, and the training objective is no longer "predict the actual next word in sequence" but one of two related, deliberately symmetric tasks described below. The result trains dramatically faster per step than a deep network, which matters enormously once you want to train on the scale of corpora Word2Vec was actually built for (the original paper used a 6-billion-word corpus).

## 2. CBOW: Continuous Bag-of-Words

CBOW predicts the **centre word from its surrounding context**. Given a window of size `m`, the context for position `t` is `{w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}}`, and the model:

1. Looks up each context word's embedding.
2. **Averages** them into a single vector `v_context = (1/2m) Σ vec(w_{t+j})` for `j ≠ 0`.
3. Projects `v_context` to a score over the vocabulary and predicts `w_t`.

![CBOW vs. Skip-gram](images/cbow_vs_skipgram_architecture.png)

The averaging step in (2) is the architecture's defining choice and its defining limitation: it deliberately discards word order within the window (the context "the ___ sat" and "sat ___ the" — nonsensical, but illustrating the point — would produce the *same* averaged vector if those were literally the only two context words), and it smooths several words' worth of signal into one vector before ever trying to predict anything, which tends to make CBOW's training task easier and faster to optimise — visible directly in `explanation.md`'s loss curves.

## 3. Skip-gram: the mirror image

Skip-gram predicts in the **opposite direction**: take the single centre word, and predict each surrounding context word independently.

```
maximize  Σ_{j≠0, -m≤j≤m}  log P(w_{t+j} | w_t)
```

Each `(centre, context)` pair within the window becomes its own independent training example — a sentence with a window of `m=2` produces up to 4 training pairs per centre word, rather than CBOW's single example. This has two consequences worth holding onto: skip-gram sees **more individual training signal per pass over the corpus** than CBOW does (more examples generated from the same raw text), but each individual example is a **harder prediction task** — guessing one specific context word from only the centre word, with no averaging to smooth out ambiguity, is intrinsically noisier than CBOW's "guess the one centre word from several context words" task. Mikolov et al. report, and this topic's own training run reproduces directly, that skip-gram's per-example loss converges to a *higher* value than CBOW's — not because it's training worse, but because it's solving a harder version of essentially the same underlying problem.

## 4. Both architectures share the same expensive bottleneck

Whichever direction the prediction runs, the final step is identical: a linear projection from the embedding dimension `d` down to the *full* vocabulary size `V`, followed by softmax.

![The cost of a full softmax](images/full_softmax_cost.png)

The embedding lookup itself costs `O(d)` — trivial. The output projection costs `O(d · V)` — and critically, this cost is paid **on every single training example**, regardless of which one word (CBOW) or which one context word (skip-gram) is actually being predicted. With a real-scale vocabulary of, say, 100,000+ words, computing and normalising 100,000 scores just to find out the gradient for one correct answer is enormous, repeated waste. This topic's vocabulary is deliberately capped at 3,000 words specifically to keep this full-softmax cost tractable to run directly in this repository — and that very cap is the motivating problem Topic 3.2 exists to solve, via two genuinely different algorithmic answers (hierarchical softmax and negative sampling) that both avoid ever scoring the entire vocabulary on a typical training step.

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop. (Introduces both CBOW and skip-gram.)
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. (Refines skip-gram with negative sampling — Topic 3.2's subject — and reports the rare-word behaviour discussed in `explanation.md`.)
3. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR. (The deeper architecture Word2Vec simplifies — see Topic 2.3.)
4. Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738. (A widely-used, very thorough derivation of both architectures' forward and backward passes.)
