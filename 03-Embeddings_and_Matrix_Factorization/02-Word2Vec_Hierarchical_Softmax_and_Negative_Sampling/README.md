# 3.2 — Word2Vec Training Mechanisms: Hierarchical Softmax and Negative Sampling

Topic 3.1 closed by identifying a real, measured bottleneck: a full softmax costs `O(d·V)` per training example, paid regardless of which single word is actually being predicted. This topic implements the two classical answers Mikolov et al. (2013b) propose, and benchmarks both against full softmax under matched conditions rather than asserting the speedup.

## 1. Hierarchical softmax: turning one V-way decision into log(V) binary decisions

Instead of scoring every vocabulary word directly, hierarchical softmax arranges the entire vocabulary as the **leaves of a binary tree** and replaces "which of V words is this?" with a sequence of binary "left or right?" decisions walking from the root down to that word's leaf. The probability of reaching a particular word is the product of the probabilities of taking the correct turn at every internal node along its unique root-to-leaf path:

```
P(w | center) = Π_{node ∈ path(w)}  σ( sign(node, w) · v_node · v_center )
```

where `v_node` is a learned vector belonging to that **internal node** (not the word itself), `v_center` is the input word's embedding, `σ` is the logistic sigmoid, and `sign(node, w)` is `+1` or `-1` depending on whether the path to `w` goes right or left at that node. Since a binary tree over `V` leaves has depth `O(log V)` when reasonably balanced, this replaces one `O(V)`-cost operation with `O(log V)` cheap binary classifications.

### Why a Huffman tree specifically, not just any binary tree

The tree's *shape* matters: a balanced tree gives every word the same path length, `⌈log₂ V⌉`. A **Huffman tree** — built by repeatedly merging the two lowest-frequency remaining nodes into a parent, exactly as in classical lossless compression — instead gives **frequent words shorter paths and rare words longer ones**, by construction. Since training spends time on a word in direct proportion to how often it occurs in the corpus, putting the most frequent words at the cheapest (shallowest) tree positions minimizes the *expected* total cost across an entire training pass, even though the *worst-case* path for a rare word is longer than `⌈log₂ V⌉` would give it under a balanced tree. `explanation.md` shows this property holding exactly on this topic's real vocabulary: the most frequent word gets a 4-step path while the rarest gets a 15-step one, against a balanced-tree expectation of about 12 steps for every word.

## 2. Negative sampling: don't normalize over the vocabulary at all

Negative sampling takes a different approach entirely: stop trying to compute a properly normalized probability distribution over the vocabulary, and instead turn the prediction task into a small set of independent binary classification problems. For one true `(center, context)` pair, the model is trained to assign **high probability to the real context word** and **low probability to `k` randomly sampled "negative" (noise) words**:

```
L = -log σ(v_context · v_center)  -  Σ_{i=1}^{k} log σ(-v_{neg_i} · v_center)
```

Both terms are plain logistic regression losses — `σ(v_context · v_center)` should be pushed toward 1 (a real co-occurrence), `σ(v_{neg_i} · v_center)` should be pushed toward 0 (a word that, statistically, almost certainly *didn't* belong in this context). The cost per example is `O(d·(k+1))` — independent of `V` entirely, with `k` typically a small constant (5–20). This topic uses `k=5`.

### The noise distribution: why words^0.75, not raw frequency

Negative words are not sampled uniformly, nor according to their raw unigram frequency — they're sampled from `P(w) ∝ count(w)^0.75`. Raising frequencies to the 0.75 power **flattens** the distribution: very frequent words (`the`, `of`) get sampled as negatives somewhat less often relative to their true frequency, while rare words get sampled somewhat *more* often than their true frequency would suggest. This is a deliberate, empirically-tuned choice from the original paper — sampling negatives purely by raw frequency would spend almost the entire negative-sampling budget contrasting against a tiny handful of extremely common words, wasting most of the available negative signal on largely redundant comparisons.

## 3. A real, well-documented failure mode: why frequent words need their own fix

Hierarchical softmax and negative sampling solve the *cost* of training, but neither one, by itself, solves a separate problem: in any natural corpus, a handful of extremely frequent function words (`the`, `of`, `to`, `and`) participate in an enormous fraction of all training pairs, simply by being everywhere. Left unchecked, the model spends most of its gradient budget learning to discriminate the contexts of `the` from the contexts of `of` — a comparison that carries very little semantic information — at the direct expense of learning the comparatively rare, but far more meaningful, co-occurrence patterns of genuine content words. `explanation.md` shows this happening directly and concretely on this topic's own trained model.

Mikolov et al. (2013b) — the same paper that introduces negative sampling — propose **subsampling**: before generating any training pairs, randomly discard each occurrence of word `w` with probability depending on how far above some threshold `t` its relative corpus frequency `f(w)` sits:

```
P_keep(w) = ( sqrt(f(w)/t) + 1 ) · ( t / f(w) )
```

For a word at or below the threshold (`f(w) ≤ t`), `P_keep(w) ≥ 1`, so (after clipping to 1) it is always kept. For a word far above the threshold, `P_keep(w)` shrinks roughly as `1/sqrt(f(w))` — heavily discarding extremely frequent words while barely touching moderately common ones. This is not a vocabulary cutoff (Topic 2.1's `<unk>` strategy, or Topic 3.1's hard 3,000-word cap) — every word stays in the vocabulary; subsampling only thins out how many of a frequent word's individual *occurrences* get to participate in training pairs. `explanation.md` reports this topic's own attempt to apply this fix, including an honest account of how much it did and did not help at the scale this repository trains at.

## References

1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. (Introduces both negative sampling and frequent-word subsampling, sections 2.2 and 2.3.)
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop. (Hierarchical softmax, building on Morin & Bengio below.)
3. Morin, F., & Bengio, Y. (2005). *Hierarchical Probabilistic Neural Network Language Model.* AISTATS. (The original proposal of a tree-structured softmax for language modeling, predating Word2Vec.)
4. Huffman, D. A. (1952). *A Method for the Construction of Minimum-Redundancy Codes.* Proceedings of the IRE. (The original Huffman coding algorithm this topic's tree-construction code implements directly.)
5. Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738. (A detailed gradient derivation for both hierarchical softmax and negative sampling.)
