# 5.1 — Introduction to Recursive Neural Networks

## 1. Why trees, not sequences

Topics 4.2 and 4.3 model sentences as **sequences** — left-to-right (or bidirectional) streams of tokens. But natural language has **hierarchical syntactic structure**: "The cat that I saw yesterday" is not just a string of six words; it is a noun phrase ("The cat") modified by a relative clause ("that I saw yesterday"), which is itself a verb phrase. This hierarchical structure is the reason that the meaning of a sentence cannot always be read off from the sum or sequence of its words' meanings — the phrase "not good" has a different meaning than either "not" or "good" alone, and that interaction is a property of their syntactic relationship (negation of an adjective), not just of their proximity as tokens.

**Recursive Neural Networks** (Socher et al., 2011) explicitly model this structure: instead of processing tokens left-to-right, they process the sentence's **parse tree** bottom-up, applying a shared composition function at every internal node. The key properties:
- The composition function `f` is **shared across all nodes** — the same weight matrix handles a `DT→NN` composition at the bottom of the tree and an `NP→VP` composition near the root.
- The network has **no fixed depth** — it can process trees of any shape, which is why the architecture is called "recursive" rather than "recurrent."
- The representation at the **root node** captures the meaning of the entire sentence, in a way that reflects the sentence's syntactic decomposition.

## 2. The composition function

For a binary-branching node with children represented by vectors `h_l` and `h_r`, the simplest composition is:

```
h_parent = tanh( W · [h_l ; h_r] + b )
```

where `[h_l ; h_r]` is the concatenation of the two child vectors. Topic 5.3 implements this as the **Tree Neural Network (TNN)**, and Topic 5.5 generalises it to the **Recursive Neural Tensor Network (RNTN)**, which adds a tensor product term to capture more expressive interactions between children.

## 3. Penn Treebank tree statistics

Penn Treebank's 3,914 parse trees have a mean depth of ~9.7 and sentences averaging ~25.6 leaves (words). The deepest trees reach depth 20, reflecting the genuine structural complexity of Wall Street Journal text. A recursive network that handles depth 20 uses the exact same weight matrix `W` as one that handles depth 2 — parameter efficiency by design.

## References

1. Socher, R., Lin, C. C., Ng, A., & Manning, C. (2011). *Parsing Natural Scenes and Natural Language with Recursive Neural Networks.* ICML.
2. Socher, R., Perelygin, A., Wu, J., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP.
