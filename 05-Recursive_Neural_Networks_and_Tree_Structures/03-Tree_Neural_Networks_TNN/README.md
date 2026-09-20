# 5.3 — Tree Neural Network (TNN) utilizing Recursion

## The TNN composition function

The Tree Neural Network (Socher et al., 2011) is the direct application of a single-layer neural network as the composition function in a recursive architecture:

```
h_parent = tanh( W [h_left ; h_right] + b )
```

where `W ∈ R^{d × 2d}` and `b ∈ R^d`. This is the same computation as a one-hidden-layer network applied to the concatenation of two child vectors, with weights **shared across all nodes in the tree** — a single `W` learns to compose any two phrases regardless of their syntactic labels or positions.

## Training signal: supervised at every node

On the Stanford Sentiment Treebank (SST), every constituent — not just the root — has a sentiment label. This means a single sentence provides supervision at dozens of tree nodes simultaneously. The loss is the sum of cross-entropy losses over all labelled nodes:

```
L = Σ_{nodes v} CE(softmax(clf(h_v)), label_v)
```

This is crucial: without intermediate supervision, gradients must propagate from the root down through every composition step, becoming vanishingly small deep in the tree. Supervising every node gives the model direct gradient signal at every level of composition.

## References

1. Socher, R., Lin, C. C., Ng, A., & Manning, C. (2011). *Parsing Natural Scenes and Natural Language with Recursive Neural Networks.* ICML.
2. Socher, R., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP.
