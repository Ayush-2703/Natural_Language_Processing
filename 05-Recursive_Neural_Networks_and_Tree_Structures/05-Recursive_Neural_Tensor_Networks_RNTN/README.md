# 5.5 — Recursive Neural Tensor Networks (RNTN): Concepts and Implementation

## 1. The problem with the plain TNN

The TNN's composition function `h = tanh(W[h_l;h_r] + b)` is a **bilinear** operation in a specific sense: it is linear in `[h_l;h_r]` but not in the *product* of `h_l` and `h_r` individually. This means the network cannot directly model multiplicative interactions between the two child representations — for instance, the way a negation word's vector should *multiply* the effect of what it modifies rather than just adding to it. Socher et al. (2013) argue that this limitation is precisely what prevents the TNN from handling sentiment-changing constructions like double negation and scope phenomena.

## 2. The RNTN composition function

The **Recursive Neural Tensor Network** adds a tensor interaction term:

```
h_parent = tanh( [h_l ; h_r]ᵀ V[i] [h_l ; h_r]  for i=1..d  +  W[h_l;h_r] + b )
```

The tensor `V ∈ R^{2d × 2d × d}` defines one bilinear form per output dimension: for output dimension `i`, `V[i]` is a `(2d × 2d)` matrix, and `[h_l;h_r]ᵀ V[i] [h_l;h_r]` is a scalar capturing the pairwise interaction between every pair of input features. Stacking `d` such scalars gives the tensor product's `d`-dimensional contribution.

The full composition is therefore:

```
h_parent = tanh( tensor_product_term + standard_linear_term + bias )
```

The standard linear term is identical to the TNN; the tensor product term adds `O(d^3)` additional parameters (versus `O(d^2)` for the linear part alone), which is what gives the RNTN its expressiveness and its computational cost.

## 3. Training on the Stanford Sentiment Treebank

Socher et al. report that the RNTN achieves 85.4% fine-grained (5-class) sentiment accuracy on SST's root nodes and 80.7% at all tree nodes — substantially outperforming the plain TNN (75.9% root / 74.6% all nodes) and matching or exceeding all non-neural baselines of the time. The gains are concentrated precisely on the negation and compositional constructions that the tensor term is designed to handle.

## References

1. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP. (The original RNTN paper; the SST dataset; all accuracy figures above.)
2. Socher, R., Lin, C. C., Ng, A., & Manning, C. (2011). *Parsing Natural Scenes and Natural Language with Recursive Neural Networks.* ICML. (The predecessor TNN this topic compares against.)
