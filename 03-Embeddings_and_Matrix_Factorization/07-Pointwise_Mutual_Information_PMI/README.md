# 3.7 — Pointwise Mutual Information (PMI) Implementations

## 1. PMI and its variants

**Pointwise Mutual Information** measures how much more often word `w` and context `c` co-occur than statistical independence would predict:

```
PMI(w, c) = log( P(w,c) / (P(w)·P(c)) )
```

where probabilities are estimated from corpus counts. When `P(w,c) > P(w)·P(c)`, the pair co-occurs more than by chance and PMI > 0. When `P(w,c) < P(w)·P(c)`, PMI < 0. PMI can theoretically range from -∞ (never co-occur) to +∞ (only ever co-occur together). In practice, with finite corpora, the range on this corpus's observed pairs is [-5.2, +8.7].

**PPMI** (Positive PMI): `max(0, PMI)`. Retains only evidence of positive association; discards noisy negative evidence. Standard for NLP; used in Topic 3.4's SVD factorisation.

**SPPMI** (Shifted PPMI): `max(0, PMI - log k)`. Subtracts `log k` from every PMI value before clipping. Topic 3.5's proof shows SGNS with `k` negatives implicitly factorises this matrix — not PPMI, but a version shifted downward by `log k`. With `k=5`, `log 5 ≈ 1.61`, and SPPMI retains only pairs whose PMI exceeds that threshold — a much sparser matrix than PPMI, as `ppmi_vs_sppmi.png` makes concrete.

**NPMI** (Normalised PMI): `PMI / (-log P(w,c))`. Bounded in [-1, +1], making values comparable across pairs with very different joint frequencies. PPMI correlates with absolute co-occurrence frequency in a way NPMI does not, but it has been found empirically to perform less well than PPMI for embedding purposes (Bullinaria & Levy, 2007).

## 2. The empirical check of Levy & Goldberg

If the proof in Topic 3.5 is correct, SGNS dot products at the global optimum equal `PMI(w,c) - log k`. In practice, models are never at an exact global optimum (finite data, finite epochs, V-dimensional optimisation). But positive correlation between SGNS dot products and PMI values on observed pairs is a real, checkable prediction. This topic computes that correlation directly (r = 0.194 on 500 randomly sampled observed pairs) and visualises the scatter in `sgns_vs_pmi.png`. The correlation is modest but clearly positive, and the scatter plot's two-cluster structure has a specific explanation — see `explanation.md`.

## References

1. Church, K. W., & Hanks, P. (1990). *Word Association Norms, Mutual Information, and Lexicography.* Computational Linguistics.
2. Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS. (Section 2 establishes SPPMI as the implicit target of SGNS.)
3. Bullinaria, J. A., & Levy, J. P. (2007). *Extracting Semantic Representations from Word Co-occurrence Statistics.* Behavior Research Methods.
