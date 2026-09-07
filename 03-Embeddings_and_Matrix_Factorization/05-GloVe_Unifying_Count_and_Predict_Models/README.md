# 3.5 — Global Vectors (GloVe): Unifying Word2Vec with GloVe

## 1. The central claim

By the middle of 2014, two research groups had independently produced strong word embedding methods that looked very different on the surface: Mikolov et al.'s Word2Vec (a neural prediction model trained on raw corpus text, Topics 3.1–3.3) and Pennington, Socher & Manning's GloVe (Topic 3.6, a direct factorisation of a co-occurrence *ratio* matrix). In August 2014, Levy & Goldberg published a result that changed how the field understood both: **skip-gram with negative sampling (SGNS) implicitly factorises a matrix of shifted PMI values.** Prediction and factorisation, far from being fundamentally different approaches, are — under this analysis — two computational routes to the *same* mathematical objective.

## 2. The derivation

Start from the negative sampling objective (Topic 3.2, theory.md section 2). SGNS trains a vector `v_w` per word (as centre word) and a vector `u_c` per word (as context word) to maximise, summed over all observed `(w, c)` pairs and `k` sampled negatives:

```
Σ_{(w,c) ∈ D} [ log σ(v_w · u_c)  +  k · E_{c'~ P_n} log σ(-v_w · u_{c'}) ]
```

where `P_n` is the noise distribution. **At the global optimum** — when every individual `(w, c)` pair's gradient is zero — Levy & Goldberg show the optimum satisfies:

```
v_w · u_c = PMI(w, c)  -  log k
```

In other words: at convergence, the dot product of any word's input and output embedding equals the pointwise mutual information of that pair, minus a constant shift of `log k` (the log of the number of negatives per positive). This means the matrix of all pairwise dot products `[v_w · u_c]` is exactly `PPMI(w,c) - log k` at the global optimum — a shifted PMI matrix. Training SGNS is, *in the limit of a global optimum*, equivalent to matrix-factorising `PMI - log k` into the product of two matrices `W Uᵀ`, just as Topic 3.4's SVD factorises a PPMI matrix.

## 3. GloVe: what changes when you factorize *ratios* rather than dot products

GloVe (Pennington, Socher & Manning, 2014) argues from a different starting point — not from a neural training objective, but from the observation that the *ratio* of co-occurrence probabilities captures semantic relationships more cleanly than the co-occurrence probabilities themselves. For words `w`, `c₁`, `c₂`:

```
P(c₁ | w) / P(c₂ | w)
```

is large (much greater than 1) if `w` is closely related to `c₁` and not to `c₂`, small if the opposite is true, and near 1 if `w` is related equally to both or neither. GloVe's model is:

```
w_i · w̃_j + b_i + b̃_j = log X_{ij}
```

where `X_{ij}` is the co-occurrence count of word `i` and context `j`, `w_i` and `w̃_j` are the word and context vectors, and `b_i`, `b̃_j` are per-word bias terms. This is linear regression of the dot product against the *log* co-occurrence count. The objective is:

```
Σ_{ij} f(X_{ij}) ( w_i · w̃_j + b_i + b̃_j - log X_{ij} )²
```

where `f(X)` is a weighting function that down-weights very frequent co-occurrences (similar in spirit to PPMI's division by marginal probabilities, but implemented as a loss weight rather than a matrix transformation). This is precisely a **weighted matrix factorisation** of the log-co-occurrence matrix — the same mathematical family as Topic 3.4's ALS, with a more carefully-designed weighting function.

## 4. The unification

What does the Levy & Goldberg proof tell us? SGNS factorises `PMI(w,c) - log k`. GloVe factorises `log X(w,c)` (log co-occurrence count). Since `log X(w,c) = PMI(w,c) + log(#(w)) + log(#(c))` (by the definition of PMI), the two targets differ only by per-word additive terms — which GloVe's bias terms `b_i + b̃_j` absorb. At a conceptual level, they are factorising essentially the same information under slightly different normalisations and with different weighting schemes. The dramatic-sounding debate between "count-based" and "predictive" methods in the 2013–2014 NLP literature was, in this light, partly a debate between two implementations of the same underlying idea — and the practical performance differences Pennington et al. documented between GloVe and Word2Vec in their original paper are attributable to differences in *weighted co-occurrence matrix design* and *training efficiency* more than to any fundamental architectural difference.

## References

1. Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS. (The proof in section 2 follows this paper's Appendix A directly.)
2. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP. (Section 3 of that paper derives the co-occurrence ratio motivation and the GloVe objective.)
3. Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop. (The SGNS objective whose implicit matrix factorisation is analysed by Levy & Goldberg.)
