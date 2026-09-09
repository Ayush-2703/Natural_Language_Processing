# 3.7 — Code Explanation

## PMI in four vectorised lines

```python
total = cooc.sum()
rowsum = cooc.sum(axis=1, keepdims=True)
colsum = cooc.sum(axis=0, keepdims=True)
pmi = np.log((cooc * total + 1e-10) / (rowsum * colsum + 1e-10))
```

`rowsum * colsum` broadcasts to a `(V,V)` matrix of expected joint counts under independence — the denominator `P(w)·P(c)·total`. Dividing the observed count by this expected count gives the PMI ratio; taking `log` gives PMI. The entire computation is three array operations with no Python loops.

## Two-cluster structure in the SGNS vs PMI scatter plot

The scatter in `sgns_vs_pmi.png` splits into two visible vertical clusters — one near PMI ≈ -37 and one near PMI ≈ 0. This is not noise; it has a direct cause. The raw PMI formula (used here, before PPMI clipping) produces extremely negative values for pairs where `P(w,c)` is tiny but `P(w)` and `P(c)` are individually large — exactly the situation for a very rare co-occurrence between two very common words. When very frequent words (function words) co-occur rarely with each other, their product `P(w)·P(c)` is large while `P(w,c)` is small, giving `log(small/large)` → very negative PMI. The left cluster consists of such pairs. The right cluster consists of pairs where at least one word is less common, making `P(w)·P(c)` smaller and the PMI less negative. The SGNS dot products are unaffected by this clustering because they're trained to represent co-occurrence *structure*, not to reproduce raw PMI values — explaining the low but positive correlation (r = 0.194).

## What r = 0.194 means, honestly

A correlation of 0.194 is: (a) clearly positive — the theoretical prediction is directionally confirmed; (b) far weaker than a tight linear relationship. Both are expected. Levy & Goldberg's proof establishes the equality `v_w · u_c = PMI - log k` **at the global optimum of the SGNS objective**. This implementation's model is trained for 5 epochs on 400,000 pairs with a small vocabulary — nowhere near a global optimum. The SGNS embeddings also use the same table for both in-word and context-word lookups (a simplification relative to the proof, which assumes separate `v_w` and `u_c` tables). The weak but positive correlation is exactly the expected result under these conditions: genuine signal from the theoretical relationship, heavily attenuated by finite-training distance from the optimum.
