# 3.5 — Code Explanation

Topic 3.5 has no `implementation.py` beyond a stub that prints three lines, because this is a purely theoretical topic. The "implementation" of what this topic establishes is split across Topics 3.6 and 3.7: 3.6 trains GloVe (the factorisation of log co-occurrence), and 3.7 builds the PMI matrix directly and checks whether SGNS embeddings correlate with it the way Levy & Goldberg's proof predicts.

## How to read the Levy & Goldberg derivation

The key step in `theory.md` section 2 is the "at the global optimum" claim. The SGNS objective, taken over the full corpus, can be written as a sum over unique `(w, c)` pairs:

```
L = Σ_{(w,c)} [ #(w,c) · log σ(v_w · u_c)  +  k · #(w) · P_n(c) · log σ(-v_w · u_c) ]
```

where `#(w,c)` is the count of the pair, `#(w)` is the count of word `w`, and `P_n(c)` is the noise probability of context `c`. Taking the derivative with respect to `v_w · u_c` and setting it to zero gives a scalar equation in one unknown `x = v_w · u_c`:

```
#(w,c) · σ(x)  +  k · #(w) · P_n(c) · (σ(x) - 1)  =  0
σ(x) = k · #(w) · P_n(c) / (#(w,c) + k · #(w) · P_n(c))
```

When `P_n(c) ∝ #(c)^{0.75}` (Topic 3.2's noise distribution), the solution simplifies to:

```
x* = log( #(w,c) / (#(w) · #(c)^{0.75}) ) - log k
   = PMI(w, c) - log k  + small correction from the 0.75 exponent
```

The 0.75 exponent on the noise distribution means the correspondence is to a *slightly shifted* PMI (proportional to `count(c)^{0.25}` as well), but for equal-weight noise (`0.75 → 1.0`) it is exactly `PMI(w,c) - log k`. This is not an approximation and not an asymptotic result — it holds at the exact global optimum of the SGNS objective, for every `(w,c)` pair simultaneously.

## Why this matters for Topics 3.6 and 3.7

If the proof is correct:
- Topic 3.6's GloVe model (which factorises log co-occurrence) and the SGNS training Topics 3.1–3.3 ran are both attempting to express the same underlying PMI-like structure as a low-rank product of word vectors. The main difference is *which* log-co-occurrence target and *which* weighting function each uses.
- Topic 3.7's PMI matrix, computed directly from the co-occurrence counts, should correlate with the dot products of the SGNS embeddings trained in Topic 3.3 — which Topic 3.7 checks empirically.

The practical takeaway: the "count-based vs. predictive" debate that occupied much of the NLP community in 2013–2014 was not wrong, but it was incomplete. The methods do differ in their *hyperparameters* (weighting, window size, dimensionality, noise distribution) and their *computational characteristics* (matrix factorisation is a one-shot linear algebra operation; SGNS trains iteratively on the corpus). But they do not differ in their *fundamental mathematical objective* — and knowing that tells you exactly where to look when one outperforms the other: not at the architecture, but at those hyperparameters.
