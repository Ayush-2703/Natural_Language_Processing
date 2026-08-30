# 2.1 — Introduction to Language Modeling and Neural Networks

## 1. What a language model is

A **language model** assigns a probability to a sequence of words `P(w_1, w_2, ..., w_T)`. That single object is useful for an enormous range of things — ranking candidate transcriptions in speech recognition, scoring machine translation output, autocomplete, and, as this whole course eventually builds toward, providing the training signal that produces useful word representations as a side effect (Topic 1.1, section 2).

The probability chain rule decomposes this exactly, with no approximation yet:

```
P(w_1, ..., w_T) = P(w_1) · P(w_2|w_1) · P(w_3|w_1,w_2) · ... · P(w_T|w_1,...,w_{T-1})
```

![Chain rule decomposition](02-Language_Modeling_and_Neural_Networks/01-Introduction_to_Language_Modeling/Image/chain_rule_diagram.png)

This is exact but useless as written — the context `w_1, ..., w_{t-1}` grows without bound, so there is no way to ever collect enough data to estimate `P(w_t | w_1, ..., w_{t-1})` for every possible history. Every practical language model is some strategy for approximating this conditional with a *manageable* amount of context or structure. This Phase covers two such strategies side by side: the classical **n-gram Markov assumption** (Topic 2.2) and a **neural network** that compresses arbitrary history into a fixed-size learned vector (Topic 2.3, and Bengio et al., 2003 before it).

## 2. The n-gram approximation

The **Markov assumption of order `n-1`** truncates the conditioning context to the last `n-1` words:

```
P(w_t | w_1, ..., w_{t-1})  ≈  P(w_t | w_{t-n+1}, ..., w_{t-1})
```

`n=1` (**unigram**) throws away *all* context — `P(w_t | history) ≈ P(w_t)`. This topic implements exactly this, as the floor every more sophisticated model in this Phase has to beat. `n=2` (**bigram**, Topic 2.2) conditions on just the immediately preceding word. Each step up in `n` captures more context but needs exponentially more data to estimate reliably — the well-known data sparsity problem that motivates everything in Topic 2.2's smoothing discussion and, ultimately, the entire neural alternative.

## 3. Perplexity: how a language model is scored

**Perplexity** is the standard evaluation metric for a language model, defined as the exponentiated average negative log-likelihood per token:

```
PP(W) = exp( -(1/T) Σ_{t=1}^{T} log P(w_t | context) )
```

Two ways to read this number. First, it is exactly `exp(cross-entropy loss)` — if you train a model with cross-entropy loss (as Topic 2.3 does), its perplexity is one `exp()` call away from the number you were already minimising, with no separate computation needed. Second, perplexity has a concrete interpretation as an **effective branching factor**: a model that places uniform probability `1/V` on every one of `V` vocabulary words, regardless of context, has perplexity exactly `V` (every prediction is equally and maximally uncertain). A model with perplexity 50 is, loosely, "as confused as if it had to guess uniformly among 50 equally likely words" at each step — lower is better, and this topic's unigram model's actual perplexity gives a concrete number to anchor that intuition against (see `explanation.md`).

## 4. Zipf's law and the vocabulary problem

Word frequencies in natural language follow an extremely skewed distribution: the `k`-th most frequent word occurs with frequency roughly proportional to `1/k` (**Zipf's law**). The practical consequence for any model that estimates a probability per word: a small number of words account for the overwhelming majority of tokens, while a very long tail of words occurs only once or twice in any corpus you'll ever train on.

![Zipf's law in the Brown corpus](02-Language_Modeling_and_Neural_Networks/01-Introduction_to_Language_Modeling/Image/zipf_law.png)

This motivates a standard preprocessing choice used throughout this Phase: cap the vocabulary at the `V` most frequent training words and map everything else to a single `<unk>` (unknown) token. Without this, a model would need to estimate parameters for tens of thousands of words it has seen only once or twice — parameters with no reliable signal behind them. With it, the model concentrates its capacity on words it actually has enough data to say something useful about, at the cost of being unable to distinguish between any two rare words it hasn't seen.

## 5. Why reach for a neural network at all

Counting-based n-gram models (Topic 2.2) have a structural weakness Zipf's law makes worse, not better, as `n` grows: most possible `n`-word combinations simply never appear in any finite training corpus, no matter how large, so their *count* is zero — even when the words involved are individually common (`theory.md` for Topic 2.2 makes this concrete). A neural language model, following **Bengio et al. (2003)** — the same paper anchoring Topic 1.1 — sidesteps this by representing each word as a dense vector and computing the next-word distribution as a *continuous function* of those vectors, rather than a *lookup table* indexed by exact word identities. Two words with similar vectors automatically get similar predictions in similar contexts, even for a specific `(context, next-word)` combination that was never literally observed during training — generalisation by interpolation in vector space, rather than the all-or-nothing matching of exact n-gram counting. Topic 2.3 builds exactly this, restricted to bigram context, and measures whether that generalisation advantage actually shows up as lower perplexity.

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR.
2. Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal. (The origin of entropy as a measure of predictive uncertainty, of which perplexity is a direct descendant.)
3. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft), Chapter 3: N-gram Language Models. (The standard textbook treatment of everything in sections 1–2 above.)
4. Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort.* Addison-Wesley.
