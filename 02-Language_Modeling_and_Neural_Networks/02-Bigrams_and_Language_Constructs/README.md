# 2.2 — Bigrams and Language Constructs

## 1. The bigram model

A bigram model applies the order-1 Markov assumption from Topic 2.1: `P(w_t | history) ≈ P(w_t | w_{t-1})`. The maximum-likelihood estimate of this conditional is the obvious one — count how often `w_{t-1}` is followed by `w_t`, divide by how often `w_{t-1}` occurs at all:

```
P_MLE(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})
```

This already captures real grammatical structure a unigram model cannot: `explanation.md`'s heatmap shows `P(was | he)` and `P(was | she)` standing out clearly (third-person singular subject → past-tense "to be"), and `P(the | of)`, `P(the | in)`, `P(the | to)` all spiking sharply (preposition → definite article is one of English's most common two-word constructs). A unigram model has no way to represent any of this; it can only ever say "the is a common word," not "the is an especially likely word *right after a preposition*."

## 2. The zero-probability problem

Here is the issue that doesn't show up until you actually try to evaluate this model on held-out text: with a 5,000-word vocabulary, there are `5000²` = 25 million *possible* bigrams, but Topic 2.2's training data — 50,000-plus sentences, around a million tokens — contains well under a million bigram *occurrences*, covering well under 1% of that table (`explanation.md` has the exact figures). The overwhelming majority of grammatically perfectly reasonable two-word combinations were simply never observed.

That would be a tolerable approximation error if it only meant individual probabilities were a little off. It is **not** tolerable, because of how the chain rule combines them: a sequence's probability is a *product* of per-token conditionals (Topic 2.1, section 1), and `P(W) = ... × 0 × ... = 0` the instant the sequence contains even **one** unseen bigram. Perplexity requires `log P(W)`, and `log(0) = -∞`. In practice: any test sentence containing a single never-before-seen word pair makes the entire model's perplexity on the whole test set undefined. This isn't a rare edge case — `explanation.md` shows it happens for roughly one bigram in seven in real held-out text from the very same corpus the model was trained on.

## 3. Laplace (add-one) smoothing

The classical fix: pretend every possible bigram was seen one more time than it actually was.

```
P_Laplace(w_t | w_{t-1}) = ( count(w_{t-1}, w_t) + 1 ) / ( count(w_{t-1}) + V )
```

Adding `1` to every numerator guarantees no probability is ever exactly zero. Adding `V` (the vocabulary size) to every denominator is what keeps each row a valid probability distribution that sums to 1 — `V` is exactly the total amount of "fake" count being injected into that row's `V` possible outcomes (Σ over all `V` possible next-words of the `+1` numerator adjustment), so it must be added to the normaliser too.

The textbook caveat about this method — confirmed with a real number in `explanation.md`, not just asserted — is that it is usually **too aggressive**. With `V = 5000`, every single context gets `5000` units of fake count added to its denominator, frequently dwarfing the real counts for any context that wasn't seen extremely often. The result is a model that has been dragged so far toward "everything is roughly equally likely" that it can come out **worse than the unigram baseline that uses no bigram information at all** — exactly what happens here.

## 4. Linear interpolation

A gentler fix: instead of inventing fake counts, **back off** to the unigram model exactly when the bigram estimate is unreliable, by blending the two:

```
P_interp(w_t | w_{t-1}) = λ · P_MLE(w_t | w_{t-1})  +  (1 - λ) · P_unigram(w_t)
```

`λ ∈ [0, 1]` controls the blend. Since `P_unigram(w_t) > 0` for every vocabulary word (Topic 2.1, section "unigram model" — no smoothing is even needed there), `P_interp` is guaranteed strictly positive as long as `λ < 1`, which solves the zero-probability problem just as completely as Laplace smoothing — but, unlike Laplace smoothing, it does so by leaning on a *real, trained* unigram distribution rather than a uniform "pretend everything is equally likely" assumption, which is why it tends to do considerably better in practice. `λ` is a hyperparameter, and the only legitimate way to choose it is to tune it on a **held-out validation split carved out of the training data** — never on the test set, which exists solely to report a final, honest number once every other decision has already been made. `explanation.md` shows the actual tuning curve and the value it selected, plus an honest discussion of what that particular result does and doesn't tell you.

## References

1. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft), Chapter 3: N-gram Language Models. (Sections on add-one smoothing's known weaknesses and on interpolation/backoff.)
2. Chen, S. F., & Goodman, J. (1999). *An Empirical Study of Smoothing Techniques for Language Modeling.* Computer Speech & Language. (The classic, rigorous comparison of exactly the smoothing methods discussed here, plus more advanced ones like Kneser-Ney.)
3. Lidstone, G. J. (1920). *Note on the General Case of the Bayes-Laplace Formula for Inductive or A Priori Probabilities.* Transactions of the Faculty of Actuaries. (The general add-k smoothing family Laplace/add-1 belongs to.)
4. Jelinek, F., & Mercer, R. L. (1980). *Interpolated Estimation of Markov Source Parameters from Sparse Data.* Workshop on Pattern Recognition in Practice. (The original statement of linear interpolation for language modeling.)
