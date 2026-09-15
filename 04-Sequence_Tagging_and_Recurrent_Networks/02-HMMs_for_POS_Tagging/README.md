# 4.2 — Hidden Markov Models (HMM) for POS Tagging

## 1. The HMM generative story

A first-order HMM for POS tagging makes two conditional independence assumptions: the tag at position `i` depends only on the tag at `i-1` (the Markov assumption), and the word at position `i` depends only on its own tag. This factorises the joint probability of a sentence and its tag sequence as:

```
P(w_1..w_n, t_1..t_n) = P(t_1) · Π_{i=2}^{n} P(t_i|t_{i-1}) · Π_{i=1}^{n} P(w_i|t_i)
```

Three sets of parameters are needed: **start probabilities** `P(t_1)` (how likely is each tag at the start of a sentence?), **transition probabilities** `P(t_i|t_{i-1})` (how likely is each tag given the previous tag?), and **emission probabilities** `P(w_i|t_i)` (how likely is a word given its tag?). All three are estimated from the training data by maximum likelihood, then smoothed with add-k smoothing to avoid zero probabilities.

## 2. Viterbi decoding in log-space

Given a sentence, the tagger wants the **most probable tag sequence** — the MAP sequence — not the marginal probability of any individual tag in isolation. Viterbi decoding computes this exactly via dynamic programming:

```
δ(t, i) = max_{t_1..t_{i-1}} log P(w_1..w_i, t_1..t_{i-1}, t_i)
         = log P(w_i|t_i) + max_{s} [ δ(s, i-1) + log P(t_i|s) ]
```

The recursion fills a `(T × n)` table where `T` is the number of tags and `n` is sentence length; a parallel backpointer table records which previous tag achieved the best score at each cell. After filling the table, the best final tag is chosen by `argmax δ(:, n)` and the full sequence is recovered by tracing back through the pointers.

All operations are in **log-space** (`log P(w_i|t_i) + max ...` rather than `P(w_i|t_i) × max ...`) to prevent numerical underflow: multiplying several hundred small probabilities together produces numbers smaller than float64 can represent, but summing their logarithms stays in a safe range throughout.

## 3. Suffix-based OOV handling

The emission table only covers words seen in training. For unknown words, this implementation tries successive suffixes from longest (length 4) to shortest (length 1): if `-ing` was seen as a suffix in training, the model uses its aggregated emission distribution (dominated by VBG), effectively implementing a lightweight morphological classifier without any explicit linguistic rules. This captures the major English suffix-to-tag patterns (`-ed` → VBD/VBN, `-ly` → RB, `-er` → NN/JJR) from training data statistics alone.

## References

1. Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* Proceedings of the IEEE. (The standard reference for HMM theory and Viterbi decoding.)
2. Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*, Chapter 9: Markov Models. MIT Press.
