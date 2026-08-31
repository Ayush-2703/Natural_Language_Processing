# 2.2 — Code Explanation

## Counting bigrams, and a subtlety that caused a real bug

```python
def count_bigrams(encoded_sentences, vocab_size):
    unigram_counts = np.zeros(vocab_size)
    bigram_counts = defaultdict(lambda: np.zeros(vocab_size))
    for seq in encoded_sentences:
        for prev, nxt in zip(seq[:-1], seq[1:]):
            unigram_counts[prev] += 1
            bigram_counts[prev][nxt] += 1
```

`zip(seq[:-1], seq[1:])` produces every consecutive `(prev, next)` pair in a sequence `[<bos>, w_1, ..., w_k, <eos>]`: `(<bos>, w_1), (w_1, w_2), ..., (w_k, <eos>)`. The `unigram_counts` accumulated here count **how many times each word appears in the "prev" position** — correct and necessary as the bigram MLE denominator, since that's literally "how many opportunities did we have to see something follow this word."

This is *not*, however, a genuine unigram distribution, and treating it as one is a real bug this implementation initially had: `<eos>` is, by construction, never the "prev" element of any pair within a sentence (nothing follows the end of a sentence), so it has a count of exactly **zero** in this array. The first version of `BigramModel`'s interpolation used `unigram_counts / unigram_counts.sum()` directly as the unigram backoff distribution — which silently gave `<eos>` a backoff probability of zero. Whenever a test sentence ended with a `(prev, <eos>)` bigram that hadn't been seen in training, *both* terms of the interpolation (`λ · 0` from the missing bigram, `(1-λ) · 0` from the broken backoff) were zero, and perplexity came back `inf` even with `λ` as low as 0.99 — exactly the failure interpolation exists to prevent. The fix, `true_unigram_distribution()`, counts every token in `seq[1:]` for every sequence — the identical recipe Topic 2.1's `UnigramModel` uses, which correctly includes `<eos>` at its real frequency (every sentence has exactly one). This is a good concrete illustration of a easy-to-miss bug class in language modelling code: two different "unigram counts" can look interchangeable and aren't, because *which positions you count* encodes an assumption about what the count means.

## The zero-probability crisis, measured rather than asserted

```python
n_observed_bigrams = sum((row > 0).sum() for row in bigram_counts.values())
```

```
Distinct (prev,next) bigram pairs observed: 219,536 out of 25,030,009 possible -- 0.8771% of the full table
```

Fewer than one in a hundred theoretically possible bigrams were ever seen — Theory's abstract argument about sparsity, as an actual number from this actual corpus.

```python
n_zero, total = raw_model.perplexity(dataset["test_encoded"], return_failure_info=True)
```

```
15,185 / 100,850 test bigrams (15.06%) have ZERO probability under raw MLE
-> test set log-likelihood is -infinity, so perplexity is undefined
```

Just over one test bigram in seven was never observed in training, each one individually enough to make the *entire* test set's likelihood zero under raw MLE. This is `theory.md` section 2 made concrete: the zero-probability problem isn't a rare pathological edge case to be quietly ignored, it's the default outcome for any bigram model evaluated honestly on genuinely held-out text.

## Comparing the fixes

```
Laplace (add-1)     : 417.4
Interpolation       : 255.7
(Topic 2.1's unigram baseline, for reference: 336.4)
```

This ordering is the real result and is worth taking seriously rather than skimming past, because it contradicts the naive assumption that "any smoothing beats no model." **Laplace smoothing is worse than ignoring bigram information entirely.** With `V = 5,000`, every context's denominator gets `5,000` units of invented count added to it (`theory.md` section 3) — for a context word that appeared, say, 50 times in training, that's two orders of magnitude more fake evidence than real evidence, and the resulting distribution is dragged most of the way back toward uniform regardless of what the bigram counts actually showed. **Interpolation, by contrast, beats the unigram baseline outright** (255.7 vs. 336.4): when a specific bigram's count is unreliable or zero, it falls back on a real, trained unigram distribution rather than a uniform guess, so the "what should I do when I'm unsure" answer is actually informed by data.

## A genuine limitation of the lambda-tuning result, worth flagging rather than hiding

```
best lambda found: 0.99
```

`lambda_tuning.png` shows validation perplexity falling *monotonically* as `λ → 1` across the entire grid searched — the validation procedure never found a point where leaning further on the bigram estimate started to hurt. That should prompt a little suspicion rather than just being reported as "the answer." The validation slice here is carved out of the *training* sentences (`theory.md` section 4's requirement to never touch the test set while tuning), which means it is drawn from text extremely similar in style and vocabulary to what the bigram counts were estimated from — a friendlier setting for high `λ` than the genuinely separate test set. The fact that `λ = 0.99` *also* beat the unigram baseline on the actual test set (255.7) is reassuring, but a more careful study would widen the search grid above 0.99, or — better — tune `λ` per-context based on how much count support that specific context has (contexts seen many times can safely lean almost entirely on their own bigram statistics; contexts seen once or twice should lean heavily on the unigram fallback), which is exactly the intuition behind the more advanced techniques (Kneser-Ney, Witten-Bell) that `theory.md`'s references point to and which a real production system would reach for next.
