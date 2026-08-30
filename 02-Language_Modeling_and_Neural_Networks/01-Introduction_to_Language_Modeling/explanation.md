# 2.1 — Code Explanation

## The shared dataset (read this even if you only care about Topics 2.2/2.3)

```python
counts = Counter(w for s in train_sents for w in s)
most_common = [w for w, _ in counts.most_common(VOCAB_SIZE)]
vocab = [BOS, EOS, UNK] + most_common
```

The vocabulary is built **strictly from `train_sents`**, after the train/test split, not from the whole corpus. This matters: if the vocabulary were built from all the data and then split, a word that appears only in the test set could "leak" into the known vocabulary, making the model's job artificially easier than it would be on genuinely unseen text. `VOCAB_SIZE = 5000` is the cutoff visible as the red dashed line in `zipf_law.png` — chosen at a point past the steepest part of the frequency curve, where additional words start contributing fewer than ~20 occurrences each across the entire training set.

```python
idx_seq = [word2idx[BOS]] + [word2idx.get(w, word2idx[UNK]) for w in s] + [word2idx[EOS]]
```

Every encoded sentence is wrapped with `<bos>` (beginning of sentence) and `<eos>` (end of sentence). `<bos>` exists purely as conditioning context — it lets a bigram or neural model condition its very first real prediction on *something*, rather than needing a special-cased "no context" rule. `<eos>` is treated as a real, predictable token: a language model that never has to predict "the sequence ends here" couldn't tell you anything about sentence length, which is itself useful information. `word2idx.get(w, word2idx[UNK])` is where Zipf's law becomes an actual code path: any training-rare or genuinely-unseen test word silently becomes `<unk>` rather than crashing or being skipped.

This dataset (vocabulary, encoded train sequences, encoded test sequences, raw word counts) is pickled to `artifacts/lm_dataset.pkl`. Topics 2.2 and 2.3 load this exact file — same vocabulary, same split, same `<unk>` policy — so that a perplexity number from one topic is directly comparable to a perplexity number from another. This is the same caching pattern Phase 1 used for its Word2Vec model, for the same reason: expensive or comparison-critical preprocessing happens once, and every topic that needs it either loads the cache or rebuilds it from scratch if run in isolation.

## The unigram model

```python
counts = np.zeros(len(word2idx))
for seq in dataset["train_encoded"]:
    for idx in seq[1:]:  # skip the leading <bos>
        counts[idx] += 1
self.probs = counts / counts.sum()
```

`seq[1:]` skips each sentence's leading `<bos>` token when accumulating counts — `<bos>` is never something the model should learn to *predict* (it never appears anywhere except as the fixed first token of every sequence, so "predicting" it would be trivial and uninformative), only something it conditions on. Every other token, including `<eos>`, is counted normally. Dividing by the total gives a proper probability distribution over the vocabulary that sums to 1, with no smoothing needed: by construction, every vocabulary word (including `<unk>`) appeared at least once in training, so no entry of `self.probs` is ever exactly zero. This is the structural reason a *unigram* model never runs into the zero-probability crisis that motivates Topic 2.2's entire discussion of smoothing — there is no notion of an "unseen unigram" once a word has made it into the vocabulary at all.

```python
def perplexity(self, encoded_sentences):
    total_log_prob, total_tokens = 0.0, 0
    for seq in encoded_sentences:
        for idx in seq[1:]:
            total_log_prob += self.log_prob(idx)
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

This is `theory.md` section 3's formula typed in directly: sum `log P(w_t)` over every scored token across the *entire* test set (not per-sentence — perplexity is normalised by total token count, so it's a single number over the whole evaluation set), negate, divide by the token count, exponentiate.

## Actual results

```
Vocabulary size (incl. special tokens): 5003
Train sentences: 50153   Test sentences: 5572
<unk> rate in training data: 11.39%

Unigram model perplexity:  train = 350.1   test = 336.4
(a uniform-random model over 5003 words would have perplexity 5003)
```

350 is a real number worth holding onto for Topics 2.2 and 2.3 — it is the number to beat. It is dramatically better than the 5,003 a model with literally no information would get (the unigram model has clearly learned *something*: that `the` is far more likely than `aardvark`), but it is also obviously weak: 350 "equally likely options" at every position in a sentence is a model that has thrown away the single most informative thing about predicting the next word — what the previous word *was*. That gap is exactly what conditioning on one word of context (Topic 2.2's bigram model) exists to close.

The 11.39% `<unk>` rate quantifies Zipf's long tail concretely: more than one word in nine, even in the *training* data the vocabulary was built from, falls outside the 5,000 most common words. (Test set `<unk>` exposure is necessarily a little different again, since test sentences can contain words never seen in training at all — not just rare ones.)
