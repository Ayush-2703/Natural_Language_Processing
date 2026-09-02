# 3.1 — Code Explanation

## Corpus prep: dropping OOV words instead of using `<unk>`

```python
filtered_sentences = []
for s in raw_sentences:
    f = [word2idx[w] for w in s if w in word2idx]
    if len(f) >= 2:
        filtered_sentences.append(f)
```

Phase 2's language models mapped rare words to a shared `<unk>` token, because a language model needs to assign *some* probability to "an unusual word occurred here." Word2Vec has no such requirement — its only goal is to produce good vectors for the words it does train on, and a `<unk>` token would itself get a vector that means nothing (it stands in for thousands of unrelated words), polluting every context window it appears in. The standard, and simpler, choice is to **drop** out-of-vocabulary words from each sentence entirely before generating any training pairs. One real consequence worth being explicit about: this changes what "the context window" means. The filtered sequence `[the, cat, on, mat]` (with "sat" dropped because it fell outside the top 3,000 words) treats "cat" and "on" as adjacent for windowing purposes, even though "sat" originally sat directly between them. This is a standard simplification for an educational implementation, not a hidden inconsistency — production word2vec implementations make similar tradeoffs (gensim, for instance, performs its own frequency-based subsampling with comparable order-distorting effects), and it's flagged here rather than left implicit.

## Generating the two architectures' training data

```python
def generate_skipgram_pairs(sentences, window, max_pairs=None, seed=SEED):
    pairs = []
    for s in sentences:
        for i, center in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    pairs.append((center, s[j]))
```

Every `(centre, context)` combination within the window becomes its own pair — exactly `theory.md` section 3's description of skip-gram generating multiple independent examples per centre-word position. `max(0, i-window)` and `min(len(s), i+window+1)` clip the window at sentence edges rather than padding, so words near the start or end of a sentence simply get fewer training pairs, with no artificial boundary tokens needed.

```python
def generate_cbow_examples(sentences, window, max_examples=None, seed=SEED):
    width = 2 * window
    ...
    ctx.append(s[j] if 0 <= j < len(s) else -1)
```

CBOW, by contrast, needs a single *fixed-width* context per example (so the averaging step has a known shape to work with), so missing positions at sentence boundaries are explicitly marked with `-1` rather than silently shortening the window — `CBOWModel.forward` then needs to handle that marker correctly (see below), rather than the data generator quietly producing variable-length examples that would break batching.

## The two models

```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, center_idx):
        e = self.in_embed(center_idx)
        return self.out_proj(e)
```

This is `theory.md`'s cost diagram as code, with no hidden layer at all — `in_embed` and `out_proj` are exactly the two weight matrices the field calls the "input embeddings" and "output embeddings" respectively (a distinction Topic 3.2 will return to). `nearest_neighbors`, used throughout evaluation, deliberately reads from `in_embed` only — the input embeddings are the ones conventionally reported as "the" word vectors, even though `out_proj`'s weight matrix is technically a second, separately-learned embedding table for the same vocabulary.

```python
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        ...
    def forward(self, context_idx):
        safe_idx = context_idx.clone()
        safe_idx[safe_idx == -1] = self.vocab_size
        e = self.in_embed(safe_idx)
        mask = (context_idx != -1).unsqueeze(-1).float()
        summed = (e * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        averaged = summed / counts
```

The embedding table is built one row larger than the vocabulary (`vocab_size + 1`), with `padding_idx=vocab_size` telling PyTorch that row is a fixed, never-updated all-zeros vector reserved for padding. `safe_idx[safe_idx == -1] = self.vocab_size` remaps every `-1` placeholder from the data generator to that dedicated row. The `mask` then ensures padding positions contribute **zero** to both the sum and the count used for averaging — `counts.clamp(min=1.0)` exists purely as a defensive guard against division by zero on a (here, impossible, since every example has at least one real context word) all-padding example. This is, in effect, a hand-built version of exactly the masked-averaging behaviour Topic 1.4's `nn.EmbeddingBag` provided automatically — written out explicitly here because CBOW's *fixed-width-with-padding* batching shape doesn't fit `EmbeddingBag`'s *flat-concatenated* convention as naturally as Topic 1.4's variable-length documents did.

## Actual results, reported honestly

```
[skip-gram] epoch 1/6  loss=7.0830  ->  epoch 6/6  loss=5.7950
[cbow]      epoch 1/6  loss=6.6236  ->  epoch 6/6  loss=5.3550
Wall-clock training time: CBOW = 81.2s   Skip-gram = 76.6s
```

`loss_curves.png` shows exactly the pattern `theory.md` section 3 predicts: CBOW converges to a noticeably lower final loss than skip-gram, consistent with CBOW solving the easier of the two prediction tasks, not with CBOW being a "better" model in any general sense.

The nearest-neighbour results are genuinely weak — `king`'s neighbours under both models (`houses`, `property`, `genre`, `lovers`) show little obvious semantic coherence, and the frequent-vs-rare-word comparison (`rare_vs_frequent.png`) shows essentially **no measurable difference** between CBOW and skip-gram (0.483 vs. 0.482 for frequent words; 0.485 vs. 0.482 for rare words) — nowhere near the clear skip-gram advantage on rare words that Mikolov et al. (2013b) report. This deserves an honest explanation rather than a quiet move-on: 320,000 training pairs is roughly 6% of the ~5 million skip-gram pairs a single full pass over this filtered corpus with `window=2` would actually generate, and a full *softmax* training step (this topic, deliberately — see `theory.md` section 4) is far more expensive per pair than the techniques real Word2Vec implementations use, which is exactly why this topic's training budget had to be capped to run in a reasonable time at all. Topic 1.1's gensim-trained Word2Vec model — trained on the *full* corpus, for *eight full epochs*, using skip-gram with negative sampling rather than full softmax — produced visibly cleaner neighbours and meaningful analogy hits, on the same underlying text. The gap between that result and this one is not a contradiction; it is the most concrete possible illustration of why Topic 3.2's hierarchical softmax and negative sampling exist: they are what makes training on enough data, for enough epochs, computationally affordable in the first place. This topic's job was to make the *architecture* concrete and inspectable; Topic 3.2 picks up exactly where its computational ceiling leaves off.
