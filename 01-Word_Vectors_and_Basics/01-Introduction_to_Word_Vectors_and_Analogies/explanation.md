# 1.1 — Code Explanation

A section-by-section walkthrough of `implementation.py`. Read `theory.md` first — this file assumes you know what cosine similarity and 3CosAdd analogy solving are, and focuses on *why the code is written the way it is*.

## Imports and paths

```python
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
IMAGE_DIR = os.path.join(HERE, "images")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "word2vec_phase1.model")
```

`os.path.dirname(os.path.abspath(__file__))` resolves paths relative to *this file's location* rather than whatever directory the script happens to be launched from — important because Topics 1.2–1.4 will `os.path.join` their way back to `MODEL_PATH` using a relative path, and that only works reliably if every script anchors its own paths the same way.

## 1. Corpus assembly

```python
def clean(tokens):
    return [w.lower() for w in tokens if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]
```

Word2Vec only cares about token co-occurrence, so the cleaning step is deliberately aggressive: lowercase everything (so `King` and `king` are the same vocabulary item — capitalisation is not a signal we want this model to spend capacity on), keep only purely alphabetic tokens (this drops standalone punctuation tokens like `"`, `--`, `)` that NLTK's corpus readers leave in, plus numbers), and drop single-character tokens (mostly cleaned-up artifacts, not meaningful words).

```python
def build_corpus():
    sentences = []
    for fid in brown.fileids():
        sentences.append(clean(brown.words(fid)))
    for fid in gutenberg.fileids():
        for sent in gutenberg.sents(fid):
            c = clean(sent)
            if len(c) > 3:
                sentences.append(c)
    for fid in movie_reviews.fileids():
        sentences.append(clean(movie_reviews.words(fid)))
    return sentences
```

Gensim's `Word2Vec` expects an iterable of token lists, where **each inner list is a unit the context window should not cross** — typically a sentence. Brown and Movie Reviews are chunked **per file** (one inner list per document) because `.words(fid)` is the natural unit there and these documents are short. Gutenberg's `.fileids()` are entire books, so chunking per-file would let the context window slide across completely unrelated chapters; `gutenberg.sents(fid)` gives NLTK's own sentence segmentation instead, and `if len(c) > 3` throws out degenerate one- or two-word "sentences" (often segmentation artifacts around chapter headings) that would otherwise inject noise with no real context to learn from.

Running this assembles 89,999 pseudo-sentences totalling 4,241,740 tokens in about 13 seconds — small by production standards (Word2Vec's original paper used a 6-billion-token corpus), but enough to produce real, inspectable structure, which is this topic's actual goal.

## 2. Training, with a cache

```python
def load_or_train_word2vec():
    if os.path.exists(MODEL_PATH):
        print(f"Loading cached Word2Vec model from {MODEL_PATH}")
        return Word2Vec.load(MODEL_PATH)
    ...
    model.save(MODEL_PATH)
    return model
```

Training takes roughly three minutes on a single CPU core. Topics 1.2, 1.3, and 1.4 all need the *same* trained vectors, and re-running a three-minute training job four times just to explore four different downstream questions would be wasteful and would make four scripts each feel slow for no good reason. So this function checks for a cached `.model` file first, and only pays the training cost once. Each of the other topics' scripts contains the identical `load_or_train_word2vec` function pointing at this same file via a relative path — so every script is still fully self-contained and runnable on its own (it will just retrain if the cache happens to be missing), but in the normal case of working through the topics in order, only the first run is slow.

```python
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, workers=1, epochs=8)
```

`vector_size=100` is a modest but standard dimensionality for a small corpus (production systems often use 200–300; more dimensions need more data to fill meaningfully). `window=5` means "5 words of context on each side" — a typical default. `min_count=5` drops any word occurring fewer than 5 times in the whole corpus, which both shrinks the vocabulary to 29,568 words and removes the noisiest, least-supported vectors (a word seen twice gets a nearly random vector — keeping it would only pollute nearest-neighbour searches). `sg=1` selects skip-gram over CBOW; Phase 3 builds both from scratch and explains the difference precisely — here it's simply "the setting that tends to do slightly better on rarer words and analogies," taken as given.

## 3. `WordVectorSpace` — the PyTorch layer

```python
raw = torch.tensor(keyed_vectors.vectors, dtype=torch.float32)
self.vectors = F.normalize(raw, dim=1)
```

`keyed_vectors.vectors` is gensim's raw embedding matrix as a NumPy array, shape `(vocab_size, 100)`. Wrapping it in a `torch.tensor` and calling `F.normalize(..., dim=1)` L2-normalises **every row independently** (`dim=1` is the embedding-dimension axis) — this is the "normalise once" optimisation from `theory.md` section 3: after this line, a plain dot product between any two rows of `self.vectors` *is* their cosine similarity, with no further division needed anywhere downstream.

```python
def nearest(self, query_vec, exclude=(), topn=8):
    q = F.normalize(query_vec.unsqueeze(0), dim=1)
    sims = (self.vectors @ q.T).squeeze(1)
    for w in exclude:
        if w in self.stoi:
            sims[self.stoi[w]] = -1.0
    top_sims, top_idx = torch.topk(sims, min(topn, sims.shape[0]))
    return [(self.itos[i], top_sims[j].item()) for j, i in enumerate(top_idx.tolist())]
```

This is the computational core of the whole topic, and it's three real lines. `query_vec.unsqueeze(0)` turns a `(100,)` vector into a `(1, 100)` row so it can be matrix-multiplied; `self.vectors @ q.T` is a `(29568, 100) @ (100, 1)` matrix–vector product producing a `(29568, 1)` column of similarity scores against *every word in the vocabulary at once* — this is the entire "nearest neighbour search," done as a single BLAS call rather than a Python loop. `.squeeze(1)` drops the now-redundant size-1 dimension. The `exclude` loop forcibly sets specific words' similarity to `-1.0` (lower than any real cosine similarity) so they can never win the top-k — essential for analogy solving, because without it, `vec(woman) - vec(man) + vec(king)` is trivially closest to `king` itself (it's literally `king` plus a small offset), which would make every analogy "solve" to one of its own inputs and tell you nothing. `torch.topk` returns the `k` largest values and their indices in one call; the final list comprehension converts tensor indices back to strings via `self.itos`.

```python
def analogy(self, a, b, c, topn=5):
    query = self.vec(b) - self.vec(a) + self.vec(c)
    return self.nearest(query, exclude={a, b, c}, topn=topn)
```

This is 3CosAdd from `theory.md`, written exactly as the math reads: `vec(b) - vec(a) + vec(c)`, then nearest-neighbour search excluding the three words that defined the query.

## 4. Evaluation

```python
hit1 = pred_words[0] == expected
hit5 = expected in pred_words
```

Two metrics are tracked because they answer different questions. Top-1 accuracy ("is the single best answer correct?") is the strict, textbook-demo version. Top-5 ("does the correct answer appear anywhere in a short shortlist?") is more forgiving and arguably more honest about what a small-corpus model can be expected to do — `man:woman :: king:?` lands `queen` at rank 3 here, which is a top-5 hit and a top-1 miss, and both numbers are worth reporting rather than picking whichever one looks better.

Running this script end-to-end produces:

```
Analogy accuracy on this 7-item curated set: top-1 = 4/7, top-5 = 6/7

  [hit]   boy:boys :: girl:girls    -> ['girls', 'women', 'females', 'ladies', 'teens']
  [hit]   good:better :: bad:worse  -> ['worse', 'payback', 'funnier', 'mediocre', 'smarter']
  [hit]   day:days :: night:nights  -> ['nights', 'months', 'mornings', 'weeks', 'afternoons']
  [top5]  man:woman :: king:queen   -> ['vashti', 'esther', 'queen', 'sennacherib', 'judah']
  [hit]   paris:france :: london:england -> ['england', 'britain', 'germany', 'sussex', 'islanders']
  [top5]  slow:slower :: fast:faster -> ['renting', 'efficiently', 'nibble', 'faster', 'vacations']
  [miss]  write:writer :: act:actor -> ['directorial', 'ronin', 'eszterhas', 'mykelti', 'mcconaughey']
```

Three of the four inflectional/comparative analogies (plurals, comparatives) hit on the first try — these patterns are extremely frequent and grammatically regular, so even ~4M tokens is enough signal. `man:woman::king:queen` is a top-5 hit, landing behind two Biblical queens for exactly the reason `theory.md` discusses. `write:writer::act:actor` is a genuine miss — derivational morphology (verb → agentive noun) is a much sparser pattern than pluralisation, and the model simply hasn't seen enough examples of it to have learned a clean linear direction. This is real behaviour, not a cherry-picked failure.

## 5. Visualisations

`plot_similarity_heatmap` computes the full pairwise cosine-similarity matrix for ten chosen words and renders it with `imshow` plus per-cell text annotations — useful for spotting structure (`man`–`woman` at 0.70, `good`–`bad` at 0.65) that single analogy tests don't show as directly: near-antonyms cluster *together*, not apart, exactly as `theory.md` warns. `plot_analogy_bars` takes one analogy's top-6 candidates and renders them as a horizontal bar chart, colouring the expected answer green wherever it appears — this is the figure that makes the "queen is rank 3, behind two Biblical names" result immediately legible rather than just a line of printed text.

Both images are saved into `images/` as `similarity_heatmap.png` and `analogy_king_queen.png`, alongside the conceptual `parallelogram_concept.png` referenced from `theory.md`.
