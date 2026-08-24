# 1.2 — Code Explanation

## Reusing Topic 1.1's embeddings

```python
W2V_PATH = os.path.join(HERE, "..", "1.1-Introduction-to-Word-Vectors-and-Analogies",
                         "artifacts", "word2vec_phase1.model")
```

`load_or_train_word2vec` here is byte-for-byte the same function as in Topic 1.1, pointed at the same cache file via a relative path. Run after 1.1, this topic loads in well under a second; run on its own with a fresh clone of the repository, it retrains in about three minutes. Either way the script is fully self-contained — see Topic 1.1's `explanation.md` for why this duplication-with-a-shared-cache pattern was chosen over a shared library module.

## Part A: building two document representations

```python
def load_brown_documents():
    texts, labels = [], []
    for fid in brown.fileids():
        texts.append(" ".join(brown.words(fid)))
        labels.append(brown.categories(fid)[0])
    return texts, labels
```

Brown's 500 files are cleanly partitioned into 15 genres with no overlap, so `brown.categories(fid)[0]` is always the single, unambiguous genre label for that file — confirmed directly rather than assumed (`brown.categories(fid)` returns a one-element list for every file in this corpus).

```python
def tfidf_document_vectors(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    return vectorizer.fit_transform(texts)
```

`stop_words="english"` removes function words *before* TF-IDF weighting even runs — belt-and-suspenders alongside IDF's own automatic down-weighting of ubiquitous terms, since extremely common words can still carry small but non-zero TF-IDF weight and add noise across 500 documents. `min_df=2` drops terms that appear in only a single document (often typos or proper nouns specific to one article, which would otherwise inflate the feature space without generalising). `max_features=5000` caps the vocabulary to the 5,000 highest-frequency surviving terms, both for speed and because the long tail beyond that contributes mostly noise to a 500-document collection.

```python
def mean_pooled_w2v_document_vectors(texts, model):
    ...
    words = [w for w in clean(text.split()) if w in model.wv]
    if words:
        X[i] = np.mean(model.wv[words], axis=0)
```

The simplest possible way to turn a *bag* of word vectors into a single *document* vector is to average them. This throws away word order entirely (Phase 4's RNNs and Phase 5's recursive networks exist precisely because order and structure matter for many tasks) but is a perfectly reasonable baseline for a topic-level signal like genre, where "which words appear" matters far more than "in what order."

## `assess_document_representation`: the t-SNE-for-pictures / silhouette-for-numbers pattern

```python
if use_svd_first:
    X_for_tsne = TruncatedSVD(n_components=50, random_state=42).fit_transform(X)
sil = silhouette_score(X, labels, metric="cosine" if use_svd_first else "euclidean")
tsne = TSNE(n_components=2, perplexity=25, random_state=42, init="pca")
X_2d = tsne.fit_transform(X_for_tsne)
```

Notice that `silhouette_score` is computed on `X` — the *original* TF-IDF or Word2Vec matrix — never on `X_2d`. This is `theory.md` section 2's central warning, applied literally. `TruncatedSVD` to 50 dimensions is applied **only as a pre-processing step for the t-SNE input** (`X_for_tsne`), not for the metric: it works directly on TF-IDF's sparse matrix without densifying it (unlike PCA, which requires mean-centring and would force a 500×5000 dense array into memory for no real benefit here), and reducing 5,000 noisy sparse dimensions to 50 before the nonlinear t-SNE step is standard practice that makes the optimisation faster and less prone to chasing noise. The Word2Vec path skips this since 100 dimensions is already low enough for t-SNE to handle directly. `metric="cosine"` is used for TF-IDF's silhouette score because cosine is the natural metric for sparse, magnitude-sensitive count vectors (two documents of very different lengths but proportionally similar content should be "close"); for the SVD-reduced matrix this still applies since SVD does not change that the original features were count-derived.

## Actual results, and why the two methods disagree

Running this script prints:

```
Document-level genre separation (higher = cleaner clusters):
  TF-IDF               silhouette = -0.007
  Mean-pooled Word2Vec silhouette = -0.084

Word2Vec-by-POS silhouette = -0.027
```

All three numbers are close to zero or slightly negative — by the strict definition in `theory.md` section 3, none of these representations cleanly separates *all* of its 15 (or 10, for POS) categories from each other in a way a generic clustering algorithm would recover unsupervised. That sounds like a disappointing result, but the actual t-SNE pictures (`tsne_tfidf_by_genre.png`, `tsne_word2vec_by_genre.png`, `word2vec_tsne_by_pos.png`, all in `images/`) tell a more specific and more useful story than the single aggregate number does:

- **TF-IDF** cleanly isolates genres with a distinctive *vocabulary and function-word signature* — `news`, `editorial`, `hobbies`, and `reviews` each form a visually tight, separate region — while `fiction`, `adventure`, `mystery`, `romance`, and `belles_lettres` are mixed into one large central blob. That blob is not a failure of the method; those five genres genuinely are all narrative prose with heavily overlapping vocabulary, and TF-IDF is doing something close to the right thing by treating them as similar.
- **Mean-pooled Word2Vec** draws a different line: there is a visible split between a "narrative/fiction" region and an "expository/informational" region that cuts cleanly across several of Brown's 15 specific genre boundaries without aligning with any single one of them. Averaging word vectors smooths over the exact-word-choice signal that TF-IDF exploits and instead surfaces a coarser semantic register — which is exactly why its silhouette score against the *fine-grained* 15-way label is worse, despite the picture showing real, coherent structure.
- **Word2Vec by POS** shows the same pattern at the word level: numbers (`NUM`) and adpositions (`ADP`) form small, very tight, clearly separated clusters (these are small, closed, highly distributionally consistent classes), verbs (`VERB`) occupy a recognisable region, while adjectives, adverbs, and determiners overlap heavily in the centre — many of these words are genuinely ambiguous in the dominant-tag sense used here (a word tagged `ADJ` 55% of the time and `NOUN` 45% of the time gets a vector that reflects *both* uses, and ends up sitting between the two clusters as a result).

The lesson worth taking from this topic is methodological as much as it is about word vectors specifically: a single global metric can hide real, locally-correct structure, and a quick "the silhouette score is bad" conclusion would have thrown away the actually-informative finding that each representation is picking up on a *different, equally real* kind of similarity.
