# 1.2 — Assessing Word Vectors using TF-IDF and t-SNE Dimensionality Reduction

This topic introduces two classical tools and, just as importantly, the right and wrong ways to use one of them together with vectors: **TF-IDF** as a non-neural baseline vector representation, and **t-SNE** as a way to *look at* high-dimensional vector spaces — whether those vectors come from TF-IDF or from a trained embedding model like the Word2Vec from Topic 1.1.

## 1. TF-IDF: weighting words by how informative they are

A raw word count vector for a document treats every word as equally important, which is wrong in an obvious way: in any English corpus, "the" will outnumber every content word, yet it tells you nothing about what the document is *about*. **TF-IDF** (term frequency–inverse document frequency) fixes this with two factors multiplied together.

**Term frequency** measures how often term `t` occurs in document `d`, usually normalised by document length:

```
tf(t, d) = count(t in d) / |d|
```

**Inverse document frequency** measures how *rare* a term is across the whole collection of `N` documents, by inverting and log-compressing its document frequency `df(t)` (the number of documents containing `t` at all):

```
idf(t) = log( N / df(t) )
```

A term in every document gets `idf = log(1) = 0` — it is weighted to nothing, regardless of how often it appears. A term confined to a single document out of many gets a large `idf`. Multiplying:

```
tfidf(t, d) = tf(t, d) × idf(t)
```

produces a vector per document where common words are suppressed and rare, document-specific words stand out. The worked example below shows this concretely on three one-sentence toy documents — note how `"the"`, present in all three, is zeroed out, while `"dog"`, present in only one, would dominate that document's vector if it appeared more than once.

![TF-IDF worked example](images/tfidf_worked_example.png)

This is, historically, the field's first systematic answer to "how do I turn a document into a vector?" (Sparck Jones, 1972; Salton & Buckley, 1988) — entirely count-based, with no learning involved. It is also the conceptual ancestor of Module III's count-based methods: a TF-IDF matrix and a PMI matrix (Topic 3.7) are both, fundamentally, reweighted co-occurrence statistics; GloVe (Topic 3.5–3.6) can be seen as factorising something in that same family.

## 2. t-SNE: the mathematics of "looking at" a high-dimensional space

A 100-dimensional Word2Vec space or a 5,000-dimensional TF-IDF space cannot be looked at directly. **t-SNE** (t-distributed Stochastic Neighbour Embedding; van der Maaten & Hinton, 2008) produces a 2D or 3D picture by trying to preserve *local neighbourhood structure*: points that were close in high dimensions should stay close in the picture, even if the picture necessarily distorts everything else.

**Step 1 — turn high-dimensional distances into probabilities.** For every pair of points `i, j`, define a conditional probability that `i` would "pick" `j` as its neighbour, using a Gaussian centred on `i`:

```
p(j|i) = exp(-‖x_i - x_j‖² / 2σ_i²) / Σ_{k≠i} exp(-‖x_i - x_k‖² / 2σ_i²)
```

`σ_i` is not fixed — it's solved for, per point, via binary search so that the resulting distribution `p(·|i)` has a target **perplexity** (loosely: "effective number of neighbours considered," typically 5–50; this topic uses 25–30). Dense regions get a small `σ_i`, sparse regions get a larger one, so every point ends up with a comparable effective neighbourhood size regardless of local density. The conditional probabilities are then symmetrised: `p_ij = (p(j|i) + p(i|j)) / 2N`.

**Step 2 — define the same kind of probability in the low-dimensional map**, but with a heavier-tailed **Student-t distribution with one degree of freedom** (equivalent to a Cauchy distribution) instead of a Gaussian:

```
q_ij = (1 + ‖y_i - y_j‖²)^(-1) / Σ_{k≠l} (1 + ‖y_k - y_l‖²)^(-1)
```

The heavy tail is the deliberate fix for the **crowding problem**: in high dimensions there is exponentially more "room" at moderate distances than there is in 2D, so if the low-dimensional similarity kernel had the same (Gaussian) shape as the high-dimensional one, moderately-distant high-D points would all be squeezed into a tiny central blob in 2D just to satisfy the close-range probabilities. The Cauchy distribution's heavy tail lets moderately-distant points stay moderately far apart in the map.

**Step 3 — move the points** `y_i` to minimise the KL divergence between the two distributions, `Σ_ij p_ij log(p_ij / q_ij)`, via gradient descent. The asymmetry of KL divergence here is itself meaningful: it penalises using a *small* `q_ij` for a pair with *large* `p_ij` (neighbours in high-D ending up far apart in the map) far more than the reverse — t-SNE is explicitly willing to sacrifice global accuracy to get local neighbourhoods right.

### The single most important practical consequence

**Never compute a downstream distance-based metric on t-SNE's output.** Cluster *sizes*, the *distances between* clusters, and even whether two visually-separate blobs were "really" separated by a lot or a little in the original space are **not reliable** in a t-SNE plot — the algorithm was never asked to preserve any of that, only local neighbour rank. This topic's code follows the only safe pattern: t-SNE is used **exclusively to produce the picture**; every actual *number* reported (the silhouette scores below) is computed in the **original, untransformed vector space**.

## 3. A real metric for "are these vectors any good?": silhouette score

To turn "do same-label points look clustered?" into an actual number, this topic uses the **silhouette score** (Rousseeuw, 1987). For a point `i` with assigned group label `g(i)`:

```
a(i) = mean distance from i to other points with the same label
b(i) = mean distance from i to points in the nearest other label
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

`s(i)` ranges from -1 (i is, on average, closer to a different group than its own — a sign it's mislabelled or the groups overlap there) to +1 (i sits comfortably inside a tight, well-separated group). The overall silhouette score is the mean of `s(i)` over all points, computed here with cosine distance (the right choice for the same reason Topic 1.1 used cosine similarity rather than Euclidean) in the **original, pre-t-SNE space**.

## 4. The two assessments this topic runs

**Document level.** Every one of the Brown corpus's 500 documents (spanning 15 genres — news, fiction, government, romance, and so on) is represented two ways: as a sparse TF-IDF vector, and as a dense vector built by mean-pooling the Word2Vec vectors (from Topic 1.1) of its words. Both representations are t-SNE'd for a picture and silhouette-scored against the genre labels for a number. The two methods turn out to disagree about *which* structure is easiest to see — see `explanation.md` for the actual results and why that disagreement is itself informative, not a bug.

**Word level.** The top ~500 frequent content words from the same Word2Vec model are coloured by their dominant part-of-speech tag (using NLTK's universal tagset over the Brown corpus) and checked the same way: visually via t-SNE, numerically via silhouette score in the original 100-dimensional space. Nothing in Word2Vec's training objective ever mentions part of speech — any clustering by POS that emerges is a side effect of words with the same grammatical role tending to occur in similar surrounding contexts. This is a preview of Module IV, where POS tagging becomes the main event.

## References

1. Sparck Jones, K. (1972). *A Statistical Interpretation of Term Specificity and Its Application in Retrieval.* Journal of Documentation.
2. Salton, G., & Buckley, C. (1988). *Term-Weighting Approaches in Automatic Text Retrieval.* Information Processing & Management.
3. van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE.* Journal of Machine Learning Research, 9, 2579–2605.
4. Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* Journal of Computational and Applied Mathematics.
5. Wattenberg, M., Viégas, F., & Johnson, I. (2016). *How to Use t-SNE Effectively.* Distill. (Practical pitfalls — cluster sizes and inter-cluster distances are not meaningful — referenced in section 2 above.)
