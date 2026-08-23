# 1.1 — Introduction to Vectors and Word Analogy

## 1. Why represent words as vectors at all?

The oldest representation of a word in an NLP system is a **one-hot vector**: pick a vocabulary of size `V`, and word `i` is the standard basis vector `e_i ∈ {0,1}^V` — all zeros except a 1 in position `i`. This is simple and exact, but it has two fatal properties for any learning system:

- **No notion of similarity.** Every pair of distinct one-hot vectors has Euclidean distance √2 and cosine similarity 0, regardless of whether the words are `cat`/`dog` or `cat`/`bureaucracy`. The representation encodes *identity*, not *meaning*.
- **No generalisation.** A model that has only ever seen "the cat sat on the mat" has learned nothing it can transfer to "the dog sat on the rug" — `cat` and `dog` are, geometrically, unrelated points.

The fix is to abandon one-hot vectors for **dense, low-dimensional, learned vectors** — typically 50 to 300 dimensions instead of `V` (often 10,000+) — where geometric closeness is made to track semantic or syntactic closeness. This is the **distributional hypothesis**, informally: *a word is characterised by the company it keeps*. Words that occur in similar contexts end up with similar vectors, because the methods that produce these vectors (count-based, as in Module III, or predictive, as in Word2Vec) are explicitly built around context co-occurrence.

## 2. Distributed representations (Bengio et al., 2003)

The idea of *learning* a dense vector per word, rather than hand-designing features, was formalised by Bengio, Ducharme, Vincent & Jauvin in **"A Neural Probabilistic Language Model"** (JMLR, 2003). Their model assigned every word in the vocabulary a feature vector `C(w) ∈ R^m`, fed the concatenation of the previous `n-1` words' vectors into a small feedforward network, and trained the whole thing — vectors and network weights together — to predict the next word.

Two consequences of this setup matter far beyond language modelling itself:

1. **The vectors are a byproduct of prediction, not an explicit target.** Nothing in the loss function says "cat and dog should be close." Closeness emerges because words that are *interchangeable in context* receive *similar gradient updates* during training.
2. **Representation is shared across exponentially many contexts.** A classical n-gram model treats "the cat sat" and "the dog sat" as unrelated events. Bengio et al.'s model represents `cat` and `dog` with vectors that can be close, so probability mass learned for one context partially transfers to the other — this is precisely the generalisation one-hot vectors cannot provide.

This is the conceptual root of every embedding method this course covers. Word2Vec (Module III) is, in essence, a heavily simplified and scaled-up version of Bengio's network with the hidden layer removed. GloVe (also Module III) shows the same vector space can be reached by factorising a co-occurrence matrix instead of running a predictive network at all. The point of this introductory topic is to get comfortable *exploring* such a vector space — we'll start training one in Phase 3.

## 3. Measuring similarity: cosine, not Euclidean distance

Given two word vectors `u, v ∈ R^d`, the natural-seeming choice — Euclidean distance `‖u - v‖` — is a poor similarity measure for word embeddings, because **vector magnitude in these spaces correlates with word frequency**, not meaning. A very common word and a very rare synonym can point in nearly the same direction yet have very different norms. **Cosine similarity** factors magnitude out and looks only at *direction*:

```
cos(u, v) = (u · v) / (‖u‖ ‖v‖)
```

`cos(u, v) = 1` means the vectors point the same way (maximally similar), `0` means orthogonal (unrelated), and `-1` means opposite. A useful implementation detail, used throughout this topic's code: if every vector in a collection is first **L2-normalised** (divided by its own norm, so `‖u‖ = 1` for all `u`), then `cos(u, v)` reduces to a plain dot product `u · v`. Normalise once, and every later similarity computation is one multiply-and-sum instead of a multiply-and-sum *plus two square roots and a division*.

One caveat worth internalising early: cosine similarity measures **relatedness**, not synonymy. `good` and `bad` are highly similar by this metric — they occur in almost identical contexts ("the food was ___") — despite being antonyms. Vector space models capture *substitutability*, not logical meaning.

## 4. Word analogy and the parallelogram model

The single most-quoted result in this entire field is:

```
vec(king) - vec(man) + vec(woman)  ≈  vec(queen)
```

Geometrically, this says the offset from `man` to `king` (call it "+royalty") is approximately the same vector as the offset from `woman` to `queen`; equivalently, `king`, `queen`, `man`, `woman` form (approximately) the four corners of a parallelogram in embedding space.

![The parallelogram model of word analogy](images/parallelogram_concept.png)

Formally, solving the analogy `a : b :: c : ?` means finding the vocabulary word `d` that maximises:

```
cos( vec(d),  vec(b) - vec(a) + vec(c) )
```

This is the **3CosAdd** method (named for the three cosine-related terms once you expand it), and it's exactly what `implementation.py`'s `analogy()` method computes: build the query vector `vec(b) - vec(a) + vec(c)`, then run nearest-neighbour search for it against the whole vocabulary (excluding `a`, `b`, `c` themselves, which would otherwise dominate the result trivially). A refinement called **3CosMul** — multiplying rather than summing the three similarity terms, which penalises a candidate that is close to any *one* of the three reference words for the wrong reason — is documented in Levy & Goldberg (2014), *Linguistic Regularities in Sparse and Explicit Word Representations*; it is not implemented here but is worth knowing about.

Why does the parallelogram structure appear at all? No one designs it in explicitly — it falls out of the fact that "royalty" and "gender" each correspond to roughly *linear, additive* directions of variation that recur across many word pairs in the training corpus. Module III's treatment of Word2Vec and GloVe gives a much more precise account of why predictive and count-based training both tend to produce this kind of linear structure.

## 5. What this topic's code actually does

`implementation.py` does **not** train Word2Vec from scratch (that's Phase 3's job) — it uses `gensim`'s production Word2Vec trainer to obtain real vectors quickly from a ~4.2-million-token corpus assembled from NLTK's Brown, Gutenberg, and Movie Reviews corpora, then hands those vectors to a small PyTorch class (`WordVectorSpace`) that implements cosine similarity, nearest-neighbour search, and 3CosAdd analogy solving as explicit tensor operations — a single L2-normalisation, a single matrix–vector product, and `torch.topk`. See `explanation.md` for the full line-by-line walkthrough.

## 6. Caveats this topic deliberately surfaces

Real results are noisier than textbook examples, and that noise is itself instructive:

- **Corpus composition shows up directly in the geometry.** Eighteen of the Gutenberg corpus's texts include the King James Bible, so this model's notion of "things similar to *king*" is dominated by Biblical monarchs (Jehoshaphat, Ahasuerus, Benhadad) ahead of the generic `queen`. A production system trained on a more balanced, much larger corpus (think billions of tokens of web text) would not show this skew — but it's a genuinely useful lesson about *what your training data actually contains* showing up in *what your model thinks is similar*.
- **Not every analogy type survives a small corpus equally well.** Inflectional patterns (`boy → boys`, `day → days`) are extremely high-frequency and robust even here. Derivational patterns (`write → writer`, `act → actor`) are rarer and noisier, and this topic's evaluation genuinely misses one of them — see `explanation.md` for the actual numbers from running this code.

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* Journal of Machine Learning Research, 3, 1137–1155.
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop. (Popularised the analogy task and the 3CosAdd method explored here; Phase 3 implements the model itself.)
3. Levy, O., & Goldberg, Y. (2014). *Linguistic Regularities in Sparse and Explicit Word Representations.* CoNLL. (3CosMul and a count-based view of why analogies work.)
4. Firth, J. R. (1957). *A Synopsis of Linguistic Theory, 1930–1955.* — origin of "you shall know a word by the company it keeps," the informal statement of the distributional hypothesis.
