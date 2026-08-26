# 1.3 — Code Explanation

## The embedding projector

```python
words = model.wv.index_to_key[:n_words]
freqs = np.array([model.wv.get_vecattr(w, "count") for w in words])
coords = PCA(n_components=3, random_state=42).fit_transform(vectors)
```

`index_to_key[:n_words]` takes the 400 most frequent words (gensim orders the vocabulary by descending frequency — see Topic 1.1's `explanation.md`). `get_vecattr(w, "count")` retrieves each word's raw training-corpus frequency, used here purely to colour the plot (more frequent words shade differently), not for anything mathematical. `PCA(n_components=3)` is the only dimensionality reduction step — for this general-purpose explorer, the goal is "give me *a* reasonable, well-understood 3D view to navigate," and PCA is the cheapest, most interpretable choice (each axis literally means "direction of decreasing variance"), not a claim that 3D PCA is uniquely correct for browsing.

```python
fig = go.Figure(data=[go.Scatter3d(..., mode="markers+text", hoverinfo="text", ...)])
fig.write_html(out_path)
```

`Scatter3d` with `mode="markers+text"` draws every point with its word label directly visible (small font, since 400 labels at once is busy — hovering, via `hoverinfo="text"`, gives the full "word (count=N)" detail one at a time without cluttering the view). `fig.write_html` bundles Plotly's JavaScript renderer directly into the output file, so `embedding_projector.html` is fully self-contained — open it in any browser, no server or internet connection required, and drag to rotate the 3D view.

## The PCA-vs-t-SNE comparison

```python
PLURAL_PAIRS = [("boy", "boys"), ("girl", "girls"), ("day", "days"),
                ("night", "nights"), ("week", "weeks"), ("month", "months")]
```

Six pairs sharing one relation (singular → plural) were chosen deliberately, rather than six unrelated word pairs, so that *if* a consistent "+plural" direction exists, there would be six chances to see it as parallel arrows.

```python
pca_coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
tsne_coords = TSNE(n_components=2, perplexity=5, random_state=42, init="pca").fit_transform(vectors)
```

`perplexity=5` is deliberately low for t-SNE here — with only 12 points total, the default perplexity of 30 (which expects roughly that many effective neighbours) would be meaningless; perplexity should always be well below the number of points being embedded.

```python
def quantify_parallelism(model, pairs):
    offsets = np.array([model.wv[b] - model.wv[a] for a, b in pairs])
    norm = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    sim_matrix = norm @ norm.T
```

This is the number that actually matters, computed entirely independently of either 2D picture: six "+plural" offset vectors, each L2-normalised, then an all-pairs cosine similarity via one matrix product (the same normalise-once trick as Topic 1.1's `WordVectorSpace`).

## The actual result

```
           boy->boys  girl->girls   day->days  night->nights  week->weeks  month->months
 boy->boys      1.00         0.63        0.07            0.09        -0.13           0.02
girl->girls     0.63         1.00        0.03            0.11        -0.17          -0.07
 day->days      0.07         0.03        1.00            0.49         0.50           0.50
night->nights   0.09         0.11        0.49            1.00         0.26           0.25
week->weeks    -0.13        -0.17        0.50            0.26         1.00           0.45
month->months   0.02        -0.07        0.50            0.25         0.45           1.00

Mean pairwise similarity among '+plural' offsets: 0.202
```

This is not "the analogy direction is consistent" and it is not "there is no plural direction at all" — it's a third, more specific finding: the offsets split into **two internally-consistent sub-groups that barely relate to each other**. `boy→boys` and `girl→girls` (similarity 0.63) are about pluralising *people*. `day/night/week/month` (similarities 0.25–0.50 among themselves) are about pluralising *time units*. Cross-group similarities are near zero or slightly negative. "+plural" in this model is better described as several related-but-distinct directions, one per broad noun category, than as one universal vector — a real, specific structural finding, not noise.

The two pictures in `pca_vs_tsne_analogy.png` partially reflect this and partially don't, which is itself the point. In the **PCA** panel, `boy→boys` and `girl→girls` sit in their own corner with short, visibly parallel arrows — consistent with their high similarity. The time-unit group's arrows are longer and visibly cross one another — also broadly consistent with their merely-moderate (0.25–0.50) mutual similarity. In the **t-SNE** panel, however, the time-unit group's arrows happen to look fairly tidy and similarly-directed — and that apparent tidiness is **not backed by anything in t-SNE's objective function**. This is `theory.md` section 1's argument made concrete: a linear projection's faithfulness to additive structure follows from algebra and can be reasoned about; a t-SNE picture's resemblance (or lack of it) to the true structure is a coincidence of that specific optimisation run, not a property you're entitled to rely on. The only number in this whole topic that's actually trustworthy evidence about the original 100-dimensional space is the cosine-similarity matrix above — both pictures are, at best, illustrations of conclusions you'd already reached some other way.
