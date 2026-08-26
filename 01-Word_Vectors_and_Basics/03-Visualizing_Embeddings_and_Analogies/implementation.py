"""
Topic 1.3 -- Visualizing Data and Analogies (t-SNE, Embedding Projectors)
CSE468: Natural Language Processing with Deep Learning

Two things this topic does that Topic 1.2 deliberately did not:

  1. Builds an interactive, rotatable 3D "embedding projector" (in the style
     of TensorFlow's actual Embedding Projector tool) as a standalone HTML
     file you can open in a browser -- exploration, not a fixed picture.

  2. Demonstrates *why* visualizing analogy structure specifically requires
     a LINEAR projection (PCA) rather than the nonlinear t-SNE used in 1.2:
     PCA preserves the additive vector relationships that make
     vec(b)-vec(a)+vec(c) ~ vec(d) work; t-SNE's nonlinear warping does not
     promise to, and the side-by-side comparison below shows the difference
     directly rather than just asserting it.

Run directly:
    python implementation.py
(also writes an interactive embedding_projector.html you can open in any browser)
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

W2V_PATH = os.path.join(
    HERE, "..", "1.1-Introduction-to-Word-Vectors-and-Analogies", "artifacts", "word2vec_phase1.model"
)


def clean(tokens):
    return [w.lower() for w in tokens if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]


def load_or_train_word2vec():
    """Same recipe and cache file as Topics 1.1/1.2 -- see 1.1's explanation.md."""
    if os.path.exists(W2V_PATH):
        print(f"Loading cached Word2Vec model from {W2V_PATH}")
        return Word2Vec.load(W2V_PATH)
    import nltk
    from nltk.corpus import brown, gutenberg, movie_reviews
    for pkg, sub in [("brown", "corpora"), ("gutenberg", "corpora"), ("movie_reviews", "corpora")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
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
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, workers=1, epochs=8)
    os.makedirs(os.path.dirname(W2V_PATH), exist_ok=True)
    model.save(W2V_PATH)
    return model


# --------------------------------------------------------------------------
# 1. Interactive 3D embedding projector (Plotly -> standalone HTML)
# --------------------------------------------------------------------------
def build_embedding_projector(model, n_words=400, out_path=None):
    words = model.wv.index_to_key[:n_words]
    vectors = np.array([model.wv[w] for w in words])
    freqs = np.array([model.wv.get_vecattr(w, "count") for w in words])

    coords = PCA(n_components=3, random_state=42).fit_transform(vectors)

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        text=words,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=4,
            color=np.log(freqs),
            colorscale="Viridis",
            colorbar=dict(title="log frequency"),
            opacity=0.85,
        ),
        hovertext=[f"{w}  (count={c})" for w, c in zip(words, freqs)],
        hoverinfo="text",
    )])
    fig.update_layout(
        title=f"Embedding Projector -- top {n_words} Word2Vec vectors (PCA to 3D)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_path)
    print(f"Saved interactive embedding projector to {out_path}")


# --------------------------------------------------------------------------
# 2. PCA vs. t-SNE for analogy structure
# --------------------------------------------------------------------------
PLURAL_PAIRS = [
    ("boy", "boys"), ("girl", "girls"), ("day", "days"),
    ("night", "nights"), ("week", "weeks"), ("month", "months"),
]


def analogy_projection_comparison(model, pairs, out_path):
    words = [w for pair in pairs for w in pair]
    vectors = np.array([model.wv[w] for w in words])

    pca_coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    tsne_coords = TSNE(n_components=2, perplexity=5, random_state=42, init="pca").fit_transform(vectors)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    for ax, coords, title in [
        (axes[0], pca_coords, "PCA (linear projection)"),
        (axes[1], tsne_coords, "t-SNE (nonlinear projection)"),
    ]:
        for i in range(0, len(words), 2):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#4C72B0", lw=2))
            ax.scatter([x0, x1], [y0, y1], color="#C0504D", s=40, zorder=3)
            ax.text(x0, y0, words[i], fontsize=9, ha="right", va="bottom")
            ax.text(x1, y1, words[i + 1], fontsize=9, ha="left", va="top")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Singular -> plural offsets ("+plural" direction) under two projections', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quantify_parallelism(model, pairs):
    """Cosine similarity between every pair of (singular->plural) offset
    vectors, computed in the ORIGINAL 100-d space -- this is the actual
    evidence for "the +plural direction is roughly consistent," independent
    of which 2D picture is drawn."""
    offsets = np.array([model.wv[b] - model.wv[a] for a, b in pairs])
    norm = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    sim_matrix = norm @ norm.T
    iu = np.triu_indices(len(pairs), k=1)
    return sim_matrix, sim_matrix[iu].mean()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    model = load_or_train_word2vec()

    build_embedding_projector(model, n_words=400, out_path=os.path.join(HERE, "embedding_projector.html"))

    analogy_projection_comparison(
        model, PLURAL_PAIRS, os.path.join(IMAGE_DIR, "pca_vs_tsne_analogy.png")
    )

    sim_matrix, mean_sim = quantify_parallelism(model, PLURAL_PAIRS)
    print("\nPairwise cosine similarity between '+plural' offset vectors (100-d space):")
    pair_labels = [f"{a}->{b}" for a, b in PLURAL_PAIRS]
    print("           " + "  ".join(f"{l:>10s}" for l in pair_labels))
    for i, label in enumerate(pair_labels):
        print(f"{label:>10s} " + "  ".join(f"{sim_matrix[i, j]:10.2f}" for j in range(len(pair_labels))))
    print(f"\nMean pairwise similarity among '+plural' offsets: {mean_sim:.3f}")
    print("(closer to 1.0 = the six singular->plural arrows point in a more consistent direction)")

    print(f"\nSaved comparison plot to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
