"""
Topic 5.2 -- Parsing Sentences as Trees and Data Description for RNNs
CSE468: Natural Language Processing with Deep Learning

Demonstrates tree preprocessing for RvNNs: loading Penn Treebank parse
trees, cleaning them (stripping function tags, removing traces, collapsing
unary chains), and computing statistics about the resulting tree corpus.

Run directly:
    python implementation.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import treebank
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def ensure_data():
    for p,s in [("treebank","corpora"),("punkt","tokenizers"),("punkt_tab","tokenizers")]:
        try: nltk.data.find(f"{s}/{p}")
        except: nltk.download(p, quiet=True)

def clean_label(label):
    """NP-SBJ -> NP"""
    return label.split("-")[0].split("=")[0]

def clean_tree(tree):
    """Remove traces and function tags, collapse unary chains."""
    if isinstance(tree, str):
        return tree
    # Filter out trace children
    children = [c for c in tree if not (isinstance(c, nltk.Tree) and c.label() == "-NONE-")]
    if not children:
        return None
    clean_children = []
    for c in children:
        if isinstance(c, str):
            clean_children.append(c)
        else:
            cc = clean_tree(c)
            if cc is not None:
                clean_children.append(cc)
    if not clean_children:
        return None
    # Collapse unary chains: single non-leaf child with same surface
    if len(clean_children) == 1 and isinstance(clean_children[0], nltk.Tree):
        # unary production - collapse
        return clean_children[0]
    return nltk.Tree(clean_label(tree.label()), clean_children)

def tree_stats(trees):
    depths, n_internal, n_leaves, n_children = [], [], [], []
    for t in trees:
        def depth(n):
            if isinstance(n,str): return 0
            return 1+max((depth(c) for c in n),default=0)
        def count(n):
            if isinstance(n,str): return 0,1
            i,l=1,0
            for c in n:
                ci,cl=count(c); i+=ci; l+=cl
            return i,l
        depths.append(depth(t))
        i,l=count(t); n_internal.append(i); n_leaves.append(l)
        def ch(n):
            if isinstance(n,str): return
            n_children.append(len(n))
            for c in n: ch(c)
        ch(t)
    return depths, n_internal, n_leaves, n_children

def main():
    ensure_data()
    raw_trees = treebank.parsed_sents()
    clean_trees = [clean_tree(t) for t in raw_trees]
    clean_trees = [t for t in clean_trees if t is not None and isinstance(t, nltk.Tree)]
    print(f"Raw trees: {len(raw_trees)}  ->  Cleaned trees: {len(clean_trees)}")
    
    depths, n_int, n_leaves, n_ch = tree_stats(clean_trees[:500])
    print(f"  Depth: mean={np.mean(depths):.1f} max={max(depths)}")
    print(f"  Leaves: mean={np.mean(n_leaves):.1f}")
    print(f"  Branching factor: mean={np.mean(n_ch):.2f} (binary={sum(1 for x in n_ch if x==2)/len(n_ch):.1%})")
    
    label_counts = Counter(
        clean_label(n.label()) 
        for t in clean_trees[:500] 
        for n in t.subtrees() 
        if not isinstance(n, str) and hasattr(n, 'label')
    )
    print(f"  Most common phrase labels: {label_counts.most_common(8)}")
    
    fig, axes = plt.subplots(1,3,figsize=(13,4),dpi=160)
    axes[0].hist(depths,bins=12,color="#4C72B0",alpha=0.8); axes[0].set_xlabel("tree depth"); axes[0].set_title("Tree depths")
    axes[1].hist(n_leaves,bins=15,color="#4F9D69",alpha=0.8); axes[1].set_xlabel("leaves per tree"); axes[1].set_title("Sentence lengths")
    axes[2].hist(n_ch,bins=range(1,8),color="#C0504D",alpha=0.8); axes[2].set_xlabel("children per node"); axes[2].set_title("Branching factor")
    plt.suptitle("Penn Treebank parse tree statistics (cleaned, first 500 trees)")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGE_DIR,"tree_statistics.png"),bbox_inches="tight",facecolor="white"); plt.close()
    
    # Label frequency plot
    top_labels = label_counts.most_common(12)
    fig,ax = plt.subplots(figsize=(8,4),dpi=160)
    ax.bar([l for l,_ in top_labels],[c for _,c in top_labels],color="#4C72B0",alpha=0.8)
    ax.set_xlabel("phrase label"); ax.set_ylabel("frequency"); ax.set_title("Most frequent phrase labels in cleaned Penn Treebank trees")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGE_DIR,"label_frequency.png"),bbox_inches="tight",facecolor="white"); plt.close()
    print(f"Saved plots to {IMAGE_DIR}")

if __name__ == "__main__":
    main()
