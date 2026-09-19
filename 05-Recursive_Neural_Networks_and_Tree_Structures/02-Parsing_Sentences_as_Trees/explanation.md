# 5.2 — Code Explanation

## Tree cleaning pipeline

```python
def clean_tree(tree):
    children = [c for c in tree if not (isinstance(c, nltk.Tree) and c.label() == "-NONE-")]
    ...
    if len(clean_children) == 1 and isinstance(clean_children[0], nltk.Tree):
        return clean_children[0]  # collapse unary chain
    return nltk.Tree(clean_label(tree.label()), clean_children)
```

Three operations in sequence: (1) filter `-NONE-` trace children before recursing, (2) recurse into each surviving child, (3) collapse unary productions (a node with a single non-leaf child is replaced by that child). The result is a cleaned tree where every internal node has at least two children or is directly above a word.

## Branching factor

The cleaned Penn Treebank trees are predominantly binary (≈62% of internal nodes have exactly 2 children) but not fully binarised — some nodes have 3, 4, or more children. The TNN in Topic 5.3 handles this by composing children left-to-right pairwise; the RNTN in Topic 5.5 restricts to strictly binary trees via binarisation.
