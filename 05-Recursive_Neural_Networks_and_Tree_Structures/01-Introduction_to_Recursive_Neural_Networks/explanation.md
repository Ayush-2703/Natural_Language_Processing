# 5.1 — Code Explanation

## Tree traversal as recursive Python

```python
def forward_tree(self, tree, word2idx):
    if isinstance(tree, str):           # leaf node
        idx = word2idx.get(tree.lower(), 0)
        return self.embedding(torch.tensor([idx])).squeeze(0)
    else:                               # internal node
        child_vecs = [self.forward_tree(c, word2idx) for c in tree]
        result = child_vecs[0]
        for cv in child_vecs[1:]:
            result = self.compose(result, cv)
        return result
```

Python's call stack IS the tree traversal. Every recursive call to `forward_tree` on a subtree returns that subtree's hidden state. PyTorch tracks every operation, so `loss.backward()` propagates gradients back through the full tree structure automatically — no special treatment needed for the recursive structure.

The toy sentiment task achieves 12/12 accuracy after 80 epochs, with the `hidden_states.png` PCA plot showing positive and negative phrases cleanly separated in the learned representation space.
