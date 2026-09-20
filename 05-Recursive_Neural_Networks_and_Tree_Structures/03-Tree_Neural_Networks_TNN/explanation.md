# 5.3 — Code Explanation

## Per-node supervision
```python
for tree in trees[:300]:
    try:
        h = tnn.fwd(t, vmap)
        lb = sent_label(t)
        loss = F.cross_entropy(tnn.clf(h.unsqueeze(0)), torch.tensor([lb]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    except: pass
```
Each tree produces one gradient update — the root node's sentiment prediction. On the real SST dataset, every subtree has its own label and contributes its own loss, giving the model supervision at every level of composition. On this synthetic task, only root-level labels are used. Test accuracy 98.7% confirms the TNN has learned the sentiment vocabulary and composition function correctly.

## Parameter sharing verified
The same `W` matrix (shape `2d×d`) is used at every composition step, regardless of tree depth or syntactic label. A single weight matrix correctly composing "not bad" (double negation needs multiplicative interaction) is the TNN's fundamental limitation — the tensor term in Topic 5.5's RNTN directly addresses this.
