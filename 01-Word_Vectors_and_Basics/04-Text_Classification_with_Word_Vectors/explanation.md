# 1.4 — Code Explanation

## Building the vocabulary and embedding matrix

```python
word2idx = {w: i + 1 for i, w in enumerate(w2v.wv.index_to_key)}
embedding_matrix = np.vstack([np.zeros((1, w2v.wv.vector_size), dtype=np.float32), w2v.wv.vectors])
```

Index `0` is deliberately reserved and never assigned to a real word (`i + 1`, not `i`) — it's the fallback for any word `ReviewDataset` encounters that isn't in Word2Vec's vocabulary (`word2idx.get(w, 0)`), and its embedding row is an all-zeros vector. Averaging in a zero vector for unknown words is a deliberate, mild choice: it dilutes a document's pooled vector slightly per unknown word rather than crashing or requiring a special-cased "unknown" embedding to be learned from very little data.

## The EmbeddingBag batching pattern

```python
def collate_batch(batch):
    indices_list, labels = zip(*batch)
    offsets = [0]
    for idx in indices_list[:-1]:
        offsets.append(offsets[-1] + len(idx))
    flat_indices = torch.tensor([i for idx in indices_list for i in idx], dtype=torch.long)
    offsets = torch.tensor(offsets, dtype=torch.long)
    ...
```

This is the part of the code most worth slowing down on if you haven't used `EmbeddingBag` before. A batch of, say, 32 movie reviews of different lengths (300 words, 850 words, 412 words, ...) is **not** stored as a padded 32×max_length matrix the way an `nn.Embedding` + RNN pipeline would need. Instead, every review's word indices are concatenated end-to-end into a single 1-D tensor (`flat_indices`), and `offsets[i]` records exactly where review `i` starts within that long tensor — `offsets = [0, 300, 1150, 1562, ...]` for the lengths above. `nn.EmbeddingBag(flat_indices, offsets)` then knows, internally, exactly which slice of `flat_indices` belongs to which document, looks up every index's embedding, and mean-pools each document's slice separately — one fused operation, zero padding, zero wasted compute on pad tokens. This is the standard, idiomatic pattern for bag-of-embeddings text classification in PyTorch (it's the same pattern used in PyTorch's own official text-classification tutorial).

## The model

```python
self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
self.embedding.weight.requires_grad = not freeze_embeddings
```

`mode="mean"` is what makes this layer compute exactly the averaging operation `theory.md` section 2 writes out mathematically — `EmbeddingBag` also supports `"sum"` and `"max"`, but mean is the right choice when you don't want longer documents to automatically get larger-magnitude vectors. `.weight.data.copy_(...)` overwrites the layer's randomly-initialised weights with Word2Vec's actual trained vectors — this single line is the entire "transfer" step the theory file's section 1 discusses; everything before it is just data plumbing to get those vectors into a PyTorch tensor of the right shape. `requires_grad` is the one-line toggle between this topic's two experiments.

```python
self.classifier = nn.Sequential(
    nn.Linear(embed_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2),
)
```

A small two-layer head, not a single linear layer, so the model has at least a little room to learn a non-trivial decision boundary over the pooled embedding even in the frozen-embedding case. `Dropout(0.3)` randomly zeroes 30% of the hidden layer's activations during training only — a standard regulariser, included because 2,000 documents is a small dataset and the fine-tuned variant in particular has a lot of effectively-trainable parameters (the entire embedding table) relative to that.

## Actual results

```
TF-IDF + LogReg     : 0.823
Frozen Word2Vec     : 0.698
Fine-tuned Word2Vec : 0.860
```

This ordering is worth sitting with rather than skimming past. **Frozen Word2Vec is the worst of the three** — worse than a plain bag-of-words logistic regression — and the loss curve in `loss_curves.png` shows why: both train and validation loss for the frozen model barely move from their starting point of ~0.69 (the loss of a model predicting 50/50 on a balanced binary task) across all 12 epochs. A small classifier head sitting on top of a *fixed* representation that was never trained with sentiment in mind has very little to work with — `good` and `bad` (Topic 1.1: cosine similarity 0.65) are close together in this space precisely *because* they're interchangeable in generic contexts, which is the opposite of what a sentiment classifier needs to tell them apart.

**Fine-tuning closes that gap and then some.** Once gradients are allowed to reshape the embedding table itself, training loss drops sharply from epoch 3 onward (visible as the steep green curve in `loss_curves.png`) and validation accuracy climbs to 86% — beating the TF-IDF baseline by about 3.7 points. The model is no longer limited to recombining a fixed, sentiment-blind geometry; it can pull `boring` and `tedious` together in a sentiment-relevant direction even if Word2Vec's original unsupervised training only ever grouped them together for unrelated distributional reasons.

The practical takeaway, and the reason this comparison was worth running rather than asserting: **a generically pretrained representation is a starting point, not a finished feature set.** Collobert & Weston's own architecture (`theory.md` section 1) updates its word lookup table during task training rather than freezing it, and this experiment reproduces, on a small scale, exactly why that choice matters. The confusion matrices (`confusion_frozen.png`, `confusion_finetuned.png`) show the same story at the level of individual predictions: the fine-tuned model's errors (35 + 21 = 56 out of 400) are much more balanced between false positives and false negatives than the frozen model's, consistent with it having learned a genuine sentiment-relevant decision boundary rather than a weak, partially-arbitrary one.
