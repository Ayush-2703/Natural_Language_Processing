# 1.4 — Text Classification utilizing Word Vectors

## 1. The central idea: reuse a representation instead of designing features

Before distributed word representations, building a text classifier meant hand-engineering features: word counts, n-gram presence, hand-built sentiment lexicons, syntactic patterns. **Collobert & Weston (2008, ICML; expanded 2011, JMLR — "Natural Language Processing (Almost) from Scratch")** made the case that has shaped NLP architecture ever since: train one good word representation, largely unsupervised, and **reuse it as the input layer for many different downstream tasks** instead of re-engineering features for each one. Their original demonstration spanned POS tagging, chunking, NER, and semantic role labelling (the tasks Module IV revisits); this topic runs the same idea on sentiment classification, with Word2Vec (Topic 1.1) standing in for their network's learned lookup table.

The practical question this raises immediately: when you plug a generically-trained representation into a new task, should you **freeze** it (treat the vectors as fixed input features) or **fine-tune** it (let the task's own gradient updates reshape the vectors)? This topic runs both and reports a real, sometimes counter-intuitive answer.

## 2. From word vectors to a document vector: mean pooling

Topic 1.1 and 1.2 give a vector per *word*. A classifier needs a vector per *document*. The simplest possible composition function is the mean:

```
doc_vector = (1/n) Σ_{i=1}^{n} vec(word_i)
```

This is a real modelling choice with a real cost: it is invariant to word order ("not good, but not bad either" and "not bad, but not good either" produce an *identical* vector) and dilutes a handful of strongly sentiment-bearing words across however many neutral words surround them. It is also, despite this, a surprisingly strong baseline for document-level tasks like topic or sentiment classification, where *which* words appear usually carries more signal than the order they appear in. Phase 4's RNNs and Phase 5's recursive networks exist specifically to stop throwing away order and structure — keep this limitation in mind as motivation for why those more complex architectures earn their complexity.

`torch.nn.EmbeddingBag` is PyTorch's fused, batched implementation of exactly this operation: given a flat list of word indices and an `offsets` tensor marking where each document starts within that list, it looks up every embedding and averages (`mode="mean"`) the ones belonging to each document, all in one operation, on documents of different lengths, with no padding required. See `explanation.md` for the index bookkeeping in detail.

## 3. Frozen vs. fine-tuned embeddings

**Frozen**: `embedding.weight.requires_grad = False`. Backpropagation never touches the embedding table; only the small classifier head on top learns. The embedding space is exactly the general-purpose, sentiment-agnostic space explored in Topics 1.1–1.3 — it knows `good` and `bad` are *related* (Topic 1.1's heatmap put their cosine similarity at 0.65) but has no notion that one is positive and the other negative, because nothing in Word2Vec's unsupervised training objective ever saw a sentiment label.

**Fine-tuned**: gradients flow into the embedding table itself. The model can now *move* `good` and `bad` apart along whatever direction the classifier finds useful, *move* `boring` toward other negative words, and so on — reshaping the representation around the specific task rather than treating it as fixed scenery. This is more expressive, and correspondingly carries more overfitting risk on a small dataset (2,000 documents is genuinely small for fine-tuning a 100-dimensional embedding table covering tens of thousands of words) — but, as `explanation.md`'s actual results show, expressiveness wins here.

## 4. The loss function

Both PyTorch models are trained with **cross-entropy loss**. For a single example with true class `y` and predicted class probabilities `p` (obtained by softmax over the classifier's two output logits):

```
L = -log( p_y )
```

This penalises confident wrong answers heavily (as `p_y → 0`, `-log(p_y) → ∞`) and confident right answers lightly — exactly the gradient signal needed to push the model toward sharp, correct decisions rather than vague, hedged ones. `nn.CrossEntropyLoss` in the code combines the softmax and the negative log-likelihood into one numerically-stable operation, which is why the model's `forward` method returns raw logits rather than probabilities.

## References

1. Collobert, R., & Weston, J. (2008). *A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning.* ICML.
2. Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., & Kuksa, P. (2011). *Natural Language Processing (Almost) from Scratch.* JMLR.
3. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR. (The original case for jointly learning representation and task — see Topic 1.1.)
4. Pang, B., Lee, L., & Vaithyanathan, S. (2002). *Thumbs up? Sentiment Classification using Machine Learning Techniques.* EMNLP. (The line of work the Movie Reviews corpus used here comes from.)
