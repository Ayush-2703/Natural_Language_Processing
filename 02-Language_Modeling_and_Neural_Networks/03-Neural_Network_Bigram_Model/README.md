# 2.3 — Implementation of the Neural Network Bigram Model

## 1. The architecture: Bengio et al. (2003), with context shrunk to one word

Topic 1.1 introduced Bengio, Ducharme, Vincent & Jauvin's neural probabilistic language model as the conceptual root of every embedding method in this course. This topic builds the actual architecture, simplified to bigram context so it is the direct neural counterpart of Topic 2.2's counting-based model:

```
prev_word  --[Embedding]-->  e  --[Linear + tanh]-->  h  --[Linear]-->  logits over vocabulary
```

`prev_word_idx` is looked up in an embedding table to get a dense vector `e` (this single table is, structurally, exactly what Topic 1.1 spent an entire topic exploring — the only difference here is that it's trained for next-word prediction directly rather than borrowed from a separately-trained Word2Vec model). `e` passes through one hidden layer with a `tanh` nonlinearity (Bengio et al.'s original choice) to get `h`, and a final linear layer projects `h` to one score per vocabulary word. Softmax (applied internally by `nn.CrossEntropyLoss` — see `explanation.md`) turns those scores into a probability distribution `P(w_t | w_{t-1})`, which is trained against the true next word with cross-entropy loss — the same loss whose relationship to perplexity Topic 2.1, section 3 already established.

With the full `n-1`-word context Bengio et al. actually used, the embedding step would look up *every* context word and concatenate their vectors before the hidden layer. With `n=2` (bigram context), there's only one word to look up, so that concatenation step simply disappears — this model **is** Bengio et al.'s architecture, at the smallest possible context size.

## 2. Why a continuous function should beat a lookup table

Topic 2.2's bigram model is, mathematically, a giant lookup table: one independent probability distribution `P(· | w_{t-1})` per distinct value of `w_{t-1}`, with no relationship enforced between the distribution for `w_{t-1} = \text{"good"}` and the distribution for `w_{t-1} = \text{"great"}` — even though those two words behave almost identically as a sentence's previous word. Every one of those distributions has to be estimated from scratch, which is exactly why sparsity (Topic 2.2, section 2) is so damaging: the table has no way to share evidence between similar contexts.

The neural model's hidden representation `h` is a *continuous function* of the embedding `e`. If `vec("good")` and `vec("great")` end up close together in embedding space — and Topic 1.1 gives every reason to expect that they would, since they occur in near-identical contexts — then `h` and therefore the predicted next-word distribution will *automatically* be similar for both, **without ever having observed every specific `("great", next-word)` pair that was observed for `"good"`**. This is generalisation by interpolation in a continuous space, replacing the all-or-nothing exact matching that makes a count table fail the instant it sees a combination it didn't memorise. `explanation.md` reports whether this theoretical advantage actually shows up as a real perplexity improvement on this specific dataset — it should not be taken on faith.

## 3. The cost this model doesn't try to hide: a softmax over the whole vocabulary

Every forward pass of this model computes a score for **every one of 5,003 vocabulary words**, for every single training example, just to find out which one was correct. This is computationally the most expensive part of the entire model by a wide margin (the hidden layer is tiny — 32 to 128 units — compared to the 5,000-way output layer), and the cost grows linearly with vocabulary size. This is not a minor implementation detail: it is precisely the bottleneck that motivated **hierarchical softmax** and **negative sampling** (Module III, Topic 3.2) — both of which exist specifically to avoid ever computing a score for the entire vocabulary on every training step. Building the plain, full-softmax version first, here, and feeling its actual computational cost directly (this topic's training had to be scaled down to fit available CPU time — see `explanation.md`), is the most honest way to understand why those later techniques were invented at all.

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR.
2. Mikolov, T., Karafiát, M., Burget, L., Černocký, J., & Khudanpur, S. (2010). *Recurrent Neural Network Based Language Model.* Interspeech. (The natural next step beyond a fixed bigram context window — full history via a recurrent hidden state, previewing Module IV's RNNs.)
3. Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL. (Tying the input embedding and output projection's weights — a well-known extension not implemented here, but worth knowing: it roughly halves this model's parameter count.)
