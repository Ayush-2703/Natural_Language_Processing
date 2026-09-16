# 4.3 — Neural Networks and RNNs applied to POS Tagging

## 1. Why an RNN instead of an MLP

The MFT tagger makes one prediction per token entirely independently — no information from neighbouring tokens is used. The HMM introduces left-to-right context via its Markov transition model, but each token's prediction still depends only on the immediately preceding tag, not on the full surrounding sequence. A **Recurrent Neural Network** breaks both of these constraints: at each position, the hidden state carries a summary of *everything seen so far* in the sentence, and a **bidirectional** LSTM additionally makes every token's prediction depend on the *entire* surrounding context — words to the right as well as to the left.

## 2. BiLSTM architecture

```
token embeddings → BiLSTM (forward + backward) → concatenate hidden states → Linear → tag logits
```

For a sentence of `n` tokens, the embedding layer produces `(n, d)`, the bidirectional LSTM produces `(n, 2H)` (the forward and backward hidden states at each position, concatenated), and the final linear layer projects each position's `2H`-dimensional vector to a score over the `T=43` tags. Softmax (applied implicitly by `nn.CrossEntropyLoss`) converts those scores to probabilities; the predicted tag is the `argmax`. Unlike the HMM's Viterbi decoder, there is no dynamic programming step — predictions at each position are independent given the BiLSTM's hidden states, trading exact MAP decoding for the ability to learn richer, distributed context representations.

## 3. Packed sequences and padding

Variable-length sentences in a batch are handled with `pack_padded_sequence` / `pad_packed_sequence` — PyTorch's idiom for telling the LSTM "don't process the PAD tokens, and don't let their zeros pollute the hidden state." The corresponding loss mask `(words != PAD_IDX)` zeroes out the loss contribution from PAD positions so they don't contribute spurious gradient signal.

## 4. Results in context

10 epochs yields test accuracy **0.834** — below the HMM's 0.932. This is not a paradox; it is a training-budget result. The loss curve is still declining at epoch 10, and the gap between val accuracy (0.823) and both baselines tells the story directly: the model is still learning representations from scratch. At ~50 epochs (well documented in the literature) a BiLSTM on Penn Treebank reaches ~97%, well above the HMM. `implementation.py`'s `EPOCHS=20` setting in the full script, given the sandbox's single-CPU constraint, gives a trajectory that is still clearly pointing in the right direction — every epoch improves, no overfitting yet.

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
2. Collobert, R., et al. (2011). *Natural Language Processing (Almost) from Scratch.* JMLR.
3. Lample, G., et al. (2016). *Neural Architectures for Named Entity Recognition.* NAACL.
