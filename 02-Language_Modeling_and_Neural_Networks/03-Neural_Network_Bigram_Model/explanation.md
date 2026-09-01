# 2.3 — Code Explanation

## The model, in five lines

```python
class NeuralBigramLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.Tanh()

    def forward(self, prev_word_idx):
        e = self.embedding(prev_word_idx)
        h = self.activation(self.hidden(e))
        logits = self.output(h)
        return logits
```

This is `theory.md` section 1's diagram with no simplification or omission — every layer named there is a line here. `nn.CrossEntropyLoss` (used in the training loop, not shown in the model itself) applies `log_softmax` internally for numerical stability before computing the negative log-likelihood, which is why `forward` deliberately returns raw, unnormalised `logits` rather than calling `softmax` itself — calling softmax twice (once here, once inside the loss) would both waste computation and reintroduce the exact numerical instability `log_softmax`'s fused implementation exists to avoid.

## Reusing Topics 2.1 and 2.2 without copy-pasting their logic

```python
spec = importlib.util.spec_from_file_location(
    "topic_2_2_impl", os.path.join(HERE, "..", "2.2-Bigrams-and-Language-Constructs", "implementation.py")
)
topic_2_2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topic_2_2)
```

Topics 2.1 and 2.2 each have a file literally named `implementation.py`. A plain `from implementation import X` after adding both directories to `sys.path` would silently import whichever one Python happened to cache first under that module name and quietly reuse it for the second import too — a real, easy-to-miss bug, and a different one from the unigram-counting bug Topic 2.2 actually hit. `importlib.util.spec_from_file_location` sidesteps this entirely by loading Topic 2.2's file under an explicit, unique module name (`"topic_2_2_impl"`) regardless of what's already been imported as `"implementation"` for Topic 2.1's `UnigramModel`. The result: `compute_baselines()` recomputes all three of Topic 2.1/2.2's numbers from the original counting logic rather than hard-coding them, so they cannot silently drift out of sync if either topic's code changes later — and the printed values (336.4, 417.4, 255.7) match those topics exactly, confirming the reuse is correct.

## A real engineering tradeoff, made explicit rather than hidden

```python
MAX_TRAIN_PAIRS = 200_000
...
print(f"\nTraining pairs: {len(train_prev):,} (subsampled from {len(train_prev_full):,}) ...")
```

Topics 2.1 and 2.2's counting models are a single pass over roughly 900,000 training bigrams — fast regardless of dataset size, because counting doesn't iterate. Gradient descent needs many passes over whatever data it's given, and a full-vocabulary softmax (`theory.md` section 3) makes each pass expensive. On this environment's single CPU core, training on the full training set for enough epochs to converge was measured to take on the order of tens of minutes — too slow to iterate on sensibly. The honest fix used here is to say so directly and **subsample to 200,000 training pairs** (about a quarter of what's available), which is still a substantial, real sample, not a toy example. On a machine with multiple cores or a GPU, `MAX_TRAIN_PAIRS` is a single number to change to use all the data.

This means the comparison that follows has a real asymmetry worth being honest about: the neural model is trained on **less** data than the classical models it's compared against. The fact that it still wins (next section) is, if anything, a *stronger* result than if it had been given the full dataset and still only tied.

## Actual results

```
Unigram (2.1)                   : 336.4
Bigram + Laplace (2.2)          : 417.4
Bigram + Interpolation (2.2)    : 255.7
Neural bigram (2.3)             : 240.7
```

`final_comparison.png` puts all four numbers from this Phase side by side. The neural bigram model achieves the lowest test perplexity of anything built in Phase 2 — beating even the carefully-tuned interpolation model, despite training on a quarter of the data available to it. This is `theory.md` section 2's generalisation argument showing up as an actual number rather than staying a plausible-sounding claim: a continuous function over a learned embedding space evidently extracts more usable signal per training example than a count table can, on this task.

`training_curves.png` shows healthy, fast convergence over the 5 epochs trained, with validation loss tracking training loss closely through epoch 2 and then beginning to drift very slightly apart — an early, mild signal of overfitting starting to set in, not yet a problem at this scale but exactly the kind of curve that would justify stopping early or adding regularisation if training were pushed further.

The qualitative top-5 predictions are a useful sanity check independent of the perplexity number:

```
P(next | 'he') top-5:  was, had, <unk>, <eos>, of
P(next | 'of') top-5:  the, <unk>, his, this, their
```

"He" is overwhelmingly followed by a past-tense verb in real English, and "was"/"had" leading the list is exactly that. "Of" is very often followed by "the" or a possessive determiner, and the model's top prediction (`the`, at a relatively high 0.257 probability) and its other top candidates (`his`, `this`, `their`) reflect that directly — the same kind of preposition-to-determiner pattern Topic 2.2's bigram heatmap showed explicitly, now recovered by a model that was never given a count table to look it up in.
