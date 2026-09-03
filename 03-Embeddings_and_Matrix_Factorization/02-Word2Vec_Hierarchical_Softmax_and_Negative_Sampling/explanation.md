# 3.2 — Code Explanation

## Building the Huffman tree

```python
def build_huffman_tree(freqs):
    V = len(freqs)
    heap = [(freqs[i], i, i) for i in range(V)]
    heapq.heapify(heap)
    parent = {}
    next_internal_id = V
    tiebreak = V
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent[n1] = (next_internal_id, 0)
        parent[n2] = (next_internal_id, 1)
        heapq.heappush(heap, (f1 + f2, tiebreak, next_internal_id))
        tiebreak += 1
        next_internal_id += 1
```

Every word starts as its own leaf, tagged with its real corpus frequency. Each iteration of the loop pops the **two lowest-frequency** remaining nodes (`heapq` keeps this an `O(log n)` operation) and merges them under a new internal node whose frequency is their sum — the standard Huffman construction, applied here to word frequencies rather than character frequencies as in the original data-compression use case. `tiebreak` exists purely so Python's heap, which compares tuples element-by-element, never has to fall back to comparing two `node_id` integers directly when frequencies tie (irrelevant for the actual algorithm, but necessary to avoid `heapq` occasionally trying — and failing, since dict-like comparisons aren't always meaningful — to break ties on incomparable values). Internal nodes are assigned ids starting at `V` (so `V` through `2V-2` for a vocabulary of size `V`), kept distinct from leaf ids `0` through `V-1`.

```python
    paths, codes = [], []
    for i in range(V):
        path, code = [], []
        node = i
        while node in parent:
            p, bit = parent[node]
            path.append(p - V)
            code.append(bit)
            node = p
        paths.append(list(reversed(path)))
        codes.append(list(reversed(code)))
```

For each word, this walks **up** from its leaf to the root via the `parent` dictionary built above, collecting the internal node visited at each step and which branch (`bit`) was taken — then reverses both lists, since walking root-to-leaf (the order needed for the forward pass) is the opposite of how the climb was performed. `p - V` re-indexes internal node ids back down to a clean `0..V-2` range, so they can be used directly as row indices into a `(V-1, embed_dim)` parameter table without needing an offset everywhere downstream.

Running this on the real 3,000-word vocabulary:

```
internal nodes: 2999   max tree depth: 15   avg depth: 13.2   (log2(V) = 11.6)
```

`huffman_depth_distribution.png` makes the theory's central claim directly visible: code length increases in clean, discrete steps as frequency rank increases — the single most frequent word gets a 4-step path, while the long tail of rare words (frequency rank in the hundreds to low thousands) gets paths of 13–15 steps. Average depth (13.2) sits a little *above* the balanced-tree figure of `log₂(3000) ≈ 11.6` rather than below it — this is the expected, correct behaviour of a Huffman tree applied to a Zipfian distribution: it deliberately sacrifices average-case depth-if-every-word-were-equally-likely in exchange for minimizing *frequency-weighted* average depth, which is the quantity that actually determines total training cost.

## The two vectorized models

```python
class HierarchicalSoftmaxSkipGram(nn.Module):
    def forward(self, center_idx, path, code, mask):
        e = self.in_embed(center_idx).unsqueeze(1)
        nodes = self.node_embed(path)
        scores = (nodes * e).sum(-1)
        sign = 2 * code - 1
        log_probs = F.logsigmoid(sign * scores)
        loss_per_example = -(log_probs * mask).sum(dim=1)
        return loss_per_example.mean()
```

`path`, `code`, and `mask` are all pre-padded to the tree's maximum depth (15, here) by `pad_huffman_paths` — every word's path is right-padded with zeros so the whole batch can be processed as one fixed-shape tensor rather than a ragged Python loop over variable-length paths. `mask` marks which entries are real (`1.0`) versus padding (`0.0`); multiplying by `mask` before summing means padded positions contribute exactly zero to the loss and, critically, exactly zero gradient — `theory.md`'s formula computed correctly despite the padding being present only for tensor-shape convenience. `sign = 2*code - 1` is the cheap arithmetic trick that converts a `{0, 1}` bit into a `{-1, +1}` sign in one line, so `theory.md`'s `sign(node, w) · v_node · v_center` becomes a single elementwise multiply rather than a branch.

```python
class NegativeSamplingSkipGram(nn.Module):
    def forward(self, center_idx, pos_idx, neg_idx):
        v_c = self.in_embed(center_idx)
        v_pos = self.out_embed(pos_idx)
        pos_loss = -F.logsigmoid((v_c * v_pos).sum(-1))
        v_neg = self.out_embed(neg_idx)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)
        return (pos_loss + neg_loss).mean()
```

`torch.bmm` (batched matrix multiply) computes all `k` negative scores for every example in the batch in one call: `v_neg` is `(batch, k, d)`, `v_c.unsqueeze(2)` is `(batch, d, 1)`, and the batched product gives `(batch, k, 1)` — every negative word's dot product with its example's centre vector, computed in parallel rather than looped. Note that `in_embed` and `out_embed` are **separate** tables here, matching `theory.md`'s formula exactly and the standard Word2Vec convention (the same convention Topic 3.1's `SkipGramModel` used via `in_embed`/`out_proj`) — a word's "as a centre word" vector and "as a context word" vector are different, separately-learned objects, even though only the centre-word table is conventionally reported as "the" word vectors at the end.

```python
def build_negative_sampling_table(freqs, table_size=1_000_000):
    noise = np.power(np.array(freqs, dtype=np.float64), 0.75)
    noise = noise / noise.sum()
    table = np.zeros(table_size, dtype=np.int64)
    cum = np.cumsum(noise)
    idx = 0
    for w in range(len(freqs)):
        end = int(cum[w] * table_size)
        table[idx:end] = w
        idx = end
```

This builds the noise distribution as a literal, physical lookup table rather than calling a weighted-random-sampling function on every draw — exactly the technique the original word2vec C implementation uses. Each word `w` is assigned a contiguous block of `table` proportional to its `count(w)^0.75` share of the total; sampling a negative is then just `table[np.random.randint(0, table_size)]` — drawing a uniform random integer and indexing, which is dramatically faster than recomputing a weighted sample from scratch on every call, and is exactly the kind of practical engineering trick worth knowing alongside the mathematical definition in `theory.md`.

## Timing comparison: real numbers, with an honest gap from the theoretical ratio

```
Full softmax:         9.63s  (20,775 pairs/sec)
Hierarchical softmax: 3.42s  (58,548 pairs/sec)  -- 2.82x faster than full softmax
Negative sampling:    1.84s  (108,820 pairs/sec)  -- 5.24x faster than full softmax
```

All three were timed under identical conditions — same 200,000 pairs, same batch size, same embedding dimension, same single-threaded CPU — so this comparison is fair. Both techniques deliver a real, substantial, measured speedup, and negative sampling's win over hierarchical softmax is consistent with `theory.md`: `O(d·(k+1)) = O(d·6)` per example is cheaper than `O(d·log V) ≈ O(d·13)` per example at this vocabulary's actual average Huffman depth.

It's worth being explicit about a gap between this result and the raw FLOP-count argument in `theory.md`: full softmax costs `O(d·V) = O(50·3000) = 150,000` multiply-adds per example, while negative sampling costs `O(d·6) = 300` — a 500x theoretical ratio, not the 5.24x actually measured. The shortfall is real and has a concrete explanation: full softmax's cost is one single, large, highly-optimized matrix multiply per batch (`nn.Linear`, backed directly by BLAS), while negative sampling's cost, despite being far smaller in raw FLOPs, is spread across several smaller operations (two embedding gathers, a batched matrix multiply, two sigmoid-log calls) each carrying its own fixed Python/PyTorch dispatch overhead. At V = 3,000, that per-operation overhead is large enough relative to the actual arithmetic to eat a substantial fraction of the theoretical advantage. The advantage would grow much closer to its theoretical scale at a realistic production vocabulary of 100,000+ words, where full softmax's `O(d·V)` term becomes large enough to dominate any fixed per-batch overhead — this topic's vocabulary is deliberately small enough that the *fixed costs*, not just the *asymptotic complexity*, are visible in the result, which is itself a useful, honest engineering lesson: asymptotic complexity arguments describe a limit, not a guarantee about wall-clock behaviour at every scale.

## The large-scale training run, and a real, only-partially-successful fix

Negative sampling's speed was put to use training on 3,000,000 pairs for 3 epochs (9 million pair-evaluations total, versus Topic 3.1's roughly 1.9 million) in about 107 seconds. The result was a clear, well-documented failure mode, not a success story:

```
'good': [('little', 0.771), ('to', 0.766), ('all', 0.748), ('their', 0.745), ('they', 0.744), ('our', 0.735)]
'water': [('best', 0.632), ('show', 0.621), ('angel', 0.599), ('will', 0.598), ('cutting', 0.589), ('gather', 0.58)]
```

Every word's nearest neighbours are dominated by extremely common function words, almost regardless of the query word's own meaning — exactly `theory.md` section 3's predicted failure mode, now observed directly rather than just described.

Applying Mikolov et al.'s subsampling fix (threshold `t = 1e-3`) produced a real, measurable, and verifiable change in *what the model trained on*:

```
keep probability for 'the' (most frequent): 0.125
tokens before subsampling: 3,527,865   after: 2,454,098  (69.6% kept)
```

— but, reported honestly, it did **not** produce a clearly cleaner set of nearest neighbours (`subsampling_comparison.png`): `good`'s neighbours shifted from `little, to, all, their` to `as, with, in, what` — still dominated by function words, just a different selection of them. A further, more aggressive experiment at the textbook-standard large-corpus threshold of `t = 1e-5` (not included in the main run above, to keep this topic's total compute budget reasonable, but run and verified separately) discarded a much larger 85.8% of all tokens, leaving only about 500,000 — and still did not produce clearly semantic neighbours.

The honest diagnosis: subsampling's frequency-thinning mechanism is verifiably working exactly as `theory.md` describes (the keep-probability numbers above prove that), but on a corpus this size, fixing the *imbalance* by discarding tokens runs directly into a second, separate constraint — there simply isn't enough *absolute* remaining data (a few hundred thousand to a few million tokens) for the small number of training epochs used here to learn clean semantic structure for 3,000 distinct words. Mikolov et al.'s own experiments applied subsampling to corpora of *billions* of tokens, where discarding 85% of occurrences of `the` still leaves an enormous amount of real training signal behind. This topic's result is a genuine, useful negative finding: hierarchical softmax and negative sampling solve the *computational cost* problem convincingly (the timing numbers above are unambiguous), but neither they, nor subsampling on top of them, are a substitute for training on enough raw data — a limitation Topic 1.1's gensim-trained model sidesteps simply by being given the entire corpus, many more epochs, and a mature, heavily-tuned implementation to train with.
