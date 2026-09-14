# 4.1 — Code Explanation

## The shared dataset cache
```python
with open(DATASET_PATH, "wb") as f:
    pickle.dump(dataset, f)
```
Everything Topics 4.2 and 4.3 need — the 80/10/10 split, vocabulary, tagset, and word-to-index mapping — is pickled once here and loaded by both downstream topics. The key design choice: the vocabulary is built from **training data only**, so no test-set words can influence which words are "known."

## MFT tagger
```python
self.word_tag[w] = counts.most_common(1)[0][0]
self.default_tag = global_tag_counts.most_common(1)[0][0]
```
`word_tag` stores the single most-frequent tag per word (lowercased). `default_tag` is NN — the globally most common tag — applied to any OOV word. This gives 87.3% token accuracy but only 10.7% sentence accuracy, establishing the floor both HMM and BiLSTM must beat.

## Transition heatmap insight
`transition_heatmap.png` shows TO→VB at ≈0.63 and DT→NN at ≈0.50. These are the exact structural patterns the HMM in Topic 4.2 exploits as explicit transition probabilities, and that the BiLSTM in 4.3 learns implicitly from labelled examples.
