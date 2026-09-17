# 4.4 — Named Entity Recognition Basics and NER utilizing RNNs

## 1. The BIO annotation scheme

NER assigns typed labels to **spans** of text — "New York" is a LOC, "Apple Inc." is an ORG, "Barack Obama" is a PER — rather than to individual tokens. Converting a span-labelling task into a token-labelling task (so it can be handled by the same sequence models as POS tagging) requires the **BIO** scheme: the first token of an entity gets a `B-TYPE` tag (B = Beginning), subsequent tokens of the same entity get `I-TYPE` (I = Inside), and non-entity tokens get `O`. The CoNLL-2002 corpus used here has four entity types: LOC, ORG, PER, MISC — giving 9 distinct tags total.

## 2. Why token accuracy is the wrong metric for NER

~87% of all tokens in CoNLL-2002's training data are tagged `O`. A degenerate model that outputs `O` for every token would achieve 87% token accuracy while correctly detecting zero entities — an F1 of 0. The correct metric is **entity-level F1**: count exact entity span matches (both start, end, and type must match), then compute:

```
Precision = TP / (TP + FP)    Recall = TP / (TP + FN)    F1 = 2·P·R / (P+R)
```

where TP is a predicted entity that exactly matches a gold entity, FP is a spurious prediction, and FN is a gold entity the model missed. A partial match (getting the type right but the span boundary wrong by one token) counts as both a FP and an FN — this strictness reflects how NER is actually used in downstream systems.

## 3. Class imbalance and weighted loss

The 87% O-token dominance is class imbalance in a concrete form. Left unaddressed, the model learns to conservatively predict O everywhere and achieves high token accuracy but negligible entity recall. The implementation applies **class-weighted cross-entropy**: `O` tokens get weight 0.3 in the loss, entity tags get weight 1.0, so the loss function values each entity tag's gradient signal more than three times as much as an O token's. This trades a few percentage points of token accuracy for substantially higher recall on rare entity classes.

## References

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.* CoNLL.
2. Tjong Kim Sang, E. F. (2002). *Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition.* CoNLL. (The Spanish dataset used here.)
3. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). *Neural Architectures for Named Entity Recognition.* NAACL. (BiLSTM-CRF — the natural next step from this topic's greedy BiLSTM.)
