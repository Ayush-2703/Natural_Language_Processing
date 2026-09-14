# 4.1 — Introduction to Parts-of-Speech (POS) Tagging

## 1. What POS tagging is and why it matters

**Part-of-speech tagging** assigns a grammatical category to every token in a sentence. This topic uses the **Penn Treebank tagset** (45 tags in full, 43 after cleaning trace tokens from the NLTK sample): `NN` (singular noun), `NNS` (plural noun), `NNP` (proper noun), `VBD` (past-tense verb), `JJ` (adjective), `IN` (preposition), and so on down to fine-grained distinctions like `VBZ` (third-person singular present verb) vs. `VBP` (non-third-person present).

POS tags are useful downstream because they resolve a great deal of lexical ambiguity without requiring any deep semantic understanding: the word "book" can be a noun ("the book") or a verb ("book a flight"), and knowing which one it is in a given sentence constrains parsing, coreference resolution, named-entity recognition, and many other tasks. Collobert & Weston (2011) — the paper this course's Phase 4 follows in spirit — listed POS tagging first among the four NLP tasks their unified architecture demonstrated, for exactly this reason: it is the simplest, cleanest sequence-labelling benchmark.

## 2. The sequence labelling problem

POS tagging is the first example in this course of a **sequence-to-sequence labelling** task: input is a sequence of tokens `x_1, ..., x_n` and output is an equally-long sequence of labels `y_1, ..., y_n`. Every token gets exactly one label, and the label boundaries align with the token boundaries. This is structurally different from the tasks in Phases 1–3, where the output was a single number (perplexity), a single vector (a word embedding), or a single class (sentiment). Sequence labelling requires the model to make `n` correlated decisions simultaneously — correlated because grammatical categories of adjacent words are strongly dependent on each other (a determiner like "the" is almost always followed by either a noun or an adjective, almost never by a verb or another determiner), which is exactly the structure the HMM in Topic 4.2 exploits explicitly and the BiLSTM in Topic 4.3 learns implicitly.

## 3. The Most-Frequent-Tag baseline and what it tells you

The simplest non-trivial tagger: look up every word's most common tag in training data; for unknown words, predict the globally most common tag (NN here). This achieves **87.3% token accuracy on the test set** — a number that is both impressive and deceptive. It is impressive because English is highly ambiguous lexically, yet most individual words in context actually *do* take their most common tag most of the time. It is deceptive because sentence accuracy is only **10.7%** — most complete sentences contain at least one wrong tag. The MFT tagger is also completely incapable of reasoning about context: it would tag both "flies" in "time flies like an arrow" and "fruit flies like a banana" identically, missing the entirely different role the word plays in each. Topic 4.2's HMM and Topic 4.3's BiLSTM exist to capture exactly that contextual sensitivity.

## References

1. Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). *Building a Large Annotated Corpus of English: The Penn Treebank.* Computational Linguistics. (The corpus this topic trains on.)
2. Collobert, R., et al. (2011). *Natural Language Processing (Almost) from Scratch.* JMLR. (POS tagging as the canonical sequence-labelling benchmark.)
