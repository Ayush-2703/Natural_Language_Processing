# 5.2 — Parsing Sentences as Trees and Data Description for RNNs

## 1. Constituency vs. dependency parse trees

**Constituency trees** (Penn Treebank format, used throughout Phase 5) represent sentences as nested phrase structure: every subtree spans a contiguous sequence of words and has a phrasal label (NP, VP, S, PP, ...). The root spans the full sentence.

**Dependency trees** represent grammatical relations between individual word pairs: each arc points from head word to dependent word, labelled with the grammatical function (subject, object, modifier). For Recursive Neural Networks, constituency trees are the natural choice because every internal node has a vector representation that corresponds to a meaningful linguistic constituent.

## 2. The Stanford Sentiment Treebank (SST)

Socher et al.'s (2013) RNTN paper introduced a dataset purpose-built for recursive sentiment models: every node in every parse tree of 11,855 sentences was manually labelled with a sentiment rating on a 5-point scale (very negative → very positive). This is what makes the RNTN trainable in a principled way — not just the root's label, but every phrasal constituent's sentiment is supervised. The SST is the standard benchmark for composition-sensitive sentiment analysis; it is used in Topic 5.5.

## 3. Converting Penn Treebank trees for use with RvNNs

Penn Treebank trees contain function tags (NP-SBJ, PP-CLR), empty nodes (-NONE-), and non-branching chains that need cleaning before an RvNN can process them. Three standard preprocessing steps:
- Strip function tags from node labels (NP-SBJ → NP)
- Remove traces and empty nodes (-NONE-)
- Collapse unary chains (an NP node with a single NP child → remove the intermediate node)

## References

1. Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). *Building a Large Annotated Corpus of English: The Penn Treebank.* Computational Linguistics.
2. Socher, R., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP.
