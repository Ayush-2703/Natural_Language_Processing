# 5.4 — Converting Trees to Sequences

## Why convert trees to sequences?

Recursive neural networks require a parse tree at test time — an external parser must first analyse each sentence, and errors in parsing propagate into the model. This dependency motivated a body of work asking whether the structural benefit of recursive processing could be achieved without explicit tree supervision, by converting tree operations into sequence operations.

## Three conversion strategies

**Left-to-right linearisation**: replace every opening bracket with a special token, output the sentence left-to-right. The sequence `( S ( NP the cat ) ( VP sat ) )` becomes `<S> <NP> the cat </NP> <VP> sat </VP> </S>`. An LSTM processing this sequence sees structure implicitly via the bracket tokens.

**Depth-first traversal**: visit every internal node label before its first child, then recurse. This is how NLTK's `tree.treepositions()` works and how Penn Treebank bracketings are written in text files.

**Shift-reduce parsing as a sequence**: represent the parser's action sequence (shift a word, reduce two nodes) as a sequence of discrete operations. A BiLSTM then learns to predict actions directly.

## ATIS and SNIPS: sequence-to-tree in NLU

Modern task-oriented dialogue systems parse user utterances into structured meaning representations — effectively converting a flat sequence into a tree (an intent plus labelled slots). This is sequence-to-tree rather than tree-to-sequence, but the same conversion ideas apply in reverse.

## References

1. Vinyals, O., Kaiser, L., Koo, T., Petrov, S., Sutskever, I., & Hinton, G. (2015). *Grammar as a Foreign Language.* NeurIPS. (Constituency parsing as sequence-to-sequence.)
2. Dyer, C., Kuncoro, A., Ballesteros, M., & Smith, N. A. (2016). *Recurrent Neural Network Grammars.* NAACL.
