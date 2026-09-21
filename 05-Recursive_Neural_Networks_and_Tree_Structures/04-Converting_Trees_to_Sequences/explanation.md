# 5.4 — Code Explanation

## Linearisation with bracket tokens
```python
def to_lin(t):
    if isinstance(t, str): return [t]
    l = t.label().split('-')[0]
    toks = [f'({l}']
    for c in t: toks.extend(to_lin(c))
    toks.append(')'); return toks
```
Every internal node becomes two tokens: an opening `(NP` and a closing `)`. A sentence like "The cat sat" under an NP becomes `(NP The cat )`. The linearised sequence is longer (more bracket tokens) but encodes the full tree structure — an LSTM processing this sequence has access to structural boundaries that the plain word sequence hides.

## Results: Words=99.1%, Linearised=99.5%
The synthetic sentiment task is dominated by keyword matching (positive/negative words), so structural information makes only a small difference. On more compositionally demanding tasks (negation scope, intensifiers) the gap would be larger — Vinyals et al. (2015) show the linearised approach achieves competitive constituency parsing accuracy, demonstrating the approach is viable at scale.
