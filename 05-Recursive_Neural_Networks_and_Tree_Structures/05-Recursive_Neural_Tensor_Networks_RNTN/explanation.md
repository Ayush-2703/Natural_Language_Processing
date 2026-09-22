# 5.5 — Code Explanation

## The tensor product term
```python
self.Vt = nn.Parameter(torch.randn(d, 2*d, 2*d) * 0.01)  # shape: (d, 2d, 2d)

def compose(self, l, r):
    c = torch.cat([l, r], -1)           # shape: (2d,)
    t = torch.einsum('i,jik,k->j', c, self.Vt, c)  # shape: (d,)
    return torch.tanh(t + self.W(c))
```
`torch.einsum('i,jik,k->j', c, Vt, c)` computes, for each output dimension `j`: the scalar `cᵀ Vt[j] c`. This is a **bilinear form** — a quadratic function of the input — unlike `self.W(c)` which is purely linear. The quadratic term lets the RNTN model interactions like "not × good → bad_sentiment" that require the product of two input values, not just their sum.

## Parameter count
- TNN: 50,210 parameters (embedding + W + classifier)
- RNTN: 181,282 parameters (+131,072 from `Vt = d × 2d × 2d = 32 × 64 × 64`)

The tensor `Vt` is `O(d³)` parameters vs. `O(d²)` for the linear weight — this cubic scaling is why the RNTN is expensive and why Socher et al. used `d=25` in their original paper.

## Results: TNN=98.7%, RNTN=100.0%
Both models perform very well on this synthetic task because the sentiment vocabulary is simple and the test/train sets share similar vocabulary. On the real SST (Socher et al., 2013), the RNTN outperforms the TNN by ~10 percentage points (85.4% vs 75.9% fine-grained accuracy), with the gains concentrated on negation and double-negation constructions — exactly what the tensor term is designed for.
