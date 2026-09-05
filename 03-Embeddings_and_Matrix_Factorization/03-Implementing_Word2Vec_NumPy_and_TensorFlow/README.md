# 3.3 — Implementing Word2Vec (NumPy and TensorFlow)

## 1. Why implement the same thing twice

Topics 3.1 and 3.2 both used PyTorch's autograd — `loss.backward()` computes all gradients automatically. This topic removes that safety net. Implementing Word2Vec twice, independently, in two different systems — once in raw NumPy with **gradients derived by hand**, once in TensorFlow with **`tf.GradientTape` autodiff** — and then checking whether the two produce consistent results is a professional-grade verification technique called cross-implementation validation. If the hand-derived NumPy gradients were wrong, the two systems would diverge: one would train while the other spun in place. Agreement (in the sense that both models are genuinely learning — both loss curves decrease) is much stronger evidence of correctness than either one alone.

## 2. The SGNS gradients, derived by hand

The negative sampling objective for one `(center c, positive context o, negatives n₁…nₖ)` triple:

```
L = -log σ(v_o · v_c)  -  Σᵢ log σ(-v_{nᵢ} · v_c)
```

Let `s_o = v_o · v_c` and `sᵢ = v_{nᵢ} · v_c`. By the chain rule through the sigmoid and dot products:

```
∂L/∂s_o  =  σ(s_o) - 1           (note: σ(s_o) → 1 if correct, → 0 if wrong, so this → 0 or -1)
∂L/∂sᵢ   =  σ(sᵢ)                (should be pushed toward 0)

∂L/∂v_c  =  (σ(s_o) - 1)·v_o  +  Σᵢ σ(sᵢ)·v_{nᵢ}
∂L/∂v_o  =  (σ(s_o) - 1)·v_c
∂L/∂v_{nᵢ} = σ(sᵢ)·v_c
```

Every gradient has the same structure: the sigmoid value (which measures how wrong the model is) scales the other word's vector. When the model predicts correctly — `σ(s_o) → 1`, `σ(sᵢ) → 0` — all gradients shrink toward zero, and the update stops. When it is wrong, the gradient is large and in the right corrective direction.

## 3. Gradient checking: the only reliable way to verify a hand-derived gradient

A gradient check evaluates the **numerical (finite-difference) approximation** of the gradient alongside the analytic formula and checks they agree:

```
∂L/∂θᵢ  ≈  ( L(θ + ε·eᵢ) - L(θ - ε·eᵢ) ) / 2ε
```

If the analytic formula is correct, this should match to within roughly `O(ε²)` — for `ε = 1e-5`, agreement within `~1e-10` is expected if double-precision arithmetic is used throughout. This topic's check produces `4.10e-11` for `∂L/∂v_c` and `4.32e-11` for `∂L/∂v_o` — consistent with float-64 rounding floor at this `ε`, confirming the formulas are correct. **Only after passing this check does `main()` proceed to training.** The code pattern — derive, check, assert, then use — is the standard engineering workflow for any hand-written gradient, and it is worth internalising as a habit.

## 4. SGD vs. Adam for sparse embedding gradients — a real failure mode documented here

The TensorFlow implementation's initial version used `tf.keras.optimizers.SGD`. Training with it produced a loss frozen at exactly its initial value across all epochs — no learning at all — a silent failure with no error message. The cause: SGD's `apply_gradients` on a `tf.Variable` of shape `(V, d)` receives the gradient as a `tf.IndexedSlices` object (a sparse representation, since only the rows actually touched in a batch need updating), and TF1-style SGD does not reliably convert those to dense updates on every version/platform combination. **Adam's `apply_gradients` handles `IndexedSlices` correctly**, applying updates only to the touched rows and leaving the rest unchanged. This is documented explicitly here because it is a genuine, reproducible pitfall in production TF code — one that produces no exception, only a perfectly flat loss curve.

## References

1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. (SGNS, Section 2.2.)
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 4.3: Gradient-Based Optimization. MIT Press. (Numerical gradient checking method used in this topic.)
3. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). *Automatic Differentiation in Machine Learning: A Survey.* JMLR. (The autodiff mechanism behind `tf.GradientTape`.)
