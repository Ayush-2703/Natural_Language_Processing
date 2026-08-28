# 1.5 — Basics of Computational Frameworks: TensorFlow and Theano

Every other topic in this Phase calls `.backward()` or relies on a framework to compute gradients without ever asking *how*. This topic is about that "how" — the computational graph abstraction that both TensorFlow and Theano are built on, and that every other deep learning framework (PyTorch included) implements some version of.

## 1. What a computational graph actually is

A computational graph represents a calculation as a directed graph: **nodes are operations** (matrix multiply, addition, sigmoid, ...), and **edges are the tensors flowing between them**. The diagram below is the exact graph this topic's logistic regression builds — every box is a real line of code in `implementation.py`.

![A computational graph](images/computational_graph_concept.png)

Two passes happen over this graph:

- **Forward pass**: evaluate every node in dependency order — feed `X` and `W` into `matmul`, add `b`, apply `sigmoid`, compare against `y` to get a scalar `loss`. This is ordinary function evaluation.
- **Backward pass**: compute `∂loss/∂W` and `∂loss/∂b` by walking the graph in reverse, applying the chain rule at every node. This is **reverse-mode automatic differentiation**, and it is the single most important algorithm in this entire course — everything from Topic 1.5's logistic regression to Phase 5's recursive networks is trained by some instance of it.

### Why reverse mode, specifically

For a function with `n` inputs (here: every entry of `W` and `b` — potentially millions of parameters in a real network) and `m` outputs (here: one scalar loss), there are two ways to mechanically apply the chain rule across a computational graph: **forward-mode**, which propagates derivatives from inputs to outputs and costs roughly `O(n)` graph traversals (one per input you want a derivative with respect to), and **reverse-mode**, which propagates derivatives from outputs back to inputs and costs roughly `O(m)` traversals (one per output). Since `m = 1` (a single scalar loss) and `n` is enormous in any real model, reverse mode computes *every* parameter's gradient in a single backward traversal, while forward mode would need millions of forward traversals to do the same job. This asymmetry — one output, many parameters — is the entire reason backpropagation (reverse-mode autodiff specialised to layered networks) became the default, and it's worth understanding once explicitly rather than treating `.backward()` / `tape.gradient()` as magic.

## 2. Theano: where explicit symbolic graphs in Python came from

**Theano** (Bergstra et al., 2010, developed at the University of Montreal) was the first widely-used framework to bring this idea to deep learning in Python: you wrote **symbolic expressions** over abstract placeholder tensors, Theano built a computational graph from those expressions, optimised and compiled that graph (it could even generate C or CUDA code from it), and only then could you actually run numbers through it via `theano.function`. This was a strict, two-phase workflow — *define the graph, then execute it* — with no way to interleave the two the way ordinary Python code does. It was also, for its time, remarkably fast and was the engine behind a great deal of foundational deep learning research in the early 2010s. Theano's official development ceased in 2017; the ideas it established (symbolic graphs, automatic differentiation as a first-class framework feature, compiled execution) did not disappear — they became the working assumptions every subsequent framework, including the one used everywhere else in this repository, was built on.

## 3. TensorFlow: the same idea, twice, on purpose

**TensorFlow 1.x** adopted Theano's exact two-phase philosophy almost without modification: build a static graph using placeholders, then run actual data through it inside a `Session`. This was powerful but famously unergonomic — a graph couldn't easily contain ordinary Python control flow (`if`, `for` over data-dependent conditions), and debugging meant inspecting an opaque graph object rather than stepping through familiar Python.

**TensorFlow 2.x** flips the default to **eager execution**: every operation runs immediately, in plain Python, the moment it's called — `tf.matmul(X, W)` returns an actual tensor of actual numbers right there, the same way NumPy does. This is what `train_step_eager` in `implementation.py` uses, and it is far easier to write and debug.

But eager mode pays a real cost: every individual operation is dispatched and executed one at a time by the Python interpreter, with no opportunity to fuse operations or skip redundant work across calls. **`@tf.function`** is TensorFlow 2.x's bridge back to graph-mode performance: decorating a Python function causes TensorFlow to **trace** it once — running it symbolically to record exactly the graph of operations it performs — and compile that trace into a static graph (via `tf.Graph` and, internally, mechanisms with direct lineage back to TF1's graph runtime — and, one level further back, to Theano's). Every subsequent call replays the compiled graph directly, skipping Python-level dispatch overhead entirely. This topic's code measures the difference directly rather than asserting it: see `explanation.md` for the actual wall-clock numbers.

## 4. `tf.GradientTape`: reverse-mode autodiff, made explicit

```python
with tf.GradientTape() as tape:
    loss = model.loss(X, y)
grads = tape.gradient(loss, model.trainable_variables())
```

Inside the `with` block, every operation that touches a watched value (by default, any `tf.Variable`) is **recorded** onto the tape — this *is* the construction of the computational graph in section 1's diagram, happening implicitly as the forward pass runs. `tape.gradient(loss, variables)` then performs the backward pass: it walks the recorded operations in reverse, applies the chain rule at each one, and returns `∂loss/∂variable` for each requested variable. This single mechanism is what makes `W` and `b` improve every training step in this topic's code, and a structurally identical mechanism (`loss.backward()` and `.grad` populating on every `requires_grad=True` tensor) is what trains every PyTorch model elsewhere in this course.

## References

1. Bergstra, J., Breuleux, O., Bastien, F., Lamblin, P., Pascanu, R., Desjardins, G., Turian, J., Warde-Farley, D., & Bengio, Y. (2010). *Theano: A CPU and GPU Math Compiler in Python.* Proceedings of the 9th Python in Science Conference (SciPy).
2. Abadi, M., et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning.* OSDI.
3. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). *Automatic Differentiation in Machine Learning: A Survey.* Journal of Machine Learning Research, 18, 1–43. (The forward- vs. reverse-mode complexity argument in section 1 is treated rigorously here.)
4. Bengio, Y. (2009). *Learning Deep Architectures for AI.* Foundations and Trends in Machine Learning. (Bengio's lab at Montreal built Theano specifically to support this kind of research at scale — direct institutional link to Topic 1.1's reference.)
