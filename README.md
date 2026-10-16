# Does DoRA Change the Value of Spectral Initialization under Per-Condition Tuning?

Adrian Adam Kalisz, Bryan Le Shook, Myeongsun Choi, Suraj Chandra Samudrala

## Getting started

```bash
uv sync --locked --extra cpu --extra models --extra analysis
.venv/bin/python -m pytest
```

See [setup](docs/SETUP.md) for ICE installation and optional notebook dependencies,
and [the experiment guide](docs/EXPERIMENT.md) for the PEFT implementation and commands.
The [test plan](docs/TEST_PLAN.md) defines tuning and confirmation; the
[background](docs/BACKGROUND.md) explains the mathematics. Execution-specific
measurements are recorded in [reports](reports).
Use the [study guide and checklist](docs/RUN_STUDY.md) for execution and milestones.

## 1. Problem Statement

LoRA is usually initialized randomly, while spectral methods initialize from singular directions of the pretrained weight matrix. However, spectral initialization changes both the update direction and potentially its scale, while DoRA changes how those updates are trained. An observed improvement could therefore come from pretrained structure, update scale, or the gradient rule. We test whether spectral initialization remains advantageous after independently tuning scale and learning rate, and whether that advantage differs between additive adaptation and DoRA. Exact versus detached DoRA is a secondary comparison.

## 2. State-of-the-Art and Gap

**PiSSA** already reports a PiSSA+DoRA comparison [[2]](#ref-2), **DuDe** studies a trainable-factor hybrid [[3]](#ref-3), and **EDoRA** directly compares SVD and random initialization within a normalized, frozen-subspace adapter [[7]](#ref-7). **DoRA** itself studies detachment [[1]](#ref-1). These establish substantial prior evidence; our contribution is a controlled reassessment, not a first hybrid.

**LoRAM** motivates scale controls [[4]](#ref-4), **Learning Rate Matters** motivates per-condition tuning [[5]](#ref-5), and **RLPO/RLMO** provides related geometry/scale analysis in RLVR [[6]](#ref-6). Our literature audit did not locate a convincing estimate of the target interaction under both independent tuning and matched initialization spectra. This is a bounded literature finding, not a guarantee of novelty; see [the audit](references/LITERATURE_REVIEW.md).

## 3. Falsifiable Hypothesis

**Primary question:** does the spectral-minus-matched-random accuracy gap differ between additive adaptation and standard detached DoRA? Let those gaps be $\Delta_A$ and $\Delta_D$. Estimate $J=\Delta_D-\Delta_A$, alongside each gap: a negative interaction can mean attenuation even when the hybrid remains useful.

**Secondary question:** does the gradient rule change the spectral benefit within DoRA? Estimate $I=\Delta_E-\Delta_D$, where $\Delta_E$ is the exact-gradient DoRA gap. Both contrasts are two-sided; there is insufficient evidence to predict their signs. Prespecify practical margins before confirmation. Narrow intervals near zero support small effects; wide intervals remain inconclusive.

## 4. Solution Direction

**Controls.** The comparison will include additive LoRA, detached DoRA, and exact DoRA, each with principal-SVD or random-orthonormal initialization. After matching factor singular values and subtracting initial adapter products from the frozen weights, we will initialize DoRA magnitudes to preserve the same starting outputs. Adapter dropout will be disabled, and other settings will remain fixed. Checks on small models will verify outputs and gradients. The additive branches will use matched nonzero initialization rather than standard zero-factor LoRA initialization. Ordinary zero-factor LoRA and detached DoRA provide practical references. Initial spectra match at a shared scale; independently selected scales and later effective steps may differ.

**Experiments.** Start with Qwen2.5-1.5B base, GSM8K SFT, and rank-16 query/value adapters, subject to a measured pilot and runtime forecast. The initial 100-hour estimate is advisory. Tune each condition with equal search effort using development data, then confirm every prespecified condition on fresh paired seeds, including unfavorable outcomes. Report the best-over-scale comparison separately from fixed-scale comparisons. A funded extension confirms the $\gamma=1$ PiSSA-construction arms and matched random controls with their own LR selection; ordinary baselines alone do not turn rescaled spectral arms into a native PiSSA comparison.

**Measurements.** Evaluation will cover held-out task performance with seed-level uncertainty, accuracy-versus-step curves, effective weight-update norms, angular movement, peak GPU memory usage, and runtime. Comparisons of exact and detached updates at shared states and batches, including equal-norm step diagnostics, will help explain any performance differences. Report $J$ as primary and $I$ as secondary.

## References

<a id="ref-1"></a>

[1] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024). Dora: Weight-decomposed low-rank adaptation. arXiv preprint [arXiv:2402.09353](https://arxiv.org/abs/2402.09353).

<a id="ref-2"></a>

[2] Meng, F., Wang, Z., & Zhang, M. (2024). Pissa: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems, 37, 121038-121072.

<a id="ref-3"></a>

[3] Han, J., Zhang, S., & Zhang, K. (2025). Dual Decomposition of Weights and Singular Value Low Rank Adaptation. arXiv preprint [arXiv:2505.14367](https://arxiv.org/abs/2505.14367).

<a id="ref-4"></a>

[4] Zhang, Z., Li, H., Zhang, Y., Gong, G., Wang, J., Hu, J., ... & Jiang, Q. (2025). The primacy of magnitude in low-rank adaptation. Advances in Neural Information Processing Systems, 38, 39-69.

<a id="ref-5"></a>

[5] Lee, Y. A., Ko, C. Y., Chen, P. Y., & Yeh, M. Y. (2026). Learning rate matters: Vanilla lora may suffice for llm fine-tuning. arXiv preprint [arXiv:2602.04998](https://arxiv.org/abs/2602.04998).

<a id="ref-6"></a>

[6] Zhang, R., Zhu, J., Zhu, H., & Shi, L. (2026). Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR. arXiv preprint [arXiv:2606.31813](https://arxiv.org/abs/2606.31813).

<a id="ref-7"></a>

[7] Nasiri, H., & Garraghan, P. (2025). EDoRA: Efficient Weight-Decomposed Low-Rank Adaptation via Singular Value Decomposition. [arXiv:2501.12067v1](https://arxiv.org/abs/2501.12067v1).
