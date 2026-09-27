# Does DoRA Change the Value of Spectral Initialization under Per-Condition Tuning?

Adrian Adam Kalisz, Bryan Le Shook, Myeongsun Choi, Suraj Chandra Samudrala

## 1. Problem Statement

LoRA is usually initialized randomly, while spectral methods initialize from singular directions of the pretrained weight matrix. However, spectral initialization changes both the update direction and potentially its scale, while DoRA’s attached or detached denominator changes how those updates are trained. An observed improvement could therefore come from pretrained structure, update scale, or the gradient rule. We test whether spectral initialization remains advantageous after independently tuning scale and learning rate, and whether that advantage differs between exact and detached DoRA.

## 2. State-of-the-Art and Gap

**DoRA** studies detachment [[1]](#ref-1); **PiSSA** uses spectral initialization [[2]](#ref-2); and **DuDe** combines spectral initialization with weight decomposition [[3]](#ref-3). **LoRAM** emphasizes update magnitude [[4]](#ref-4), **Learning Rate Matters** highlights independent tuning [[5]](#ref-5), and **RLPO/RLMO** separates geometry from scaling in RLVR [[6]](#ref-6). What this team is trying to do is to bridge these areas together: spectrum-matched initialization, exact vs detached DoRA, and separate learning-rate searches all combined.

## 3. Falsifiable Hypothesis

We hypothesize that DoRA’s gradient detachment changes the real-world advantage of spectral initialization, even when each setup gets its own hyper-parameter tuning. To test this, we will compare the performance gap between spectral initialization and random initialization under exact and detached DoRA. A meaningful difference supports the hypothesis while similar advantages do not. We will run an initial pilot phase to establish what counts as a meaningful threshold before running the confirmation experiments.

## 4. Solution Direction

**Controls.** The comparison will include additive LoRA, detached DoRA, and exact DoRA, each with principal-SVD or random-orthonormal initialization. After matching factor singular values and subtracting initial adapter products from the frozen weights, we will initialize DoRA magnitudes to preserve the same starting outputs. Adapter dropout will be disabled, and other settings will remain fixed. Checks on small models will verify outputs and gradients. The additive branches will use matched nonzero initialization rather than standard zero-factor LoRA initialization.

**Experiments.** Initial experiments will use a small open-weight language model and a supervised fine-tuning benchmark with evaluation metrics. Pilot tests will be conducted to select the model, dataset, adapter rank, and experiment budget based on memory, runtime, and measurable learning progress. This setup will remain fixed for the main comparison. We will tune learning rates independently with equal search budgets and select configurations using validation data. Promising results will be evaluated across multiple random seeds.

**Measurements.** Evaluation will cover held-out task performance, effective weight-update norms, angular movement, peak GPU memory usage, and runtime. Comparisons of exact and detached updates on shared batches will help explain any performance differences.

## 5. Timeline

**Sep 16–Oct 09:** implement, validate, and pilot; finalize evaluation rules. **Oct 10–30:** run learning-rate searches. **Oct 31–Nov 13:** confirm with fresh seeds. **Nov 14–27:** analyze results and costs. **Nov 28–Final:** submit the report and release code.

## References

<a id="ref-1"></a>

[1] Liu, S. Y., Wang, C. Y., Yin, H., Molchanov, P., Wang, Y. C. F., Cheng, K. T., & Chen, M. H. (2024). Dora: Weight-decomposed low-rank adaptation. arXiv preprint [arXiv:2402.09353](https://arxiv.org/abs/2402.09353).

<a id="ref-2"></a>

[2] Meng, F., Wang, Z., & Zhang, M. (2024). Pissa: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems, 37, 121038-121072.

<a id="ref-3"></a>

[3] Han, J., Zhang, S., & Zhang, K. (2025). Dual Decomposition of Weights and Singular Value Low Rank Adaptation. arXiv preprint [arXiv:2505.14367](https://arxiv.org/abs/2505.14367).

<a id="ref-4"></a>

[4] Zhang, Z., Li, H., Zhang, Y., Gong, G., Wang, J., Hu, J., ... & Jiang, Q. (2026). The primacy of magnitude in low-rank adaptation. Advances in Neural Information Processing Systems, 38, 39-69.

<a id="ref-5"></a>

[5] Lee, Y. A., Ko, C. Y., Chen, P. Y., & Yeh, M. Y. (2026). Learning rate matters: Vanilla lora may suffice for llm fine-tuning. arXiv preprint [arXiv:2602.04998](https://arxiv.org/abs/2602.04998).

<a id="ref-6"></a>

[6] Zhang, R., Zhu, J., Zhu, H., & Shi, L. (2026). Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR. arXiv preprint [arXiv:2606.31813](https://arxiv.org/abs/2606.31813).
