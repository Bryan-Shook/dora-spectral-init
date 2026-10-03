# Test plan: spectral initialization and DoRA detachment

Prepared 2026-10-02 from [README.md](README.md). Hardware confirmed by the project owner: **one NVIDIA H100 80 GB, with 100–200 total GPU-hours, including pilots, tuning, and evaluation**.

**Recommended starting point:** Qwen2.5-1.5B base, GSM8K supervised fine-tuning, rank 16 adapters on query and value projections, six controlled conditions plus a standard LoRA baseline. Complete the 100-hour study before adding breadth. Use additional resources to strengthen tuning and seed replication.

This is a proposed protocol, not a record of completed experiments. Model suitability, memory, throughput, statistical sensitivity, and all runtime estimates remain to be measured. Numeric choices below are operational defaults selected for this budget unless explicitly attributed to a source. Freeze any pilot changes before the main search.

## 1. Research question and intended contribution

**Does allowing gradients through DoRA's normalization denominator change the benefit of spectral initialization after each condition receives its own learning-rate and initialization-scale search?**

There are three related questions:

1. **Mechanism:** Does detachment change actual effective weight steps differently for spectral and random initialization?
2. **Task performance:** Does that interaction survive separate hyperparameter tuning and repeated training?
3. **Cost:** Is any observed benefit useful relative to initialization, tuning, training, and evaluation cost?

DoRA derives different gradients with and without denominator detachment [R1]. PiSSA uses principal singular components and a frozen residual [R2]. DuDe already combines spectral initialization with weight decomposition [R3]. LoRAM motivates controlling magnitude [R4], and Learning Rate Matters motivates separate tuning [R5]. RLPO/RLMO provide related initialization analysis in reinforcement learning; their results do not establish the answer for this supervised setting [R6].

The intended contribution is a controlled interaction study. Combining SVD and DoRA alone is already covered by related work. The cited papers motivate the experiment; this document does not establish an exhaustive novelty claim.

## 2. What to do, why, and how

| Work item | Why it is necessary | How to do it | Data/model and deliverable |
| --- | --- | --- | --- |
| Validate the parameterization | An incorrect norm axis, residual, or gradient would invalidate the study | Analytic gradients, finite differences where appropriate, forward equality, and checkpoint round trips | Synthetic matrices, then a tiny randomly initialized transformer; correctness report |
| Establish a viable task | The budget is useful only if the model learns and completes enough repeats | Short runs for all four DoRA conditions, then one complete run of the slowest condition | Qwen2.5-1.5B and the GSM8K pilot split; measured cost forecast |
| Match starting conditions | Spectral direction should not be confounded with initial factor singular values or initial predictions | Balanced SVD factors versus random orthonormal directions with the same singular values; subtract initial products | Every adapted matrix; initialization audit |
| Tune each condition | A shared learning rate can favor one parameterization | Equal numbers of candidates and training epochs, separate selection on development data | Six controlled cells and standard LoRA; complete search table |
| Repeat every final condition | Selecting only promising cells creates an incomplete and potentially biased comparison | Three new, paired seed blocks for all seven conditions | Locked configuration files and 21 confirmation runs |
| Test the interaction | Ranking methods does not answer whether detachment changes the spectral advantage | Difference of spectral-minus-random gaps, with uncertainty | Per-example predictions, paired seed estimates, interaction plot |
| Explain and cost the result | A gradient difference may disappear in effective weights or final accuracy | Counterfactual updates at the same state, effective movement, synchronized timing and peak memory | Small shared training batches; mechanism and cost tables |

## 3. Model, data, and training defaults

### 3.1 Model choice

Use **[`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B), the base checkpoint**. The official model card lists 1.54B parameters, 28 layers, and grouped-query attention; the saved configuration has hidden size 1,536 and 12 query/2 key-value heads [R8, R10]. Pin the model and tokenizer to an immutable repository revision when implementation begins.

This size leaves room for repeated tuning and confirmation on one H100. A base model also avoids adding instruction fine-tuning as another unknown in the adaptation comparison. This does not establish that GSM8K was absent from pretraining. Report results as relative adaptation behavior on a public benchmark, without claims of contamination-free mathematical generalization.

The default adapter placement is `q_proj` and `v_proj` in all 28 layers. At rank 16, the saved dimensions imply **2,179,072 trainable factor parameters**; DoRA adds **50,176 magnitude parameters**. These are calculated counts to check against the implementation. Additive versus DoRA therefore also differs in parameterization and parameter count; the exact-versus-detached comparison has equal trainable parameters.

Use `Qwen/Qwen2.5-0.5B` as a throughput fallback only if the main model cannot pass the budget gate and the smaller model passes the learning gate. Switch the entire experiment before the main search, not individual conditions. The 0.5B model is also an optional replication model in the 200-hour track [R12]. A 7B model is unnecessary for the initial question and would reduce the resources available for controls and repeats.

### 3.2 Dataset and split

Use **[`openai/gsm8k`, configuration `main`](https://huggingface.co/datasets/openai/gsm8k)**. The official dataset metadata has 7,473 training examples and 1,319 test examples [R9, R11]. Use the supplied worked solutions as supervised targets.

| Partition | Proposed examples | Allowed use |
| --- | ---: | --- |
| Training | 6,473 | All optimization, small overfit checks, and shared-batch gradient diagnostics |
| Pilot validation | 500 | Prompt/parser checks, runtime and learning feasibility, preliminary variability |
| Main development | 500 | Hyperparameter screening and final configuration selection |
| Official test | 1,319 | Final evaluation after the protocol, configurations, and seed count are frozen |

Create the three training-derived partitions once: sort examples by `SHA256("dora-gsm8k-v1\n" + question)`, take the first 500 for pilot validation, the next 500 for development, and the remainder for training. Save IDs, dataset revision, and hashes. Check exact normalized-question duplicates across partitions; if found, keep duplicate groups together and report revised counts. Do not move examples based on their answers or difficulty.

Use the same plain-text format for all conditions:

```text
Question: {question}
Answer: {worked solution ending in #### final_number}
```

At inference, provide only `Question: ...\nAnswer:`. Remove GSM8K's `<<...>>` calculator annotations from training solutions, retain the surrounding reasoning and `####` answer, and append the tokenizer's EOS token. Compute loss on answer/EOS tokens only; mask prompt and padding tokens.

Default maximum training length is 1,024 tokens, with dynamic padding and no example packing. Inspect lengths before training. Never silently truncate away final answers: if examples exceed the limit, increase the common limit to 2,048 if the pilot budget allows; otherwise exclude those training examples consistently and publish their count. Evaluation keeps every test question in the denominator.

### 3.3 Shared training settings

| Setting | Initial choice and rationale |
| --- | --- |
| Training horizon | Three epochs, fixed final checkpoint; small dataset and a manageable number of steps |
| Rank | 16; enough adapter capacity for an initial comparison without a rank sweep |
| Adapter multiplier | `s = alpha / rank = 1`, so `alpha = 16`; keep this distinct from initialization scale |
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`; decay would otherwise add another scale-dependent effect |
| Learning rate | Independently selected per condition; DoRA magnitudes and factors use the same selected rate |
| Schedule | 3% warmup followed by cosine decay over the full three-epoch horizon |
| Batch | 32 examples per optimizer step; start with microbatch 8 and accumulation 4 |
| Dropout | Adapter and attention dropout disabled; any other model dropout also disabled |
| Precision | BF16 transformer computation; FP32 adapter parameters, norm calculations, residual construction, and optimizer state |
| Frozen parameters | Backbone, biases, embeddings, and language-model head; train only factors and DoRA magnitudes where applicable |
| Gradient clipping | Global trainable-gradient norm 1.0, identical in all conditions; log how often clipping activates |
| Attention/checkpointing | Use the same supported attention backend; enable activation checkpointing for all conditions only if the pilot requires it |

With 6,473 training examples and batch 32, there are 203 optimizer steps per epoch and **609 steps per full run**, including each epoch's partial final batch. Normalize loss by the number of non-masked target tokens across an entire accumulation window. Preserve example order and accumulation semantics when changing microbatch size.

Pin Python, PyTorch, CUDA, Transformers, PEFT, tokenizer, and evaluation code versions. Hardware profiling must record H100 variant, allocated memory, power limit if available, and whether the GPU is shared.

## 4. Exact definitions of the experimental conditions

Use PyTorch's stored linear-weight orientation: $W_0\in\mathbb{R}^{d_{out}\times d_{in}}$, with $y=xW_0^T+b$. Define $B\in\mathbb{R}^{d_{out}\times r}$ and $A\in\mathbb{R}^{r\times d_{in}}$. Norms below are **per output row**, corresponding to `weight.norm(dim=1)` in this orientation. This prevents confusion with papers that use transposed weight conventions.

### 4.1 Matched initialization and the scale being tuned

For each target matrix, compute its top-rank SVD, $U_r\Sigma_rV_r^T$. Use one FP32 decomposition cache per model revision and rank. Start with exact reduced SVD; profile its one-time cost and record its device and precision. A faster approximation is permissible only after checking its singular values and reconstruction against exact SVD on representative layers and freezing that choice for all cells.

The tunable scalar **$\gamma$ controls the initial adapter-product magnitude**:

$$
B_{S,0}=\sqrt{\gamma}\,U_r\Sigma_r^{1/2},\qquad
A_{S,0}=\sqrt{\gamma}\,\Sigma_r^{1/2}V_r^T.
$$

For the random control, draw independent Gaussian matrices, take thin QR decompositions with a consistent diagonal-sign convention, and obtain $Q_L^TQ_L=Q_R^TQ_R=I_r$. Set

$$
B_{R,0}=\sqrt{\gamma}\,Q_L\Sigma_r^{1/2},\qquad
A_{R,0}=\sqrt{\gamma}\,\Sigma_r^{1/2}Q_R^T.
$$

At a shared $\gamma$, both choices have factor singular values $\sqrt{\gamma\sigma_i}$ and product singular values $\gamma\sigma_i$. The orthonormal bases are random; the final scaled factors themselves need not be orthonormal. Do not replace this control with unscaled Gaussian factors.

Let $C_0=B_0A_0$ be a **frozen copy**, $W_{res}=W_0-C_0$, and

$$Z_t=W_{res}+B_tA_t.$$

Then $Z_0=W_0$. An equivalent implementation is $Z_t=W_0+(B_tA_t-C_0)$; this can avoid rounding a residual into BF16. Keep the correction and recombination in FP32, and verify the resulting forward computation. Never recompute $C_0$ from the current trainable factors.

This matches initial spectra, not future gradient norms or effective weight steps. Those subsequent differences are measurements, not properties assumed away. Also, independently tuned cells can select different $\gamma$ values: the primary result measures performance after this specified search. Use shared $(\gamma,\eta)$ comparisons for a stricter mechanism interpretation.

### 4.2 Forward rule and detachment

For additive conditions, $W_{eff,t}=Z_t$. For both DoRA conditions, define a trainable magnitude $m_i$, initialized to $\|W_{0,i:}\|_2$, and use

$$
W_{eff,t,i:}=m_i\frac{Z_{t,i:}}{\|Z_{t,i:}\|_2}.
$$

The denominator is recomputed on every forward pass. Exact DoRA differentiates through it. Detached DoRA applies stop-gradient to it. Bias $b$ remains frozen and is added after weight rescaling. Use a common numerical norm floor, document it, and verify that it is inactive for the actual weights.

The inspected PEFT implementation detaches both the low-rank product used in norm computation and the final weight norm [R13]. Therefore, merely setting `use_dora=True`, or deleting only the final `.detach()`, does not establish an exact-gradient implementation. Inspect every path used by the pinned version and compare against a small explicit reference implementation.

| ID | Forward/gradient rule | Initialization | Role |
| --- | --- | --- | --- |
| A-S | Additive | Principal SVD, matched nonzero factors | Additive spectral control |
| A-R | Additive | Random directions, matched nonzero factors | Additive random control |
| E-S | DoRA, exact denominator gradient | Principal SVD | Primary interaction cell |
| E-R | DoRA, exact denominator gradient | Random directions | Primary interaction cell |
| D-S | DoRA, detached denominator | Principal SVD | Primary interaction cell |
| D-R | DoRA, detached denominator | Random directions | Primary interaction cell |
| L0 | Standard additive LoRA | Kaiming-uniform A, zero B | Separately tuned practical reference [R7] |

A-S at $\gamma=1$ uses the PiSSA initialization construction; the spectral DoRA cells are related to DuDe. These are controlled adaptations, not reproductions of every published training recipe. A-R is not standard zero-factor LoRA. L0 is outside the interaction contrast.

Reuse the same random bases across A-R, E-R, and D-R within a seed block and across that block's scale candidates. Use separate random-number streams for initialization and data order so that adapter initialization does not accidentally change the batches seen by a condition.

## 5. Elaborated hypotheses and decision rules

### H1 — Primary: detachment changes the tuned spectral advantage

Let $a_{g,i,k}$ be test accuracy in **percentage points**, for gradient rule $g$ (E or D), initialization $i$ (S or R), and confirmation seed block $k$. Each condition uses its own development-selected hyperparameters. Define

$$
\Delta_E=\operatorname{mean}_k(a_{E,S,k}-a_{E,R,k}),\quad
\Delta_D=\operatorname{mean}_k(a_{D,S,k}-a_{D,R,k}),\quad
I=\Delta_E-\Delta_D.
$$

**Prediction:** the interaction $I$ may be practically meaningful. This is two-sided: there is insufficient evidence to precommit that exact or detached DoRA benefits more. For illustration only, spectral gains of 3 points under exact DoRA and 1 point under detached DoRA give $I=2$ points.

**Rationale:** the gradient rule changes how a loss gradient reaches the low-rank factors. Their initial orientation can change the resulting effective movement. That is a plausible mechanism, not a theorem about final task accuracy.

Use a proposed practical margin **$\delta=2$ percentage points**. This is a project decision threshold, not a published constant or a promised detectable effect; two points correspond to about 26 correct answers out of 1,319. Review whether this improvement would justify the extra work before the main search. Do not enlarge the margin because measured variance makes equivalence difficult.

Interpret the primary 95% interval for $I$ as follows:

| Interval outcome | Interpretation |
| --- | --- |
| Entirely above $+\delta$ or below $-\delta$ | Evidence for a practically meaningful interaction in this setting |
| Entirely inside $[-\delta,+\delta]$ | Evidence that the interaction is practically small in this setting |
| Excludes zero but overlaps a practical boundary | Detectable interaction; practical size remains uncertain |
| Includes zero and extends outside the practical band | Inconclusive; it does not establish equivalence |

### H2 — Secondary: spectral direction retains value after matching and tuning

Estimate $\Delta_A$, $\Delta_E$, and $\Delta_D$ separately. A positive gap that remains meaningful with uncertainty supports a spectral advantage for that rule and search protocol. A well-resolved near-zero gap supports the view that the matched random alternative is sufficient here. A negative gap favors the random alternative. H1 can hold even if both gaps are negative; it concerns their difference.

LoRAM and Learning Rate Matters motivate this alternative explanation [R4, R5]. Neither proves that the advantage must disappear in this model, dataset, rank, and optimizer setting. Plot the searched performance surfaces as well as their selected points. This also reveals whether a claimed advantage depends on one scale or a search boundary.

### H3 — Mechanistic: initialization changes the consequence of detachment

For a single normalized row, let $u=z/\|z\|$ and let $g$ be the loss gradient with respect to the effective weight row. Exact and detached gradients with respect to $z$ are respectively

$$
g_E=\frac{m}{\|z\|}(g-u(u^Tg)),\qquad
g_D=\frac{m}{\|z\|}g.
$$

At the same initialized model and batch, $z$, $m$, and the effective-weight loss gradient are shared between spectral and random starts. The initialization-dependent effect enters through the factor parameterization and optimizer. A large raw gradient difference alone does not prove a large prediction or performance difference.

Run four dedicated 100-step diagnostic trajectories (E-S, E-R, D-S, D-R) with seed block `40`, common $\gamma=1$ and $\eta=10^{-4}$. At steps 0, 50, and 100, clone each checkpoint and its optimizer state. On the same diagnostic batch, take one hypothetical exact step and one detached step; do not apply these steps to the training run. Compare the merged weight-step directions, their relative distance, and prediction change. Use the same magnitude-learning rule and optimizer state within each counterfactual pair. Also inspect final confirmation checkpoints if the diagnostic allocation permits.

Report $d=\|\Delta W_E-\Delta W_D\|_F/(\|\Delta W_E\|_F+\|\Delta W_D\|_F+\epsilon)$ and the cosine between the steps, aggregated across layers. Explore whether S and R differ and whether those patterns accompany H1. These diagnostics support an explanation; a correlation across a few cells is not a validated predictor of final accuracy.

### H4 — Systems: any accuracy gain must be assessed with its cost

Measure peak GPU memory, training tokens/second, initialization time, complete search cost, and end-to-end run time. Detachment's savings reported in DoRA were measured in other settings [R1]; measure them again here. A supported outcome can be that accuracy is similar while one implementation is cheaper.

Use equal training exposure for scientific comparisons. Separately show quality against total measured GPU-hours. Report SVD cost once for this cached experimental campaign and again as a cold-start cost for a new model; include SVD cost for the matched random control, which also needs the singular values. Do not claim that this controlled random method avoids SVD.

## 6. Correctness checks before spending the main budget

Start on CPU with rectangular synthetic matrices, including different input/output sizes; then use a tiny transformer. These are meaningful implementation gates, not full training experiments.

| Check | Procedure | Pass criterion |
| --- | --- | --- |
| Spectrum and factor balance | Compare singular values of S/R factors and products at the same scale | FP64 toy relative errors below `1e-8`; FP32 real-layer relative errors below `1e-5` |
| Starting weight/output equality | Reconstruct every condition at initialization and compare with the frozen model | FP32 relative effective-weight error below `1e-5`; actual BF16 output drift within a pre-measured numerical baseline and with no systematic S/R difference |
| Exact gradient | Finite differences/gradcheck on the fully differentiable normalized layer; compare A, B, m to analytic chain rule | FP64 tolerances such as `atol=1e-6`, `rtol=1e-4`; investigate failures |
| Detached gradient | Compare autograd to the analytic stopped-denominator rule and a surrogate with the denominator held fixed at the reference point | Same toy tolerance; do not finite-difference a recomputed denominator and expect the detached gradient to match |
| Forward agreement E/D | Clone all parameters and run the same batch | Same forward result within numerical tolerance, but the expected gradient difference on a nondegenerate example |
| Frozen backbone and nonzero learning | Inspect `requires_grad`, optimizer membership, and post-step tensors | Only intended parameters change; nonzero gradients reach both factors in nonzero-initialized cells |
| Bias, merge, save/load | Compare explicit dense and efficient forwards before and after several steps; reload model plus residual/init metadata | Agreement within precision-specific tolerances; frozen bias is not magnitude-rescaled |
| Loss and evaluation | Check masking, shifted labels, accumulation normalization, answer parser, and seed streams | Hand-computed toy cases agree; invalid/truncated generations are retained as wrong answers |
| Small overfit | Train on 32 training examples | Clear loss reduction and ability to fit the subset; failure triggers debugging before a search |

For BF16 validation, first measure drift from repeat execution and from switching between the reference and intended arithmetic paths on fixed prompts. Record maximum and relative RMS logit errors and next-token agreement. Set and justify the acceptance tolerance before comparing trained outcomes; do not silently excuse initialization differences as noise. If needed, retain target-layer computation in FP32 and reprofile every condition.

## 7. Pilot and freeze gate: up to 10 GPU-hours including device checks

1. Complete CPU checks; use the first device allocation for numerical checks, model loading, and the SVD cache.
2. Run all four DoRA cells for 100 steps with pilot seed blocks `0, 1, 2`, initially at $\gamma=1$, $\eta=10^{-4}$. Also profile additive and standard LoRA. Use training examples for optimization and only pilot validation for learning decisions.
3. Run one complete three-epoch job for the slowest measured cell. Short-run extrapolation alone is insufficient because checkpoints, generation, and dataloading also cost time.
4. Time generation on the complete pilot validation split, including loading/merging and answer parsing. Use the intended generation backend and batch sizes. Profile the cost of development scoring separately from training.
5. Freeze the dataset partitions, model revision, adapter targets, rank, horizon, precision, search ranges, practical margin, parser, seed count, and analysis before the main search.

**Learning gate:** the small overfit check must pass, and the complete pilot run must reduce held-out answer loss and produce usable, non-saturated answer accuracy. Proposed diagnostic flags are below 5% or above 90% pilot accuracy, or more than 1% generations ending at the output limit; they are prompts for investigation, not reported scientific findings. First check masking, prompt, parsing, and optimization. If query/value adapters cannot learn, test all seven transformer projections during the pilot and reprice the complete design before choosing that placement for every cell.

**Memory gate:** the longest supported batch in the slowest condition must fit with at least 10% of the device memory available as headroom. Record both allocated and reserved peaks. If it fails, reduce the common microbatch while preserving the effective batch, or enable checkpointing consistently, and measure throughput again.

**Budget gate:** forecast all remaining jobs using measured times, including failures and evaluation, and keep the confirmation reserve intact. If the 100-hour track does not fit, first improve batching/measurement overhead consistently, then consider a common two-epoch horizon or the 0.5B fallback before freezing. Recheck the learning gate after a change. Do not keep a fast condition's three epochs while shortening a slower one, and do not remove a primary cell to fit the budget.

If the pilot cannot establish both learning and affordability within its allocation, document the failure and revised scope. Do not spend the remainder on a large, uninterpretable search.

## 8. Search and confirmation protocol

### 8.1 Equal search allocation

For each of the six controlled cells, search:

- Initialization scale $\gamma\in\{0.1,0.3,1.0\}$.
- Learning rate $\eta\in\{2\times10^{-5},10^{-4},5\times10^{-4},2\times10^{-3}\}$.
- Twelve combinations per cell, tuning seed block `10`, with identical data order across cells.

These ranges are starting proposals, informed by the broad rate sensitivity in [R5], not proven optima. Pilot behavior can justify shifting them for **all six cells** before freezing. Factor scale is tuned; the adapter multiplier, rank, batch size, weight decay, and magnitude-to-factor LR ratio are fixed.

Give L0 twelve learning rates logarithmically spaced from `2e-6` through `2e-3`, inclusive, and the same total epoch allocation. Its zero-factor initialization remains standard. This gives it an equal candidate budget without inventing a spectral-scale parameter for it.

Use two stages:

1. **Screen:** all 84 candidates train for one epoch. Score greedy final-answer accuracy on a fixed 256-example subset of development data; break ties by lower answer-token NLL on the full 500 development examples, then lower LR. Use the same subset for all candidates.
2. **Promote:** in each controlled cell retain the best LR at each of the three scales, giving three candidates per cell. Retain L0's top three rates. Continue these 21 jobs for two additional epochs. Select one final configuration per cell using full-development accuracy, breaking ties by answer NLL and then lower LR/scale.

Every screened job uses the **three-epoch LR schedule from the beginning**. A promoted job resumes model, optimizer, scheduler, random-number generators, and dataloader state; restarting with a new schedule would change the comparison. Score the final three-epoch checkpoint, not whichever checkpoint happens to maximize development accuracy.

This costs 84 one-epoch starts plus 42 continuation epochs: **126 epochs, or 42 complete three-epoch training equivalents**. It is a bounded search with early screening, not a guarantee of global optimality. Screening can discard slow starters. If an optimum is on a rate boundary, or neighboring rates are unstable, either finance an equal expansion for the compared cells from reserve before confirmation, or explicitly report that the tuning range limits the conclusion. An expansion is never free.

### 8.2 Confirmation

Freeze selected configurations. Run **all seven conditions with fresh seed blocks `101, 202, 303`**, giving 21 complete runs. Do not reuse tuning runs as seed replicates. Within each block, pair example order and the random bases across the relevant cells. Keep all results, including disappointing cells.

The 100-hour result is conditional on a single tuning seed and a restricted search. The expanded track below repeats search finalists to assess selection stability. Failed numerical runs remain in the search/failure record; rerun infrastructure interruptions from checkpoint, and count both allocations. Do not repeatedly retry divergent configurations until a favorable seed appears.

## 9. Evaluation and statistical analysis

Use greedy decoding, one completion per question, no tools, no few-shot demonstrations, and a common default maximum of **512 new tokens**. Choose any increase to 1,024 from the pilot only, reprice evaluation, and apply it to all conditions. Merge a copy of each trained adapter into its effective weights for generation; verify merge correctness and do not overwrite training states. Use one pinned generation backend for all cells.

The primary metric is **100 × correctly parsed numeric answers / all 1,319 test questions**. Extract the number after the last `####`, allow commas/sign/decimal formatting, and compare using an exact decimal/rational normalization rather than an arbitrary floating-point tolerance. Missing, malformed, or truncated answers count as incorrect. Save raw text, extracted answer, correctness, and truncation status for every question. A permissive last-number score may be reported only as a labeled secondary metric.

Evaluate the untouched base model once on the locked test as context after the freeze. Its formatting behavior may differ, so it is not the main adaptation comparison. Do not use official-test results to adjust prompts, parsers, search ranges, or select a condition.

For each seed block compute

$$I_k=(a_{E,S,k}-a_{E,R,k})-(a_{D,S,k}-a_{D,R,k}).$$

Report the mean, every individual value, and the paired Student-t 95% interval across seed blocks. With three seeds, the multiplier is approximately 4.303 and the half-width is $4.303\,s_I/\sqrt{3}$. For example, seed SD 1.5 points would give a half-width about 3.7 points; three seeds cannot reliably establish a tiny effect. Pilot variability is only a rough guide because short trajectories may underestimate final variability.

This primary interval describes training randomness conditional on the locked test set and selected configurations. Also provide a paired question-bootstrap interval for item variability, resampling the same question IDs across all cells, plus an optional two-way seed/question bootstrap as a sensitivity analysis. Ten thousand resamples are inexpensive on CPU. With only three seeds, bootstrap precision does not create additional independent training evidence.

Use H1 as the sole primary inferential claim. Treat H2–H4 and per-layer correlations as secondary/exploratory, or apply an explicitly stated multiplicity correction if making additional confirmatory claims. Never treat 28 layers, many checkpoints, or 1,319 questions as 28, many, or 1,319 independent training runs.

Required figures/tables:

- Seven-condition test table: selected LR/scale, accuracy mean/SD, seed values, answer NLL, failure count, peak memory, and measured time.
- Interaction plot: $\Delta_E$, $\Delta_D$, and $I$ with intervals and the practical band.
- Search surfaces by LR and scale, marking promoted candidates and boundary selections.
- Effective movement versus step and versus wall time, alongside task learning curves.
- Cost table separating initialization, tuning, confirmation, generation, diagnostics, and retries.

## 10. Measurements that distinguish scale from direction

At initialization, steps 1/10/50/100, and the final checkpoint, measure selected representative layers; compute all-layer summaries at the final checkpoint. Use common training batches for diagnostics, not test questions.

| Measurement | Definition and purpose |
| --- | --- |
| Initial adapter product | $\|B_0A_0\|_F$ and factor singular values; verifies the promised matching |
| Net model movement | $\|W_{eff,t}-W_0\|_F/\|W_0\|_F$; this is the meaningful change from the pretrained layer |
| One-step movement | $\|W_{eff,t+1}-W_{eff,t}\|_F$; distinguishes step size from accumulated movement |
| Angular movement | Mean output-row angle to the corresponding row of $W_0$, with cosine arguments clamped to `[-1,1]` |
| Parallel/perpendicular movement | Project each effective row step onto its current effective row; report parallel and orthogonal norms |
| Magnitude learning | Relative change in m and its gradient/step norm for DoRA |
| Counterfactual detachment effect | Shared-state step distance/cosine from H3, including the actual AdamW update |
| Prediction change | Answer-token NLL and a fixed-probe logit/KL comparison; report clearly that it is a diagnostic |
| Training stability | Gradient norms, clipping fraction, non-finite events, loss spikes, invalid-answer and truncation rates |

For nonzero initialization, $\|BA\|$ is not the net model update: even a large initial $BA$ is canceled by the frozen residual. For DoRA, $BA-B_0A_0$ also omits normalization and magnitude effects. Always calculate the effective merged weight for movement claims.

Measure elapsed GPU work with synchronization at timing boundaries, warm up before reporting steady-state throughput, and reset peak allocated/reserved memory statistics per run. Report active-token and padded-token throughput separately. Include time spent on loading, SVD/QR, validation, merging, and checkpoint I/O in end-to-end costs. Match diagnostic frequency across conditions and include its overhead.

## 11. GPU-hour budget and stop rules

### 11.1 A complete study capped at 100 hours

One full training equivalent below means three epochs with training and routine logging, excluding generation and one-time initialization. **The assumed 0.75 hour per equivalent is an engineering allowance to validate, not measured H100 performance.** Actual times differ across cells; use the slowest measured cell for a conservative forecast and record per-cell costs.

| Allocation | Planned workload | GPU-hour allowance |
| --- | --- | ---: |
| Device correctness, initialization, profiling | CPU checks first; real-model checks and decomposition cache | 4.00 |
| Pilot and freeze | Short paired pilots plus one full slow-cell pilot | 6.00 |
| Six-cell search | 36 full training equivalents × 0.75 h | 27.00 |
| Standard LoRA search | 6 full training equivalents × 0.75 h | 4.50 |
| Confirmation | 7 conditions × 3 seeds × 0.75 h | 15.75 |
| All scoring and associated I/O | Development and final-test generation/NLL, merges, load/save overhead | 18.00 |
| Mechanism diagnostics | Short common-setting trajectories and shared-state counterfactuals | 4.00 |
| Contingency | Boundary extensions, failures, slower jobs, additional scoring | 20.75 |
| **Hard cap** | **63 full training equivalents for search + confirmation** | **100.00** |

There are **84 search starts and 21 fresh confirmation starts**, with 21 search jobs continued. These are not 105 full training runs. Do not count continuation epochs twice.

The default schedule generates about **70,203 completions** across search screening (84 × 256), promoted-candidate scoring (21 × 500), confirmation-development scoring (21 × 500), and confirmation testing (21 × 1,319). This excludes pilot/base-model scoring and repeats, which also need allowance. Token lengths and generation throughput determine the actual cost; short training alone does not establish affordability.

At average padded length 512, a full run processes approximately $3\times6473\times512=9.94$ million padded tokens. Fitting training into 0.75 hour requires about 3,682 padded tokens/second; at padded length 1,024 it requires about 7,365. These are **required rates calculated from the allowance**, not predictions. Replace them with measured length distributions and rates in the pilot.

Maintain a ledger of reserved/actual GPU-hours for every job, including interrupted allocations. Charge GPU time while the accelerator is allocated, even during I/O; do preprocessing and CPU analysis outside a reserved GPU session when possible. At the freeze gate, require `spent pilot + forecast remaining training + forecast scoring/I/O + diagnostics + at least 10 h failure reserve <= 100 h`. Update the forecast before every job and keep unfinished confirmation/evaluation funded. The table's larger initial contingency can absorb measured differences, but it is not an extra budget beyond the cap.

If the cap is 100 hours, do not start optional work until the seven-condition result can be completed. Never spend confirmation time chasing the best preliminary score. If the main study was frozen and later slows down, preserve complete paired seed blocks and report any reduction in precision explicitly; do not report a partial block as a completed interaction estimate. Fewer than three complete confirmation blocks means the planned confirmatory study is unfinished and any available estimates are exploratory.

### 11.2 Expansion toward 200 hours

Choose this track **before confirmation/test inspection** if the extra allocation is available. Up to 100 additional hours are reserved as follows:

| Priority | Extra work | Additional allowance |
| --- | --- | ---: |
| 1 | Repeat the top two full-run candidates in each of the seven conditions on two new tuning seeds (`20, 30`): 28 full runs. Select by mean development accuracy over tuning seeds `10, 20, 30`, then mean NLL. Use these winners in the already budgeted confirmation runs. | 50 h including scoring and reserve |
| 2 | Increase all seven conditions from three to five confirmation seeds by adding `404, 505`: 14 full runs plus scoring. | 20 h |
| 3 | Replicate the four primary DoRA cells on Qwen2.5-0.5B/GSM8K, with its own pilot and tuning: 12 screened candidates/cell, three promotions/cell, and three fresh confirmation seeds/cell. This is 36 full training equivalents. Launch only if the measured total fits. | 30 h including pilot, scoring, and reserve |
| **Additional cap** | Added to the completed core allocation | **100 h** |

The replication model is cheaper and tests sensitivity to model size; it is not evidence across model families or tasks. Its 30-hour allowance needs training near 0.5 h per full equivalent with the remainder covering other costs. If it fails that gate, use remaining time for the primary study's uncertainty and search coverage, and omit the replication.

Repeated finalist tuning tests stability within the screened candidate set; it does not repair all possible early-screening bias. If slow-starter concerns dominate, use the expansion allocation to finish discarded candidates symmetrically and reforecast confirmation instead of claiming robust global tuning.

If extra hours arrive only after confirmation, changing selected configurations requires new confirmation runs and a new budget. Existing results cannot be relabeled as confirmation of a different configuration. For an intermediate budget such as 150 hours, prioritize tuning stability and complete additional seed blocks over another model.

## 12. Work sequence and deliverables

| Target window, 2026 | Work and completion evidence |
| --- | --- |
| Oct 2–9 | Implement parameterizations and checks; prepare splits, parser, and cached SVD; complete pilot and freeze a versioned protocol |
| Oct 10–23 | Run equal-budget search; review the entire search table for boundaries, failures, and promotion bias |
| Oct 24–30 | Complete any funded search extension or repeated finalist tuning; lock winners and confirmation seeds |
| Oct 31–Nov 13 | Run complete confirmation blocks, evaluate frozen final checkpoints, finish shared-state diagnostics |
| Nov 14–27 | Analyze interaction and costs, perform CPU resampling, audit artifacts, write the report |
| Nov 28–course deadline | Finalize reproducibility package and submission; confirm the actual course deadline separately |

Calendar windows follow the proposal and are not a request to reserve a GPU continuously. At one GPU, 100/200 allocated GPU-hours equal approximately 4.2/8.3 days of device occupancy, spread over the schedule.

Suggested implementation artifacts, **to be created during execution**, are:

- `configs/protocol_locked.yaml`: model/data revisions, splits, all fixed settings, search candidates, seeds, hypotheses, margin, and analysis version.
- `data/splits.json`: exact example IDs and checksums, plus preprocessing/length statistics.
- `reports/correctness.md` and `reports/pilot.md`: actual checks, numerical errors, memory, learning curves, throughput, and revised cost forecast.
- `runs/<run_id>/`: configuration, logs, selected checkpoint, initialization metadata, per-example predictions, timings, failure status, and git commit.
- `reports/results.md`: all conditions and seeds, effect intervals, search limitations, plots, and cost accounting.

Save residual/correction metadata with adapters: these nonzero-initialized adapters cannot be loaded correctly onto an untouched base weight by blindly applying a standard additive loader. Keep optimizer and RNG state for jobs that may resume. Keep the original model revision and initialization cache hashes so checkpoints can be reconstructed.

Completion means passing the implementation gates, reporting all four primary cells and their interaction, retaining the additive controls and standard baseline, and accounting for total cost. A null or inconclusive result is acceptable. A general claim that spectral initialization is always useful/useless, or that exact DoRA is universally better/worse, is outside this experiment's scope.

## 13. References and reading list

See [the reference index](references/README.md) for online paper links, exact versions, recommended sections, and relevance. [The manifest](references/manifest.json) records paper URLs, versions, and page counts, plus metadata for the retained official source snapshots. No local paper PDFs are retained.

| ID | Reference and online source | Specific use in this plan |
| --- | --- | --- |
| R1 | Liu et al., **DoRA: Weight-Decomposed Low-Rank Adaptation**, ICML 2024. [Paper](https://arxiv.org/abs/2402.09353v6) | Exact/stopped normalization gradients; original detachment ablation |
| R2 | Meng et al., **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models**. [Referenced arXiv revision](https://arxiv.org/abs/2404.02948v4) | Principal SVD factorization and frozen residual |
| R3 | Han et al., **Dual Decomposition of Weights and Singular Value Low Rank Adaptation**, 2025. [Paper](https://arxiv.org/abs/2505.14367v2) | Closest combination of spectral initialization and weight decomposition |
| R4 | Zhang et al., **The Primacy of Magnitude in Low-Rank Adaptation**, NeurIPS 2025. [Paper](https://arxiv.org/abs/2507.06558v2); [proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/0010665e949927b74faf6e3ada6d7f72-Abstract-Conference.html) | Magnitude as a competing explanation; motivates scale controls |
| R5 | Lee et al., **Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning**, 2026. [Paper](https://arxiv.org/abs/2602.04998v2) | Independent tuning, broad search ranges, implementation settings |
| R6 | Zhang et al., **Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR**, 2026. [Paper](https://arxiv.org/abs/2606.31813v1) | Related geometry/scale distinction; RLVR scope limitation |
| R7 | Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**, arXiv 2021 / ICLR 2022. [Paper](https://arxiv.org/abs/2106.09685v2) | Standard additive low-rank reference |
| R8 | Qwen Team, **Qwen2.5 Technical Report**, 2025 revision. [Paper](https://arxiv.org/abs/2412.15115v2) | Model family context |
| R9 | Cobbe et al., **Training Verifiers to Solve Math Word Problems**, 2021. [Paper](https://arxiv.org/abs/2110.14168v2) | GSM8K task and dataset origin |
| R10 | Official Qwen2.5-1.5B [model card](https://huggingface.co/Qwen/Qwen2.5-1.5B) and [configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json) | Exact model identity and dimensions; saved snapshots under `references/metadata/` |
| R11 | Official GSM8K [dataset card](https://huggingface.co/datasets/openai/gsm8k) and [repository](https://github.com/openai/grade-school-math) | Split sizes, format, calculator annotations |
| R12 | Official [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Fallback/replication checkpoint |
| R13 | Hugging Face PEFT, [DoRA implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py) | Norm axis, bias handling, and multiple detachment sites; archived source snapshot |

Bibliographic clarification: the proposal's LoRAM entry uses 2026; the official proceedings identify **NeurIPS 2025**. This plan uses that record. The linked PiSSA paper is arXiv v4, which is a later revision than its original 2024 publication. Treat revision-specific results accordingly. References are motivation and implementation evidence; no published benchmark has been presented here as an H100 timing measurement for this proposed setup.
