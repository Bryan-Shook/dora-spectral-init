# Literature review: spectral initialization × DoRA

Literature coverage: **2026-10-11**. This document assesses prior evidence and the research gap. [TEST_PLAN.md](../TEST_PLAN.md) specifies the experimental protocol. The repository contains no completed training results.

**Verdict:** The broad question is partly answered, with direct positive evidence that spectral information can remain useful inside weight-normalized adaptation. Calling it simply “unanswered” is inaccurate. The reviewed sources do not establish a convincing, independently tuned, scale-controlled estimate of the interaction for trainable PiSSA-style factors and DoRA. That narrower gap is plausible, but neither guaranteed novel nor a large unexplored field. The defensible project is a controlled reassessment of existing hybrid claims.

## 1. Separate the claims before judging novelty

| Claim or question | Assessment |
|---|---|
| Can spectral initialization and DoRA work together? | Already answered positively in published experiments. |
| Can spectral initialization retain an advantage within a DoRA-style normalized model? | Directly investigated; positive evidence exists. Do not present this as untouched. |
| Does the hybrid beat the individual methods under particular recipes? | Already investigated. |
| Does DoRA amplify, attenuate, or reverse the spectral advantage after independent tuning of every cell? | No adequately documented direct answer located in this audit. Existing four-cell results are substantial prior evidence, not irrelevant baselines. |
| Is any remaining advantage attributable specifically to pretrained singular-vector orientation after matching scales? | Not resolved for the targeted DoRA/PiSSA setting by the audited studies. Related additive and RLVR studies substantially overlap. |
| Does exact versus detached normalization change these effects? | A narrower auxiliary question. It should not replace the original objective merely to produce a less crowded novelty claim. |

Independent tuning is part of the target research question. Matched spectra, multiple tasks, five seeds, and particular gradient implementations must not become arbitrary extra requirements used to dismiss all prior evidence. A study can substantially answer the scientific question without matching this repository's exact protocol.

## 2. The closest evidence

### PiSSA: an existing four-cell experiment

Appendix A, Table 5 reports the following GSM8K results on LLaMA-3-8B. The reported means already supply a descriptive interaction. [PiSSA, v4](https://arxiv.org/html/2404.02948v4#A1).

| Parameterization | Ordinary initialization | PiSSA initialization |
|---|---:|---:|
| Additive | 71.01 | 76.75 |
| DoRA | 72.38 | 77.51 |

Calculated from the reported means: the spectral gain is 5.74 points without DoRA and 5.13 with it; the difference is **−0.61 points**. The hybrid is best, while the incremental spectral gain is slightly smaller. Thus “best hybrid” does not establish positive statistical interaction.

Appendix L documents method-specific hyperparameter searches for GLUE. It does not establish that the GSM8K hybrid table received separate searches for all four cells. That experiment-specific tuning provenance remains unverified. The table also lacks a matched-spectrum random counterfactual. This is direct partial evidence, not a blank literature gap.

### EDoRA: direct SVD-versus-random evidence under normalization

EDoRA freezes SVD-derived outer factors, trains an intermediate square matrix plus magnitude, and normalizes the reconstructed weight. Section 4.3/Table 3 directly compares SVD and random initialization. At rank 32, the reported aggregate GLUE score is 84.48 versus 82.68. Section 4.1 reports grid search; Section 4.2 reports five seeds. Appendix A lists chosen EDoRA and classifier learning rates. [EDoRA, v1](https://arxiv.org/html/2501.12067v1).

This substantially answers the broad “can spectral information help despite directional normalization?” question. Its frozen-subspace architecture differs from trainable PiSSA factors. The paper does not clearly document independently selected hyperparameters for both initialization ablations or a matched-spectrum control. Those limits narrow its answer; they do not erase it.

The official launcher explicitly loops over five adapter LRs, five classifier LRs, two initialization sigmas, and five seeds. The initialization utility's random branch replaces both outer factors with Kaiming samples; it does not preserve their SVD spectra. The inspected launcher does not itself switch between the two initialization arms. This verifies a substantial tuning mechanism, but does not establish the selection provenance for Table 3. [Launcher](https://raw.githubusercontent.com/Hamid-Nasiri/EDoRA/main/scripts/run_glue.py), [initialization code](https://raw.githubusercontent.com/Hamid-Nasiri/EDoRA/main/utils/initialization_utils.py).

### DuDe: the closest trainable-factor hybrid

DuDe combines SVD initialization, a frozen residual, and magnitude/direction decomposition. Sections 4.1 and 4.2 explicitly share hyperparameters across methods. Section 4.6 varies whether singular values are allocated to A or B; performance changes despite retaining the selected singular directions. This is useful evidence that factor scaling matters inside the hybrid. It is not an independently tuned orientation-by-normalization experiment. [DuDe, v2, §§3–4 and Appendix A](https://arxiv.org/html/2505.14367v2).

An additional OpenReview copy was discovered, but its direct PDF endpoint returned a browser challenge. The conclusions here concern the inspected arXiv v2, not an asserted audit of every later submission. [Discovered OpenReview record](https://openreview.net/forum?id=JVOEgPgXSC).

### LoRAM: much of the magnitude-versus-subspace question is prior work

Section 4.3 substitutes bases and uses a tracking mode matching initialization-product variance to reference methods. These experiments support the claim that magnitude explains much of the observed advantage of certain spectral recipes. The reported exception for gradient-informed initialization also matters. DoRA appears in related work, but the audited experiments contain no DoRA × spectral/random factorial. One cannot transfer an additive-adapter result to normalized adaptation as a theorem. [The Primacy of Magnitude in Low-Rank Adaptation, v2](https://arxiv.org/html/2507.06558v2).

### Learning Rate Matters: strong tuning evidence, missing hybrid cell

The study sweeps learning rates for ten methods, including DoRA, PiSSA, OLoRA, MiLoRA, and nonzero initialization. Its method list and implementation appendix do not include PiSSA+DoRA or DuDe as an evaluated condition. Therefore separate tuned results for DoRA and PiSSA do not determine their interaction. Section 4 describes roughly tenfold higher initial PiSSA **curvature**, not tenfold higher loss, in the highlighted analysis. [Learning Rate Matters, v2, §§3–4 and Appendix E](https://arxiv.org/html/2602.04998v2).

### LoRA-XS: tuned orientation ablations already exist outside DoRA

The revised paper compares random factors, factors derived from a random matrix's SVD, and pretrained-weight SVD, with detailed adapter/classifier LR tables. Results depend on task; pretrained SVD is not universally best. This is substantial overlap with “does subspace identity survive tuning?” for frozen-subspace additive adaptation. It cannot be combined numerically with EDoRA's tables to construct a valid cross-paper interaction: training recipes and comparisons differ. [LoRA-XS, v3, initialization ablations and appendix tables](https://arxiv.org/html/2405.17604v3).

## 3. Related work that constrains the research claim

These are relevant-section audits unless marked **screened**. A paper being listed here does not mean every proof, table, and implementation has been reproduced.

| Work | Relevant evidence and boundary |
|---|---|
| [DoRA, v6](https://arxiv.org/html/2402.09353v6) | Defines total-weight normalization and studies denominator detachment. Detachment itself is established prior work; its interaction with spectral initialization is the possible extension. |
| [DeLoRA, v1](https://arxiv.org/html/2503.18225v1) | Normalizes low-rank factors and separates adaptation strength. Studies LR robustness; Appendix C compares DoRA with fixed and learned magnitude. Its nonzero factors are randomly initialized. Different normalization operator; no targeted spectral factorial found. |
| [NoRA, v1](https://arxiv.org/html/2608.31036v1) | Table 5 applies initialization-only and persistent rank-dimension normalization to both additive LoRA and DoRA. Appendix B reports a shared SFT LR. This is a closely related normalization×initialization experiment, but the initialization is not pretrained SVD. |
| [BiDoRA, v1](https://arxiv.org/html/2410.09758v1) | Separates magnitude and direction through bilevel optimization. Relevant to optimizer coupling; no PiSSA hybrid comparison located. |
| [DisLoRA, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.694.pdf) | SVD task-specific adaptation with orthogonality regularization and normalized loss weighting. “Directional” and “normalization” here do not mean the DoRA total-weight operator. Inspected method, baseline, and hyperparameter sections. |
| [Balanced LoRA, v1](https://arxiv.org/html/2605.31484v1) | Independently sweeps LR and adapter scaling, including a 10×10 grid on Wikitext. Removes factorization imbalance. The paper explicitly leaves alternative initialization combinations as future work. Its benchmark does not answer the target hybrid interaction. |
| [Which LoRA?, v1](https://arxiv.org/html/2606.10428v1) | Method-specific tuning and parameter-budget comparisons for multilingual instruction tuning. DoRA* means a budget-matched DoRA configuration, not PiSSA+DoRA. Important negative evidence about broad method superiority. |
| [Evaluating Parameter Efficient Methods for RLVR, v1](https://arxiv.org/html/2512.23165v1) | Structural variants succeed and spectral variants fail under the reported RLVR recipe. Main comparison shares hyperparameters; LR ablation tables do not establish individually optimized hybrid comparisons. Reinforcement learning results cannot settle the SFT question. |
| [Geometry-Preserving Orthonormal Initialization for RLVR, v1](https://arxiv.org/html/2606.31813v1) | Separates singular-vector geometry and singular-value scaling; evaluates principal/minor orthonormal initialization and LR sensitivity. Strong overlap with the mechanism question, different regime and no target DoRA crossing. Its Eq. 7 calls a two-sided SVD construction “OLoRA”; distinguish that construction from the original QR-based OLoRA. |
| [ULoRA, v1](https://arxiv.org/html/2607.26247v1) | Gradient/curvature-informed initialization family with LR sweeps. Geometry after tuning is already an active question. DoRA is discussed as context, rather than crossed with the initialization family in the audited experiments. |
| [Initialization and Regularization of Factorized Neural Layers, 2021, §3.3](https://arxiv.org/html/2105.01029v1) | Explicitly studies spectral initialization in the presence of normalization and explains effects through effective directional step size. Predates LoRA; different factorized networks and normalization, but undermines any broad claim that this conceptual issue is new. |
| [The Impact of Initialization on LoRA Finetuning Dynamics, 2024](https://arxiv.org/abs/2406.08447) | **Screened.** Swapping which factor is zero changes stable LRs and learning dynamics. Equal starting functions do not imply equal optimization. |
| [Beyond Zero Initialization, 2025](https://arxiv.org/abs/2505.23194) | **Screened.** Nonzero initialization changes robustness to LR choice. Necessary conceptual control for attributing nonzero-factor benefits to pretrained directions. |
| [OLoRA](https://arxiv.org/abs/2406.01775) | **Screened.** QR-based initialization, not principal-singular-subspace selection. Do not group it with PiSSA under an identical mathematical description. |
| [MiLoRA](https://arxiv.org/abs/2406.09044) | **Screened.** Minor rather than principal singular components. Native comparisons change singular-value scales too; they are not clean orientation-only interventions. |
| [LoRA-GA](https://arxiv.org/abs/2407.05000) | **Screened.** Gradient-informed rather than pretrained-weight-informed initialization. Different alignment target. |
| [LoRA+](https://arxiv.org/abs/2402.12354) | **Screened.** Factor-specific LRs show why one scalar LR does not exhaust optimizer tuning. |
| [TaRA, v1](https://arxiv.org/html/2609.02639v1) | **Screened method/related work.** Approximates full-weight training gradients at initialization. No target DoRA initialization crossing located. |
| [SAD-LoRA, v1](https://arxiv.org/html/2607.04306v1) | **Screened method/related work.** Spectral alignment for knowledge distillation. DoRA citation is contextual; distillation objective differs from the proposal. |

Further optimizer context, screened at abstract level: [Riemannian Preconditioned LoRA](https://arxiv.org/abs/2402.02347v3), [LoRA meets Riemannion](https://arxiv.org/abs/2507.12142v2), [ISO-LoRA](https://arxiv.org/abs/2609.12123v1), and [A Riemannian Geometry for Low-rank Adaptation](https://arxiv.org/abs/2610.08049v1). These make optimizer-dependent geometry an important limitation; their presence does not establish the targeted empirical interaction.

Additional search leads were screened as adjacent rather than decisive: [CORA](https://arxiv.org/abs/2607.02576), [ReCoLoRA](https://arxiv.org/abs/2607.07719), and [Prism-LoRA](https://arxiv.org/abs/2609.15643). These were discovery-level screens, not full negative audits. No conclusion of absence depends on them.

## 4. What follows scientifically—and what does not

There are three meanings of “conflict,” and they require different evidence:

1. **Incompatibility:** the hybrid cannot train or cannot be useful. Existing positive hybrid experiments argue against this broad claim.
2. **Negative interaction:** spectral initialization helps less when DoRA is present. This can happen even when the hybrid is the best method.
3. **Harmful combination:** spectral initialization actually lowers the tuned DoRA score. This is stronger than a negative interaction.

Similarly, three meanings of “magnitude” must be separated: initial adapter-factor/product scale, norm of the reconstructed weight, and size of the optimizer-induced effective weight step. Matching one does not match the others.

For a column vector, the DoRA form is

\[
w=m\,z/\|z\|,\qquad z=w_{\mathrm{res}}+Ba.
\]

Normalizing z does not normalize Ba in isolation. A unit-length direction can rotate by a small or large angle in one optimizer step. The magnitude parameter can also change. Consequently, the inference “spectral methods benefit from scale, therefore DoRA must remove their benefit” is not logically valid. This is an algebraic interpretation, consistent with the older normalization literature, not a new empirical finding.

For the repository's additive residual construction, small simultaneous SGD steps produce

\[
\Delta W=-\eta(BB^\top G+GA^\top A)+O(\eta^2).
\]

Changing factor orientation changes the mapping from G to the effective update even with matched factor spectra. AdamW adds coordinatewise adaptation, so a scalar LR search cannot generally make these mappings identical. This motivates measuring effective steps; it does not predict a performance advantage.

Do not infer equivalence from nonsignificance. Do not infer a mechanism solely from a correlation between norms and accuracy. Do not infer general subspace irrelevance from a successful scalar match in one experimental regime.

## 5. Research question and contribution

> Does DoRA amplify, attenuate, or reverse the benefit of spectral initialization relative to spectrum-matched random initialization after per-condition learning-rate and initialization-scale tuning?

The secondary question asks whether exact versus detached denominator gradients change that benefit. Shared-state effective-step diagnostics investigate how initialization and the gradient rule interact.

Prior studies establish successful spectral/DoRA hybrids and initialization ablations. This project reassesses their complementarity through an explicit interaction, independently selected hyperparameters, and controlled random initializations. It measures effective weight updates to distinguish local scale effects from directional differences. Matched starting spectra do not establish matched effective updates throughout training.

This is a worthwhile course project with a falsifiable target. Its value as a standalone research publication depends on the precision, explanatory evidence, and replication of the results. A small single-task leaderboard difference alone would offer limited novelty.

## 6. Search coverage, limitations, and unresolved checks

The search started from all seven original references, then used combination/initialization/normalization/tuning terminology, exact-title citation searches, related-work links, later benchmark papers, and official code. Primary sources included arXiv, OpenReview, ACL Anthology, conference proceedings, and author repositories. Secondary aggregators were discovery aids only. Relevant sections and appendices of the nearest candidates were inspected; wider candidates are explicitly marked screened. See [LITERATURE_SEARCH_LOG.md](LITERATURE_SEARCH_LOG.md).

This was a broad, question-directed audit, **not a complete export and screening of every citation in a bibliographic database**. Search-engine indexing can miss papers, author code may be newer than the experiment, and inaccessible submissions remain a limitation. The audit does not establish universal absence of an answer or exhaustive coverage of the literature.

The remaining provenance checks are narrow: independent retuning of PiSSA Table 5's four cells; independent retuning of EDoRA Table 3's random/SVD arms; and whether an accessible later DuDe revision changes its tuning protocol. None should be represented as “definitely not done” merely because documentation was not located. No authors were contacted.

**Assessment:** direct prior evidence argues against universal incompatibility. The independently tuned, spectrum-controlled interaction is a plausible research gap and a worthwhile target for empirical reassessment. Publication priority remains unverified; a strong contribution requires informative estimates and explanatory evidence.

## Appendix: detachment and effective weight steps

Exact and detached DoRA share the forward function at identical parameters. They differ in whether gradients pass through the norm denominator.

Using the factor map $\mathcal K(H)=BB^\top H+HA^\top A$ and effective-weight loss gradient $G$, at initialization let $m_i=\|W_{0,i:}\|_2$, and let $\mathcal P$ remove each row's component parallel to the corresponding row of $W_0$. For small simultaneous SGD steps, without clipping, decay, or momentum, the directional contributions are

$$\Delta W_E=-\eta\mathcal P\mathcal K(\mathcal P G)+O(\eta^2),$$
$$\Delta W_D=-\eta\mathcal P\mathcal K(G)+O(\eta^2).$$

With the same magnitude step in both arms, that contribution cancels in the difference:

$$\Delta W_D-\Delta W_E=-\eta\mathcal P\mathcal K((\mathrm{Id}-\mathcal P)G)+O(\eta^2).$$

A radial gradient component can pass through the factor map and acquire a tangential component. Whether that happens depends on factor orientation; neither its sign nor a performance advantage follows generally. This motivates the secondary experiment without assuming exact gradients are better.

AdamW, clipping, and later optimizer states change this calculation. The experiment therefore measures actual effective steps at shared parameters, batches, and optimizer states. Comparing native steps and equal-norm counterfactuals tests whether a local difference is solely scalar. It does not equalize entire training trajectories or establish a causal explanation of final accuracy by itself.
