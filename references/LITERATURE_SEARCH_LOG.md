# Literature audit search record

Date: 2026-10-11. Scope: spectral/pretrained-subspace initialization × magnitude/direction normalization, with particular attention to experiment-specific tuning. This is a readable query and disposition record, not a PRISMA census or a complete raw search export. Queries below were issued during the audit; related papers were also followed from source references and official repositories.

## Query families

Direct combinations:

```text
"DoRA" "PiSSA" "learning rate" combination
"DoRA" "spectral initialization" normalization
"DuDe" "Dual Decomposition" low rank adaptation
"DoRA" "LoRAM"
"DoRA" "PiSSA" hybrid tuning ablation
"DoRA" "SVD" initialization normalization learning rate
"DoRA" "spectral" "magnitude"
"PiDoRA" OR "SVD-DoRA" OR "spectral DoRA"
"DoRA" "OLoRA" combination
"DoRA" "MiLoRA" combination
"DoRA" "PiSSA" "interaction"
"DoRA" "PiSSA" "independently"
"DoRA" "spectral" "conflict"
"DoRA" "PiSSA" "factorial"
"DoRA" "initialization" "magnitude matched"
"DoRA" "PiSSA" combination
"DoRA" "PiSSA" "tuning" "combined"
"DoRA" "PiSSA" "synergy"
"DoRA" "spectral" "matched"
"weight-decomposed" "spectral" initialization
```

Mechanism, tuning, and exact-title follow-ups:

```text
"spectral initialization" magnitude direction low rank adaptation
"low-rank adaptation" "normalization" "initialization" "learning rate"
"EDoRA" "random" "learning rate"
"EDoRA" "initialization"
"Which LoRA" "Multilingual"
"Balanced LoRA" "Removing Parameter"
"SAD-LoRA" spectral alignment
"DoRA" "SVD" "grid search"
"DuDe" "learning rate" adaptation tuning
"Dual Decomposition of Weights" -site:researchgate.net
"The Primacy of Magnitude" "DoRA"
BiDoRA low rank adaptation initialization
DoRA spectral initialization synergy PiSSA normalization study
"spectral initialization" "normalization" LoRA
```

Selected queries were restricted to arxiv.org, openreview.net, and aclanthology.org. Broad queries produced considerable acronym noise. Results about the DORA DevOps survey, orexin antagonists, the DUDE document benchmark, unrelated spectral analysis, and unrelated LoRAM acronyms were excluded.

## Evidence inspection record

| Source | Material checked | Disposition |
|---|---|---|
| PiSSA v4 | Appendix A Table 5, Appendix L tuning, gradient discussion; official repository landing page | Direct four-cell evidence; hybrid-specific independent tuning unresolved. |
| EDoRA v1 | Method §3, §4.1–4.3, Tables 2–4, Appendix A; official main script, launcher and initialization utility | Direct SVD/random normalized-adaptation ablation; substantial grid-search code verified; scale control absent from inspected random construction. |
| DuDe v2 | Method, comparison recipes §§4.1–4.2, seed/rank studies, §4.6, Appendix A | Direct hybrid; shared-hyperparameter comparisons. |
| DuDe OpenReview copy | Search discovery; direct PDF requested | Browser challenge; not treated as a fully audited revision. |
| LoRAM v2 | §4.3 tracking/basis tests; DoRA occurrences and limitations context | Direct magnitude/subspace evidence for additive methods. |
| Learning Rate Matters v2 | Evaluated method list, LR grids, §4 Hessian, implementation appendix | Direct independent-tuning evidence without hybrid cell. |
| LoRA-XS v3 | Initialization ablations, random/SVD-of-random/SVD-of-W LR tables, DoRA occurrences | Important tuned subspace evidence without DoRA factorial. |
| DeLoRA v1 | Method/initialization, LR robustness, tuning appendix, fixed-magnitude DoRA ablation | Related normalization; different experiment. |
| NoRA v1 | Parameterization table, Table 5 DoRA combination, Appendix B settings | Related initialization×DoRA evidence; shared SFT LR, not pretrained SVD. |
| DisLoRA EMNLP PDF | SVD method, normalization definition, baselines, appendix hyperparameters | Different normalization; not DoRA hybrid. |
| Balanced LoRA v1 | Initialization and balancing discussion, independent LR/scale search, baseline setup | Strong tuning precedent, no targeted combination. |
| Which LoRA? v1 | Method list, DoRA* definition, parameter budgets, Appendix D | Tuned multilingual comparison; no hybrid. |
| RLVR evaluation v1 | Method categories, common hyperparameters, LR ablation tables | Different regime; no hybrid comparison. |
| RLPO/RLMO v1 | Subspace/scale sections, LR sensitivity, DoRA occurrences | Mechanistic overlap outside DoRA and SFT. |
| ULoRA v1 | Related work, initialization family, LR sweep rationale | Geometry/tuning overlap; no target hybrid. |
| Factorized Neural Layers 2021 | §3.3 normalization and effective-step-size discussion | Strong conceptual predecessor. |
| BiDoRA v1 | Method/related-work screening, PiSSA search | Bilevel magnitude/direction optimization; not target comparison. |
| TaRA v1; SAD-LoRA v1 | Method/related-work screens and DoRA occurrences | Gradient initialization/distillation context. |

## Other discovered leads and exclusions

- LOST, an EDoRA-citing ICLR submission: inspected abstract and motivation in its [accessible PDF](https://openreview.net/pdf/8cc571601e64badd90da744946ac3e11ac09d971.pdf). It concerns low-rank/sparse pretraining, not the target interaction.
- Bi-LoRA: inspected retrieved integration table/section. [Table 4](https://openreview.net/pdf?id=zoYPlgX1bH) combines Bi-LoRA separately with PiSSA and DoRA; those are not PiSSA+DoRA cells.
- MoG: [retrieved table](https://openreview.net/pdf/3d2b6347d0bae25c3cf3b0ffb291f4aeb62177d0.pdf) similarly combines MoG separately with DoRA and PiSSA. Discovery screen only.
- Structure Over Scale: [retrieved RLVR discussion](https://openreview.net/pdf?id=XGZHI14m8P) concerns effective rank and model maturity; distinct from the similarly themed PEFT-RLVR evaluation. Not assumed to be the same paper.
- HD-PiSSA, TLoRA, Adapters Selector, FAAR, LieRA, nested NoRA, ConsNoTrainLoRA, QR-LoRA, and AIRA appeared as nearby search leads. They were not all audited in full and are not used to certify absence of prior art.
- Newer optimizer papers listed in the audit were screened at abstract level for context; they were not independently reproduced.

## Access and interpretation limits

The terminal could not resolve external hosts. Web access supplied primary HTML/PDFs and raw author source files. GitHub tree/API views for EDoRA failed, but direct raw launcher and initialization files succeeded. No code was executed from these repositories, and no training logs were independently reproduced.

The review explicitly distinguishes “documented shared settings,” “documented tuning,” and “experiment-specific tuning not verified.” Finding a tuning loop does not prove which runs produced a table. Finding no named hybrid in a method list is stronger than merely failing to retrieve it in keyword search, but neither establishes universal absence across the entire literature.
