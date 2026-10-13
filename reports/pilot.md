# H100 pilot results — 2026-10-13

> Historical report: the source used here is preserved in the local
> `runs/legacy-pilot-source.tar.gz` archive. Current implementation and status:
> [experiment guide](../docs/EXPERIMENT.md).

The authorized five-hour pilot completed in approximately 30.1 minutes of accounted allocation time. All twelve 100-step DoRA runs (seeds 0/1/2), four additive/ordinary profiles, the overfit and longest-batch checks, one complete three-epoch run, and all 500 pilot-validation predictions finished. No main search, confirmation, or official-test scoring ran.

## Complete pilot run

| Measurement | Result |
| --- | --- |
| Condition | D-S: detached DoRA, spectral initialization |
| Settings | Qwen2.5-1.5B base; q/v rank 16; gamma 1; LR 1e-4; effective batch 32; three epochs |
| Training examples / optimizer steps | 6,473 / 609 |
| Pilot-validation accuracy | 72.8% (364/500) |
| Held-out answer NLL | 0.889465 → 0.429690 |
| Output-limit truncation | 1/500 (0.2%); counted incorrect |
| Complete training | 248.12 s (4.14 min) |
| 500-question generation/scoring | 382.66 s (6.38 min) |
| Peak allocated / reserved memory | 20.19 / 36.86 GiB |
| Active / padded training tokens per second | 12105 / 19149 |
| Gradient clipping fraction | 1.81% |

This is a feasibility result from one full condition and one seed on pilot validation. It cannot establish a spectral benefit, a DoRA interaction, or a method ranking. Public-benchmark results do not establish absence of pretraining contamination.

## Correctness, learning, and memory

- All eight conditions had zero initial logit differences against the common FP32-target reference path on the two-example probe; initial effective-weight and factor-spectrum checks passed. Initial merged logits also agreed exactly.
- The 32-example overfit NLL fell from 0.875376 to 0.000587 in 100 steps.
- The longest-example exact-DoRA batch passed the 10% reserved-memory-headroom gate. Both longest-batch and full-run peaks were comfortably below the device capacity.
- Learning and generation feasibility flags passed: held-out loss declined, accuracy was within the prespecified diagnostic range, and truncation was below 1%.

## Numerical findings and limits

The first attempt stopped before training because the default CUDA SVD driver produced insufficiently orthogonal singular vectors. On the first q projection, spectral-factor errors were approximately 2.8e-4/4.8e-4. Switching to the accurate QR-based `gesvd` driver reduced these errors to approximately 5.6e-7/5.4e-7. The original 1e-5 tolerance was retained. The failed cache and measured driver comparison are preserved. CUDA decomposition now uses `gesvd`; CPU decomposition is unchanged.

The common FP32 target-projection arithmetic differs from the untouched BF16 model. On the two-example probe, max logit error was 8.795898, relative RMS drift was 0.059578, and next-token agreement was 97.30%. All eight initialized conditions matched the common reference exactly, but this does not establish equivalence to standard PEFT/BF16 execution. Accept/document this common path or validate a PEFT-based path before freezing the main study. Runtime numbers describe the custom dense FP32 adapter implementation.

## Training profiles

| Condition | Steps per run | Seeds | Mean training seconds | Max reserved GiB |
| --- | ---: | --- | ---: | ---: |
| D-S | 100 | 0, 1, 2 | 42.48 | 38.92 |
| D-R | 100 | 0, 1, 2 | 42.02 | 45.00 |
| E-S | 100 | 0, 1, 2 | 43.86 | 44.99 |
| E-R | 100 | 0, 1, 2 | 42.54 | 44.97 |
| A-S | 20 | 0 | 7.89 | 34.71 |
| A-R | 20 | 0 | 7.93 | 34.71 |
| L0 | 20 | 0 | 7.93 | 34.71 |
| D0 | 20 | 0 | 8.34 | 34.96 |

These are training profiles at common settings, not independently tuned accuracy comparisons. Timing includes routine logging and checkpointing, but standalone per-run training times exclude model loading/attachment and generation. Those costs contribute to the overall ledger.

## Provisional full-study cost

The runner forecasts 4.96 training hours for the protocol's 72 three-epoch equivalents. It uses the slower of the base-model generation probe and the trained-model evaluation to forecast 60.56 scoring hours. Including accounted pilot time and a ten-hour failure reserve gives 76.02 hours, before separate diagnostic, remaining initialization, and additional I/O costs.

The base-model generation probe had 93.75% truncation and zero strict accuracy on just 16 examples; it was slower than the trained model and is a conservative timing input. The forecast is provisional: output lengths, cold-start overhead, condition-specific scoring, and full search/resume behavior require continued accounting. The pilot supports feasibility within a 100-hour track but does not automatically freeze the protocol.

## Artifacts and verification

- `runs/pilot-5h/ledger.json`: completed work, accounted time, and preserved failure history.
- `runs/pilot-5h/numerical-checks.json` and `svd-driver-check.json`: numerical evidence.
- `runs/pilot-5h/full/`: 609-step checkpoint, training logs/summary, all predictions, and evaluation.
- `runs/pilot-5h/environment.json`: exact model/data revisions, software versions, source hashes, and GPU allocation.
- `runs/pilot-5h/study-forecast.json`: provisional cost projection.

The saved 500 IDs and per-example correctness were reconciled against the prepared pilot-validation partition using the strict scorer. All twelve short-run step counts and the full checkpoint (609 steps, 56 adapters) were verified. Raw results/caches remain gitignored; this report is reviewable repository content.

After completion, the runner status reporting was corrected so a resumed run immediately shows `running` rather than retaining its previous failure status. This changes reporting only and was not used to alter completed experimental computations.

Final post-run verification: all 45 CPU tests pass; Ruff, shell syntax, and patch whitespace checks pass. GPU memory returned to zero. Allocation elapsed at this verification was 35.3 minutes, including post-run CPU verification/reporting; subsequent idle allocation time is additional cost.

## Subsequent arithmetic investigation

The original drift summary above included padding. On valid tokens in the same two-example probe, maximum logit error was 1.21875, relative RMS drift was 2.978%, and next-token agreement was 100%. A larger probe confirmed small prediction differences remain. A native-BF16 forward prototype matched the original model exactly for all eight initialized conditions. The measured controls and metric-correction evidence are preserved in `runs/arithmetic-investigation/` and the archived report in `runs/legacy-pilot-source.tar.gz`. The completed pilot and its training arithmetic were not changed.
