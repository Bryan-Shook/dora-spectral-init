# Experiment progress checklist

Last reviewed: **2026-10-16**, using the results available in this repository.
Unchecked items have no completion evidence recorded here yet; cluster work may be ahead.

**Current position: H100 validation passed. Next: initialize the study, then run the measured pilot.**

The [test plan](TEST_PLAN.md) defines the protocol. The [run guide](RUN_STUDY.md) explains every
command and its outputs. This checklist tracks completion of the rank-16, eight-condition core.

## Completed preparation

- [x] Implement the study workflow and verify CPU checks: 92 tests passed in the locked
  PyTorch 2.6 environment during implementation. Save subsequent test results using run-guide step 1.
- [x] Prepare the pinned training-derived partitions. The successful validation records their
  [data checksum and shared configuration](../reports/mchoi345/validation-h100-v2/validation.json).
- [x] Complete the earlier eight-condition, 100-step smoke check:
  [saved evidence](../reports/mchoi345/h100-check). These are preliminary runs, separate from
  the measured pilot and confirmation replicas below.
- [x] Pass real-model H100 validation:
  [validation gates](../reports/mchoi345/validation-h100-v2/validation.json) and
  [execution status](../reports/mchoi345/validation-h100-v2/status.json).
  Initialization/gradients, overfit/frozen weights, memory, and checkpoint resume all passed.

## Study milestones

- [ ] **Initialize the study** — run-guide step 4, `init`. Record the actual previous GPU
  allocation hours and lock the proposed configuration, budget, and fresh seeds.
  Evidence: `study.json`, `config.yaml`, and `validation-evidence.json`.
- [ ] **Pass the measured pilot** — step 5, `pilot`. Finish the short profiles and the slowest
  condition's full three-epoch run. Require improved pilot NLL, usable accuracy, and at most
  1% truncation. Evidence: `pilot.json` with `passed: true`.
- [ ] **Account for idle time and freeze the search** — step 6, `account` and `freeze`.
  Check the measured forecast fits the budget with the failure reserve.
  Evidence: `ledger.jsonl` and `frozen.json`.
- [ ] **Complete mechanism diagnostics** — step 7, `diagnose`. Finish the four shared-setting
  runs and retain the exact/detached counterfactual measurements.
  Evidence: completed `diagnostics/` runs and their `counterfactuals.jsonl` files.
- [ ] **Complete all 96 screening candidates** — step 8, `screen`. Retain completed results
  and recorded numerical failures. Evidence: `search/` run metadata, metrics, and statuses.
- [ ] **Complete promotion** — step 8, `promote`. Continue up to 24 finalists through three
  epochs, retaining at least one usable finalist per condition.
  Evidence: `promotions.json` and final development metrics in the promoted `search/` directories.
- [ ] **Lock eight winning configurations** — step 9, `select`.
  Evidence: `selection.json` and `selection-lock.json`.
- [ ] **Complete fresh-seed confirmation** — step 9, `confirm`. Finish all eight conditions
  on every prespecified seed: 24 full runs for the default three seeds, including the development
  evaluations after each epoch. Evidence: complete `confirmation/` runs and development predictions.
- [ ] **Complete official-test evaluation** — step 10, `test`. After every confirmation run
  finishes, score the untouched base and all final replicas on all 1,319 questions.
  Evidence: `test-opened.json` and all `test/` evaluations/predictions.
- [ ] **Generate and inspect the final analysis** — step 11, `analyze`. Verify pairing and
  scoring, then inspect the primary/secondary contrasts, uncertainty, diagnostics, and costs.
  Evidence: `analysis/statistics.json`, CSV tables, and PNG figures.

Evidence paths in this section are relative to the study directory, normally `runs/study-main`.
The small evidence files are also copied under `reports/<execution-user>/study-main`.

## Keeping this current

Once a study has been initialized, run this on the cluster:

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main status
```

This displays accounted GPU hours and whether the protocol is frozen, winners are selected, and
official-test access has opened. It does not update this Markdown checklist or report per-job
completion counts. Repeat a stage's original command to verify/skip finished jobs and resume
unfinished jobs, as explained in the run guide.

Check a milestone only after its required evidence is complete and available in the repository.
Update the review date and current-position sentence when recording new results. Keep long stages
unchecked while any required jobs remain; preserve failed attempts with the successful results.
