# Study guide and checklist

Use the checkboxes below as a template; record completion with each study's report data.
Keep this shared guide unchecked. See the [test plan](TEST_PLAN.md) for experiment details.
Core: rank 16, eight conditions, three confirmation seeds.

Run from the repository root. Set these in **each new terminal**; `study` is a shortcut for the runner:

```bash
export DORA_DATA=runs/data
export DORA_VALIDATION=runs/validation-h100
export DORA_STUDY=runs/study-main
study() {
  .venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" "$@"
}
```

Keep code, dependencies, and data fixed. For GPU work, set time caps below the remaining allocation.
Repeat a stage command to resume; finished jobs are skipped. Finish each stage before the next.
Choose fresh validation/study directories for new runs, or point to compatible existing evidence.

## 1. Install and check

```bash
bash scripts/setup_pace.sh cuda
.venv/bin/python -m pytest -q --junitxml="reports/$USER/unit-tests.xml"
```

- [ ] Install locked dependencies, pass CPU tests, and save their JUnit results.

## 2. Prepare data

```bash
.venv/bin/python -m dora_spectral.experiment prepare --data "$DORA_DATA"
```

- [ ] Create or verify the pinned training/pilot/development splits (`prepared.json`).

Official-test data stays unopened.

## 3. Validate on H100

```bash
.venv/bin/python -m dora_spectral.validation \
  --data "$DORA_DATA" --directory "$DORA_VALIDATION" --seconds 7200
```

- [ ] Pass initialization, gradient, overfit, memory, and resume gates (`validation.json`: `passed: true`).

Reuse a passed validation only when its code, configuration, data, and package versions match.
Otherwise validate in a fresh directory; compatible prepared data/SVD caches can be reused.

## 4. Initialize the study

```bash
study init --data "$DORA_DATA" --validation "$DORA_VALIDATION" \
  --budget-hours 100 --previous-hours 1
```

- [ ] Save the configuration, grid, budget, and seeds in a fresh study directory (`study.json`).

**Replace `1` with actual earlier allocated GPU hours**, including idle time; exclude the selected validation, counted automatically.

## 5. Run the measured pilot

```bash
study pilot --wall-hours 3 --job-seconds 3600
```

- [ ] Finish short profiles and the slowest condition's three-epoch pilot (`pilot.json`: `passed: true`).

Require improved NLL, usable accuracy, and ≤1% truncation.
`--wall-hours` caps this stage invocation; `--job-seconds` caps each child run.

## 6. Account for idle time and freeze

```bash
study account --hours 0.25 --note "additional allocated setup/idle time"
study freeze
```

Replace `0.25` with additional uncounted allocated hours; skip `account` if none.
Repeat accounting after later allocations as needed. `freeze` checks pilot/budget feasibility and preserves a ten-hour failure reserve.

- [ ] Record uncounted allocation time and pass the budget gate (`ledger.jsonl`, `frozen.json`).

## 7. Run mechanism diagnostics

```bash
study diagnose --wall-hours 1 --job-seconds 1800
```

- [ ] Finish four DoRA runs comparing exact/detached updates at steps 0/50/100 (`counterfactuals.jsonl`).

## 8. Screen, then promote

```bash
study screen --wall-hours 3 --job-seconds 3600
study promote --wall-hours 3 --job-seconds 3600
```

- [ ] `screen`: finish 96 one-epoch candidates, scoring 256 development questions (`search/`).
- [ ] `promote`: continue up to 24 finalists to three epochs and score all 500 (`promotions.json`).

Retain numerical failures; require a usable finalist for every condition.

## 9. Select, then confirm

```bash
study select
study confirm --wall-hours 3 --job-seconds 3600
```

- [ ] `select`: lock eight winning configurations (`selection.json`, `selection-lock.json`).
- [ ] `confirm`: finish 24 full runs on seeds 101/202/303, with development checks each epoch (`confirmation/`).

## 10. Evaluate the official test

```bash
study test --wall-hours 3 --job-seconds 3600
```

- [ ] After all confirmations finish, score the base and final replicas on all 1,319 questions (`test/`).

Keep settings, scoring rules, and seeds unchanged after opening the test.

## 11. Analyze on CPU

```bash
study analyze --resamples 10000 --two-way
```

- [ ] Verify paired predictions and inspect statistics, CSV tables, and PNG figures (`analysis/` in the report folder).

<details>
<summary>Analyze copied results on your laptop instead</summary>

```bash
uv sync --locked --extra cpu --extra models --extra analysis
.venv/bin/python -m dora_spectral.study \
  --directory "reports/<execution-user>/<study>" analyze --resamples 10000 --two-way
```

Replace the placeholders with the original execution username and study name.

</details>

## Status and saving

```bash
study status
.venv/bin/python -m dora_spectral.reporting --runs "$DORA_STUDY"
```

`status` shows accounted hours and lock state. Evidence copies automatically to `reports/<execution-user>/`;
`reporting` copies it manually after a hard interruption. Commit the report folder when ready.
Back up checkpoints in `runs/` separately for resume. Check a milestone only after all required jobs finish.
