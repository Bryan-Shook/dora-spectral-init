# Study guide and checklist

Use the checkboxes below as a template; record completion with each study's report data.
Keep this shared guide unchecked. See the [test plan](TEST_PLAN.md) for experiment details.
Core: rank 16, eight conditions, three confirmation seeds.

Run from the repository root. Commands use explicit example paths; edit them for your study.
No environment-variable setup or shell shortcut is required.

Keep code, dependencies, and data fixed. For GPU work, set time caps below the remaining allocation.
Repeat a stage command to resume; finished jobs are skipped. Finish each stage before the next.
Choose fresh validation/study directories for new runs, or point to compatible existing evidence.
Runtime is recorded automatically. Unknown allocated GPU-hours and idle time are omitted.

## 1. Install and check

```bash
bash scripts/setup_pace.sh cuda
.venv/bin/python -m pytest -q --junitxml="reports/$USER/unit-tests.xml"
```

- [x] Install locked dependencies, pass CPU tests, and save their JUnit results.

## 2. Prepare data

```bash
.venv/bin/python -m dora_spectral.experiment prepare --data runs/data
```

- [x] Create or verify the pinned training/pilot/development splits (`prepared.json`).

Official-test data stays unopened.

## 3. Validate on H100

```bash
.venv/bin/python -m dora_spectral.validation \
  --data runs/data --directory runs/validation-h100 --seconds 7200
```

- [x] Pass initialization, gradient, overfit, memory, and resume gates (`validation.json`: `passed: true`).

Reuse a passed validation only when its runtime code, configuration, data, and package versions match.
Orchestration/analysis-only edits do not require another GPU validation.
Otherwise validate in a fresh directory; compatible prepared data/SVD caches can be reused.

## 4. Initialize the study

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main init \
  --data runs/data --validation runs/validation-h100 \
  --target-hours 100
```

- [ ] Save the configuration, grid, runtime estimate, and seeds in a fresh study directory (`study.json`).

Point `--validation` to the validation directory that passed for this study.
100 hours is a planning estimate, not a limit: exceeding it never blocks jobs or shortens training.
Each option uses the specified epochs; per-job/allocation time limits allow checkpoint resume.

## 5. Run the measured pilot

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main pilot \
  --wall-hours 3 --job-seconds 3600
```

- [ ] Finish short profiles and the slowest condition's three-epoch pilot (`pilot.json`: `passed: true`).

Require improved NLL, usable accuracy, and ≤1% truncation.
`--wall-hours` caps this stage invocation; `--job-seconds` caps each child run.

## 6. Freeze the study

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main freeze
```

- [ ] Pass the pilot, save a runtime forecast, and freeze the protocol (`frozen.json`).

## 7. Run mechanism diagnostics

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main diagnose \
  --wall-hours 1 --job-seconds 1800
```

- [ ] Finish four DoRA runs comparing exact/detached updates at steps 0/50/100 (`counterfactuals.jsonl`).

## 8. Screen, then promote

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main screen \
  --wall-hours 3 --job-seconds 3600
.venv/bin/python -m dora_spectral.study --directory runs/study-main promote \
  --wall-hours 3 --job-seconds 3600
```

- [ ] `screen`: finish 96 one-epoch candidates, scoring 256 development questions (`search/`).
- [ ] `promote`: continue up to 24 finalists to three epochs and score all 500 (`promotions.json`).

Retain numerical failures; require a usable finalist for every condition.

## 9. Select, then confirm

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main select
.venv/bin/python -m dora_spectral.study --directory runs/study-main confirm \
  --wall-hours 3 --job-seconds 3600
```

- [ ] `select`: lock eight winning configurations (`selection.json`, `selection-lock.json`).
- [ ] `confirm`: finish 24 full runs on seeds 101/202/303, with development checks each epoch (`confirmation/`).

## 10. Evaluate the official test

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main test \
  --wall-hours 3 --job-seconds 3600
```

- [ ] After all confirmations finish, score the base and final replicas on all 1,319 questions (`test/`).

Keep settings, scoring rules, and seeds unchanged after opening the test.

## 11. Analyze on CPU

```bash
.venv/bin/python -m dora_spectral.study --directory runs/study-main analyze \
  --resamples 10000 --two-way
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
.venv/bin/python -m dora_spectral.study --directory runs/study-main status
.venv/bin/python -m dora_spectral.reporting --runs runs/study-main
```

`status` shows recorded runtime and lock state. Evidence copies automatically to `reports/<execution-user>/`;
`reporting` copies it manually after a hard interruption. Commit the report folder when ready.
Back up checkpoints in `runs/` separately for resume. Check a milestone only after all required jobs finish.

## Different rank

**Current limitation:** the automated runner requires rank 16. Before another full study, update
the rank check in [study.py](../src/dora_spectral/study.py) to accept positive integer ranks,
keep the three-epoch/diagnostics requirements, and rerun the CPU tests. Validation and individual
training commands already support other ranks.

Example for rank 8:

1. Copy the configuration, then change `rank: 16` to `rank: 8` in the copy.

   ```bash
   cp configs/experiment.yaml configs/rank8.yaml
   ```

2. Validate that rank on H100 using a fresh directory.

   ```bash
   .venv/bin/python -m dora_spectral.validation --config configs/rank8.yaml \
     --data runs/data --directory runs/validation-r8 --seconds 7200
   ```

3. After enabling other ranks in the runner and passing validation, initialize a separate study.

   ```bash
   .venv/bin/python -m dora_spectral.study --directory runs/study-r8 init \
     --config configs/rank8.yaml --data runs/data --validation runs/validation-r8
   ```

Repeat steps 5–11 with `runs/study-r8`, independently tuning LR × gamma for that rank.
Reuse prepared data; SVD caches are separated by rank. Alpha automatically equals rank.
Choose the ranks to compare before official-test inspection, and keep each rank's results separate.
