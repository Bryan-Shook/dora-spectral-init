# Running the fixed-rank LR × gamma study

Use the [progress checklist](PROGRESS.md) to see completed milestones and the next step.

Run these commands from the repository root, using the same Git revision throughout a study.
This implements the eight-condition, rank-16 core of [TEST_PLAN.md](TEST_PLAN.md): six controlled
conditions plus ordinary LoRA and DoRA. The optional 200-hour fixed-scale/finalist-retuning
extensions are not launched by these commands. Five confirmation seeds can be chosen at `init`.

The commands save data, not narrative summaries. Raw files and resumable checkpoints stay under
`runs/`; small evidence is copied automatically under `reports/<execution-user>/`. Exceptions,
interruptions, predictions, diagnostics, source hashes, package versions, and GPU details are kept.
No Git commit or push happens automatically.

## 1. Install and run the CPU checks

```bash
bash scripts/setup_pace.sh cuda
.venv/bin/python -m pytest -q --junitxml="reports/$USER/unit-tests.xml"
```

The first command installs the locked CUDA, model, and analysis dependencies. The second exercises gradients,
effective-weight measurements, reversible counterfactuals, checkpoint resume, selection rules,
test-access gates, reporting, and a simulated complete study. It saves JUnit results. These small
tests use CPU models; passing them does not establish H100 memory or throughput.

Set working paths at the start of each shell session:

```bash
export DORA_DATA=runs/data
export DORA_VALIDATION=runs/validation-h100-v3
export DORA_STUDY=runs/study-main
```

Use fresh validation/study directory names if these already exist. The earlier `h100-check` runs
remain useful smoke-test evidence, and `validation-h100-v2` passed for its recorded implementation.
The named-stream implementation changes random draws and source hashes, so rerun validation in
`validation-h100-v3` before study initialization. The new study does not resume older offset-seed
checkpoints. The existing pinned tokenized data and SVD cache can be reused when their
checksums/revisions match.

Randomness uses stable, versioned hashes of seed block, purpose, and matrix name. This pairs
controlled random directions across A-R/D-R/E-R and gamma/LR candidates, pairs ordinary L0/D0
factors, and keeps data ordering separate. `run.json` records the stream version and shuffle seed;
`initialization.json` records per-matrix seeds. Layer registration order cannot change these keys.

## 2. Prepare the training-derived partitions

```bash
.venv/bin/python -m dora_spectral.experiment prepare --data "$DORA_DATA"
```

This downloads/tokenizes the pinned training split or verifies an existing preparation. It creates
the training, 500-question pilot, and 500-question development partitions, answer-only labels,
length/exclusion records, and a checksum. It does not download the official test split. CPU is
sufficient for preparation.

## 3. Validate on an allocated H100

```bash
.venv/bin/python -m dora_spectral.validation \
  --data "$DORA_DATA" --directory "$DORA_VALIDATION" --seconds 7200
```

This runs the real-model gates:

- Compare all eight initial models with untouched native-BF16 outputs and effective weights.
  Logit tolerances are recorded from repeat-execution noise before training comparisons.
- Compare independent dense and PEFT gradients on a real projection slice, plus FP64 gradcheck.
- Fit 32 training examples for 100 updates and check that frozen target weights stay unchanged.
- Exercise the longest effective training batch, including optimizer state/diagnostics, requiring
  at least 10% device-memory headroom.
- Compare six uninterrupted updates with three updates followed by checkpoint reload and three
  more updates.
- Measure initial pilot NLL and a 16-question untouched-base generation timing probe.

`validation.json` must say `passed: true`. Failed gates retain their data and exception; investigate
them before a search. A failed validation rerun uses a new directory. `--seconds` is a work cap,
not a runtime estimate: set it below the allocation's remaining time, leaving cleanup time.

## 4. Create the proposed study

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" init \
  --data "$DORA_DATA" --validation "$DORA_VALIDATION" \
  --budget-hours 100 --previous-hours 1
```

**Replace `1` with the actual earlier GPU allocation hours**, including previous pilots and idle
allocated time. Do not include the new validation here: its measured duration is charged
automatically. This writes a source/version/data-bound proposed protocol, snapshots the common
configuration and passed validation, and prespecifies the grid, practical margin, and fresh seed
blocks `101, 202, 303`. It does not start training.

Choose a different total cap or five seeds here, before running confirmation. For example,
`--budget-hours 200 --seeds 101 202 303 404 505` requests the same core with five seeds; it does not
implicitly request the other optional extensions. The default failure reserve is ten hours.

## 5. Run the measured pilot

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" pilot \
  --wall-hours 3 --job-seconds 3600
```

This runs the four controlled DoRA conditions for 100 steps on pilot seeds `0, 1, 2`, and profiles
A-S/A-R/L0/D0 for 20 steps. Each short run generates 16 pilot answers. It locks the slowest
condition using training time per update, resumes that condition to all three epochs, and scores
all 500 pilot questions. The full run must improve held-out NLL, avoid unusable/saturated accuracy,
and have no more than 1% output-limit truncation. No hyperparameters are selected from the pilot.

For every GPU stage, set `--wall-hours` **below the remaining allocation time**. It caps the whole
stage invocation; `--job-seconds` caps an individual child invocation. Repeat the same command in
a later allocation to continue: completed jobs are verified/skipped and interrupted jobs resume.
Keep the code, environment, data, and study configuration fixed. Model errors stop the stage;
numerical divergence is retained as a failed candidate only during screening/promotion.

## 6. Account for idle allocation and freeze the search

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" account \
  --hours 0.25 --note "allocated setup/idle time outside measured jobs"
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" freeze
```

The accounting command is an example: replace `0.25` with **only the additional allocated time not
already covered** by earlier allocations, validation duration, or the driver ledger. Repeat after
each allocation as needed. The driver measures child-job wall time, including startup and I/O;
it cannot automatically know scheduler billing or idle time.

`freeze` checks the passed validation and full pilot, forecasts remaining training, scoring,
diagnostics, and invocation overhead from measured timings, and preserves the failure reserve.
It refuses a forecast beyond the cap. No search starts until this gate passes. If the core does
not fit, revise the common setup/scope before starting a new validated study. The runner does not
silently remove conditions. The forecast is updated before subsequent jobs and can stop a stage.

## 7. Run shared-state mechanism diagnostics

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" diagnose \
  --wall-hours 1 --job-seconds 1800
```

This runs E-S/E-R/D-S/D-R for 100 steps at common LR `1e-4`, gamma `1`, seed `40`. At steps 0, 50,
and 100 it compares hypothetical exact/detached **clipped AdamW** updates from identical parameters
and optimizer moments. It restores the actual training state afterward. Saved measurements include
step distance/cosine, per-layer results, and local prediction/loss probes at actual and equal small
effective-step norms. These local interventions do not establish matched full training trajectories.

## 8. Screen and promote LR × gamma candidates

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" screen \
  --wall-hours 3 --job-seconds 3600
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" promote \
  --wall-hours 3 --job-seconds 3600
```

`screen` trains 96 candidates for one epoch: 12 LR/gamma combinations per controlled condition
and 12 logarithmically spaced LRs for each ordinary baseline. It uses the fixed 256-question
development subset and computes answer NLL on all 500 development examples. Every candidate uses
the full three-epoch LR schedule from its first update.

After screening is complete, `promote` retains the best LR at each gamma in each controlled
condition and the top three LRs per ordinary baseline, then continues those checkpoints through
epochs two and three. It evaluates all 500 development questions. Normally there are 24 promoted
candidates; numerically failed groups can have fewer. All eight conditions must retain a usable
finalist. Ties use lower full-development NLL, then lower LR (and lower gamma for final selection).

Each stage can require several allocations. Finish/repeat `screen` until complete before calling
`promote`; then finish/repeat `promote`. The checkpoints, optimizer moments, LR schedule, and data
order continue from their saved states.

## 9. Lock winners and confirm on fresh seeds

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" select
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" confirm \
  --wall-hours 3 --job-seconds 3600
```

`select` chooses one full-run winner per condition and writes immutable selected settings and a
checksum lock. It retains every full-run finalist and flags LR/gamma boundary selections. This is
a bounded search with a single tuning seed; boundary selections limit the claim. Expanding the
grid requires a new frozen protocol/budget before confirmation/test inspection.

`confirm` trains all eight winners from fresh starts on each prespecified paired seed block:
24 full training runs for the default three seeds. It saves fixed-128-question development curves
after epochs one and two and full-500 development results after epoch three. It implements these
as three resumable invocations per replica, with the same full-horizon schedule throughout. It
does not choose an intermediate checkpoint or reuse tuning runs as seed replicates.

Ordinary and controlled training runs also record effective merged-weight movement, angles,
parallel/perpendicular step components, and magnitude changes at steps 0/1/10/50/100/final.
Representative layers are followed during training; final measurements include all adapted layers.

## 10. Evaluate the official test once settings are locked

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" test \
  --wall-hours 3 --job-seconds 3600
```

This checks that **every paired confirmation replica is complete** and the selection checksum is
unchanged. Only then can it download/prepare the pinned 1,319-question official test split. It
scores the untouched base model once and every final confirmation checkpoint with greedy decoding.
All questions remain in the denominator; malformed/missing/truncated answers count as incorrect.
It saves raw text, normalized reference/prediction, question hashes, correctness, and generation
lengths. Test interruption is resumable, though an incomplete generation pass is regenerated.
Official-test results cannot change selected settings, parser, generation settings, or seed count.

## 11. Analyze on CPU

On the cluster, install the optional analysis dependencies while preserving the CUDA environment:

```bash
uv sync --locked --extra cuda --extra models --extra analysis
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" analyze \
  --resamples 10000 --two-way
```

This checks full test-question pairing and strict saved scoring, then writes machine-readable
statistics, per-seed and condition CSV tables, cost/failure tables, and PNG scientific figures.
The primary contrast is `J = (D-S − D-R) − (A-S − A-R)`; `I = (E-S − E-R) − (D-S − D-R)` is
secondary. It reports paired Student-t intervals across training seeds, paired-question bootstrap
intervals, and optional two-way seed/question bootstrap intervals. Layers/questions are not counted
as independent training runs. Only J receives the primary interpretation against the fixed ±2 pp
practical band; other comparisons remain secondary.

Figures show interactions with intervals, screening surfaces, development learning curves, and
effective-weight movement. Results remain conditional on the searched settings, model/task/rank,
screening procedure, and selected configurations. The script does not create extra narrative
reports during normal experiment runs.

After pushing copied evidence, the same analysis can run locally without cluster checkpoints:

```bash
uv sync --locked --extra cpu --extra models --extra analysis
.venv/bin/python -m dora_spectral.study \
  --directory reports/mchoi345/study-main analyze --resamples 10000 --two-way
```

Replace `mchoi345` with the execution username. Analysis outputs go into that study's report folder.

## Status and evidence transfer

```bash
.venv/bin/python -m dora_spectral.study --directory "$DORA_STUDY" status
.venv/bin/python -m dora_spectral.reporting --runs "$DORA_STUDY"
git add reports/
git commit -m "Record study evidence"
git push
```

`status` prints accounting and stage-lock status. The copy command is useful after hard termination
before cleanup, or for exporting existing runs. Stage cleanup normally performs it automatically.
It copies JSON/JSONL/YAML/log/XML data while retaining raw originals and excluding checkpoints and
weight files. Keep a separate checkpoint backup for recovery; a Git-only evidence copy supports
analysis but cannot resume GPU training.
