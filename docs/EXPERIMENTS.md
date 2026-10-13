# Running the Qwen/GSM8K experiments

Run commands from the repository root, on the machine holding the environment and assets.
Use `.venv/bin/python` explicitly; a shell `pip` can point to PACE's shared Anaconda.

## What is implemented

- Revision-pinned model/tokenizer download and GSM8K download; saved, hashed partitions.
- All eight conditions: A-S/A-R, D-S/D-R, E-S/E-R, ordinary L0/D0.
- Query/value adapters, exact SVD caching, matched factor spectra and layer-specific paired seeds.
- Frozen backbone/biases, disabled dropout, FP32 adapters/norms inside a BF16 transformer.
- Answer/EOS-only loss; token-weighted accumulation; AdamW, clipping, fixed-horizon cosine schedule.
- Atomic resumable checkpoints including residuals, optimizer, scheduler, RNG, epoch and batch position.
- Greedy generation on a separate merged model, strict numeric scoring, raw predictions and answer NLL.
- Equal-budget search specs, per-scale promotion, configuration selection, frozen confirmation specs.
- Seed-level J/I intervals and paired question bootstrap; timing, memory and movement logs.

This is a **reference implementation**: target projections build full FP32 effective weights.
Do not assume the proposed H100 budget fits. The GPU numerical, learning, memory and budget
pilot gates in TEST_PLAN.md still have to pass. No full-size results have been produced by
adding this code. H3 optimizer counterfactuals/equal-norm probes, publication figures, and an
allocation-wide budget ledger are not automated here. Fixed-scale/native-PiSSA extensions
need separately budgeted job specifications.

## 1. Environment and offline checks

On PACE with a compatible NVIDIA driver:

```bash
bash scripts/setup_pace.sh cuda
.venv/bin/python -m pytest -q
```

Tests include a locally constructed tiny Qwen model and tokenizer; no model download is
needed to run them. CPU installation uses `bash scripts/setup_pace.sh cpu`.

## 2. Download once, outside a reserved GPU session

On a machine with internet access and sufficient storage:

```bash
.venv/bin/python -m dora_spectral.prepare --output data/qwen-gsm8k
```

Defaults are `Qwen/Qwen2.5-1.5B` and `openai/gsm8k` (`main` configuration).
The command resolves repository revisions to immutable commit IDs **before** downloading.
It downloads model/tokenizer assets and saves train/pilot/dev/test JSONL files and
`manifest.json`. Subsequent training/evaluation loads only local files. Transfer the entire
asset directory if downloads and training happen on different machines.

Pass `--model-revision` and `--dataset-revision` to reproduce previously chosen revisions.
An existing manifest cannot be overwritten; use a new asset directory to change versions.
An interrupted download without a manifest can be retried.

The train-derived split uses the hash ordering specified in TEST_PLAN.md. Normalized
question duplicates stay together, so partition counts can differ if duplicates exist.
Train questions overlapping the official test are removed and listed. Saved partition
hashes are checked before every run. Dataset solutions retain reasoning and `####` answers;
calculator annotations are removed at tokenization time.

## 3. Initial numerical check on an allocated GPU

```bash
.venv/bin/python -m dora_spectral.check_model \
  --assets data/qwen-gsm8k --output runs/qwen-initial-check.json
```

This measures repeated base execution, the change from FP32 target arithmetic, all eight
initializations, and merged/unmerged agreement. It reports maximum and relative RMS logit
errors and next-token agreement. By default it reports evidence without declaring BF16
acceptance. Set and justify a common `--max-logit-tolerance` from the baseline evidence,
then repeat the check before training. A failed acceptance exits nonzero.

CUDA initialization uses the accuracy-oriented `gesvd` driver. Factor spectra are measured
independently on CPU in float64 after construction, retaining the `1e-5` relative-error
threshold. Versioned cache filenames prevent reusing decompositions made by the old
default CUDA driver. A spectrum failure prints condition, A/B errors and measurement
method; it must be investigated rather than bypassed by widening the tolerance.

## 4. Short pilot, full pilot, and 32-example overfit

Inside a GPU allocation:

```bash
.venv/bin/python -m dora_spectral.experiment train \
  --config configs/pilot.yaml --output runs/pilot-D-S-seed0 --stop-after-steps 100

.venv/bin/python -m dora_spectral.experiment evaluate \
  --run runs/pilot-D-S-seed0 --split pilot --output runs/pilot-D-S-seed0/pilot-score
```

Copy `configs/pilot.yaml` and change condition/seed to run D-S/D-R/E-S/E-R at seeds 0/1/2;
profile A-S/A-R and L0/D0 as required by the plan. Each run needs its own output directory.
For L0/D0 use `gamma: 1.0`; the baseline has zero B and Kaiming-uniform A.
For the small overfit check use a copied config with `train_limit: 32` and a new output.
That option changes the training subset and labels the run in its metadata; it cannot be
used for official-test confirmation.

Continue the slowest pilot to the full three-epoch horizon:

```bash
.venv/bin/python -m dora_spectral.experiment train \
  --config configs/pilot.yaml --output runs/pilot-D-S-seed0 --resume
```

Configuration and asset identities must match on resume. A short pilot or one-epoch screen
uses the full three-epoch schedule from its first step; stopping early does not shorten it.
Checkpoints are saved every 25 steps and at each epoch. An interruption resumes from the
last saved optimizer boundary. Work since that checkpoint is repeated. Logs retain that
work; use checkpoint progress to identify repeated records.

Training examples longer than `max_length` are excluded with their IDs recorded in
`data.json`; no answer is silently truncated. Inspect that file before freezing the
protocol, consider a common 2,048 limit, and publish any remaining exclusions. Held-out
NLL retains complete solutions up to the model context limit. All evaluation questions
remain in the accuracy denominator; malformed or output-limit generations count as wrong.

`run.json`, `source.json`, `training.jsonl`, `checkpoint.pt`, `epoch-*.pt`, and `status.json`
describe a run. The source snapshot is saved and resume rejects implementation changes.
Logs include gradient clipping, active/target/padded token counts, elapsed session time, peak GPU
memory, and selected-layer effective movement. The final step records all-layer movement.
Timing includes model loading and initialization for each session. SLURM allocation time,
failed jobs and scoring still need to be included in the study's external budget ledger.

To evaluate the untouched base on pilot data, omit `--run` and pass
`--base-config configs/pilot.yaml`. Official-test base evaluation uses the frozen settings.

## 5. ICE submission

Create `logs/` **before** submission. Specify your allocation's partition, GPU and time
using the site's supported sbatch options; the scripts do not guess them.

```bash
mkdir -p logs
sbatch [your allocation options] scripts/train_ice.sbatch \
  --config configs/pilot.yaml --output runs/pilot-D-S-seed0 --stop-after-steps 100
```

For evaluation, use `scripts/evaluate_ice.sbatch` with the same arguments as the evaluate
command above. Both scripts enforce offline Hub/dataset operation.

## 6. Search specifications; submit only after the pilot gates pass

```bash
.venv/bin/python -m dora_spectral.protocol search --output runs/search-configs
```

This creates 96 YAML configs and `jobs.json`, using seed block 10. It does **not** launch
training. Controlled cells receive 3 scales × 4 learning rates; each ordinary baseline
receives 12 logarithmically spaced learning rates. Shared settings come from the pilot
config (`--base` can select a revised configuration frozen after the pilot).

For every job name, train with its YAML into `runs/search/<name>` using
`--stop-after-epochs 1`, then score:

```bash
.venv/bin/python -m dora_spectral.experiment evaluate \
  --run runs/search/A-S-00 --split dev --limit 256 \
  --output runs/search/A-S-00/screen
```

Generation uses the same first 256 hash-ordered development questions in every screen;
answer NLL uses the full development partition even when generation has `--limit`.
Once all screen results exist:

```bash
.venv/bin/python -m dora_spectral.protocol promote \
  --jobs runs/search-configs/jobs.json --output runs/promoted-configs
```

This selects the best LR at each controlled scale and the top three baseline rates (24
jobs). Resume those jobs in their **original run directories** with their original YAMLs.
After three epochs, evaluate all development questions into each run's `final-dev`:

```bash
.venv/bin/python -m dora_spectral.experiment evaluate \
  --run runs/search/A-S-00 --split dev --output runs/search/A-S-00/final-dev

.venv/bin/python -m dora_spectral.protocol select \
  --jobs runs/promoted-configs/jobs.json --output runs/selection.json
```

Selection uses development accuracy, then answer NLL, then lower LR/scale. It also preserves
per-scale LR winners. Failed candidates need to be investigated and recorded before
promotion; missing scores fail explicitly rather than silently dropping a condition.

## 7. Freeze, confirm, then evaluate the official test

Choose the practical margin **before** confirmation; the number below is a placeholder
that you must replace with the prespecified percentage-point margin.

```bash
.venv/bin/python -m dora_spectral.protocol freeze \
  --selection runs/selection.json --margin YOUR_MARGIN \
  --output runs/confirmation-configs
```

The command writes 24 fresh-seed configs (all eight conditions × 101/202/303) and
`frozen-protocol.json`. Train each config for the complete horizon into
`runs/confirmation/<condition>-seed<seed>`. Official-test evaluation requires the frozen
protocol, an eligible completed checkpoint, and all test questions:

```bash
.venv/bin/python -m dora_spectral.experiment evaluate \
  --run runs/confirmation/D-S-seed101 --split test \
  --frozen-protocol runs/confirmation-configs/frozen-protocol.json \
  --output runs/confirmation/D-S-seed101/test
```

The official test cannot be used with `--limit`. The untouched base can be scored once
with the same protocol by omitting `--run`. Do not change settings based on test outcomes.
Epoch checkpoints support the planned development learning curves (e.g.,
`--checkpoint epoch-1.pt --split dev --limit 128`), with no checkpoint selection.

When every confirmation result exists:

```bash
.venv/bin/python -m dora_spectral.analyze \
  --protocol runs/confirmation-configs/frozen-protocol.json \
  --output runs/confirmation-analysis.json
```

The output includes all eight accuracy tables, paired seed-level spectral gaps/J/I with
Student-t intervals, and 10,000 matched-question bootstrap resamples. It needs SciPy,
which the project's notebook dependency group installs. Three seeds remain only three
independent training runs; question bootstrap intervals describe different uncertainty.

API references: [Hub revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download),
[dataset loading](https://huggingface.co/docs/datasets/loading), and
[Qwen2 integration](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2).
