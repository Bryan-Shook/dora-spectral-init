# Main experiment

This guide describes the implementation and execution workflow for comparing spectral and
spectrum-matched random initialization under additive adaptation, detached DoRA, and attached DoRA.
[TEST_PLAN.md](TEST_PLAN.md) defines the research hypotheses, tuning allocation, confirmation
design, and statistical analysis. [SETUP.md](SETUP.md) covers installation and cluster setup.
The [step-by-step study instructions](RUN_STUDY.md) explain validation, pilot, staged search,
confirmation, locked testing, and CPU analysis commands.

## Conditions

| Condition | Adaptation rule | Initialization |
| --- | --- | --- |
| A-S / A-R | Additive | Spectral / spectrum-matched random |
| D-S / D-R | DoRA, detached denominator | Spectral / spectrum-matched random |
| E-S / E-R | DoRA, attached denominator (exact gradient) | Spectral / spectrum-matched random |
| L0 | Ordinary PEFT LoRA | Random A, zero B |
| D0 | Ordinary PEFT detached DoRA | Random A, zero B |

At identical parameters, the controlled attached and detached DoRA variants compute the same forward
outputs. Attached DoRA additionally propagates gradients through the direction's row norm. Both
variants use the same frozen initial correction.

## Configuration and arithmetic

[configs/experiment.yaml](../configs/experiment.yaml) pins the model and dataset revisions and
defines the shared training and generation settings.

| Setting | Default |
| --- | --- |
| Model / dataset | Qwen2.5-1.5B base / GSM8K |
| Adapter targets / rank | Query and value projections / 16 |
| Training horizon | Three epochs |
| Effective batch / microbatch | 32 / 4 |
| Base computation | BF16 |
| Adapter parameters, magnitudes, norms, optimizer state | FP32 |
| Adapter scaling / dropout | Alpha equals rank (multiplier 1) / zero |
| Evaluation | Greedy decoding, at most 512 new tokens, unmerged adapters |

Training requires a GPU with native BF16 support. Validate memory headroom and measure training and
generation costs on the hardware used for the study. Apply any change to shared settings
consistently across conditions before freezing the protocol.

Controlled initialization preserves the pretrained direction through `W0 + (BA - B0A0)`. Frozen
factors A0/B0 store the initial correction; their forward remains differentiable with respect to
inputs. DoRA magnitudes start at the FP32 row norms of W0. Base projections retain native BF16
arithmetic, while adapter branches and norm calculations use FP32.

## Implementation

- [model.py](../src/dora_spectral/model.py) uses PEFT 0.18.1 linear layers.
  L0/D0 use the stock LoRA/DoRA forwards. The six controlled conditions use a
  PEFT variant for frozen initial-factor subtraction and the attached/detached
  norm gradient. The supported configuration has one adapter and zero dropout.
- [initialization.py](../src/dora_spectral/initialization.py) computes exact SVD
  (`gesvd` on CUDA), balanced spectral factors, and spectrum-matched random
  factors. SVD caches are shared by model revision and rank.
- [streams.py](../src/dora_spectral/streams.py) derives stable 64-bit seeds from
  SHA-256 of the version, seed block, purpose, and target-matrix name. Controlled random
  directions use `init`; ordinary reference factors use `ref-factor`; shuffling uses
  `data-order`. Conditions, LR, and gamma are excluded so paired runs share the intended draws.
- [data.py](../src/dora_spectral/data.py) provides deterministic grouped splits,
  answer-only masking, and exact numeric scoring. Calculator annotations are
  removed from training answers, EOS is appended, and examples are not packed.
- [training.py](../src/dora_spectral/training.py) provides token-normalized
  accumulation, the full-horizon LR schedule, resumable checkpoints, and greedy
  evaluation using the same unmerged arithmetic as training.
- [experiment.py](../src/dora_spectral/experiment.py) prepares data and executes
  one selected candidate. Its evaluation splits are `pilot` and `development`;
  `score-test` evaluates locked confirmation checkpoints after the study's test-access gate.
- [validation.py](../src/dora_spectral/validation.py) implements the real-model validation gates.
- [diagnostics.py](../src/dora_spectral/diagnostics.py) measures effective weights and reversible
  exact/detached AdamW counterfactuals.
- [study.py](../src/dora_spectral/study.py) implements pilot, freeze, screening, promotion,
  selection, fresh-seed confirmation, automatic runtime tracking, and official-test orchestration.
- [scripts/run_study.py](../scripts/run_study.py) adds an NLL/generation phase message,
  completed-answer counts, and evidence-export messages to the terminal and each run's console log.
  It observes saved files while delegating to the original launcher, preserving frozen study sources.
- [analysis.py](../src/dora_spectral/analysis.py) validates paired test evidence and writes
  statistics, CSV tables, and scientific figures when explicitly requested through `analyze`.
- [tests/reference.py](../tests/reference.py) contains the dense mathematical
  reference used by tests and the [initialization notebook](../notebooks/01_initialization_checks.ipynb).

Controlled checkpoints contain frozen corrections as well as trainable adapters. Load them with the
repository's checkpoint utilities and the pinned base model. Standard PEFT export/merge does not
reconstruct the controlled correction.

Each named projection receives the same initialization regardless of module registration order.
The local data-order generator advances continuously across epochs and is reconstructed on resume;
initialization and global RNG consumption cannot change its permutations. `run.json` records the
stream version and data-order seed; `initialization.json` records the per-projection seeds.
Checkpoints commit completed updates before optional diagnostic probes. If a probe is interrupted,
resume retries the probe at that completed state; repeated probe records retain retry evidence.

## Study workflow

1. **Validate and freeze.** Check factor spectra, preserved starting outputs,
   forward/gradient behavior, checkpoint reloads, learning, and memory headroom.
   Record measured costs and freeze the model/data revisions, shared settings,
   search ranges, practical margins, condition scope, and runtime estimate.
2. **Tune each condition independently.** Give A-S/A-R/D-S/D-R/E-S/E-R equal search
   effort over learning rate and initialization scale. Give L0/D0 equal candidate
   effort over learning rate. Use the same development data and paired example
   order; each condition selects its own winning configuration.
3. **Screen and promote.** Screen candidates for one epoch, then continue selected
   candidates to the full three-epoch horizon. Use the three-epoch LR schedule
   from step zero and resume optimizer and RNG state. Apply the selection and
   tie-breaking rules in TEST_PLAN section 8.
4. **Confirm.** Freeze each condition's selected settings and run fresh paired
   seed blocks. Keep official-test data out of tuning and protocol decisions.
5. **Analyze.** Report the primary additive-versus-detached spectral interaction
   and secondary attached-versus-detached interaction, together with seed values,
   uncertainty, fixed-scale analyses, failures, and measured costs as specified
   in TEST_PLAN section 9.

## Commands

Prepare the pinned training-derived partitions:

```bash
.venv/bin/python -m dora_spectral.experiment prepare --data runs/data
```

Preparation writes tokenized examples, partition IDs, length/exclusion summaries, configuration,
revisions, and a checksum. An existing compatible preparation can be supplied through `--data`. Use
a separate output directory for each candidate.

Screen one candidate on the fixed 256-example development subset:

```bash
.venv/bin/python -m dora_spectral.experiment run \
  --data runs/data --directory runs/search/A-S-g1-lr1e-4-s10 \
  --condition A-S --seed 10 --gamma 1 --lr 1e-4 \
  --steps 203 --seconds 3600 --split development --evaluation-count 256
```

For 6,473 training examples and batch 32, one epoch is 203 updates and three epochs are 609 updates.
`--steps` specifies the total target, including updates already saved in the checkpoint. The runner
also computes answer-token NLL on all 500 development examples, including when generation uses the
256-example subset.

Promote that candidate by resuming the same directory to 609 updates and scoring the full
development partition:

```bash
.venv/bin/python -m dora_spectral.experiment run \
  --data runs/data --directory runs/search/A-S-g1-lr1e-4-s10 \
  --condition A-S --seed 10 --gamma 1 --lr 1e-4 \
  --steps 609 --seconds 3600 --split development
```

Resume checks reject changed source, package versions, configuration, condition, seed, or data. Keep
these fixed throughout a candidate's screen and promotion. The time cap applies per invocation. Set
it below the allocation's remaining wall time to allow checkpoint cleanup; `invocations.jsonl`
records costs across resumes.

## Outputs and evidence

Each candidate saves `run.json` (provenance), `initialization.json`, `checkpoint.pt`,
`training.jsonl`, `training-summary.json`, and per-split/step predictions and metrics. Checkpoints
store adapters, frozen corrections, optimizer state, RNG state, and step count; the base-model
weights are reconstructed from the pinned revision.

After each completed, interrupted, or failed invocation, the runner automatically copies JSON,
JSONL, YAML, log, and XML evidence into `reports/<user>/`, preserving the path below `runs/`. For example,
`runs/<study>/A-S-s0` is copied to `reports/<execution-user>/<study>/A-S-s0/`.
The OS username is recorded with each invocation. This includes configuration,
source hashes, initialization checks, training logs, predictions, and metrics. No narrative summary
is generated. `status.json` and `invocations.jsonl` also retain the execution-time Git revision,
package versions, GPU details, target steps, elapsed time, and any exception. Invocation time
excludes idle GPU allocation and evidence copying. Study cost reports automatically sum the selected
validation and recorded child-job runtimes, including failed/interrupted attempts. Earlier runs,
idle allocation, and unrecorded terminations are omitted; total allocated GPU-hours remain unknown.
The planning forecast uses these measured durations and requires no manual hour entries.
The total-hour target is advisory: it never blocks jobs or changes per-option epochs.
Per-job and allocation time limits still pause work for checkpoint resume.

Use `run --reports PATH` to choose a different destination root. Run directories outside `runs/`
are placed under `reports/<user>/external/<name>-<path-hash>/` to avoid name collisions. Original files
stay in `runs/` so checkpoint resumes continue to work. Checkpoints, model weights, prepared
examples, and SVD caches are not copied; back those up separately when resumability is needed.

Copy already completed cluster runs without retraining, model dependencies, or a GPU:

```bash
.venv/bin/python -m dora_spectral.reporting --runs runs/study-main
```

The report path uses the latest recorded execution user, falling back to the current OS username
for older results. To label an older run explicitly, or when exporting on another person's behalf,
add `--user USERNAME`. The override changes only the destination path, preserving original metadata.
Existing exports without a username directory are left in place; the command creates the new path.

The command accepts one candidate directory or a campaign root containing multiple `run.json`
files. Repeating it updates the copied evidence and retains older evaluation directories. It
preserves original metadata; it cannot recover execution-time Git/GPU information for older runs.
Use the same command on partial results if a process was killed before its cleanup could run.
Commit and push the copied `reports/` files to share results through Git. Nothing is committed or
pushed automatically. If copying fails, the runner reports the failure and leaves raw files intact.

Keep implementation source fixed between screening and promotion: the existing resume provenance
checks still apply. Copying old results does not require resuming their checkpoints.

Notebook dependencies are an optional installation group; see SETUP.md for the launch command.
