# Main experiment

This guide describes the implementation and execution workflow for comparing spectral and
spectrum-matched random initialization under additive adaptation, detached DoRA, and attached DoRA.
[TEST_PLAN.md](TEST_PLAN.md) defines the research hypotheses, tuning allocation, confirmation
design, and statistical analysis. [SETUP.md](SETUP.md) covers installation and cluster setup.

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
- [data.py](../src/dora_spectral/data.py) provides deterministic grouped splits,
  answer-only masking, and exact numeric scoring. Calculator annotations are
  removed from training answers, EOS is appended, and examples are not packed.
- [training.py](../src/dora_spectral/training.py) provides token-normalized
  accumulation, the full-horizon LR schedule, resumable checkpoints, and greedy
  evaluation using the same unmerged arithmetic as training.
- [experiment.py](../src/dora_spectral/experiment.py) prepares data and executes
  one selected candidate. Its evaluation splits are `pilot` and `development`;
  it has no sweep, winner-selection, official-test, or statistical-analysis command.
- [tests/reference.py](../tests/reference.py) contains the dense mathematical
  reference used by tests and the [initialization notebook](../notebooks/01_initialization_checks.ipynb).

Controlled checkpoints contain frozen corrections as well as trainable adapters. Load them with the
repository's checkpoint utilities and the pinned base model. Standard PEFT export/merge does not
reconstruct the controlled correction.

## Study workflow

1. **Validate and freeze.** Check factor spectra, preserved starting outputs,
   forward/gradient behavior, checkpoint reloads, learning, and memory headroom.
   Record measured costs and freeze the model/data revisions, shared settings,
   search ranges, practical margins, condition scope, and study budget.
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

Execution-specific measurements and findings belong in [reports](../reports), including the [PEFT
validation report](../reports/peft-validation.md). Raw artifacts in `runs/` are gitignored and must
be transferred separately when reproducing or publishing results. Notebook dependencies are an
optional installation group; see SETUP.md for the launch command.
