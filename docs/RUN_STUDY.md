# Study guide and checklist

Use the checkboxes below as a template; record completion with each study's report data.
Keep this shared guide unchecked. See the [test plan](TEST_PLAN.md) for experiment details.
Core: rank 16, eight conditions, three confirmation seeds.

Run from the repository root. Commands use explicit example paths; edit them for your study.
No environment-variable setup or shell shortcut is required.

Keep code, dependencies, and data fixed. GPU stage commands use a 16-hour cap;
reduce it if the remaining allocation is shorter.
Repeat a stage command to resume; finished jobs are skipped. Finish each stage before the next.
Choose fresh validation/study directories for new runs, or point to compatible existing evidence.
Runtime is recorded automatically. Unknown allocated GPU-hours and idle time are omitted.
GPU stages use `scripts/run_study.py` to show evaluation counts and evidence-copy progress.
It calls the existing runner unchanged, preserving study/checkpoint compatibility.

### Shared settings and condition names

These are the current [configuration](../configs/experiment.yaml) and [runner](../src/dora_spectral/study.py) defaults;
an existing study uses its saved `config.yaml` and `study.json`.

| Setting | Used throughout, unless a step says otherwise |
| --- | --- |
| Model/data | Revision-pinned Qwen2.5-1.5B **base** and GSM8K `main` |
| Adapters | Rank 16 on every `q_proj`/`v_proj`; alpha = 16 (multiplier 1); backbone frozen |
| Training | Three epochs; batch 32, microbatch 4 (eight accumulation chunks per full batch); answer/EOS-token loss only |
| Optimizer | AdamW: betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0; global gradient clipping 1.0; one LR for factors and DoRA magnitudes |
| Schedule | 3% warmup, then cosine decay over the **full three-epoch horizon**, including during short runs |
| Numerics | BF16 backbone computation; FP32 adapters/norms; dropout off; SDPA; gradient checkpointing off |
| Lengths/scoring | Training limit 1,024 tokens; greedy generation, at most 512 new tokens, evaluation batch 4; exact numeric answer after the last `####`; malformed/truncated outputs count as wrong |

`A-S/A-R` = additive adapters; `E-S/E-R` = DoRA with exact denominator gradients;
`D-S/D-R` = DoRA with the denominator detached. `S` uses principal SVD directions;
`R` uses random orthonormal directions with matching singular values. In these six controlled
arms, gamma scales the initial factor product (each factor scales by √gamma), and the frozen
initial product is subtracted so every arm starts at the base model's effective weights.
`L0/D0` are ordinary LoRA/detached DoRA with Kaiming-uniform A and zero B.

Evaluation counts below refer to generated answers. Pilot/development answer NLL
(negative log-likelihood) always uses the full corresponding 500-example split.

## 1. Install and check

```bash
bash scripts/setup_pace.sh cuda
.venv/bin/python -m pytest -q --junitxml="reports/$USER/unit-tests.xml"
```

- [x] Install locked dependencies, pass CPU tests, and save their JUnit results.

Install Python 3.11 dependencies with the PyTorch 2.6.0 CUDA 12.6 build and PEFT 0.18.1.
CPU tests check adapter initialization/gradients, data and scoring, training/resume,
diagnostics, study locks, reporting, and launcher behavior. No hyperparameters are tuned here.

## 2. Prepare data

```bash
.venv/bin/python -m dora_spectral.experiment prepare --data runs/data
```

- [x] Create or verify the pinned training/pilot/development splits (`prepared.json`).

Hash-sort GSM8K's 7,473 training examples into 500 pilot, 500 development, and 6,473 training
examples before length filtering; keep normalized-question duplicate groups together.
Remove calculator annotations, mask prompt/padding tokens, and append EOS. Exclude training
examples exceeding 1,024 tokens and record their IDs; retain every pilot/development example.
Save split IDs, lengths, revisions, and checksums. Official-test data stays unopened.

## 3. Validate on H100

```bash
.venv/bin/python -m dora_spectral.validation \
  --data runs/data --directory runs/validation-h100 --seconds 7200
```

- [x] Pass initialization, gradient, overfit, memory, and resume gates (`validation.json`: `passed: true`).

Use LR `1e-4`, gamma `1`, seed `0`, and the shared settings:

- Check base-weight/logit equality for all eight conditions and cache rank-16 SVDs. Verify additive/exact/detached gradients against independent dense references, plus an FP64 exact-DoRA gradcheck.
- Overfit `E-S` on the same 32 training examples for 100 updates (schedule horizon 100); require final answer NLL < `0.1` and < 20% of its initial value, with backbone weights unchanged.
- Run one update on the 32 longest retained training examples; require ≥10% GPU memory headroom.
- Compare `E-R` trained uninterrupted for six updates with a three-update checkpoint resumed to six; require matching adapter states.
- Record base NLL on all 500 pilot examples and generation on the first 16 for later feasibility/timing comparisons.

Reuse a passed validation only when its runtime code, configuration, data, and package versions match.
Orchestration/analysis-only edits do not require another GPU validation.
Otherwise validate in a fresh directory; compatible prepared data/SVD caches can be reused.

## 4. Initialize the study

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main init \
  --data runs/data --validation runs/validation-h100 \
  --target-hours 100
```

- [x] Save the configuration, grid, planning target, and seeds in a fresh study directory (`study.json`).

Register all eight conditions, the search grid in step 8, tuning seed `10`, confirmation seeds
`101/202/303`, and a practical interaction margin of **2 percentage points** (`--margin 2`).
Compute steps per epoch as `ceil(retained_training_count / 32)` and the full horizon as three
times that count: normally **203/609 updates**. Bind the study to validation, source, package,
and data hashes; this step launches no training.

Point `--validation` to the validation directory that passed for this study.
100 hours is a planning estimate, not a limit: exceeding it never blocks jobs or shortens training.
Each option uses the specified epochs; per-job/allocation time limits allow checkpoint resume.

## 5. Run the measured pilot

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main pilot \
  --wall-hours 16 --job-seconds 3600
```

- [x] Finish short profiles and the slowest condition's three-epoch pilot (`pilot.json`: `passed: true`).

All pilot jobs use LR `1e-4`, gamma `1`:

- Run `D-S/D-R/E-S/E-R` for 100 updates each on seeds `0/1/2` (12 profiles).
- Run `A-S/A-R/L0/D0` for 20 updates each on seed `0` (four profiles).
- Generate answers for the first 16 pilot questions per profile. Choose the condition with
  the highest mean training seconds/update, resume its seed-0 run to three epochs, and score all 500 pilot questions.

The full pilot must beat base-model pilot NLL, achieve **5–90% accuracy**, and have ≤1% truncation.
`--wall-hours` caps this stage invocation; `--job-seconds` caps each child attempt, including evaluation.
Training resumes from checkpoints; unfinished evaluation restarts. Increase `--job-seconds`
if evaluation alone cannot finish within that limit.

## 6. Freeze the study

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main freeze
```

- [x] Pass the pilot, save a runtime forecast, and freeze the protocol (`frozen.json`).

Verify validation/pilot success and lock the study manifest. Forecast remaining search,
confirmation, diagnostic, and test cost using the slowest measured training/evaluation rates
and measured overhead. No training or hyperparameter selection occurs here.

## 7. Run mechanism diagnostics

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main diagnose \
  --wall-hours 16 --job-seconds 1800
```

- [x] Finish four DoRA runs comparing exact/detached updates at steps 0/50/100 (`counterfactuals.jsonl`).

Train `E-S/E-R/D-S/D-R` for 100 updates with **seed `40`, LR `1e-4`, gamma `1`**;
generate answers for the first 16 pilot questions afterward. At updates 0/50/100, use the
same two training examples and saved model/AdamW state to try both exact and detached,
clipped updates, then restore the state. Compare effective-weight step norms/directions,
answer-NLL changes, and token KL, including local probes scaled to 0.1× the smaller step norm.
These probes explain local update behavior; they do not select settings.

## 8. Screen, then promote

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main screen \
  --wall-hours 16 --job-seconds 3600
.venv/bin/python scripts/run_study.py --directory runs/study-main promote \
  --wall-hours 16 --job-seconds 3600
```

- [x] `screen`: finish 96 one-epoch candidates, scoring 256 development questions (`search/`).
- [x] `promote`: continue up to 24 finalists to three epochs and score all 500 (`promotions.json`).

| Conditions | Settings screened per condition, all on seed `10` |
| --- | --- |
| Six controlled arms (`A-S/A-R/D-S/D-R/E-S/E-R`) | Gamma `{0.1, 0.3, 1}` × LR `{2e-5, 1e-4, 5e-4, 2e-3}` = 12 candidates each |
| Ordinary baselines (`L0/D0`) | 12 log-spaced LRs from `2e-6` to `2e-3` inclusive (`2e-6 × 1000^(i/11)`, `i=0…11`); gamma fixed at 1 and unused by zero-B initialization |

Screen for one epoch (normally 203 updates), using the first 256 saved development IDs for
accuracy. Rank by higher accuracy, then lower full-development NLL, lower LR, and lower gamma.
For each controlled arm promote the best LR **at each gamma**; for each baseline promote its
top three LRs. Resume those checkpoints/optimizer states to three total epochs (normally 609
updates), then rerank using all 500 development questions. Initialization and data order are
paired across conditions/settings within each seed; their random streams are separate.

Retain numerical failures; require a usable finalist for every condition.

## 9. Select, then confirm

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main select
.venv/bin/python scripts/run_study.py --directory runs/study-main confirm \
  --wall-hours 16 --job-seconds 3600
```

- [x] `select`: lock eight winning configurations (`selection.json`, `selection-lock.json`).
- [x] `confirm`: finish 24 full runs on seeds 101/202/303, with development checks each epoch (`confirmation/`).

Select one three-epoch finalist per condition by the same accuracy/NLL/LR/gamma ranking;
record whether its LR/gamma lies at a grid boundary. Freeze each winner's LR and gamma.
Train **eight conditions × three fresh paired seeds** from initialization for three epochs;
score the first 128 development questions after epochs 1 and 2 and all 500 after epoch 3.
Use the final checkpoint for every run. Intermediate checks do not trigger early stopping,
checkpoint selection, or retuning.

## 10. Evaluate the official test

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main test \
  --wall-hours 16 --job-seconds 3600
```

- [ ] After all confirmations finish, score the base and final replicas on all 1,319 questions (`test/`).

Verify all locked confirmation results, then evaluate **25 models**: the unadapted base once
and all 24 final confirmation checkpoints. Use the shared greedy/512-token/strict-answer
rules and compute answer NLL on all 1,319 examples. Save per-question predictions, correctness,
truncation, and provenance for paired analysis.

Keep settings, scoring rules, and seeds unchanged after opening the test.

## 11. Analyze on CPU

```bash
.venv/bin/python scripts/run_study.py --directory runs/study-main analyze \
  --resamples 10000 --two-way
```

- [ ] Verify paired predictions and inspect statistics, CSV tables, and PNG figures (`analysis/` in the report folder).

Recompute strict correctness and verify identical question IDs/references across all models.
Compute spectral-minus-random accuracy gaps `ΔA`, `ΔD`, and `ΔE`; the primary interaction is
`J = ΔD − ΔA`, and the secondary detachment interaction is `I = ΔE − ΔD`.
Report paired-seed Student-t 95% intervals against the frozen ±2-point margin, 10,000 paired-question
bootstrap resamples (seed `722`), and, with `--two-way`, joint seed/question bootstrap intervals.
Seed intervals condition on this test set and selected settings; question bootstrap intervals
condition on the trained models. Export accuracy/seed variability, search boundaries, failures,
mechanism measurements, runtime/memory tables, and figures; no further model tuning occurs.

<details>
<summary>Analyze copied results on your laptop instead</summary>

```bash
uv sync --locked --extra cpu --extra models --extra analysis
.venv/bin/python scripts/run_study.py \
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

## Telegram alerts (optional)

Create a bot with [@BotFather](https://t.me/BotFather), then open your bot and press **Start**.
Get your chat ID from `message.chat.id` in the bot's
[getUpdates response](https://core.telegram.org/bots/api#getupdates).

```bash
cp configs/telegram.example.json configs/telegram.local.json
chmod 600 configs/telegram.local.json
```

This creates a Git-ignored credentials file readable only by your account. Edit its `bot_token`
and `chat_id`, then send a test alert from the cluster:

```bash
.venv/bin/python scripts/run_study.py --telegram-test
```

Run the study commands above normally. Each invocation from `init` through `analyze` sends one
completion or stop alert with its stage, study path, execution user/host, and elapsed time.
`screen` and `promote` each send their own alert. Installation, preparation, and validation run separately.
Telegram failures do not change experiment outcomes. Forced termination or node loss may prevent alerts.
Alerts apply when a command is next launched or resumed; already-running processes keep their old code.
Remove the local credentials file to disable alerts; no environment variables are required.
These launcher-only changes preserve existing studies and checkpoints.

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
   .venv/bin/python scripts/run_study.py --directory runs/study-r8 init \
     --config configs/rank8.yaml --data runs/data --validation runs/validation-r8
   ```

Repeat steps 5–11 with `runs/study-r8`, independently tuning LR × gamma for that rank.
Reuse prepared data; SVD caches are separated by rank. Alpha automatically equals rank.
Choose the ranks to compare before official-test inspection, and keep each rank's results separate.
