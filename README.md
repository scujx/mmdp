# When to Commit and When to Defer: Maturing Markov Decision Processes under Refining Information and Expiring Opportunities

Research code accompanying the paper above.

Maturing Markov Decision Processes (MMDPs) study sequential decisions in which
information is refined while action opportunities expire. The paper uses this
structure to organize when to commit and when to defer, motivating stage-local
policy interfaces, action abstraction, and policy-guided planning.

This package provides three synthetic benchmarks:

- **Staged replenishment:** Flat and MMDP policies with retained or updated
  delivery-week forecasts.
- **Five-account cash management:** PPO, Random, and rolling-horizon MILP
  controllers under Flat, MMDP, and fixed route-mask controls.
- **Ten-account cash management:** PPO with and without action abstraction,
  policy-guided Search with frozen policies, planning-budget comparisons,
  Random, and a rolling-horizon MILP reference.

The package contains implementations and necessary configurations, **not saved
experimental results**. It excludes model weights, plotting code, production
systems/data, and training logs. See `LICENSE` for the
non-commercial research and review permissions; commercial deployment and
operational use are not permitted by this license.

## Getting the code

```sh
git clone https://github.com/scujx/mmdp.git
cd mmdp
```

The repository provides the synthetic benchmarks only. The production-scale
system and its underlying operational data are not distributed.

## Structure and installation

```text
reproduce.py                Root entry point: smoke checks or full fresh reruns
experiments/milp_planner.py  Shared rolling-horizon MILP controller
experiments/controller_timing.py  Randomized serial CPU timing suite
experiments/replenishment/   Environment, PPO, evaluation, paired analysis
experiments/cash5/           PPO, Random, MILP, fixed route masks, paired analysis
experiments/cash10/          PPO/AA, frozen-policy Search, MILP, budget analysis
requirements.txt            NumPy dependency for statistical analysis
```

Each benchmark uses the same three subdirectories:

```text
configs/       Static environment and experiment configurations
envs/          Environment dynamics and synthetic data generation
wrappers/      Policy interfaces, route masks, or training wrappers
```

Each also provides `run.py`, `analyze.py`, `tests.py`, and `runtime.py`.
Cash5 additionally provides `training.py` for PPO and mask training,
`evaluation.py` for final-policy evaluation, `random_policy.py` for the Random
reference, `milp.py` for the optimization reference, `learning.py` for offline
fixed-bank checkpoint scoring, and `generate_masks.py` for the route controls.
Cash10's `search.py` implements policy-guided planning and
its `milp.py` evaluates the optimization reference, while `random_policy.py`
evaluates the ten-account Random reference. Its `validation.py` records the fixed
validation bank during training. The MILP adapters share
`experiments/milp_planner.py`. Replenishment's policy interfaces are selected
inside its environment; its wrappers handle episode logging and reward scaling.

Use Python 3.10 and run commands from this package's root. All three benchmarks
pin Stable-Baselines3 2.7.1. Separate virtual environments retain each benchmark's
other pinned dependencies. Install the relevant experiment's `requirements.txt`,
which includes its runtime dependencies. For analysis alone, the root
`requirements.txt` suffices.
The cash-management MILP references use `scipy.optimize.milp` (SciPy 1.15.3),
which invokes the HiGHS solver supplied with SciPy; no separate solver package
or license is required to run these scripts.
For example:

```sh
python3.10 -m venv ../rep-env
source ../rep-env/bin/activate
python -m pip install -r experiments/replenishment/requirements.txt
```

## One-command entry point

After installing the benchmark dependencies, `reproduce.py` defaults to short
runtime checks, not a full experiment. For one benchmark in its active environment:

```sh
python -B reproduce.py --benchmark replenishment
```

To check all three benchmarks with their separate environments:

```sh
python -B reproduce.py --replenishment-python ../rep-env/bin/python \
  --cash5-python ../cash5-env/bin/python --cash10-python ../cash10-env/bin/python
```

Full mode explicitly runs all training seeds, final/selected-policy evaluations,
Random and MILP references for the cash benchmarks, and paired statistical analyses for the
selected benchmark. It can take substantial
time. Preview the commands first, then remove `--dry-run` to execute:

```sh
python -B reproduce.py --mode full --benchmark replenishment --output ../rep-full --dry-run
```

Use `--benchmark cash5` or `--benchmark cash10` for the other experiments. The
default `--benchmark all` executes all three sequentially with the interpreter
options above: 40 replenishment models, 120 Cash5 models (including appendix mask
controls), and 40 Cash10 models. No dependencies are installed automatically.
Output paths must be new and outside this package; failures stop the run, and
there is no automatic resume. Use the individual commands below to recover a
partially completed run without overwriting its outputs. Smoke checks use temporary
directories. Neither mode requires pretrained checkpoints. Cash-benchmark smoke
mode also checks one MILP episode per interface.

## Replenishment

The staged replenishment benchmark uses a shared 24-dimensional observation
under both interfaces, including remaining flexible capacity. Every training
reset draws a fresh episode. Orders
arrive the following Monday; the final delivery week is fully settled. Updated
reveals an additional forecast for that same delivery week and retains the
coarse signal. Retained withholds the fine signal, not physical observations.

The bundled configuration is the main-text 80% fine-category accuracy setting,
with coarse accuracy fixed at 57.875%. Demand and coarse signals are matched
across information conditions; fine signals are sampled conditionally on demand
category and coarse signal. Internal latent values are not policy observations.
The values in `configs/environment.json` document the
fixed physical and signal configuration, and `configs/experiment.json` records
the training/evaluation protocol. The environment is a fixed benchmark, not a
general configurable inventory library; changing configuration fields alone is
not a supported way to define another experiment. The separate accuracy
sensitivity entry point changes the fine-signal law while preserving the shared
coarse-signal distribution. PPO-configuration sensitivity variants are not run
by the default reproduction command.

```sh
python -B -m experiments.replenishment.tests
python -B -m experiments.replenishment.run train --cell mmdp_updated --seed 42 --output ../rep-smoke --smoke
python -B -m experiments.replenishment.run train --cell mmdp_updated --seed 42 --output ../rep-runs/mmdp_updated_42
```

The four cells are `flat_retained`, `flat_updated`, `mmdp_retained`, and
`mmdp_updated`, each using training seeds 42–51. Full training uses nominal 100K
transitions (100,096 actual), with validation at 5K intervals through 100K on
20 shared instances (9241000–9241019). Highest mean validation return selects
the checkpoint; ties favor the earliest. Initialization and the final 100,096
transition policy are not selection candidates. The package saves the selected
checkpoint, not all intermediate weights. Smoke training uses 512 transitions
and development instances; it is not a paper result.

Train all forty cells/seeds before running their test evaluations. For example:

```sh
python -B -m experiments.replenishment.run evaluate --run ../rep-runs/mmdp_updated_42 \
  --trust-checkpoint --output ../rep-evaluations/mmdp_updated_42.jsonl
```

Evaluation uses deterministic actions on 200 shared instances
(9242000–9242199). Use a separate output file for each policy. Evaluations are
in unscaled economic units; the 0.01 reward scale is applied only in training.
Only load trusted checkpoints: model deserialization can execute Python code.

After all forty training runs and evaluations:

```sh
python -B -m experiments.replenishment.analyze --runs ../rep-runs \
  --evaluations ../rep-evaluations --output ../rep-analysis
```

The analyzer checks the complete matrix, selected-checkpoint identities,
validation selection, instance fingerprints, and reward accounting. It writes
one `summary.json` containing seed-level metrics and paired contrasts. Normalized
validation AUC integrates only 5K–100K and divides by 95K, with no extrapolation
to step zero. Test returns first average instances within each trained policy.
Marginal 95% percentile intervals use 100,000 resamples of complete paired
training-seed rows (PCG64 seed 9245000), conditional on the shared instance bank.
They do not adjust for exploratory setting selection or multiple comparisons.

The root command `reproduce.py --mode full --benchmark replenishment` automates
all training, then all testing, then analysis. Its default `smoke` mode instead
checks signal release, capacity observability, daily accounting, lossless
deferral fixtures, four short training runs, checkpoint evaluation, and statistics.

Forecast-accuracy sensitivity (the appendix's 75%, 80%, and 85% fine-signal
settings) has its own fresh-run entry point. It trains and evaluates all four
cells at the selected accuracy, then writes paired statistics. Demand and the
coarse-signal law remain fixed; run each accuracy into a distinct new directory.
The 80% main-setting run is also covered by `reproduce.py --mode full`.

```sh
python -B -m experiments.replenishment.sensitivity --fine-accuracy 0.75 --output ../rep-accuracy-75 --dry-run
python -B -m experiments.replenishment.sensitivity --fine-accuracy 0.75 --output ../rep-accuracy-75 --smoke
# For the formal 75% cell, use a different new output and omit --smoke.
```

## Five-account cash management

In a separate Python 3.10 environment:

```sh
python -m pip install -r experiments/cash5/requirements.txt
python -B -m experiments.cash5.tests
python -B -m experiments.cash5.run train --setting True --seed 42 --output ../cash5-true42-smoke --smoke
python -B -m experiments.cash5.run train --setting True --seed 42 --output ../cash5-true42
```

Settings are `Flat`, `True` (MMDP), C1–C5 (funding-preserving masks averaged as
**Mask** in the main table), and U1–U5 (appendix controls). Repeat each setting
with seeds 42–51, using separate output directories. Full runs end at 500,224
transitions and evaluate final policies, not validation-selected policies.
`configs/mask_inventory.json` contains the fixed masks; `configs/design.md`
explains their construction and interpretation.

### Fixed-bank learning curves (Figure 4a)

Full training saves 100 scheduled checkpoints at 5,000-transition increments
from 5,000 through 500,000. Score these checkpoints offline with twenty fixed
environment seeds `training_seed + 1000` through `training_seed + 1019`.
Each episode constructs a fresh wrapper and explicitly resets it (reset ordinal
two); policy actions are deterministic and use Raw execution. This is separate
from the continuing ten-episode evaluation callback during training. Neither
evaluation selects the final policy, which remains the 500,224-transition endpoint.

```sh
python -B -m experiments.cash5.run evaluate-learning --run ../cash5-true42 \
  --setting True --seed 42 --inventory-sha256 YOUR_CHECKPOINT_INVENTORY_SHA256 \
  --reward-convention native --trust-checkpoint --output ../cash5-learning/true42
python -B -m experiments.cash5.learning_tests
```

Use the SHA-256 of your trusted run's `checkpoint_inventory.json`; the scorer
verifies its checkpoint hashes before loading. Repeat for Flat and True at
seeds 42--51. `learning_curve_points.csv` contains unsmoothed checkpoint means;
`learning_curve_episodes.csv` records the twenty episodes per point, their reset
identities, and action hashes. `learning_curve_metadata.json` records the bank,
checkpoint schedule, reward convention, and completion status. Across-seed curves
use equal-weight means and sample standard deviations of the ten seed means.

Choose `native` for Python-float sums of the environment rewards, or `vec-float32`
for per-step float32 reward transport followed by Python-float summation. Both
are recorded on the same trajectory; neither uses a float32 episode accumulator.
Full root reproduction uses `native`. These commands produce fresh evaluations,
not an archive of the reported curves. They refuse incomplete formal schedules,
modified checkpoints, and existing output directories. Short fixture evaluation
requires `--smoke` and is marked as non-performance output.

### Final-policy independent tests

```sh
mkdir -p ../cash5-tests
python -B -m experiments.cash5.run evaluate-final --setting True --seed 42 \
  --checkpoint ../cash5-true42/final_model.zip --sha256 YOUR_CHECKPOINT_SHA256 \
  --trust-checkpoint --output ../cash5-tests/true42.jsonl
python -B -m experiments.cash5.run random --output ../cash5-random.jsonl
```

Evaluate every trained policy. Each evaluation uses 200 common synthetic
instances (980000–980199). Random uses ten action-RNG replicates per instance;
`--smoke` reduces its evaluation to one environment/replicate per setting.
Analyze the 120 policy-evaluation files together, and Random separately:

```sh
python -B -m experiments.cash5.analyze --controller ppo --input ../cash5-tests/*.jsonl --output ../cash5-ppo-statistics.json
python -B -m experiments.cash5.analyze --controller random --input ../cash5-random.jsonl --output ../cash5-random-statistics.json
```

Here `../cash5-tests/` should contain your twelve settings x ten seeds, and
nothing else. The analyzer also accepts CSV files with the same episode columns
and rejects incomplete or duplicate inputs. PPO inference uses paired training
seeds. Random inference uses environment means after averaging action-RNG
replicates, not artificial training seeds. Masks can be regenerated using
`run.py generate-masks --output-directory ../new-masks`.

The same forecast-based rolling-horizon MILP formulation can be evaluated
independently of PPO checkpoints. It uses the current observation, forecasts,
and public model coefficients, then replans after each transition. `True` is
the MMDP route interface; C1--C5 are the five funding-preserving masks.

```sh
python -B -m experiments.cash5.milp --smoke
python -B -m experiments.cash5.milp --output ../cash5-milp
```

The full command evaluates all seven route settings on the same 200 test
instances and reports paired instance-bootstrap contrasts. The smoke command
uses one test instance per setting and is not a paper result.

## Ten-account cash management

In a separate Python 3.10 environment:

```sh
python -m pip install -r experiments/cash10/requirements.txt
python -B -m experiments.cash10.tests
python -B -m experiments.cash10.run train --policy mmdp-aa --seed 42 --output ../cash10-mmdpaa42-smoke --smoke
python -B -m experiments.cash10.run train --policy mmdp-aa --seed 42 --output ../cash10-mmdpaa42
```

Policies are `flat`, `mmdp`, `flat-aa`, and `mmdp-aa`, with seeds 42–51. Full
runs train in two consecutive segments and end at 2,000,384 transitions. Smoke
runs use 512. AA uses `zero_passthrough`: a valid route with zero amount is not
a STOP. The environment and PPO configuration are in `configs/experiment.json`.

Every full training run also writes `validation.csv` and `validation.json`.
The same callback spans both training segments and records 400 points at absolute
transition counts 5,000 through 2,000,000. Training seed `i` uses the fixed
20-instance bank `i+1000` through `i+1019`, explicit reset ordinal two, and
deterministic Raw execution. Curve returns use float32 per-step accumulation;
native sums are recorded separately. Validation preserves the learner's random
states and policy mode and does not select checkpoints. The final updated policy
at 2,000,384 transitions is distinct from the last validation point. Smoke mode
records only two smoke-only points at 256 and 512 transitions.

This callback is a fresh implementation of the documented protocol, not an
archive of the reported curves. Across-seed learning curves use equal-weight
means and sample standard deviations of the ten seed-level validation means.
Its isolation and continuous-grid tests require no historical checkpoints:

```sh
python -B -m experiments.cash10.validation_tests
```

Read `checkpoint_sha256` from the new training run's `run.json`:

```sh
python -B -m experiments.cash10.run evaluate --policy mmdp-aa \
  --checkpoint ../cash10-mmdpaa42/final.zip --sha256 YOUR_CHECKPOINT_SHA256 \
  --trust-checkpoint --caps 0 576 1152 2304 --output ../cash10-evaluations/mmdpaa42
```

Repeat for all forty final policies. Defaults use the same 200 instances
(994000–994199) across all budgets. Cap 0 is Raw; 576/1152/2304 are the 1x/2x/4x
planning budgets. For a diagnostic, use `--instances 1 --seed-start 880000
--caps 0 576`. Search uses public-observation-conditioned scenario worlds, not
the instance's hidden future; equal caps need not mean equal realized work.

```sh
python -B -m experiments.cash10.analyze --input ../cash10-evaluations/* --output ../cash10-statistics.json
```

Each input directory must contain the generated `evaluation.json` and
`episodes.csv`. The analyzer requires the complete forty-policy/four-budget
matrix and uses paired training-seed inference.

The independent rolling-horizon MILP reference uses the same fixed test bank
under the Flat and MMDP route interfaces. It requires no learned checkpoint:

```sh
python -B -m experiments.cash10.milp --smoke
python -B -m experiments.cash10.milp --output ../cash10-milp
```

The full command reports the two 200-instance mean returns and a paired
instance-bootstrap interface contrast. The smoke command is only a short
controller check.

The ten-account Random reference uses the same 200 test instances and ten
action-sampling replicates per instance/interface; it needs no model weights.
Its summary averages replicates within each instance before paired instance
bootstrap inference.

```sh
python -B -m experiments.cash10.random_policy --smoke --output ../cash10-random-smoke
python -B -m experiments.cash10.random_policy --output ../cash10-random
```

## Result entry points

| Paper result | Fresh-run entry point | Output |
| --- | --- | --- |
| Replenishment main learning/test contrasts | `reproduce.py --mode full --benchmark replenishment` | `replenishment/analysis/summary.json` |
| Replenishment forecast-accuracy sensitivity | `experiments.replenishment.sensitivity --fine-accuracy 0.75/0.80/0.85` | `analysis/summary.json` in each accuracy's output directory |
| Five-account PPO/Mask and Random | `reproduce.py --mode full --benchmark cash5` | `ppo-statistics.json`, `random-statistics.json` |
| Five-account learning curves (Figure 4a) | `experiments.cash5.run evaluate-learning`, also invoked by full reproduction | `learning_curve_points.csv`, `learning_curve_episodes.csv`, and `learning_curve_metadata.json` per Flat/MMDP training seed |
| Five-account MILP | `experiments.cash5.milp` | `summary.json` and `episodes.jsonl` |
| Ten-account PPO/AA, Search, and planning-budget sensitivity | `reproduce.py --mode full --benchmark cash10` | `statistics.json`; per-policy evaluation directories also record each cap |
| Ten-account learning curves (Figure 4b) | `experiments.cash10.run train`, also invoked by full reproduction | `validation.csv` and `validation.json` in each policy/seed training directory |
| Ten-account Random and MILP | `experiments.cash10.random_policy`, `experiments.cash10.milp` | Each writes `summary.json` and episode records |
| Five-/ten-account serial controller time | `experiments.controller_timing --manifest` | Randomized schedule, episode/decision records, and aggregate timing summaries |

Timing uses one CPU worker, batch size one, single-thread PyTorch/BLAS/HiGHS,
and the same fixed 20-instance subset of the test bank. Use a manifest to measure
all requested controller/interface/replicate blocks in one session. Models are
verified and preloaded before measurement. Each of three rounds randomizes block
and instance order using an independent recorded PCG64 scheduler. Two off-bank
episodes warm each block in the first round; one warms each later block.
Warm-up, loading, resets, hashing, and output writing are excluded from the
controller intervals. Actual physical environment advancement is timed separately;
Search's hypothetical transitions remain part of controller computation.

The supplied checkpoint-free two-interface MILP manifest can be run immediately:

```sh
python -B -m experiments.controller_timing --manifest experiments/controller_timing_manifest.example.json --smoke --output ../cash10-milp-suite-smoke.json
python -B -m experiments.controller_timing --manifest experiments/controller_timing_manifest.example.json --output ../cash10-milp-suite.json
python -B -m experiments.controller_timing_tests
```

To time learned controllers, create a separate version-1 JSON manifest with a
`blocks` list. Each block needs a unique `id`, `benchmark`, `controller`, and
`interface`. Learned blocks also require `training_seed`, `checkpoint`, and
the checkpoint's `sha256`; paths are relative to the manifest. Random blocks
use `random_replicate` (0--9). Run the manifest with `--trust-checkpoint` only
for models you produced or independently trust. Measure Cash5 and Cash10 in
separate manifests. For a complete table, include all ten learned-policy seeds,
all ten Random replicates where applicable, and each deterministic MILP interface
once; Cash5 Mask requires all C1--C5 blocks with matching replicates.

Single-controller calls remain available for installation checks, but separately
invoking them does not provide the randomized multi-controller comparison:

```sh
python -B -m experiments.controller_timing --benchmark cash5 --controller milp --interface True --smoke --output ../cash5-milp-time-smoke.json
python -B -m experiments.controller_timing --benchmark cash10 --controller milp --interface mmdp --output ../cash10-milp-time.json
```

Cash5 controller names are `ppo`, `random`, and `milp`; its interfaces are
`Flat`, `True` (MMDP), and `C1`--`C5`. Cash10 controller names are `raw`,
`search`, and `milp`; its policy/interface labels are `flat`, `mmdp`,
`flat-aa`, and `mmdp-aa` (`milp` uses only `flat` or `mmdp`). For example, add
`--checkpoint PATH --sha256 DIGEST --trust-checkpoint` for `ppo`, `raw`, or
`search`. Outputs contain the actual randomized schedule, per-episode and
per-decision timing records, checkpoint hashes, and reset/trajectory identities.
Repeated runs of a block/instance must preserve its return, length, and action
sequence. Summaries take repetition medians first, then equal-weight episode
means over instances and controller replicates. Per-decision summaries pool
visited decisions after repetition medians. Cash5 Mask episode times average
C1--C5 equally. A partial manifest reports only its requested blocks, not the
complete paper table.

This is fresh tooling based on the documented and archived measurement protocol,
not a distribution of the archived timing records. Numerical latencies depend
on CPU hardware, library, solver versions, and system load; new measurements do
not automatically replace the paper's reported values.

## Scope and safety

Short tests check installation and behavior; they do not reproduce reported
performance. Full training and evaluation are explicit operations and can take
substantial time. Analysis consumes **your newly generated outputs**; the
reported PPO results cannot be regenerated instantly from this code-only
package. The deterministic MILP reference results can be recomputed without
training, subject to platform and solver-version differences.

Load only your own or independently trusted model archives: serialized models
can execute Python code. A checksum verifies identity, not safety; obtain it
using `shasum -a 256 MODEL.zip`. Evaluation/analysis outputs require new paths.

Exact numerical agreement is not guaranteed across platforms and dependency
versions. The benchmark requirements pin the versions for fresh reruns. No
pretrained models or production data are included in this package.
