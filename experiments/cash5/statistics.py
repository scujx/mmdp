"""Paired PPO and Random statistics; imports no policies or environments.

PPO resamples training seeds; Random resamples instance means after averaging
action-sampling replicates. The analysis entry point checks input coverage.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("Flat", "True", "U1", "U2", "U3", "U4", "U5", "C1", "C2", "C3", "C4", "C5")
TRAINING_SEEDS = tuple(range(42, 52))
MASK_TYPES = {
    "Flat": "flat",
    "True": "true",
    **{f"U{i}": "uniform" for i in range(1, 6)},
    **{f"C{i}": "conditional" for i in range(1, 6)},
}
ANALYSIS_SEED = 982301
BOOTSTRAP_REPLICATES = 100000
CI_PROBABILITIES = (0.025, 0.975)
PRIMARY = "True_minus_mean_all_masks"
SECONDARIES = ("True_minus_Flat", "True_minus_mean_U", "True_minus_mean_C")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite_array(values, ndim=None):
    array = np.asarray(values, dtype=np.float64)
    require(np.all(np.isfinite(array)), "Statistical input contains a nonfinite value")
    require(
        ndim is None or array.ndim == ndim,
        f"Expected {ndim} dimensions, received {array.shape}",
    )
    return array


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_means(episodes):
    """Input axes: training seed, fixed setting, evaluation environment."""
    cube = finite_array(episodes, 3)
    require(
        cube.shape[0] >= 2 and cube.shape[1] == len(SETTINGS) and cube.shape[2] >= 1,
        f"Invalid paired endpoint cube: {cube.shape}",
    )
    return cube.mean(axis=2)


def contrast_vectors(setting_means):
    matrix = finite_array(setting_means, 2)
    require(
        matrix.shape[1] == len(SETTINGS), "Contrast matrix must retain all 12 settings"
    )
    true = matrix[:, 1]
    return {
        PRIMARY: true - matrix[:, 2:12].mean(axis=1),
        SECONDARIES[0]: true - matrix[:, 0],
        SECONDARIES[1]: true - matrix[:, 2:7].mean(axis=1),
        SECONDARIES[2]: true - matrix[:, 7:12].mean(axis=1),
    }


def exact_sign_flip(differences):
    """Two-sided |mean| statistic, all 2**n sign assignments, ties included.

    A 64-epsilon arithmetic tolerance includes numerical representations of
    exact ties; it is scaled by mean absolute paired difference. There is no
    Monte Carlo approximation, continuity correction, or plus-one correction.
    """
    values = finite_array(differences, 1)
    n = len(values)
    require(1 <= n <= 20, "Exact sign enumeration supports 1 to 20 paired rows")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=np.float64)
    null = (signs * values[None, :]).mean(axis=1)
    observed = float(abs(values.mean()))
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.abs(values).mean()), observed)
    )
    extreme = int(np.count_nonzero(np.abs(null) >= observed - tolerance))
    return dict(
        p_value=extreme / len(null),
        extreme_assignments=extreme,
        total_assignments=len(null),
        observed_absolute_mean=observed,
        ties_included=True,
        arithmetic_tie_tolerance=tolerance,
    )


def holm_adjust(p_values):
    """Holm adjustment over exactly the supplied family; output order unchanged."""
    values = finite_array(p_values, 1)
    require(
        len(values) >= 1 and np.all((values >= 0.0) & (values <= 1.0)),
        "Invalid p-values",
    )
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, float((len(values) - rank) * values[index]))
        adjusted[index] = min(1.0, previous)
    return adjusted


def resampling_plan(
    n_training_seeds, replicates=BOOTSTRAP_REPLICATES, analysis_seed=ANALYSIS_SEED
):
    """Exactly four draws, in this order, from a single PCG64 stream.

    1. Ordinary paired training-row indices (shared by all settings/contrasts).
    2. Independent hierarchical training-row indices.
    3. U-mask column indices, shared by every sampled training row.
    4. C-mask column indices, shared by every sampled training row.
    """
    require(n_training_seeds >= 2 and replicates >= 1, "Invalid resampling dimensions")
    rng = np.random.Generator(np.random.PCG64(int(analysis_seed)))
    return {
        "paired_rows": rng.integers(
            0, n_training_seeds, size=(replicates, n_training_seeds), dtype=np.int64
        ),
        "hierarchical_rows": rng.integers(
            0, n_training_seeds, size=(replicates, n_training_seeds), dtype=np.int64
        ),
        "hierarchical_u_columns": rng.integers(
            0, 5, size=(replicates, 5), dtype=np.int64
        ),
        "hierarchical_c_columns": rng.integers(
            0, 5, size=(replicates, 5), dtype=np.int64
        ),
    }


def checked_indices(indices, upper, expected_width):
    values = np.asarray(indices)
    require(
        values.ndim == 2
        and values.shape[1] == expected_width
        and np.issubdtype(values.dtype, np.integer)
        and np.all((values >= 0) & (values < upper)),
        "Invalid bootstrap index matrix",
    )
    return values


def paired_bootstrap_setting_means(setting_means, rows):
    matrix = finite_array(setting_means, 2)
    require(
        matrix.shape[1] == len(SETTINGS), "Bootstrap requires all 12 setting columns"
    )
    indices = checked_indices(rows, len(matrix), len(matrix))
    # A single index matrix indexes the whole setting vector for each seed.
    return matrix[indices].mean(axis=1)


def hierarchical_mask_bootstrap(setting_means, rows, u_columns, c_columns):
    """Sensitivity to the five observed masks in each family.

    Each replicate uses one U-column sample and one C-column sample for ALL
    sampled training seeds. True is averaged once per sampled seed, never once
    per mask as if those copies supplied additional training repetitions.
    """
    means = paired_bootstrap_setting_means(setting_means, rows)
    u_indices = checked_indices(u_columns, 5, 5)
    c_indices = checked_indices(c_columns, 5, 5)
    require(
        len(means) == len(u_indices) == len(c_indices),
        "Bootstrap replicate-count mismatch",
    )
    u = np.take_along_axis(means[:, 2:7], u_indices, axis=1).mean(axis=1)
    c = np.take_along_axis(means[:, 7:12], c_indices, axis=1).mean(axis=1)
    true = means[:, 1]
    return {
        PRIMARY: true - 0.5 * (u + c),
        SECONDARIES[1]: true - u,
        SECONDARIES[2]: true - c,
    }


def percentile_ci(bootstrap_values):
    values = finite_array(bootstrap_values, 1)
    require(len(values) >= 1, "Empty bootstrap distribution")
    return np.quantile(values, CI_PROBABILITIES, method="linear").tolist()


def empirical_lower_cvar(values, alpha=0.05):
    """Mean of the lowest alpha probability mass of the empirical distribution.

    Fractionally weight the boundary order statistic when alpha*n is not an
    integer. This avoids including too much mass when observations tie at p05.
    """
    ordered = np.sort(finite_array(values, 1))
    require(len(ordered) > 0 and 0.0 < alpha <= 1.0, "Invalid empirical CVaR inputs")
    mass = len(ordered) * float(alpha)
    full = int(math.floor(mass))
    remainder = mass - full
    total = math.fsum(float(x) for x in ordered[:full])
    if remainder > 0.0:
        total += remainder * float(ordered[full])
    return total / mass


def descriptive_episode_statistics(values):
    pooled = finite_array(values).reshape(-1)
    require(len(pooled) > 0, "Empty descriptive sample")
    p05, q1, median, q3 = np.quantile(pooled, [0.05, 0.25, 0.5, 0.75], method="linear")
    return dict(
        episode_count=len(pooled),
        pooled_mean=float(pooled.mean()),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        p05=float(p05),
        cvar05_lower_return=float(empirical_lower_cvar(pooled)),
        interpretation="Pooled episode distribution description; not an inferential sample of independent training runs",
    )


def contrast_summary(differences, bootstrap_values):
    values = finite_array(differences, 1)
    require(len(values) >= 2, "At least two paired training seeds are required")
    average, sd = float(values.mean()), float(values.std(ddof=1))
    return dict(
        n_training_seeds=len(values),
        paired_differences=values.tolist(),
        mean_difference=average,
        sample_sd_difference=sd,
        percentile_95_ci=percentile_ci(bootstrap_values),
        dz=average / sd if sd > 0.0 else None,
        dz_status="defined" if sd > 0.0 else "undefined_zero_sample_sd",
        positive_seed_count=int(np.count_nonzero(values > 0.0)),
        zero_seed_count=int(np.count_nonzero(values == 0.0)),
        negative_seed_count=int(np.count_nonzero(values < 0.0)),
        sign_flip=exact_sign_flip(values),
    )


def analyze_cube(
    episodes,
    training_seeds=TRAINING_SEEDS,
    replicates=BOOTSTRAP_REPLICATES,
    analysis_seed=ANALYSIS_SEED,
):
    """Pure analysis; caller supplies an already-authorized complete endpoint cube."""
    cube = finite_array(episodes, 3)
    require(
        len(training_seeds) == cube.shape[0]
        and len(set(training_seeds)) == len(training_seeds),
        "Training labels do not match the input cube",
    )
    matrix = seed_means(cube)
    plan = resampling_plan(len(matrix), replicates, analysis_seed)
    differences = contrast_vectors(matrix)
    ordinary = contrast_vectors(
        paired_bootstrap_setting_means(matrix, plan["paired_rows"])
    )
    inference = {
        key: contrast_summary(differences[key], ordinary[key])
        for key in (PRIMARY, *SECONDARIES)
    }
    adjusted = holm_adjust(
        [inference[key]["sign_flip"]["p_value"] for key in SECONDARIES]
    )
    for key, p in zip(SECONDARIES, adjusted):
        inference[key]["holm_adjusted_p_value"] = float(p)
        inference[key]["ci_multiplicity_adjusted"] = False
    inference[PRIMARY]["role"] = "sole_primary"
    hierarchical = hierarchical_mask_bootstrap(
        matrix,
        plan["hierarchical_rows"],
        plan["hierarchical_u_columns"],
        plan["hierarchical_c_columns"],
    )
    summaries = []
    for column, setting in enumerate(SETTINGS):
        values = matrix[:, column]
        summaries.append(
            dict(
                setting_id=setting,
                mask_type=MASK_TYPES[setting],
                training_seed_means={
                    str(s): float(v) for s, v in zip(training_seeds, values)
                },
                mean_return=float(values.mean()),
                training_seed_sd=float(values.std(ddof=1)),
                pooled_episode_description=descriptive_episode_statistics(
                    cube[:, column, :]
                ),
            )
        )
    per_mask = []
    for column, setting in enumerate(SETTINGS[2:], 2):
        values = matrix[:, 1] - matrix[:, column]
        per_mask.append(
            dict(
                setting_id=setting,
                mask_type=MASK_TYPES[setting],
                paired_true_minus_mask={
                    str(s): float(v) for s, v in zip(training_seeds, values)
                },
                mean_difference=float(values.mean()),
                sample_sd_difference=float(values.std(ddof=1)),
                positive_seed_count=int(np.count_nonzero(values > 0.0)),
                role="all-mask descriptive comparison; no individual-mask significance selection",
            )
        )
    return dict(
        status="PASS",
        settings=list(SETTINGS),
        training_seeds=list(training_seeds),
        environments_per_setting_seed=int(cube.shape[2]),
        independent_inference_unit="paired training-seed row",
        ordinary_bootstrap_estimand="conditional on these 10 fixed masks and the fixed shared test bank",
        inferential_caveat="Exact sign-flip calibration requires exchangeability under sign reversal; masks and episodes are not training repetitions",
        analysis_rng=dict(
            bit_generator="PCG64",
            seed=int(analysis_seed),
            numpy_version=np.__version__,
            replicates_per_procedure=int(replicates),
            call_order=list(plan),
            index_array_sha256={
                key: hashlib.sha256(
                    values.astype("<i8", copy=False).tobytes(order="C")
                ).hexdigest()
                for key, values in plan.items()
            },
            confidence_interval="2.5th and 97.5th percentiles, NumPy linear quantiles",
        ),
        contrasts=inference,
        secondary_holm_family=list(SECONDARIES),
        hierarchical_mask_sensitivity={
            key: dict(
                percentile_95_ci=percentile_ci(values),
                bootstrap_mean=float(values.mean()),
                role="secondary sensitivity interval; not a replacement primary test",
                scope="training rows resampled; U and C each resampled among their five observed fixed columns; shared columns across all rows",
            )
            for key, values in hierarchical.items()
        },
        setting_summaries=summaries,
        all_mask_comparisons=per_mask,
    )


def load_endpoint_csv(path, environment_seeds):
    """Read only after a calling completion gate has passed."""
    envs = tuple(int(seed) for seed in environment_seeds)
    require(
        len(envs) == 200 and len(set(envs)) == 200,
        "Exactly 200 distinct, locked test seeds required",
    )
    lookup = {
        (setting, seed, env): (i, j, k)
        for i, seed in enumerate(TRAINING_SEEDS)
        for j, setting in enumerate(SETTINGS)
        for k, env in enumerate(envs)
    }
    cube = np.empty((10, 12, 200), dtype=np.float64)
    seen = set()
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            set(
                (
                    "setting_id",
                    "mask_type",
                    "training_seed",
                    "environment_seed",
                    "full_return",
                )
            ).issubset(reader.fieldnames or []),
            "Missing required endpoint CSV columns",
        )
        for row in reader:
            key = (
                row["setting_id"],
                int(row["training_seed"]),
                int(row["environment_seed"]),
            )
            require(
                key in lookup and key not in seen,
                f"Unknown or duplicate endpoint row {key}",
            )
            require(
                row["mask_type"] == MASK_TYPES[key[0]],
                f"Mask-family label mismatch {key}",
            )
            value = float(row["full_return"])
            require(math.isfinite(value), f"Nonfinite endpoint return {key}")
            cube[lookup[key]] = value
            seen.add(key)
    require(
        seen == set(lookup) and len(seen) == 24000,
        "Endpoint exact-key coverage is incomplete",
    )
    return cube


RANDOM_SETTINGS = ["Flat", "True", "C1", "C2", "C3", "C4", "C5"]


def analyze_random(arr):
    assert arr.shape == (7, 200, 10) and np.isfinite(arr).all()
    diff = np.stack(
        [arr[1] - arr[0], arr[1] - arr[2:].mean(axis=0)]
    )  # contrast x env x RNG
    envdiff = diff.mean(axis=2)
    n = 100000
    ci_rng = np.random.Generator(np.random.PCG64(2026091702))
    indices = ci_rng.integers(0, 200, size=(n, 200), dtype=np.int64)
    boots = envdiff[:, indices].mean(axis=2)
    cis = np.quantile(boots, [0.025, 0.975], axis=1).T
    signs_rng = np.random.Generator(np.random.PCG64(2026091703))
    signs = signs_rng.integers(0, 2, size=(n, 200), dtype=np.int64) * 2 - 1
    obs = envdiff.mean(axis=1)
    null = signs @ envdiff.T / 200
    assert np.isfinite(null).all(), "Nonfinite sign-flip statistics"
    tol = 64 * np.finfo(float).eps * np.maximum(1, np.abs(envdiff).mean(axis=1))
    p = (np.sum(np.abs(null) >= np.abs(obs)[None, :] - tol, axis=0) + 1) / (n + 1)
    order = np.argsort(p)
    holm = np.empty(2)
    holm[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.array([2, 1])))
    del indices, boots, signs, null
    cross_rng = np.random.Generator(np.random.PCG64(2026091704))
    mask_rng = np.random.Generator(np.random.PCG64(2026091705))
    crossed = []
    mask_sens = []
    for start in range(0, n, 1000):
        es = cross_rng.integers(0, 200, size=(1000, 200), dtype=np.int64)
        rs = cross_rng.integers(0, 10, size=(1000, 10), dtype=np.int64)
        ew = np.stack([np.bincount(x, minlength=200) for x in es]) / 200
        rw = np.stack([np.bincount(x, minlength=10) for x in rs]) / 10
        agg = np.einsum("be,ser,br->bs", ew, arr, rw, optimize=True)
        crossed.append(
            np.column_stack(
                [agg[:, 1] - agg[:, 0], agg[:, 1] - agg[:, 2:].mean(axis=1)]
            )
        )
        cs = mask_rng.integers(0, 5, size=(1000, 5), dtype=np.int64)
        mask_sens.append(
            agg[:, 1] - np.take_along_axis(agg[:, 2:], cs, axis=1).mean(axis=1)
        )
    crossci = np.quantile(np.concatenate(crossed), [0.025, 0.975], axis=0).T
    maskci = np.quantile(np.concatenate(mask_sens), [0.025, 0.975])
    contrasts = {}
    for j, name in enumerate(["MMDP_minus_Flat", "MMDP_minus_Mask"]):
        contrasts[name] = dict(
            mean=float(obs[j]),
            ci95=cis[j].tolist(),
            ci_unit="200 paired environments after averaging ten RNG seeds",
            ci_multiplicity_adjusted=False,
            paired_sd=float(envdiff[j].std(ddof=1)),
            positive_environments=int((envdiff[j] > 0).sum()),
            monte_carlo_p=float(p[j]),
            holm_p=float(holm[j]),
            crossed_environment_rng_ci95=crossci[j].tolist(),
            rng_seed_gaps=diff[j].mean(axis=0).tolist(),
        )
    contrasts["MMDP_minus_Mask"][
        "environment_rng_mask_sensitivity_ci95"
    ] = maskci.tolist()
    setting_means = {s: float(arr[i].mean()) for i, s in enumerate(RANDOM_SETTINGS)}
    setting_means["Mask"] = float(arr[2:].mean())
    return dict(random_means=setting_means, random_contrasts=contrasts)
