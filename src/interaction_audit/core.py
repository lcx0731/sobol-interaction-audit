"""Jansen pick-freeze estimates; replication over whole randomized designs."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import qmc


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def _array(value, name):
    data = np.asarray(value)
    if data.dtype.kind not in "fiu" or not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must contain finite real numbers")
    return data.astype(float)


@dataclass(frozen=True)
class Estimate:
    first: np.ndarray
    total: np.ndarray
    variance: float
    normalization: str
    evaluations: int


def jansen(f_a, f_b, f_hybrid, variance=None):
    """Hybrid i uses A with column i replaced by B. Never clip estimates.

    Independent input coordinates and a deterministic scalar model are required.
    Passing a known positive population variance removes the plug-in ratio bias.
    A low-level call cannot establish that the evaluations came from this design.
    """
    a, b, c = _array(f_a, "f_a"), _array(f_b, "f_b"), _array(f_hybrid, "f_hybrid")
    if a.ndim != 1 or a.shape != b.shape or not 2 <= len(a) <= 65536:
        raise ValueError("f_a and f_b must have equal length in 2..65536")
    if c.ndim != 2 or c.shape[1] != len(a) or not 1 <= c.shape[0] <= 32:
        raise ValueError("f_hybrid must have shape (dimension,n), dimension in 1..32")
    normalization = "pooled_sample" if variance is None else "known_population"
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if variance is None:
                variance = float(np.var(np.concatenate((a, b)), ddof=1))
            if type(variance) not in (int, float) or not 0 < variance <= 1e200 or not math.isfinite(variance):
                raise ValueError("normalizing variance must be finite and in (0,1e200]")
            total = np.mean((c - a) ** 2, axis=1) / (2 * variance)
            first = 1 - np.mean((c - b) ** 2, axis=1) / (2 * variance)
    except FloatingPointError as error:
        raise ValueError("evaluation scale exceeds numerical range") from error
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(total)):
        raise ValueError("non-finite sensitivity estimate")
    return Estimate(first, total, float(variance), normalization, len(a) * (c.shape[0] + 2))


def draw_design(dimension, power, scheme="scrambled-sobol", seed=1729):
    """Two matrices on [-1,1]^d from one joint 2d-dimensional design."""
    _integer(dimension, 1, 32, "dimension")
    _integer(power, 1, 16, "power")
    _integer(seed, 0, 2**32-1, "seed")
    rng = np.random.Generator(np.random.PCG64(seed))
    if scheme == "iid":
        joint = rng.random((2**power, 2 * dimension))
    elif scheme == "scrambled-sobol":
        # Do not skip, thin, optimize, or use a non-power-of-two prefix.
        joint = qmc.Sobol(2 * dimension, scramble=True, bits=52, rng=rng).random_base2(power)
    else:
        raise ValueError("scheme must be iid or scrambled-sobol")
    joint = 2 * joint - 1
    return joint[:, :dimension].copy(), joint[:, dimension:].copy()


def evaluate_design(function, a, b, variance=None):
    a, b = _array(a, "a"), _array(b, "b")
    if a.ndim != 2 or a.shape != b.shape or not 2 <= len(a) <= 65536 or not 1 <= a.shape[1] <= 32:
        raise ValueError("a and b must be equal-shape (n,d) arrays in supported bounds")
    if np.any(np.abs(a) > 1) or np.any(np.abs(b) > 1):
        raise ValueError("input coordinates must lie in [-1,1]")

    def call(x):
        values = _array(function(x.copy()), "model output")
        if values.shape != (len(a),):
            raise ValueError("model output must be a scalar per row, shape (n,)")
        return values

    y_a, y_b = call(a), call(b)
    hybrids = []
    for j in range(a.shape[1]):
        mixed = a.copy()
        mixed[:, j] = b[:, j]
        hybrids.append(call(mixed))
    return jansen(y_a, y_b, np.array(hybrids), variance)


def replicate(function, dimension, power=10, repetitions=32, scheme="scrambled-sobol", seed=1729, variance=None):
    """Independent randomizations, not row bootstrap within a Sobol sequence."""
    _integer(repetitions, 2, 256, "repetitions")
    _integer(seed, 0, 2**32-1, "seed")
    _integer(dimension, 1, 32, "dimension")
    _integer(power, 1, 16, "power")
    if repetitions * 2**power * (dimension + 2) > 20_000_000:
        raise ValueError("replicated evaluation budget exceeds 20000000")
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(repetitions)]
    if len(set(seeds)) != repetitions:
        raise ArithmeticError("replicate seed collision")
    results = [evaluate_design(function, *draw_design(dimension, power, scheme, child), variance=variance)
               for child in seeds]
    first, total = np.array([x.first for x in results]), np.array([x.total for x in results])

    def summarize(values):
        sd = values.std(axis=0, ddof=1)
        return {"mean": values.mean(axis=0).tolist(), "replicate_sd": sd.tolist(),
                "mean_mcse": (sd / math.sqrt(repetitions)).tolist(), "estimates": values.tolist()}

    return {"scheme": scheme, "power": power, "rows_per_matrix": 2**power, "repetitions": repetitions,
            "seed": seed, "replicate_seeds": seeds, "dimension": dimension,
            "normalization": results[0].normalization,
            "normalizing_variances": [x.variance for x in results],
            "evaluations_per_replicate": results[0].evaluations,
            "total_evaluations": sum(x.evaluations for x in results),
            "first": summarize(first), "total": summarize(total)}
