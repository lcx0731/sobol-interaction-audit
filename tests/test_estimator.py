from itertools import product

import numpy as np
import pytest

from interaction_audit import benchmark, draw_design, evaluate_design, jansen, replicate


def test_complete_cube_pairing_has_exact_discrete_anova_indices():
    # Independent Rademacher inputs are an independent finite oracle for the
    # algebra; the Uniform[-1,1] polynomial reference is not used in this test.
    cube = np.array(list(product([-1., 1.], repeat=4)))
    a = np.repeat(cube, len(cube), axis=0)
    b = np.tile(cube, (len(cube), 1))
    f = lambda x: x[:, 0] + 2*x[:, 1] + 3*x[:, 0]*x[:, 2]
    result = evaluate_design(f, a, b, variance=14)
    np.testing.assert_allclose(result.first, [1/14, 4/14, 0, 0], atol=1e-15)
    np.testing.assert_allclose(result.total, [10/14, 4/14, 9/14, 0], atol=1e-15)


def test_orientation_and_estimates_are_not_clipped():
    result = jansen([0, 2], [2, 0], [[2, 0], [0, 2]], variance=1)
    np.testing.assert_array_equal(result.first, [1, -1])
    np.testing.assert_array_equal(result.total, [2, 0])


def test_evaluation_accounting_no_input_mutation_and_affine_invariance():
    model = benchmark()
    a, b = draw_design(4, 8, "iid", 4)
    before = a.copy(), b.copy()
    calls = []

    def spy(x):
        calls.append(len(x))
        return model(x)

    base = evaluate_design(spy, a, b, variance=float(model.anova().variance))
    shifted = evaluate_design(lambda x: 10000 - 3*model(x), a, b,
                              variance=9*float(model.anova().variance))
    assert calls == [256] * 6 and base.evaluations == sum(calls)
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])
    np.testing.assert_allclose(base.first, shifted.first, atol=1e-12)
    np.testing.assert_allclose(base.total, shifted.total, atol=1e-12)
    assert base.total[3] == 0


@pytest.mark.parametrize("scheme", ["iid", "scrambled-sobol"])
def test_sampling_reproducibility_and_convergence_to_reference(scheme):
    a, b = draw_design(4, 13, scheme, 39)
    again = draw_design(4, 13, scheme, 39)
    np.testing.assert_array_equal(a, again[0])
    np.testing.assert_array_equal(b, again[1])
    assert np.all(np.abs(a) <= 1) and np.all(np.abs(b) <= 1) and not np.array_equal(a, b)
    model = benchmark()
    truth = model.anova()
    observed = evaluate_design(model, a, b, variance=float(truth.variance))
    # A loose fixed-seed integration check, not a performance/coverage guarantee.
    np.testing.assert_allclose(observed.first, [float(x) for x in truth.first], atol=.06)
    np.testing.assert_allclose(observed.total, [float(x) for x in truth.total], atol=.06)


def test_pooled_variance_and_replicate_uncertainty_use_the_right_units():
    a, b = draw_design(4, 5, "iid", 12)
    model = benchmark()
    result = evaluate_design(model, a, b)
    assert result.variance == pytest.approx(np.var(np.concatenate([model(a), model(b)]), ddof=1))
    assert result.normalization == "pooled_sample"
    report = replicate(model, 4, power=5, repetitions=5, scheme="iid", seed=12)
    assert len(set(report["replicate_seeds"])) == 5
    assert report["total_evaluations"] == 5*32*6
    values = np.array(report["first"]["estimates"])
    expected_mcse = np.sqrt(np.sum((values-values.mean(axis=0))**2, axis=0) / (5*4))
    np.testing.assert_allclose(report["first"]["mean_mcse"], expected_mcse)
    first_a, first_b = draw_design(4, 5, "iid", report["replicate_seeds"][0])
    np.testing.assert_array_equal(values[0], evaluate_design(model, first_a, first_b).first)
    assert report == replicate(model, 4, power=5, repetitions=5, scheme="iid", seed=12)


@pytest.mark.parametrize("args", [(0, 4), (4, 0), (4, 17), (True, 4), (4, 3, "other"), (4, 3, "iid", -1)])
def test_invalid_design_settings(args):
    with pytest.raises(ValueError):
        draw_design(*args)


def test_invalid_evaluations_variance_and_budget():
    for variance in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            jansen([1, 2], [2, 1], [[2, 1]], variance)
    with pytest.raises(ValueError):
        jansen([1, 1], [1, 1], [[1, 1]])
    for a, b, c in [([1], [2], [[1]]), ([1, 2], [3], [[1, 2]]),
                    ([1, 2], [3, float("inf")], [[1, 2]]),
                    ([1j, 2], [3, 4], [[1, 2]]), ([1, 2], [3, 4], [[1]])]:
        with pytest.raises(ValueError):
            jansen(a, b, c)
    a, b = draw_design(2, 2)
    for function in [lambda x: np.zeros((4, 1)), lambda x: np.full(4, np.nan)]:
        with pytest.raises(ValueError):
            evaluate_design(function, a, b)
    with pytest.raises(ValueError, match="budget"):
        replicate(benchmark(), 4, power=16, repetitions=256)
