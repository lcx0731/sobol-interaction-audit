from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from interaction_audit import Polynomial, benchmark
from interaction_audit.reference import moment


def test_benchmark_has_exact_main_and_interaction_components():
    truth = benchmark().anova()
    assert truth.mean == 0 and truth.variance == F(43, 15)
    assert truth.first == (F(5, 43), F(20, 43), F(3, 43), F(0))
    assert truth.total == (F(20, 43), F(20, 43), F(18, 43), F(0))
    assert truth.components[5] == 1
    assert sum(truth.components) == truth.variance
    x = np.random.default_rng(10).uniform(-1, 1, (30, 4))
    expected = x[:, 0] + 2*x[:, 1] + 3*x[:, 0]*x[:, 2] + (3*x[:, 2]**2-1)/2
    np.testing.assert_allclose(benchmark()(x), expected, atol=1e-14)


def test_moments_and_noncentered_polynomial_terms():
    assert [moment(i) for i in range(5)] == [1, 0, F(1, 3), 0, F(1, 5)]
    # x0^2*x1^2 has main effects too; classifying each monomial by support is wrong.
    truth = Polynomial(2, {(2, 2): 1, (0, 0): 7}).anova()
    assert truth.mean == F(64, 9)
    assert truth.components[1] == truth.components[2] == F(4, 405)
    assert truth.components[3] == F(16, 2025)


def test_fraction_projection_against_tensor_gauss_quadrature():
    poly = Polynomial(3, {(3, 0, 1): F(2, 3), (0, 2, 0): 2, (1, 1, 1): -3,
                          (0, 0, 2): 1, (0, 0, 0): 4})
    truth = poly.anova()
    nodes, weights = leggauss(5)
    weights = weights / 2
    points = np.array(list(product(nodes, repeat=3)))
    values = poly(points).reshape(5, 5, 5)
    mean = np.einsum("i,j,k,ijk", weights, weights, weights, values)
    variance = np.einsum("i,j,k,ijk", weights, weights, weights, (values-mean)**2)
    assert mean == pytest.approx(float(truth.mean), abs=2e-13)
    assert variance == pytest.approx(float(truth.variance), abs=2e-13)
    for mask in range(1, 8):
        conditional = values
        for axis in reversed(range(3)):
            if not mask & (1 << axis):
                conditional = np.tensordot(conditional, weights, axes=(axis, 0))
        squared = (conditional - mean)**2
        for axis in reversed(range(mask.bit_count())):
            squared = np.tensordot(squared, weights, axes=(axis, 0))
        component_sum = sum(v for s, v in enumerate(truth.components) if s & mask == s)
        assert float(squared) == pytest.approx(float(component_sum), abs=2e-12)


def test_single_input_and_constant_models():
    truth = Polynomial(1, {(0,): 1, (7,): -2}).anova()
    assert truth.first == truth.total == (1,)
    for terms in ({(0,): 7}, {(1,): 0}):
        with pytest.raises(ValueError, match="positive population variance"):
            Polynomial(1, terms).anova()


@pytest.mark.parametrize("dimension,terms", [(0, {(0,): 1}), (7, {(0,): 1}),
    (True, {(0,): 1}), (1, {}), (2, {(1,): 1}), (1, {(-1,): 1}),
    (1, {(9,): 1}), (1, {(1,): 0.1}), (1, {(True,): 1})])
def test_invalid_reference_inputs(dimension, terms):
    with pytest.raises(ValueError):
        Polynomial(dimension, terms)
