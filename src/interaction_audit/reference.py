"""Exact monomial moments and ANOVA for independent Uniform[-1,1] inputs."""

from dataclasses import dataclass
from fractions import Fraction
from math import prod

import numpy as np


def moment(power):
    if type(power) is not int or not 0 <= power <= 16:
        raise ValueError("moment power must be an integer in 0..16")
    return Fraction(0) if power % 2 else Fraction(1, power + 1)


@dataclass(frozen=True)
class Anova:
    mean: Fraction
    variance: Fraction
    components: tuple
    first: tuple
    total: tuple

    def as_dict(self):
        return {"mean": str(self.mean), "variance": str(self.variance),
                "components_by_bitmask": {str(k): str(v) for k, v in enumerate(self.components) if k and v},
                "first": [str(x) for x in self.first], "total": [str(x) for x in self.total]}


class Polynomial:
    """A bounded-size exact reference; exponents index ordinary monomials."""

    def __init__(self, dimension, terms):
        if type(dimension) is not int or not 1 <= dimension <= 6:
            raise ValueError("reference dimension must be an integer in 1..6")
        if not isinstance(terms, dict) or not 1 <= len(terms) <= 64:
            raise ValueError("terms must be a dictionary of 1..64 monomials")
        validated = []
        for powers, coefficient in terms.items():
            if not isinstance(powers, tuple) or len(powers) != dimension or any(type(p) is not int or not 0 <= p <= 8 for p in powers):
                raise ValueError("monomial powers must be a dimension-length tuple of integers in 0..8")
            if type(coefficient) not in (int, Fraction):
                raise ValueError("coefficients must be integers or Fraction values")
            if coefficient:
                validated.append((powers, Fraction(coefficient)))
        self.dimension = dimension
        self.terms = tuple(sorted(validated))

    def __call__(self, x):
        x = np.asarray(x)
        if x.dtype.kind not in "fiu" or x.ndim != 2 or x.shape[1] != self.dimension or not np.all(np.isfinite(x)):
            raise ValueError("x must be a finite real (n,dimension) array")
        with np.errstate(over="raise", invalid="raise"):
            result = np.zeros(len(x))
            for powers, coefficient in self.terms:
                result += float(coefficient) * np.prod(x.astype(float) ** np.array(powers), axis=1)
        return result

    @staticmethod
    def _moments(terms):
        mean = sum((c * prod(moment(p) for p in powers) for powers, c in terms), Fraction(0))
        second = sum((c * e * prod(moment(a+b) for a, b in zip(powers, other))
                      for powers, c in terms for other, e in terms), Fraction(0))
        return mean, second - mean * mean

    def anova(self):
        mean, variance = self._moments(self.terms)
        if variance <= 0:
            raise ValueError("Sobol indices require positive population variance")
        components = [Fraction(0)] * (1 << self.dimension)
        for mask in range(1, 1 << self.dimension):
            projected = {}
            for powers, coefficient in self.terms:
                kept = tuple(p if mask & (1 << j) else 0 for j, p in enumerate(powers))
                integrated = coefficient * prod(moment(p) for j, p in enumerate(powers) if not mask & (1 << j))
                projected[kept] = projected.get(kept, Fraction(0)) + integrated
            _, closed_variance = self._moments(tuple(projected.items()))
            proper_sum = Fraction(0)
            sub = (mask - 1) & mask
            while sub:
                proper_sum += components[sub]
                sub = (sub - 1) & mask
            components[mask] = closed_variance - proper_sum
        if any(v < 0 for v in components) or sum(components) != variance:
            raise ArithmeticError("exact ANOVA partition failed")
        first = tuple(components[1 << j] / variance for j in range(self.dimension))
        total = tuple(sum(v for mask, v in enumerate(components) if mask & (1 << j)) / variance
                      for j in range(self.dimension))
        return Anova(mean, variance, tuple(components), first, total)


def benchmark():
    """x0 + 2*x1 + 3*x0*x2 + (3*x2**2-1)/2; x3 is inactive."""
    return Polynomial(4, {(1, 0, 0, 0): 1, (0, 1, 0, 0): 2, (1, 0, 1, 0): 3,
                          (0, 0, 2, 0): Fraction(3, 2), (0, 0, 0, 0): Fraction(-1, 2)})
