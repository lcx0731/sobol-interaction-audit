"""Synthetic sensitivity experiments with exact polynomial reference values."""

from .core import draw_design, evaluate_design, jansen, replicate
from .reference import Polynomial, benchmark

__all__ = ["Polynomial", "benchmark", "draw_design", "evaluate_design", "jansen", "replicate"]
