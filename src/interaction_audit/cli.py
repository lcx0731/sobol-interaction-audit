"""Run the fixed synthetic interaction benchmark with reproducible settings."""

import argparse
import json

from .core import replicate
from .reference import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--scheme", choices=["iid", "scrambled-sobol"], default="scrambled-sobol")
    parser.add_argument("--normalization", choices=["known", "pooled"], default="known")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        model = benchmark()
        truth = model.anova()
        result = replicate(model, model.dimension, args.power, args.repetitions, args.scheme, args.seed,
                           float(truth.variance) if args.normalization == "known" else None)
        print(json.dumps({"data": "Deterministic seeded synthetic polynomial; independent Uniform[-1,1] inputs.",
                          "uncertainty": "MCSE describes the randomized replicate mean, not bias or a confidence guarantee.",
                          "reference": truth.as_dict(), "result": result}, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
