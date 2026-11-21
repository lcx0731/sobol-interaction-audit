"""Reproduce synthetic sensitivity, randomized integration, and clipping audits."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from interaction_audit import benchmark, replicate


SEED = 1729
REPETITIONS = 64
POWERS = [6, 8, 10, 12]
SCHEMES = ["iid", "scrambled-sobol"]
COLORS = {"iid": "#b76e36", "scrambled-sobol": "#227f80"}


def add_errors(report, reference):
    for kind in ["first", "total"]:
        values = np.asarray(report[kind]["estimates"])
        truth = np.array([float(x) for x in getattr(reference, kind)])
        errors = values - truth
        report[kind]["empirical_bias"] = errors.mean(axis=0).tolist()
        report[kind]["rmse_per_index"] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
        report[kind]["rmse_over_replicates_and_indices"] = float(np.sqrt(np.mean(errors**2)))
    return report


def plot(metrics, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "axes.facecolor": "#fafbf8", "figure.facecolor": "#fafbf8"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    fig.subplots_adjust(top=.85, bottom=.13, left=.09, right=.96, wspace=.30, hspace=.58)
    fig.suptitle("Which inputs matter together?", x=.07, y=.97, ha="left", fontsize=24, weight="bold")
    fig.text(.07, .916, "Exact polynomial truth  |  Independent uniform inputs  |  64 randomized replicates per setting",
             fontsize=11, color="#526066")
    exact = metrics["reference"]
    ax = axes[0, 0]
    positions = np.arange(4)
    ax.bar(positions-.18, exact["first_float"], width=.36, color="#4c617d", label="First order")
    ax.bar(positions+.18, exact["total_float"], width=.36, color="#227f80", label="Total order")
    ax.set_xticks(positions, ["x0", "x1", "x2", "x3 (inactive)"])
    ax.set(title="A  Main effects do not capture all influence", ylabel="Exact share of output variance", ylim=(0, .64))
    ax.legend(frameon=False, fontsize=9)
    ax.text(.02, .94, "Interaction {x0, x2}: 15/43 of total variance", va="top",
            transform=ax.transAxes, fontsize=9, color="#526066")

    ax = axes[0, 1]
    for scheme in SCHEMES:
        rows = [r for r in metrics["known_variance"] if r["scheme"] == scheme]
        ax.loglog([r["evaluations_per_replicate"] for r in rows],
                  [r["first"]["rmse_over_replicates_and_indices"] for r in rows],
                  "o-", color=COLORS[scheme], lw=2, label=scheme)
    ax.set(title="B  Equal model-evaluation budgets", xlabel="Model evaluations per replicate: 6n",
           ylabel="RMSE of first-order estimates")
    ax.legend(frameon=False, fontsize=9)
    ax.text(.02, .04, "RMSE averages 64 replicates and 4 inputs\nKnown population variance in denominator",
            transform=ax.transAxes, fontsize=9, color="#526066")
    ax.grid(which="both", alpha=.15)

    ax = axes[1, 0]
    clip = metrics["clipping_diagnostic"]
    ax.hist(clip["raw_estimates"], bins=16, color=COLORS["iid"], alpha=.65,
            edgecolor="white", label="Raw inactive-input estimates")
    ax.axvline(0, color="#22323b", lw=1.3, label="Truth = 0")
    ax.axvline(clip["clipped_mean"], color="#227f80", ls="--", lw=2, label="Mean after clipping to [0,1]")
    ax.set(title="C  Clipping trades bias against squared error", xlabel="First-order estimate for inactive x3; iid, n=64",
           ylabel="Replicate count")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(.98, .70, f"Raw mean: {clip['raw_mean']:.4f}\nClipped mean: {clip['clipped_mean']:.4f}\n"
            f"Raw MSE: {clip['raw_mse']:.4f}\nClipped MSE: {clip['clipped_mse']:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#526066",
            bbox={"facecolor": "#fafbf8", "edgecolor": "none", "alpha": .9, "pad": 4})

    ax = axes[1, 1]
    for index, scheme in enumerate(SCHEMES):
        known = next(r for r in metrics["known_variance"] if r["scheme"] == scheme and r["power"] == 10)
        pooled = next(r for r in metrics["pooled_variance"] if r["scheme"] == scheme)
        vals = [known["first"]["rmse_over_replicates_and_indices"],
                pooled["first"]["rmse_over_replicates_and_indices"]]
        ax.bar(np.array([0, 1]) + (index-.5)*.32, vals, width=.32, color=COLORS[scheme], label=scheme)
    ax.set_xticks([0, 1], ["Known population variance", "Pooled sample variance"])
    ax.set(title="D  Normalization is part of the estimator", ylabel="First-order RMSE; n=1024", ylim=(0, .052))
    ax.legend(frameon=False, fontsize=9)
    ax.text(.02, .94, "Paired seeds within each sampling scheme", transform=ax.transAxes,
            va="top", fontsize=9, color="#526066")
    for ax in (axes[0, 0], axes[1, 1]):
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.07, .05,
             "Synthetic benchmark, seed 1729. Results describe these functions and budgets, not a universal advantage for one sampler.\n"
             "MCSE is computed across whole randomized replicates; rows within a scrambled Sobol design are dependent.",
             fontsize=9, color="#526066", linespacing=1.6)
    fig.savefig(directory / "sensitivity_audit.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = benchmark()
    truth = model.anova()
    metrics = {
        "data": "Deterministic seeded synthetic polynomial; no external data or fitted model.",
        "function": "x0 + 2*x1 + 3*x0*x2 + (3*x2^2-1)/2; x3 inactive",
        "input_law": "Four independent Uniform[-1,1] coordinates",
        "design": "n rows in each A,B; each hybrid replaces exactly one A column by B; n*(d+2) scalar model evaluations",
        "reference": {**truth.as_dict(), "first_float": [float(x) for x in truth.first],
                      "total_float": [float(x) for x in truth.total]},
        "uncertainty": "MCSE estimates the standard deviation of the randomized replicate mean. It is not an error bound, does not cover bias, and is not a simultaneous confidence interval.",
        "known_variance": [], "pooled_variance": [],
    }
    for scheme in SCHEMES:
        for power in POWERS:
            report = add_errors(replicate(model, 4, power, REPETITIONS, scheme, SEED,
                                         variance=float(truth.variance)), truth)
            metrics["known_variance"].append(report)
            print(f"{scheme:16} n={2**power:4} first RMSE={report['first']['rmse_over_replicates_and_indices']:.6f}", flush=True)
        pooled = replicate(model, 4, 10, REPETITIONS, scheme, SEED)
        metrics["pooled_variance"].append(add_errors(pooled, truth))
    raw = np.array(metrics["known_variance"][0]["first"]["estimates"])[:, 3]
    clipped = np.clip(raw, 0, 1)
    metrics["clipping_diagnostic"] = {
        "scheme": "iid", "n": 64, "input": "x3", "truth": 0,
        "raw_estimates": raw.tolist(), "raw_mean": float(raw.mean()),
        "clipped_mean": float(clipped.mean()), "raw_mse": float(np.mean(raw**2)),
        "clipped_mse": float(np.mean(clipped**2)), "negative_count": int(np.sum(raw < 0)),
        "interpretation": "Clipping changes bias and the sampling distribution. Projection to [0,1] cannot increase squared error when the true index lies in [0,1]; here its empirical mean is farther from the zero truth.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    plot(metrics, args.output)
    print(f"Wrote metrics.json and sensitivity_audit.png to {args.output}")


if __name__ == "__main__":
    main()
