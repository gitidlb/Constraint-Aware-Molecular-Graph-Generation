import os
import subprocess
import json

RESULTS_ROOT = "results"

METHODS = [
    "baseline",
    "delete",
    "gradual",
    "rerank"
]

OUTPUT_FILE = "evaluation/evaluation_summary.json"


def run_valency_eval(method_path):
    result = subprocess.run(
        ["python", "cometh/src/evaluate_valency_metrics.py", "--folder", method_path],
        capture_output=True,
        text=True
    )
    return result.stdout


def parse_output(output):
    metrics = {}
    for line in output.split("\n"):
        if "molecules_fully_valency_valid_rate" in line:
            metrics["valency_valid_rate"] = float(line.split(":")[-1])
        elif "atom_level_valency_valid_rate" in line:
            metrics["atom_valency_valid_rate"] = float(line.split(":")[-1])
        elif "avg_total_excess_valency_per_molecule" in line:
            metrics["avg_total_excess"] = float(line.split(":")[-1])
        elif "avg_max_excess_valency_per_molecule" in line:
            metrics["avg_max_excess"] = float(line.split(":")[-1])
    return metrics


def main():
    all_results = {}

    for method in METHODS:
        path = os.path.join(RESULTS_ROOT, method)

        if not os.path.exists(path):
            continue

        output = run_valency_eval(path)
        metrics = parse_output(output)

        all_results[method] = metrics

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()