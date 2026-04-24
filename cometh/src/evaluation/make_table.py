import json

with open("evaluation/evaluation_summary.json") as f:
    data = json.load(f)

print(f"{'Method':<12} {'Valency%':<10} {'Atom%':<10} {'TotalEx':<10} {'MaxEx':<10}")

for method, metrics in data.items():
    print(f"{method:<12} "
          f"{metrics.get('valency_valid_rate', 0):<10.3f} "
          f"{metrics.get('atom_valency_valid_rate', 0):<10.3f} "
          f"{metrics.get('avg_total_excess', 0):<10.3f} "
          f"{metrics.get('avg_max_excess', 0):<10.3f}")