import argparse
import csv
import sys
from rdkit import Chem
from rdkit.Chem import QED, Descriptors


def load_smiles(file_path: str) -> list[str]:
    smiles_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Take only the first token if the file is delimited
            token = line.split()[0] if "\t" not in line else line.split("\t")[0]
            smiles_list.append(token)
    return smiles_list

def save_results(results: list[dict], output_path: str, fmt: str = "tsv") -> None:
    delimiter = "\t" if fmt == "tsv" else ","
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["smiles", "qed", "mw", "score"],
            delimiter=delimiter,
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "smiles": r["smiles"],
                    "qed": f"{r['qed']:.4f}",
                    "mw": f"{r['mw']:.2f}",
                    "score": f"{r['score']:.4f}",
                }
            )

def compute_properties(smiles_list: list[str]) -> list[dict]:
    results = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        try:
            qed = QED.qed(mol)
            mw = Descriptors.MolWt(mol)
            results.append({"smiles": sm, "qed": qed, "mw": mw})
        except Exception:
            continue
    return results

def filter_by_mw( results: list[dict], min_mw: float | None = None, max_mw: float | None = None,) -> list[dict]:
    if min_mw is None and max_mw is None:
        return results
    filtered = []
    for r in results:
        if min_mw is not None and r["mw"] < min_mw:
            continue
        if max_mw is not None and r["mw"] > max_mw:
            continue
        filtered.append(r)
    return filtered

def add_score(results: list[dict], score_mode: str, mw_target: float | None) -> list[dict]:
    """
    Attach a scalar 'score' field used for ranking.

    Modes
    -----
    qed: score = QED  (higher is better)
    mw: score = -|MW - mw_target| / mw_target  (closer to target = higher)
    composite: score = 0.5 * QED + 0.5 * (1 - |MW - mw_target| / mw_target)
                Requires --mw_target.
    """
    if score_mode in ("mw", "composite") and mw_target is None:
        print(
            "ERROR: --score mw or composite requires --mw_target to be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    for r in results:
        if score_mode == "qed":
            r["score"] = r["qed"]
        elif score_mode == "mw":
            r["score"] = -abs(r["mw"] - mw_target) / mw_target
        elif score_mode == "composite":
            mw_score = 1.0 - abs(r["mw"] - mw_target) / mw_target
            mw_score = max(0.0, mw_score)          # clip to [0, 1]
            r["score"] = 0.5 * r["qed"] + 0.5 * mw_score
    return results

def rerank(results: list[dict], top_k: int) -> list[dict]:
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft-constraint reranking for Cometh-generated molecules.")
    parser.add_argument("--input", required=True, help="Path to SMILES file produced by Cometh sampling.",)
    parser.add_argument("--output", default="top_molecules.tsv", help="Output file path (default: top_molecules.tsv).",)
    parser.add_argument("--top_k", type=int, default=20, help="Number of top molecules to retain (default: 20).",)
    # MW hard filter
    parser.add_argument("--min_mw", type=float, default=None, help="Minimum molecular weight (hard filter).")
    parser.add_argument("--max_mw", type=float, default=None, help="Maximum molecular weight (hard filter).")
    # Soft scoring
    parser.add_argument(
        "--score", choices=["qed", "mw", "composite"], default="qed",
        help=(
            "Scoring function for reranking:\n"
            "  qed       - rank by QED alone (default)\n"
            "  mw        - rank by proximity to --mw_target\n"
            "  composite - 0.5*QED + 0.5*MW-proximity (requires --mw_target)"
        ),
    )
    parser.add_argument("--mw_target", type=float, default=None, help="Target MW for 'mw' and 'composite' scoring modes.",)
    parser.add_argument("--output_format", choices=["tsv", "csv"], default="tsv", help="Output delimiter format (default: tsv).",)
    args = parser.parse_args()


    smiles = load_smiles(args.input)
    print(f"Loaded         : {len(smiles)} SMILES")

    results = compute_properties(smiles)
    print(f"Valid molecules: {len(results)}")

    results = filter_by_mw(results, args.min_mw, args.max_mw)
    print(f"After MW filter: {len(results)}")

    results = add_score(results, args.score, args.mw_target)

    top_results = rerank(results, args.top_k)

    save_results(top_results, args.output, args.output_format)
    print(f"Saved top-{len(top_results)} molecules → {args.output}  (score={args.score})")


if __name__ == "__main__":
    main()