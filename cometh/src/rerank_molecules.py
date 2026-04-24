import argparse
from rdkit import Chem
from rdkit.Chem import QED, Descriptors


def load_smiles(file_path):
    """Load SMILES strings from a file (one per line)."""
    with open(file_path, "r") as f:
        smiles = [line.strip() for line in f if line.strip()]
    return smiles


def compute_properties(smiles_list):
    """Compute QED and Molecular Weight for valid molecules."""
    results = []

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue  # skip invalid molecules

        try:
            qed = QED.qed(mol)
            mw = Descriptors.MolWt(mol)
            results.append({
                "smiles": sm,
                "qed": qed,
                "mw": mw
            })
        except:
            continue  # skip edge-case failures

    return results


def filter_by_mw(results, min_mw=None, max_mw=None):
    """Filter molecules by molecular weight range."""
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


def rerank_by_qed(results, top_k=20):
    """Sort molecules by QED and return top-K."""
    sorted_results = sorted(results, key=lambda x: x["qed"], reverse=True)
    return sorted_results[:top_k]


def save_results(results, output_path):
    """Save results to file."""
    with open(output_path, "w") as f:
        f.write("SMILES\tQED\tMW\n")
        for r in results:
            f.write(f"{r['smiles']}\t{r['qed']:.4f}\t{r['mw']:.2f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to SMILES file")
    parser.add_argument("--output", default="top_molecules.txt")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_mw", type=float, default=None)
    parser.add_argument("--max_mw", type=float, default=None)

    args = parser.parse_args()

    # Load molecules
    smiles = load_smiles(args.input)

    # Compute properties
    results = compute_properties(smiles)

    print(f"Loaded {len(smiles)} molecules")
    print(f"Valid molecules: {len(results)}")

    # Optional MW filtering
    results = filter_by_mw(results, args.min_mw, args.max_mw)
    print(f"After MW filtering: {len(results)}")

    # Rerank
    top_results = rerank_by_qed(results, args.top_k)

    # Save
    save_results(top_results, args.output)

    print(f"Saved top {len(top_results)} molecules to {args.output}")


if __name__ == "__main__":
    main()