import os
import glob
import argparse
from collections import Counter

import torch
from rdkit import Chem
from rdkit import RDLogger

from metrics.molecular_metrics import allowed_bonds


RDLogger.DisableLog("rdApp.*")

ATOM_DECODER_QM9 = ['C', 'N', 'O', 'F']
ATOM_DECODER_MOSES = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br']


def parse_generated_samples_file(path: str):
    molecules = []

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if not line.startswith("N="):
            i += 1
            continue

        n = int(line.split("=")[1])

        i += 1
        if i >= len(lines) or lines[i].strip() != "X:":
            raise ValueError(f"Expected 'X:' in {path} at line {i+1}")

        i += 1
        atom_vals = [int(x) for x in lines[i].strip().split()]
        if len(atom_vals) != n:
            raise ValueError(f"Atom count mismatch in {path}: expected {n}, got {len(atom_vals)}")

        i += 1
        if i >= len(lines) or lines[i].strip() != "E:":
            raise ValueError(f"Expected 'E:' in {path} at line {i+1}")

        edge_rows = []
        for _ in range(n):
            i += 1
            row = [int(x) for x in lines[i].strip().split()]
            if len(row) != n:
                raise ValueError(f"Edge row length mismatch in {path}: expected {n}, got {len(row)}")
            edge_rows.append(row)

        atom_types = torch.tensor(atom_vals, dtype=torch.long)
        edge_types = torch.tensor(edge_rows, dtype=torch.long)
        molecules.append((atom_types, edge_types))

        i += 1

    return molecules


def find_sample_files(folder):
    pattern = os.path.join(folder, "**", "generated_samples*.txt")
    files = sorted(glob.glob(pattern, recursive=True))

    resolved = []
    broken_symlinks = []

    for f in files:
        if os.path.exists(f):
            resolved.append(f)
        elif os.path.islink(f):
            broken_symlinks.append(f)

    if resolved:
        return resolved

    # Fallback: try to resolve broken symlinks via run folders
    if broken_symlinks:
        print(f"Warning: Found {len(broken_symlinks)} broken symlink(s), attempting to resolve via run folders...")

        wandb_dir = os.path.join(folder, "wandb")
        if not os.path.isdir(wandb_dir):
            wandb_dir = None
            for root, dirs, _ in os.walk(folder):
                if "wandb" in dirs:
                    wandb_dir = os.path.join(root, "wandb")
                    break

        if wandb_dir:
            run_folders = sorted(glob.glob(os.path.join(wandb_dir, "run-*")), reverse=True)
            for run_folder in run_folders:
                candidate_files = sorted(glob.glob(os.path.join(run_folder, "**", "generated_samples*.txt"), recursive=True))
                real_files = [f for f in candidate_files if os.path.exists(f)]
                if real_files:
                    print(f"Resolved to run folder: {run_folder}")
                    return real_files

    raise FileNotFoundError(
        f"No valid generated_samples*.txt found in {folder}. "
        f"Found {len(broken_symlinks)} broken symlink(s) and could not resolve to a real run folder."
    )


def max_allowed_valency(atom_name: str) -> int:
    allowed = allowed_bonds.get(atom_name, 0)
    if isinstance(allowed, int):
        return allowed
    if isinstance(allowed, list):
        return max(allowed) if allowed else 0
    if isinstance(allowed, dict):
        vals = []
        for v in allowed.values():
            if isinstance(v, list):
                vals.extend(v)
            else:
                vals.append(v)
        return max(vals) if vals else 0
    return 0


def compute_valencies(edge_types: torch.Tensor) -> torch.Tensor:
    bond_vals = edge_types.clone().float()
    bond_vals[bond_vals == 4] = 1.5
    return bond_vals.sum(dim=1)


def build_rdkit_mol(atom_types: torch.Tensor, edge_types: torch.Tensor, atom_decoder):
    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    mol = Chem.RWMol()

    for atom_idx in atom_types.tolist():
        atom_symbol = atom_decoder[int(atom_idx)]
        mol.AddAtom(Chem.Atom(atom_symbol))

    n = atom_types.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            b = int(edge_types[i, j].item())
            if b > 0:
                mol.AddBond(i, j, bond_map[b])

    mol = mol.GetMol()

    try:
        Chem.SanitizeMol(mol)
        return mol, None
    except Exception as e:
        msg = str(e).lower()
        if "explicit valence" in msg:
            return None, "AtomValence"
        if "kekulize" in msg:
            return None, "Kekulize"
        return None, "Other"


def analyze_folder(folder: str, atom_decoder, max_molecules=None):
    sample_files = find_sample_files(folder)
    if not sample_files:
        raise FileNotFoundError(f"No generated_samples*.txt files found in {folder}")

    total_molecules = 0
    atom_total = 0
    violating_atom_total = 0

    violating_atoms_per_mol = []
    max_excess_per_mol = []
    sum_excess_per_mol = []

    violation_by_atom = Counter()
    total_by_atom = Counter()
    rdkit_error_counts = Counter()

    for sf in sample_files:
        mol_graphs = parse_generated_samples_file(sf)

        for atom_types, edge_types in mol_graphs:
            if max_molecules is not None and total_molecules >= max_molecules:
                break

            total_molecules += 1
            n_atoms = atom_types.shape[0]

            valencies = compute_valencies(edge_types)

            violating_count = 0
            max_excess = 0.0
            sum_excess = 0.0

            for idx in range(n_atoms):
                atom_name = atom_decoder[int(atom_types[idx].item())]
                total_by_atom[atom_name] += 1
                atom_total += 1

                allowed = max_allowed_valency(atom_name)
                excess = max(0.0, float(valencies[idx].item()) - float(allowed))

                if excess > 0:
                    violating_count += 1
                    violating_atom_total += 1
                    max_excess = max(max_excess, excess)
                    sum_excess += excess
                    violation_by_atom[atom_name] += 1

            violating_atoms_per_mol.append(violating_count)
            max_excess_per_mol.append(max_excess)
            sum_excess_per_mol.append(sum_excess)

            _, err = build_rdkit_mol(atom_types, edge_types, atom_decoder)
            if err is None:
                rdkit_error_counts["NoError"] += 1
            else:
                rdkit_error_counts[err] += 1

        if max_molecules is not None and total_molecules >= max_molecules:
            break

    molecules_with_any_violation = sum(v > 0 for v in violating_atoms_per_mol)
    molecules_with_2plus_violations = sum(v >= 2 for v in violating_atoms_per_mol)
    molecules_fully_valency_valid = total_molecules - molecules_with_any_violation

    results = {
        "folder": folder,
        "total_molecules": total_molecules,
        "molecules_fully_valency_valid": molecules_fully_valency_valid,
        "molecules_fully_valency_valid_rate": molecules_fully_valency_valid / total_molecules if total_molecules > 0 else 0.0,
        "molecules_with_any_violation": molecules_with_any_violation,
        "molecules_with_any_violation_rate": molecules_with_any_violation / total_molecules if total_molecules > 0 else 0.0,
        "molecules_with_2plus_violations": molecules_with_2plus_violations,
        "molecules_with_2plus_violations_rate": molecules_with_2plus_violations / total_molecules if total_molecules > 0 else 0.0,
        "violating_atom_total": violating_atom_total,
        "atom_total": atom_total,
        "atom_level_valency_valid_rate": 1.0 - (violating_atom_total / atom_total if atom_total > 0 else 0.0),
        "avg_violating_atoms_per_molecule": sum(violating_atoms_per_mol) / total_molecules if total_molecules > 0 else 0.0,
        "avg_max_excess_valency_per_molecule": sum(max_excess_per_mol) / total_molecules if total_molecules > 0 else 0.0,
        "avg_total_excess_valency_per_molecule": sum(sum_excess_per_mol) / total_molecules if total_molecules > 0 else 0.0,
        "rdkit_error_counts": dict(rdkit_error_counts),
        "violation_by_atom": dict(violation_by_atom),
        "total_by_atom": dict(total_by_atom),
    }

    per_atom_violation_rate = {}
    for atom_name, total_count in total_by_atom.items():
        per_atom_violation_rate[atom_name] = violation_by_atom[atom_name] / total_count if total_count > 0 else 0.0
    results["per_atom_violation_rate"] = per_atom_violation_rate

    return results


def print_results(res):
    print("\n=== Valency Diagnostic Report ===")
    print(f"folder: {res['folder']}")
    print(f"total_molecules: {res['total_molecules']}")
    print(f"molecules_fully_valency_valid: {res['molecules_fully_valency_valid']}")
    print(f"molecules_fully_valency_valid_rate: {res['molecules_fully_valency_valid_rate']:.6f}")
    print(f"molecules_with_any_violation: {res['molecules_with_any_violation']}")
    print(f"molecules_with_any_violation_rate: {res['molecules_with_any_violation_rate']:.6f}")
    print(f"molecules_with_2plus_violations: {res['molecules_with_2plus_violations']}")
    print(f"molecules_with_2plus_violations_rate: {res['molecules_with_2plus_violations_rate']:.6f}")
    print(f"violating_atom_total: {res['violating_atom_total']}")
    print(f"atom_total: {res['atom_total']}")
    print(f"atom_level_valency_valid_rate: {res['atom_level_valency_valid_rate']:.6f}")
    print(f"avg_violating_atoms_per_molecule: {res['avg_violating_atoms_per_molecule']:.6f}")
    print(f"avg_max_excess_valency_per_molecule: {res['avg_max_excess_valency_per_molecule']:.6f}")
    print(f"avg_total_excess_valency_per_molecule: {res['avg_total_excess_valency_per_molecule']:.6f}")

    print("\nRDKit error counts:")
    for k, v in sorted(res["rdkit_error_counts"].items()):
        print(f"  {k}: {v}")

    print("\nPer-atom violation rates:")
    for atom_name in sorted(res["per_atom_violation_rate"].keys()):
        rate = res["per_atom_violation_rate"][atom_name]
        viol = res["violation_by_atom"].get(atom_name, 0)
        total = res["total_by_atom"].get(atom_name, 0)
        print(f"  {atom_name}: {viol}/{total} = {rate:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to generated folder")
    parser.add_argument("--dataset", default="qm9", choices=["qm9", "moses"])
    parser.add_argument("--max_molecules", type=int, default=None,
                        help="Maximum number of molecules to analyze")
    args = parser.parse_args()
    
    if "qm9" in args.folder:
        atom_decoder = ATOM_DECODER_QM9
    elif "moses" in args.folder:
        atom_decoder = ATOM_DECODER_MOSES
    res = analyze_folder(args.folder, atom_decoder, max_molecules=args.max_molecules)
    print_results(res)


if __name__ == "__main__":
    main()