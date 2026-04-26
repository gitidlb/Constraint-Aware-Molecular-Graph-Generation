import os
import glob
import argparse
from rdkit import Chem
from rdkit import RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

ATOM_DECODER = ["C", "N", "O", "F"]


def find_sample_files(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "generated_samples*.txt"), recursive=True))


def parse_generated_samples_file(path):
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
        if lines[i].strip() != "X:":
            raise ValueError(f"Expected X: in {path}")

        i += 1
        atom_vals = [int(x) for x in lines[i].strip().split()]

        i += 1
        if lines[i].strip() != "E:":
            raise ValueError(f"Expected E: in {path}")

        edge_rows = []
        for _ in range(n):
            i += 1
            edge_rows.append([int(x) for x in lines[i].strip().split()])

        atom_types = torch.tensor(atom_vals, dtype=torch.long)
        edge_types = torch.tensor(edge_rows, dtype=torch.long)
        molecules.append((atom_types, edge_types))

        i += 1

    return molecules


def build_rdkit_mol(atom_types, edge_types):
    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    mol = Chem.RWMol()

    for atom_idx in atom_types.tolist():
        mol.AddAtom(Chem.Atom(ATOM_DECODER[int(atom_idx)]))

    n = atom_types.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            b = int(edge_types[i, j].item())
            if b > 0:
                mol.AddBond(i, j, bond_map[b])

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def has_carbonyl(mol):
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if bond.GetBondType() == Chem.BondType.DOUBLE:
            symbols = {a1.GetSymbol(), a2.GetSymbol()}
            if symbols == {"C", "O"}:
                return True

    return False


def is_connected(mol):
    frags = Chem.GetMolFrags(mol)
    return len(frags) == 1


def write_samples(molecules, output_file):
    with open(output_file, "w") as f:
        for atom_types, edge_types in molecules:
            n = atom_types.shape[0]

            f.write(f"N={n}\n")
            f.write("X: \n")
            for x in atom_types.tolist():
                f.write(f"{int(x)} ")
            f.write("\n")

            f.write("E: \n")
            for row in edge_types.tolist():
                for e in row:
                    f.write(f"{int(e)} ")
                f.write("\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--max_molecules", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    sample_files = find_sample_files(args.input_folder)

    if len(sample_files) == 0:
        raise FileNotFoundError(f"No generated_samples*.txt found in {args.input_folder}")

    total = 0
    rdkit_valid = 0
    carbonyl_valid = 0
    connected_valid = 0
    kept = []

    for sample_file in sample_files:
        molecules = parse_generated_samples_file(sample_file)

        for atom_types, edge_types in molecules:
            if args.max_molecules is not None and total >= args.max_molecules:
                break

            total += 1

            mol = build_rdkit_mol(atom_types, edge_types)

            if mol is None:
                continue

            rdkit_valid += 1

            if not has_carbonyl(mol):
                continue

            carbonyl_valid += 1

            if not is_connected(mol):
                continue

            connected_valid += 1

            kept.append((atom_types, edge_types))

        if args.max_molecules is not None and total >= args.max_molecules:
            break

    output_file = os.path.join(args.output_folder, "filtered_valid_carbonyl_samples.txt")
    write_samples(kept, output_file)

    print("\n=== Post-hoc Filtering Report ===")
    print(f"input_folder: {args.input_folder}")
    print(f"output_folder: {args.output_folder}")
    print(f"total_molecules_checked: {total}")
    print(f"rdkit_valid: {rdkit_valid}")
    print(f"rdkit_valid_rate: {rdkit_valid / total if total else 0:.6f}")
    print(f"rdkit_valid_and_carbonyl: {carbonyl_valid}")
    print(f"rdkit_valid_and_carbonyl_rate: {carbonyl_valid / total if total else 0:.6f}")
    print(f"kept_rdkit_valid_carbonyl_connected: {len(kept)}")
    print(f"kept_rate: {len(kept) / total if total else 0:.6f}")
    print(f"saved_to: {output_file}")


if __name__ == "__main__":
    main()