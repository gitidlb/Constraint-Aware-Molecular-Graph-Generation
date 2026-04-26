import os
import glob
import argparse
from collections import Counter

import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ATOM_DECODER_QM9 = ["C", "N", "O", "F"]


SMARTS_PATTERNS = {
    "carbonyl_C_eq_O": "[CX3]=[OX1]",
    "C_O_single": "[C]-[O]",
    "C_N_single": "[C]-[N]",
    "C_C_single": "[C]-[C]",
    "N_O_bond": "[N]-[O]",
    "has_fluorine": "[F]",
    "nitrile_C_triple_N": "[C]#[N]",
}


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
        assert lines[i].strip() == "X:"
        i += 1
        atom_types = torch.tensor([int(x) for x in lines[i].strip().split()], dtype=torch.long)

        i += 1
        assert lines[i].strip() == "E:"
        edge_rows = []
        for _ in range(n):
            i += 1
            edge_rows.append([int(x) for x in lines[i].strip().split()])

        edge_types = torch.tensor(edge_rows, dtype=torch.long)
        molecules.append((atom_types, edge_types))

        i += 1

    return molecules


def find_sample_files(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "generated_samples*.txt"), recursive=True))


def build_rdkit_mol(atom_types, edge_types, atom_decoder):
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
        return None, str(e)


def graph_ring_count(edge_types):
    """
    Cyclomatic number = E - V + C.
    For connected molecules, number of independent cycles = edges - nodes + 1.
    """
    n = edge_types.shape[0]
    num_edges = int((edge_types > 0).sum().item() // 2)
    num_components = count_components(edge_types)
    return max(0, num_edges - n + num_components)


def count_components(edge_types):
    n = edge_types.shape[0]
    visited = [False] * n

    def dfs(start):
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            neighbors = (edge_types[u] > 0).nonzero(as_tuple=False).flatten().tolist()
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

    comps = 0
    for i in range(n):
        if not visited[i]:
            comps += 1
            dfs(i)

    return comps


def analyze(folder, max_molecules=None):
    files = find_sample_files(folder)
    if not files:
        raise FileNotFoundError(f"No generated_samples*.txt found in {folder}")

    smarts = {name: Chem.MolFromSmarts(pattern) for name, pattern in SMARTS_PATTERNS.items()}

    total = 0
    valid_rdkit = 0
    rdkit_errors = 0

    substructure_counts = Counter()
    ring_count_hist = Counter()
    rdkit_ring_count_hist = Counter()
    max_ring_size_hist = Counter()
    component_hist = Counter()

    atom_count_hist = Counter()

    for path in files:
        molecules = parse_generated_samples_file(path)

        for atom_types, edge_types in molecules:
            if max_molecules is not None and total >= max_molecules:
                break

            total += 1
            atom_count_hist[int(atom_types.shape[0])] += 1

            n_components = count_components(edge_types)
            component_hist[n_components] += 1

            approx_rings = graph_ring_count(edge_types)
            ring_count_hist[approx_rings] += 1

            mol, err = build_rdkit_mol(atom_types, edge_types, ATOM_DECODER_QM9)

            if mol is None:
                rdkit_errors += 1
                continue

            valid_rdkit += 1

            for name, patt in smarts.items():
                if mol.HasSubstructMatch(patt):
                    substructure_counts[name] += 1

            ring_info = mol.GetRingInfo()
            rdkit_num_rings = ring_info.NumRings()
            rdkit_ring_count_hist[rdkit_num_rings] += 1

            ring_sizes = [len(r) for r in ring_info.AtomRings()]
            if ring_sizes:
                max_ring_size_hist[max(ring_sizes)] += 1
            else:
                max_ring_size_hist[0] += 1

        if max_molecules is not None and total >= max_molecules:
            break

    print("\n=== Structural Constraint Diagnostic Report ===")
    print(f"folder: {folder}")
    print(f"total_molecules: {total}")
    print(f"rdkit_valid_molecules: {valid_rdkit}")
    print(f"rdkit_valid_rate: {valid_rdkit / total if total else 0:.6f}")
    print(f"rdkit_errors: {rdkit_errors}")

    print("\n--- Atom Count ---")
    for k, v in atom_count_hist.most_common():
        print(f"n_atoms={k}: {v} ({v/total:.3f})")

    print("\n--- Connectivity ---")
    for k, v in sorted(component_hist.items()):
        print(f"components={k}: {v} ({v/total:.3f})")

    print("\n--- Approximate Graph Ring Count ---")
    for k, v in sorted(ring_count_hist.items()):
        print(f"rings={k}: {v} ({v/total:.3f})")

    print("\n--- RDKit Ring Count (valid RDKit molecules only) ---")
    denom = max(valid_rdkit, 1)
    for k, v in sorted(rdkit_ring_count_hist.items()):
        print(f"rings={k}: {v} ({v/denom:.3f})")

    print("\n--- Max Ring Size (valid RDKit molecules only) ---")
    for k, v in sorted(max_ring_size_hist.items()):
        print(f"max_ring_size={k}: {v} ({v/denom:.3f})")

    print("\n--- Substructure Presence (valid RDKit molecules only) ---")
    for name in SMARTS_PATTERNS:
        count = substructure_counts[name]
        print(f"{name}: {count}/{valid_rdkit} = {count/denom:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--max_molecules", type=int, default=None)
    args = parser.parse_args()

    analyze(args.folder, args.max_molecules)


if __name__ == "__main__":
    main()