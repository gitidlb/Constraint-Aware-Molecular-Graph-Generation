import os
import glob
import random
import numpy as np

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ATOM_DECODER = ["C", "N", "O", "F"]


def parse_generated_samples_file(path):
    molecules = []
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        if not lines[i].startswith("N="):
            i += 1
            continue

        n = int(lines[i].split("=")[1])

        i += 1
        if lines[i] != "X:":
            raise ValueError(f"Expected X: in {path}")

        i += 1
        atom_vals = list(map(int, lines[i].split()))

        i += 1
        if lines[i] != "E:":
            raise ValueError(f"Expected E: in {path}")

        i += 1
        edge_rows = []
        for _ in range(n):
            edge_rows.append(list(map(int, lines[i].split())))
            i += 1

        molecules.append((atom_vals, edge_rows))

    return molecules


def build_mol(atom_types, edge_types):
    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    mol = Chem.RWMol()

    for atom_idx in atom_types:
        mol.AddAtom(Chem.Atom(ATOM_DECODER[int(atom_idx)]))

    n = len(atom_types)
    for i in range(n):
        for j in range(i + 1, n):
            b = edge_types[i][j]
            if b > 0:
                mol.AddBond(i, j, bond_map[b])

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ✅ FIXED: limit number of molecules
def load_molecules(folder, max_molecules=2000):
    files = sorted(glob.glob(os.path.join(folder, "**", "generated_samples*.txt"), recursive=True))
    mols = []

    for f in files:
        data = parse_generated_samples_file(f)

        for atom_types, edge_types in data:
            if len(mols) >= max_molecules:
                return mols

            mol = build_mol(atom_types, edge_types)
            if mol is not None:
                mols.append(mol)

    return mols


def compute_uniqueness(mols):
    smiles = [Chem.MolToSmiles(m, canonical=True) for m in mols]

    if len(smiles) == 0:
        return 0.0, 0, 0

    unique = len(set(smiles))
    return unique / len(smiles), unique, len(smiles)


def compute_diversity(mols, max_pairs=10000):
    if len(mols) < 2:
        return 0.0, 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]

    pairs = [(i, j) for i in range(len(fps)) for j in range(i + 1, len(fps))]

    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    sims = []
    for i, j in pairs:
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        sims.append(sim)

    avg_sim = float(np.mean(sims))
    diversity = 1.0 - avg_sim

    return diversity, avg_sim


if __name__ == "__main__":
    folder = "/scratch/dye7jx/Constraint-Aware-Molecular-Graph-Generation/carbonyl_t06_p05_filtered"

    # ✅ limit to 2000 molecules
    mols = load_molecules(folder, max_molecules=2000)

    print("\n=== Uniqueness and Diversity Report ===")
    print(f"folder: {folder}")
    print(f"Total valid molecules used: {len(mols)}")

    uniq, n_unique, n_total = compute_uniqueness(mols)
    print("\n--- Uniqueness ---")
    print(f"Unique valid SMILES: {n_unique}/{n_total}")
    print(f"Uniqueness: {uniq:.6f}")

    div, avg_sim = compute_diversity(mols)
    print("\n--- Diversity ---")
    print(f"Average Tanimoto similarity: {avg_sim:.6f}")
    print(f"Fingerprint diversity: {div:.6f}")