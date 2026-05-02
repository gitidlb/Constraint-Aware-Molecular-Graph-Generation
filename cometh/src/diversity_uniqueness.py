import os
import glob
import random
import numpy as np
import argparse
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ATOM_DECODER_QM9 = ["C", "N", "O", "F"]
ATOM_DECODER_MOSES = ["C", "N", "O", "F", "S", "Cl", "Br"]


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


def build_mol(folder, atom_types, edge_types):
    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    mol = Chem.RWMol()

    for atom_idx in atom_types:
        if "qm9" in folder:
            mol.AddAtom(Chem.Atom(ATOM_DECODER_QM9[int(atom_idx)]))
        elif "moses" in folder:
            mol.AddAtom(Chem.Atom(ATOM_DECODER_MOSES[int(atom_idx)]))

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


def find_sample_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "generated_samples*.txt"), recursive=True))
    
    resolved = []
    broken_symlinks = []
    
    for f in files:
        if os.path.exists(f):
            resolved.append(f)
        elif os.path.islink(f):
            broken_symlinks.append(f)
    
    if resolved:
        return resolved
    
    # Fallback: try to resolve broken symlinks by finding real run folders
    if broken_symlinks:
        print(f"Warning: Found {len(broken_symlinks)} broken symlink(s), attempting to resolve via run folders...")
        
        wandb_dir = os.path.join(folder, "wandb")
        if not os.path.isdir(wandb_dir):
            # walk up to find wandb dir
            wandb_dir = None
            for root, dirs, _ in os.walk(folder):
                if "wandb" in dirs:
                    wandb_dir = os.path.join(root, "wandb")
                    break
        
        if wandb_dir:
            run_folders = sorted(glob.glob(os.path.join(wandb_dir, "run-*")))
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


# Limit number of molecules
def load_molecules(folder, max_molecules=2000):
    files = find_sample_files(folder)
    mols = []

    for f in files:
        data = parse_generated_samples_file(f)

        for atom_types, edge_types in data:
            if len(mols) >= max_molecules:
                return mols

            mol = build_mol(folder, atom_types, edge_types)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to folder of generated sample files")
    args = parser.parse_args()

    folder = args.folder

    # Limit to 2000 molecules
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