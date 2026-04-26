# Constraint-Aware Molecular Graph Generation via Projection-Based Diffusion

## How to run Cometh sampling

These instructions are for running Cometh on Rivanna.
Complete Cometh conda env setup in cometh folder.

### Setup
- Perform additional pip installs:
```
rdkit
hydra-core
imageio
pytorch_lightning
torch_geometric
```

- Perform additional conda installs:
```
graph-tool
graph-tool-base
```

Make a checkpoints folder in the cometh folder, and download the QM9, MOSES, and GuacaMol checkpoints from the Cometh GitHub repository.

### Additional setup
- Create a Wandb account.
Provide account to Sabrina to join constrainedGenAI team or create your own Wandb team.
- Perform pip install of the following packages:
```
pip install wandb weave
```
- Create a .env file. It is recommend to have a .env with team as username and personal API key.

Example of .env:
```
WANDB_USERNAME = {teamName}
WANDB_API_KEY = {apiKey}
```

### Run sampling
Example to run sampling:
```
python main.py +experiment=qm9_sampling.yaml encoding=rrwp general.test_only=/home/{computingID}/Constraint-Aware-Molecular-Graph-Generation/cometh/checkpoints/qm9.ckpt hydra.run.dir=/home/{computingID}/outputs
```
Sampling can be performed on any of the three dataset, but the MOSES and GuacaMol datasets need the following argument replacement for the sampling compared to QM9:
```
encoding=rrwp_moses
```

### Additional Note
Make sure python environment is 3.9 to ensure that graph-tools import works.
The likely solution is to ensure you are not in a stacked conda environment system, so perform the following until no new changes to the terminal are presented:
```
conda deactivate
```
Afterwards, you should be able to run the cometh conda environment and run the sampling. This solution is especially key if you notice that either of the following provides 3.11 and the cometh conda environment's bin folder has python 3.9.
```
which python
python --version
```

## Hard Constraint (Valency) – Projection-Based Methods

### Overview
We implement two **hard constraint methods** to enforce valency during diffusion sampling.  
These methods modify intermediate graphs during the reverse process and are **inspired by projection-based constraint handling**.

Both methods:
- Detect atoms that violate valency constraints
- Apply a repair step during sampling
- Continue the reverse diffusion process



## Method 1: Delete-Based Constraint

### Description
- Identifies atoms whose valency exceeds the allowed limit
- Removes offending bonds completely
- Applies repair during the final steps of reverse sampling

This is a **simple and aggressive strategy** that enforces constraints by removing invalid connections.

### How to Run

Modify:

```
cometh/src/diffusion_models.py
```

Change:

```python
from models.abstract_diffusion_model import AbstractDiffusionModel
```

to:

```python
from models.abstract_diffusion_model_delete import AbstractDiffusionModel
```

Run:

```bash
python main.py +experiment=qm9_sampling.yaml \
encoding=rrwp \
general.test_only=/home/{computingID}/Constraint-Aware-Molecular-Graph-Generation/cometh/checkpoints/qm9.ckpt \
hydra.run.dir=/home/{computingID}/outputs_delete
```

### Adjusting When the Constraint is Applied

The constraint is applied during the **last portion of the reverse diffusion steps**.

In `abstract_diffusion_model_delete.py`, locate:

```python
if t_int <= max(1, int(0.1 * self.T)):
    z_s = self.project_to_valency_constraint(z_s)
```

- `0.1 * self.T` means the constraint is applied in the **last 10% of sampling steps**

You can modify this value to control when the constraint is applied:

| Value | Effect |
|------|--------|
| `0.05 * self.T` | very late (minimal interference) |
| `0.1 * self.T`  | late (default) |
| `0.2 * self.T`  | earlier |
| `0.5 * self.T`  | much earlier (stronger constraint influence) |

Example (apply in last 20%):

```python
if t_int <= max(1, int(0.2 * self.T)):
```



## Method 2: Gradual Constraint (Bond Adjustment)

### Description
- Detects valency violations
- Reduces bond order step-by-step (e.g., double → single)
- Only removes bonds if necessary
- Applied during reverse sampling

This is a **softer and more controlled approach** that attempts to preserve molecular structure.

### How to Run

Modify:

```
cometh/src/diffusion_models.py
```

Change:

```python
from models.abstract_diffusion_model import AbstractDiffusionModel
```

to:

```python
from models.abstract_diffusion_model_gradual import AbstractDiffusionModel
```

Run:

```bash
python main.py +experiment=qm9_sampling.yaml \
encoding=rrwp \
general.test_only=/home/{computingID}/Constraint-Aware-Molecular-Graph-Generation/cometh/checkpoints/qm9.ckpt \
hydra.run.dir=/home/{computingID}/outputs_gradual
```



## Important Notes

- Only **one constraint method can be used at a time**  
  (controlled via the import in `diffusion_models.py`)

- No changes to `main.py` or training are required

- Use separate output directories to avoid overwriting results:
  - `outputs_delete`
  - `outputs_gradual`

## Soft Constraints - Reranking Molecules

## How to run
1. Run hard constraint sampling.
2. Run the soft constraint script.

You could run with the outputs from solely the sampling without the projection-based hard constraint on valency as well.

## Method 1: Rank by QED only
To run soft constraint script for this method, use the following:
```
python rerank_molecules.py --input /home/{computingID}/outputs_delete/generated_smiles.txt --output top_molecules.tsv --top_k 20
```
### Description
- Computes the Quantitative Estimate of Drug-likeness (QED) for each generated molecule.
- Ranks all valid molecules from highest to lowest QED.
- Returns the top-K most drug-like molecules.

QED is a composite score between 0 and 1 that combines multiple molecular properties (molecular weight (MW), lipophilicity (logP), hydrogen bond donors/acceptors, etc.) into a single drug-likeness estimate. This is the simplest and most general reranking strategy, useful as a baseline for evaluating the overall quality of generated molecules without imposing size or structure constraints.

## Method 2: Rank by proximity to a target Molecular Weight
```
python rerank_molecules.py --input /home/{computingID}/outputs_delete/generated_smiles.txt --output top_mw_target.tsv --top_k 20 --score mw --mw_target 350
```
### Description
- Computes the molecular weight of each generated molecule.
- Scores each molecule by how close its MW is to the specified target (--mw_target).
- Returns the top-K molecules nearest to the target MW.

This method is useful when the desired molecule must fall within a specific size range for the mass of the molecule, such as matching a known scaffold or satisfying fragment-based design criteria. It does not consider drug-likeness, so MW is a suitable soft constraint.

## Method 3: Rank by QED with Molecular Weight filter
```
python rerank_molecules.py --input /home/{computingID}/outputs_delete/generated_smiles.txt --output qed_mwfiltered.tsv --top_k 20 --min_mw 200 --max_mw  500
```
### Description
- Removes any molecules outside the specified MW window (--min_mw, --max_mw) as a hard post-hoc filter.
- Ranks the remaining molecules by QED.
- Returns the top-K most drug-like molecules within the MW range.

This method combines a hard MW boundary with soft QED ranking. It is useful when the target application has a firm size requirement but otherwise wants to maximise drug-likeness among the valid candidates.

## Method 4: Rank by composite score based on QED and Molecular Weight proximity
```
python rerank_molecules.py --input /home/{computingID}/outputs_delete/generated_smiles.txt --output top_composite.tsv --top_k 20 --score composite --mw_target 350 --min_mw 200 --max_mw  500
```
### Description
- Computes both QED and MW proximity to a target for each molecule.
- Combines them into a single score: 0.5 × QED + 0.5 × MW-proximity.
- MW-proximity is clipped to [0, 1] so molecules far from the target do not produce negative scores.
- Optionally applies a hard MW window before scoring (--min_mw, --max_mw).
- Returns the top-K molecules with the highest composite score.

This is the most balanced reranking strategy, rewarding molecules that are simultaneously drug-like and close to the desired size. It is recommended when neither QED nor MW alone is sufficient to characterise the target molecule, and a trade-off between the two properties is acceptable.