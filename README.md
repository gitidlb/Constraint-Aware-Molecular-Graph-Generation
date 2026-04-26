# How to run Cometh sampling

These instructions are for running Cometh on Rivanna.
Complete Cometh conda env setup in cometh folder.

extra pip installs:
```
rdkit
hydra-core
imageio
pytorch_lightning
torch_geometric
```

extra conda installs:
```
graph-tool
graph-tool-base
```

Make a checkpoints folder in the cometh folder, and download the QM9, MOSES, and GuacaMol checkpoints from the Cometh GitHub repository.

Additional setup:
Create a Wandb account.
Provide account to Sabrina to join constrainedGenAI team or create your own Wandb team.
```
pip install wandb weave
```
Recommend a .env with team as username and personal API key.

Example of .env:
```
WANDB_USERNAME = {teamName}
WANDB_API_KEY = {apiKey}
```

Example to run sampling:
```
python main.py +experiment=qm9_sampling.yaml encoding=rrwp general.test_only=/home/{computingID}/Constraint-Aware-Molecular-Graph-Generation/cometh/checkpoints/qm9.ckpt hydra.run.dir=/home/{computingID}/outputs
```
Sampling can be performed on any of the three dataset, but the MOSES and GuacaMol datasets need the following argument replacement for the sampling compared to QM9:
```
encoding=rrwp_moses
```

Extra note: 
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

# Structural Constraint Sampling (COMETH)

## Overview

We implement carbonyl-based structural constraints during diffusion sampling.

The constraint enforces:

```
atom 0 = C
atom 1 = O
bond(0,1) = double bond
```

## Constraint Implementations

We use two versions of the diffusion model:

### 1. Hard Constraint
```
abstract_diffusion_model_carbonyl_hard.py
```
- Constraint applied at every selected step (always enforced)

### 2. Probabilistic Constraint
```
abstract_diffusion_model_carbonyl_soft.py
```
- Constraint applied with a probability at each step

---

## How to Switch Constraint Mode

Replace the default diffusion model with the desired constraint file:

```bash
cp abstract_diffusion_model_carbonyl_hard.py models/abstract_diffusion_model.py
```

or

```bash
cp abstract_diffusion_model_carbonyl_soft.py models/abstract_diffusion_model.py
```

---

## Constraint Settings

### Timing (when constraint is applied)

In both files:

```python
self.carbonyl_start_frac = getattr(cfg.model, "carbonyl_start_frac", 0.6)
```

We tested the following values:

```
0.9 → early
0.6 → mid
0.3 → late
```

---

### Probability (only for probabilistic constraint)

```python
self.carbonyl_apply_prob = getattr(cfg.model, "carbonyl_apply_prob", 0.5)
```

We tested the following values:

```
0.25, 0.5, 0.75
```

---

## Run Sampling

### Base command

```bash
python main.py +experiment=qm9_sampling.yaml \
encoding=rrwp \
general.test_only="../checkpoints/qm9.ckpt" \
hydra.run.dir="/path/to/output_folder" \
general.final_model_samples_to_generate=2000
```

- `hydra.run.dir` → output folder name  
- `general.final_model_samples_to_generate` → number of samples to generate  


---

## Evaluation

### Structural Evaluation

File:
```
evaluate_structural_constraints.py
```

Reports:
- RDKit validity
- Connectivity
- Atom count
- Ring statistics
- Carbonyl presence
- Substructures

Run:

```bash
python evaluate_structural_constraints.py \
  --folder /path/to/generated_samples \
  --max_molecules 2000
```

---

### Valency Evaluation

File:
```
evaluate_valency_metrics.py
```

Reports:
- Valency validity
- Violation rate
- Violation magnitude
- Per-atom violations

Run:

```bash
python evaluate_valency_metrics.py \
  --folder /path/to/generated_samples \
  --max_molecules 2000
```

---

## Provided Samples

We include generated outputs for:

```
baseline_2000
carbonyl_t03_*
carbonyl_t06_*
carbonyl_t09_*
```

These can be directly evaluated using the commands above.

