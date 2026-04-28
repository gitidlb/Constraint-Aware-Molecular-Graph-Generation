# Constraint-Aware Molecular Graph Generation via Projection-Based Diffusion

## How to run COMETH sampling

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

Make a checkpoints folder in the cometh folder, and download the QM9 checkpoint from the COMETH GitHub repository.

### Additional setup
There are two ways.

#### Method 1: Wandb
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

#### Method 2: No Wandb
Go to the qm9_sampling.yaml file in cometh/configs/experiment, and then change
```
wandb: 'online'
```
to
```
wandb: 'disabled'
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

## How to Switch Constraint Mode

Replace the default diffusion model with the desired constraint file:

```bash
cp abstract_diffusion_model_carbonyl_hard.py models/abstract_diffusion_model.py
```

or

```bash
cp abstract_diffusion_model_carbonyl_soft.py models/abstract_diffusion_model.py
```

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

### Probability (only for probabilistic constraint)

```python
self.carbonyl_apply_prob = getattr(cfg.model, "carbonyl_apply_prob", 0.5)
```

We tested the following values:

```
0.25, 0.5, 0.75
```

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

## Post-hoc Filtering

We also apply a post-hoc filtering step to retain only high-quality molecules.

This filtering keeps molecules that are:
- RDKit valid  
- Connected  
- Contain a carbonyl group (C=O)  

File used:
```
posthoc_filter_carbonyl.py
```

### Run filtering

```bash
python posthoc_filter_carbonyl.py \
  --input_folder /path/to/generated_samples \
  --output_folder /path/to/output_filtered \
  --max_molecules 2000
```

### Output

The script saves:

```
filtered_valid_carbonyl_samples.txt
```

Rename it for evaluation:

```bash
mv filtered_valid_carbonyl_samples.txt generated_samples1.txt
```

### Note

The filtering script uses a simple C=O detector, while the evaluation uses a stricter SMARTS-based definition.  
Therefore, the final carbonyl percentage may be slightly below 100%.


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

## Method 2: Rank by QED with Molecular Weight filter
```
python rerank_molecules.py --input /home/{computingID}/outputs_delete/generated_smiles.txt --output qed_mwfiltered.tsv --top_k 20 --min_mw 200 --max_mw  500
```
### Description
- Removes any molecules outside the specified MW window (--min_mw, --max_mw) as a hard post-hoc filter.
- Ranks the remaining molecules by QED.
- Returns the top-K most drug-like molecules within the MW range.

This method combines a hard MW boundary with soft QED ranking. It is useful when the target application has a firm size requirement but otherwise wants to maximise drug-likeness among the valid candidates.
