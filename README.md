# How to run Cometh sampling

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

Make a checkpoints folder in the cometh folder, and download the QM9 checkpoints from the Cometh GitHub repository.

Additional setup:
Create a Wandb account.
Provide account to Sabrina to join constrainedGenAI team.
```
pip install wandb weave
```
Recommend a .env with team as username and personal API key.

Example to run sampling:
```
python main.py +experiment=qm9_sampling.yaml encoding=rrwp general.test_only=/home/{computingID}/Constraint-Aware-Molecular-Graph-Generation/cometh/checkpoints/qm9.ckpt hydra.run.dir=/home/{computingID}/outputs
```

Example of .env:
WANDB_USERNAME = {teamName}
WANDB_API_KEY = {apiKey}

Extra note: 
Make sure python environment is 3.9 to ensure that graph-tools import works.
Can do the following as a potential solution:
```
export PATH=/home/{computingID}/.conda/envs/cometh/bin:$PATH
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
