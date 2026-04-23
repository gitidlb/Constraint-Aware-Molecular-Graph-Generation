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
Provide account to Sabrina to join constrainedGenAI team or create your own Wandb team.
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

