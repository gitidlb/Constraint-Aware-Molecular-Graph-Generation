Complete Cometh conda env setup.

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
