#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=1:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=8   # number of CPU cores per task
#SBATCH --gres=gpu:a6000:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=12G   # memory
#SBATCH -J "qm9"   # job name
#SBATCH --mail-user=ryp8rx@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=genai-spring26   # allocation name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Initialize conda for non-interactive shell
source /sfs/weka/applications/202506_build/software/standard/core/miniforge/24.3.0-py3.11/etc/profile.d/conda.sh

conda deactivate; conda deactivate; conda activate cometh; cd Constraint-Aware-Molecular-Graph-Generation/cometh/src; clear; python main.py +experiment=qm9_sampling.yaml encoding=rrwp general.test_only=/home/ryp8rx/Constraint-Aware-Molecular-Graph-Generation/cometh/checkpoints/qm9.ckpt hydra.run.dir=/home/ryp8rx/outputs_qm9_hard_06_05 general.final_model_samples_to_generate=2000 model.carbonyl_start_frac=0.6 model.carbonyl_apply_prob=0.5
