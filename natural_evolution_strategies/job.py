"""
This program aims at launching jobs on the CentraleSupelec cluster
"""
import os

def makejob(model, nruns):
    return f"""#!/bin/bash 

    #SBATCH --job-name=emnist-{model}
    #SBATCH --nodes=1
    #SBATCH --partition=gpu_prod_night
    #SBATCH --time=1:00:00
    #SBATCH --output=logslurms/slurm-%A_%a.out
    #SBATCH --error=logslurms/slurm-%A_%a.err
    #SBATCH --array=0-{nruns}

    python3 bash_nes.py --model {model} --dataset_dir $TMPDIR train
    """