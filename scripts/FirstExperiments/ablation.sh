#!/bin/sh
#SBATCH --job-name=VLG-C__
#SBATCH --time=0-14:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/soldanm/projects/cvpr-2021/VLG-Net_best_model/
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-23
#SBATCH --constraint=v100
##SBATCH --account=conf-cvpr-2021.11.23-ghanembs

module load gcc

echo "######################### SLURM JOB ########################"
echo HOST NAME
echo `hostname`
echo "############################################################"

environment=vlg
conda_root=$HOME/anaconda3

source $conda_root/etc/profile.d/conda.sh
conda activate $environment

set -ex

# ------------------------ need not change -----------------------------------

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID))"p scripts/ablation_joblist.txt)
python3 Certify.py  $LINE 
