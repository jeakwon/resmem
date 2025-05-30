#!/bin/bash -l
#SBATCH -o /ptmp/jekwo/2025/logs/resmem/%j.out
#SBATCH -e /ptmp/jekwo/2025/logs/resmem/%j.err
#SBATCH -J resmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=12:00:00

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate resmem

cd /ptmp/jekwo/2025/resmem/

python main_resmem.py