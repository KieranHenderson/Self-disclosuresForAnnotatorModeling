#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=2:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/author_percentage.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate
python src/author_percentage.py \
    --comment_embeddings_path='data/embeddings/precomputed_comment_embeddings_postlevel.pkl' \
    --thresholds 5 10 15 20 25 30
