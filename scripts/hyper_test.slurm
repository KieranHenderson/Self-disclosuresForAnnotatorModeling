#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=ft_bert_%j.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate
chmod +x scripts/hyperparameter_tests/test_20.sh
./scripts/hyperparameter_tests/test_20.sh