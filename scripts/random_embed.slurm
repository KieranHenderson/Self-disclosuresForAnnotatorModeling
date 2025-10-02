#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/random_sent_5_embed_new.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate

## Random 5
python src/users_sbert_embeddings_random.py \
--path_to_data='data/' \
--dirname='data/amit_filtered_history/' \
--output_dir='data/embeddings/' \
--embed_sentences='true' \
--posts_per_author=5 \
--output_file_name='random_sent_embeddings_5_new' \
--random_sampling='true' \
--all_posts='false' \