#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=10:00:00
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
python src/diverse_sampling.py \
  --comment_embeddings_path="data/embeddings/precomputed_comment_embeddings_postlevel.pkl" \
  --post_embeddings_path="data/embeddings/precomputed_post_embeddings_postlevel.pkl" \
  --path_to_data="data/" \
  --output_dir="data/final_embeddings/" \
  --output_file_name="diverse_sampling_manual" \
  --persona_data_dir="data/persona_data" \
  --bert_model="sentence-transformers/all-distilroberta-v1" \
  --use_manual=False \
  --use_clusters=True
