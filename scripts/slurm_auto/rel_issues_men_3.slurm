#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/rel_issues_men_3_slurm.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate
python src/ft_bert_no_verdicts_topk.py --use_authors='true' --author_encoder='average' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --user_dim=768 --model_name='sbert' --split_type='sit' --situation='text' --authors_embedding_path='data/final_embeddings/verdict_embeddings_cluster_4.pkl' --plot_title='Automatic Clustering - Rel. Issues w/Men_3' --path_to_data='data/' --social_norm='true' --log_file='rel_issues_men_3'
