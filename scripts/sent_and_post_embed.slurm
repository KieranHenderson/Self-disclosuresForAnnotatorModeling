#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=ft_bert_%j.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

## Sentence based tests

## Random 5
source venv/bin/activate
python src/users_sbert_embeddings_random.py \
--path_to_data='data/' \
--dirname='data/amit_filtered_history/' \
--output_dir='data/embeddings/' \
--embed_sentences='true' \
--max_posts_per_author=5 \
--filter_cond=5 \
--output_file_name='user_sent_embeddings_5' \

## Random 10
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='true' \
##--max_posts_per_author=10 \
##--filter_cond=10 \
##--output_file_name='user_sent_embeddings_10' \

## Random 15
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='true' \
##--max_posts_per_author=15 \
##--filter_cond=15 \
##--output_file_name='user_sent_embeddings_15' \

## Random 20
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='true' \
##--max_posts_per_author=20 \
##--filter_cond=20 \
##--output_file_name='user_sent_embeddings_20' \

## Random 25
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='true' \
##--max_posts_per_author=25 \
##--filter_cond=25 \
##--output_file_name='user_sent_embeddings_25' \

## Random 30
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='true' \
##--max_posts_per_author=30 \
##--filter_cond=30 \
##--output_file_name='user_sent_embeddings_30' \




## Post based tests
## Random 5
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='false' \
##--max_posts_per_author=5 \
##--filter_cond=5 \
##--output_file_name='user_sent_embeddings_5' \

## Random 10
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='false' \
##--max_posts_per_author=10 \
##--filter_cond=10 \
##--output_file_name='user_sent_embeddings_10' \

## Random 15
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='false' \
##--max_posts_per_author=15 \
##--filter_cond=15 \
##--output_file_name='user_sent_embeddings_15' \

## Random 20
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='false' \
##--max_posts_per_author=20 \
##--filter_cond=20 \
##--output_file_name='user_sent_embeddings_20' \

## Random 25
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='false' \
##--max_posts_per_author=25 \
##--filter_cond=25 \
##--output_file_name='user_sent_embeddings_25' \

## Random 30
##source venv/bin/activate
##python src/users_sbert_embeddings_random.py \
##--path_to_data='data/' \
##--dirname='data/amit_filtered_history/' \
##--output_dir='data/embeddings/' \
##--embed_sentences='false' \
##--max_posts_per_author=30 \
##--filter_cond=30 \
##--output_file_name='user_sent_embeddings_30' \