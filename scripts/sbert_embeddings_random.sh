#!/bin/bash
#

## Random 5
python src/users_sbert_embeddings_random.py \
--path_to_data='data/' \
--dirname='data/amit_filtered_history/' \
--output_dir='data/embeddings/' \
--embed_sentences='false' \
--posts_per_author=15 \
--output_file_name='user_sent_embeddings_15' \
--random_sampling='true' \