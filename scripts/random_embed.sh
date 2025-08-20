#!/bin/bash
#

## Random 5
python src/users_sbert_embeddings_random.py \
--path_to_data='data/' \
--dirname='data/amit_filtered_history/' \
--output_dir='data/embeddings/' \
--embed_sentences='false' \
--posts_per_author=5 \
--output_file_name='test_random_embeddings_5' \
--random_sampling='true' \