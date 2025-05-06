#!/bin/bash
#

python src/users_sbert_embeddings_random.py \
--path_to_data='data/' \
--dirname='data/amit_filtered_history/' \
--output_dir='data/embeddings/' \
--embed_sentences='true' \