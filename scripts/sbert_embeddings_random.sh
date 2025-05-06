#!/bin/bash
#

## Random 5
python src/users_sbert_embeddings_random.py \
--path_to_data='data/' \
--dirname='data/demographics/' \
--output_dir='data/embeddings/' \
--embed_sentences='false' \
--posts_per_author=0 \
--output_file_name='identities_embeddings' \
