# This script should be run once with --embed_sentences=false and once with --embed_sentences=true
# Adjust --output_file_name accordingly

# python ./src/precompute_comment_embeddings.py \
# --path_to_data='data/' \
# --dirname='data/amit_filtered_history/' \
# --output_dir='data/embeddings/' \
# --embed_sentences=false \
# --output_file_name='precomputed_comment_embeddings_postlevel' \

python ./src/precompute_comment_embeddings.py \
--path_to_data='data/' \
--dirname='data/amit_filtered_history/' \
--output_dir='data/embeddings/' \
--embed_sentences=true \
--output_file_name='precomputed_comment_embeddings_sentlevel' \
