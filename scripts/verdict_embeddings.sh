python src/verdict_embeddings.py \
--comment_embeddings_path='data/embeddings/precomputed_comment_embeddings_postlevel.pkl' \
--post_embeddings_path='data/embeddings/precomputed_post_embeddings_postlevel.pkl' \
--path_to_data='data/' \
--output_dir='data/final_embeddings/' \
--top_k=5 \
--embed_sentences=false \
--output_file_name='verdict_embeddings_postlevel_5' \
--json_embeddings_path='data/manual/attitudes.json' \