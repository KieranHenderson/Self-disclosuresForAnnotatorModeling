#!/bin/bash

python src/diverse_sampling.py \
  --comment_embeddings_path="data/embeddings/precomputed_comment_embeddings_postlevel.pkl" \
  --post_embeddings_path="data/embeddings/precomputed_post_embeddings_postlevel.pkl" \
  --path_to_data="data/" \
  --output_dir="data/final_embeddings/" \
  --output_file_name="diverse_sampling_manual" \
  --persona_data_dir="data/persona_data" \
  --bert_model="sentence-transformers/all-distilroberta-v1" \
  --use_manual=True \
  --use_clusters=False


