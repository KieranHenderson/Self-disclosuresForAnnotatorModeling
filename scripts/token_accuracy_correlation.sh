python token_accuracy_correlation.py \
  --results_json_path="path/to/results.json" \
  --comment_embeddings_path="data/embeddings/precomputed_comment_embeddings_postlevel.pkl" \
  --post_embeddings_path="data/embeddings/precomputed_post_embeddings_postlevel.pkl" \
  --path_to_data="data/" \
  --top_k=5 \
  --bert_model="sentence-transformers/all-distilroberta-v1"
