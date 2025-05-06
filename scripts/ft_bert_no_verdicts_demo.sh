#!/bin/bash

python src/ft_bert_no_verdicts_demographics.py \
--use_authors='true' \
--use_demos='true' \
--demo_embedding_path='data/embeddings/identities_embeddings.pkl' \
--author_encoder='average' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=768 \
--model_name='sbert' \
--split_type='author' \
--situation='title' \
--authors_embedding_path='data/embeddings/user_post_embeddings_similar_5.pkl' \
--path_to_data='data/' \
--social_norm='true'
