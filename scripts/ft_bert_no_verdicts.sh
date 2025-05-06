#!/bin/bash

python src/ft_bert_no_verdicts.py \
--use_authors='true' \
--author_encoder='average' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=384 \
--model_name='sbert' \
--split_type='author' \
--situation='title' \
--authors_embedding_path='data/embeddings/user_embeddings.pkl' \
--path_to_data='data/'
