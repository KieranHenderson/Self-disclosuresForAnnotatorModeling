#!/bin/bash
#

python src/ft_bert_with_users.py \
--use_authors='false' \
--author_encoder='none' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=-1 \
--model_name='judge_bert' \
--split_type='sit' \
--path_to_data='data/' \
--authors_embedding_path='data/embeddings/user_embeddings.pkl' \

