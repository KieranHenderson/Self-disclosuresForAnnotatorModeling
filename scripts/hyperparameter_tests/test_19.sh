
python src/ft_bert_no_verdicts_topk.py \
--use_authors='true' \
--author_encoder='average' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=768 \
--model_name='sbert' \
--split_type='verdicts' \
--situation='text' \
--authors_embedding_path='data/final_embeddings/verdict_embeddings_cluster_4.pkl' \
--path_to_data='data/' \
--social_norm='true' \
--learning_rate=0.0029893692599463396 \
--dropout=0.2875903836491357 \
--weight_decay=0.014348308268503203 \
--plot_title='Hyperparameter Test 19' \
