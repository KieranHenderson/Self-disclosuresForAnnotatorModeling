
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
--learning_rate=3.15055525725062e-05 \
--dropout=0.17323706496191205 \
--weight_decay=0.0007378999159283696 \
--plot_title='Hyperparameter Test 10' \
