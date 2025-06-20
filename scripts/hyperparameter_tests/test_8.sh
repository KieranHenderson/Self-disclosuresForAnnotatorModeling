
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
--learning_rate=1.4309715323682422e-05 \
--dropout=0.21852028072078888 \
--weight_decay=0.004476825475689488 \
--plot_title='Hyperparameter Test 8' \
