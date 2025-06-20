## This file is used to create random parameters for the model to test which values are most optimal for each hyperparameter.
## Hyperparameters are: learning rate, regularization strength (dropout), and weight decay.
## Generates a file with 20 sets of random parameters for the model to test, 10 on ada and 10 on nibi.
## All tests are run on cluster 4 because it is consistently the worst performing cluster.

import random
import math

for i in range(20):

    log_lr = random.uniform(math.log10(1e-5), math.log10(1e-2))
    lr = 10 ** log_lr

    log_dropout = random.uniform(math.log10(0.1), math.log10(0.4))
    dropout = 10 ** log_dropout

    log_decay = random.uniform(math.log10(1e-4), math.log10(1e-1)) # default pytorch setting is 1e-2
    decay = 10 ** log_decay

    with open(f"scripts/hyperparameter_tests/test_{i+1}.sh", "w") as f:
        f.write(f"""
python src/ft_bert_no_verdicts_topk.py \\
--use_authors='true' \\
--author_encoder='average' \\
--loss_type='focal' \\
--num_epochs=10 \\
--sbert_model='sentence-transformers/all-distilroberta-v1' \\
--bert_tok='sentence-transformers/all-distilroberta-v1' \\
--sbert_dim=768 \\
--user_dim=768 \\
--model_name='sbert' \\
--split_type='verdicts' \\
--situation='text' \\
--authors_embedding_path='data/final_embeddings/verdict_embeddings_cluster_4.pkl' \\
--path_to_data='data/' \\
--social_norm='true' \\
--learning_rate={lr} \\
--dropout={dropout} \\
--weight_decay={decay} \\
--plot_title='Hyperparameter Test {i+1}' \\
""")

    print(f"Generated test_{i+1}.sh with lr={lr}, dropout={dropout}, decay={decay}")