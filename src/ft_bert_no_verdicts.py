import glob
import sys
print(sys.executable)
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast, DataCollatorWithPadding, get_scheduler
from datasets import DatasetDict, Dataset, Features, Value
from torch_geometric.data import Data


from dataset import SocialNormDataset
from utils.read_files import *
from utils.utils import *
from utils.loss_functions import *
from utils.train_utils import *
from models import SentBertClassifier
from constants import *
from tqdm.auto import tqdm
from argparse import ArgumentParser
import logging

TIMESTAMP = get_current_timestamp()

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
    
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         #logging.FileHandler(os.path.join(os.path.expanduser("~/PycharmProjects/perspectivism-personalization/logs"), f"{TIMESTAMP}.log")),
#         logging.FileHandler(os.path.join("logs", f"{TIMESTAMP}.log")),
#         logging.StreamHandler()
#     ]
# )


parser = ArgumentParser()

parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

parser.add_argument("--use_authors", dest="use_authors", required=True, type=str2bool)
parser.add_argument("--author_encoder", dest="author_encoder", required=True, type=str) # ['average', 'priming', 'graph', 'none']
parser.add_argument("--social_norm", dest="social_norm", required=True, type=str2bool) # True or False

parser.add_argument("--split_type", dest="split_type", required=True, type=str) # ['author', 'sit', 'verdicts']
parser.add_argument("--situation", dest="situation", required=True, type=str) # ['text', 'title']

parser.add_argument("--sbert_model", dest="sbert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--authors_embedding_path", dest="authors_embedding_path", required=True, type=str)
parser.add_argument("--sbert_dim", dest="sbert_dim", default=768, type=int)
parser.add_argument("--user_dim", dest="user_dim", default=768, type=int)
parser.add_argument("--graph_dim", dest="graph_dim", default=384, type=int)
parser.add_argument("--concat", dest="concat", default='true', type=str2bool)
parser.add_argument("--num_epochs", dest="num_epochs", default=10, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-5, type=float)
parser.add_argument("--dropout_rate", dest="dropout_rate", default=0.2, type=float)
parser.add_argument("--weight_decay", dest="weight_decay", default=1e-2, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='softmax', type=str)
parser.add_argument("--verdicts_dir", dest="verdicts_dir", default='../data/verdicts', type=str)
parser.add_argument("--bert_tok", dest="bert_tok", default='bert-base-uncased', type=str)
parser.add_argument("--dirname", dest="dirname", type=str, default='../data/amit_filtered_history')
parser.add_argument("--results_dir", dest="results_dir", type=str, default='../results')
parser.add_argument("--model_name", dest="model_name", type=str, required=True) # ['judge_bert', 'sbert'] otherwise exception
parser.add_argument("--plot_title", dest="plot_title", type=str, default='') # for plotting the results
parser.add_argument("--log_file", dest="log_file", type=str, default=None) # for logging the results


if __name__ == '__main__':
    args = parser.parse_args()


    
    if args.log_file:
        log_path = args.log_file
    else:
        TIMESTAMP = get_current_timestamp()
        log_path = os.path.join("logs", f"{TIMESTAMP}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logs", f"{log_path}.log")),
            logging.StreamHandler()
        ]
    )



    print_args(args, logging)
    path_to_data = args.path_to_data
    dirname = args.dirname
    bert_checkpoint = args.bert_tok
    model_name = args.model_name
    results_dir = args.results_dir
    verdicts_dir = args.verdicts_dir
    graph_dim = args.graph_dim
    # NOTE: had to change b/c mixed forward and backward slash
    #checkpoint_dir = os.path.join(results_dir, 'best_models', f'{TIMESTAMP}_best_model_sampled.pt')
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #results_dir = os.path.abspath('../results')
    #results_dir = os.path.join(script_dir, 'results')
    #checkpoint_dir = os.path.join(r'C:\Users\User\PycharmProjects\perspectivism-personalization\results', 'best_models', f'{TIMESTAMP}_best_model_sampled.pt')
    checkpoint_dir = os.path.join('results/best_models', f'{TIMESTAMP}_best_model_sampled.pt')
    #checkpoint_dir = os.path.normpath(os.path.join(results_dir, 'best_models', f'{TIMESTAMP}_best_model_sampled.pt'))
    #checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_model_sampled.pt')
    graph_checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_graphmodel.pt')
    dropout_rate = args.dropout_rate

    
    authors_embedding_path = args.authors_embedding_path
    USE_AUTHORS = args.use_authors
    author_encoder = args.author_encoder
    social_norm = args.social_norm
    if USE_AUTHORS:
        assert author_encoder in {'average', 'graph', 'attribution'}
    else:
        assert author_encoder.lower() == 'none' or author_encoder.lower() == 'priming'  or author_encoder.lower() == 'user_id'
        
    split_type = args.split_type
    
    
    logging.info("Device {}".format(DEVICE))

    # NOTE: same note as before
    # social_norm = True #@TODO: Fixed for EMNLP, should make it a parameter 
    # replaced with parameter
    social_chemistry = pd.read_pickle(path_to_data + 'social_chemistry_clean_with_fulltexts')
    print(social_chemistry.shape)
    # save to csv
    social_chemistry.to_csv(path_to_data + 'social_chemistry_clean_with_fulltexts.csv', index=False)

    with open(path_to_data+'social_norms_clean.csv', encoding="utf8") as file:
        social_comments = pd.read_csv(file)

    print(social_comments.shape)
    
    dataset = SocialNormDataset(social_comments, social_chemistry)

    
    
    if split_type == 'sit':
        logging.info("Split type {}".format(split_type))
        train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_situations_split(dataset)
    elif split_type == 'author':
        logging.info("Split type {}".format(split_type))
        train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_author_split(dataset)
    elif split_type == 'verdicts':
        logging.info("Split type {}".format(split_type))
        verdict_ids = list(dataset.verdictToLabel.keys())
        labels = list(dataset.verdictToLabel.values())
        train_verdicts, test_verdicts, train_labels, test_labels = train_test_split(verdict_ids, labels, test_size=0.2, 
                                                                            random_state=SEED)

        train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(train_verdicts, train_labels, test_size=0.15, 
                                                                            random_state=SEED)
    else:
        raise Exception("Split type is wrong, it should be either sit or author")    
   
    
    train_size_stats = "Training Size: {}, NTA labels {}, YTA labels {}".format(len(train_verdicts), train_labels.count(0), train_labels.count(1))
    logging.info(train_size_stats)
    val_size_stats = "Validation Size: {}, NTA labels {}, YTA labels {}".format(len(val_verdicts), val_labels.count(0), val_labels.count(1))
    logging.info(val_size_stats)
    test_size_stats = "Test Size: {}, NTA labels {}, YTA labels {}".format(len(test_verdicts), test_labels.count(0), test_labels.count(1))
    logging.info(test_size_stats)
    
    graph_model = None
    data = None

    if USE_AUTHORS and (author_encoder == 'average' or author_encoder == 'attribution'):
        print(f"Loaded authors embeddings from {authors_embedding_path}")
        embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=args.user_dim)
    else:
        embedder = None
    
    
    raw_dataset = {'train': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}, 
            'val': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}, 
            'test': {'index': [], 'text': [], 'label': [], 'author_node_idx': [] }}

    
    for i, verdict in enumerate(train_verdicts):
        if args.situation == 'text':
            situation_title = dataset.postIdToText[dataset.verdictToParent[verdict]]
        else:
            assert args.situation == 'title', print(args.situation)
            situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        
        if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
            author = dataset.verdictToAuthor[verdict]
            
            if author != 'Judgement_Bot_AITA':
                raw_dataset['train']['index'].append(dataset.verdictToId[verdict])
                
                if author_encoder == 'user_id':
                    author = dataset.verdictToAuthor[verdict]
                    raw_dataset['train']['text'].append('[' + author + ']' + ' [SEP] ' + situation_title)
                else:
                    raw_dataset['train']['text'].append(situation_title)
                    
                raw_dataset['train']['label'].append(train_labels[i])
                
                
                raw_dataset['train']['author_node_idx'].append(-1)
                    
                assert train_labels[i] == dataset.verdictToLabel[verdict] 
        
    for i, verdict in enumerate(val_verdicts):
        if args.situation == 'text':
            situation_title = dataset.postIdToText[dataset.verdictToParent[verdict]]
        else:
            assert args.situation == 'title', print(args.situation)
            situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
            
        if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
            author = dataset.verdictToAuthor[verdict]
            
            if author != 'Judgement_Bot_AITA': 
                raw_dataset['val']['index'].append(dataset.verdictToId[verdict])
                # Priming logic
                if author_encoder == 'user_id':
                    author = dataset.verdictToAuthor[verdict]
                    raw_dataset['val']['text'].append('[' + author + ']' + ' [SEP] ' + situation_title)
                else:
                    raw_dataset['val']['text'].append(situation_title)
                
                raw_dataset['val']['label'].append(val_labels[i])
                
                raw_dataset['val']['author_node_idx'].append(-1)
                
                assert val_labels[i] == dataset.verdictToLabel[verdict]          
        
    for i, verdict in enumerate(test_verdicts):
        if args.situation == 'text':
            situation_title = dataset.postIdToText[dataset.verdictToParent[verdict]]
        else:
            assert args.situation == 'title', print(args.situation)
            situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        
        if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
            author = dataset.verdictToAuthor[verdict]
            
            if author != 'Judgement_Bot_AITA': 
                raw_dataset['test']['index'].append(dataset.verdictToId[verdict])
                # Priming logic
                if author_encoder == 'user_id':
                    author = dataset.verdictToAuthor[verdict]
                    raw_dataset['test']['text'].append('[' + author + ']' + ' [SEP] ' + situation_title)
                else:
                    raw_dataset['test']['text'].append(situation_title)
                    
                raw_dataset['test']['label'].append(test_labels[i])
                
                raw_dataset['test']['author_node_idx'].append(-1)
                
                assert test_labels[i] == dataset.verdictToLabel[verdict] 
    

    if model_name == 'sbert':
    
        logging.info("Training with SBERT, model name is {}".format(model_name))

        local_path = "/home/kieranh/projects/def-cfwelch/kieranh/Self-disclosuresForAnnotatorModeling/.cache/huggingface/hub/models--sentence-transformers--all-distilroberta-v1/snapshots/842eaed40bee4d61673a81c92d5689a8fed7a09f"  # Use actual snapshot hash
        # model = AutoModel.from_pretrained(local_path)
        # tokenizer = AutoTokenizer.from_pretrained(local_path)

        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = SentBertClassifier(users_layer=USE_AUTHORS, user_dim=args.user_dim, sbert_model=local_path, sbert_dim=args.sbert_dim, dropout_rate=dropout_rate)
    # elif model_name == 'judge_bert':
    #     logging.info("Training with Judge Bert, model name is {}".format(model_name))
    #     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    #     model = JudgeBert()
    else:
        raise Exception('Wrong model name')
    
    
    model.to(DEVICE)

    print("raw_dataset size: ", {k: len(v['text']) for k, v in raw_dataset.items()})
    
    ds = DatasetDict()

    for split, d in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'), 
        'text': Value(dtype='string'), 'index': Value(dtype='int64'), 'author_node_idx': Value(dtype='int64')}))
    
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    logging.info("Tokenizing the dataset")
    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    batch_size = args.batch_size
    #class_sample_count = [train_labels.count(0), train_labels.count(1)]
    #weights = 1 / torch.Tensor(class_sample_count)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle = True
    )

    eval_dataloader = DataLoader(
        tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator
    )
    
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )

    print("Train dataloader size: ", len(train_dataloader))
    print("Eval dataloader size: ", len(eval_dataloader))
    print("Test dataloader size: ", len(test_dataloader))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logging.info("Number of training steps {}".format(num_training_steps))
    loss_type=args.loss_type
    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0
    val_metrics = []
    train_loss = []
    
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            verdicts_index = batch.pop("index")
            author_node_idx = batch.pop("author_node_idx")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            if USE_AUTHORS and  (author_encoder == 'average' or author_encoder == 'attribution'):
                authors_embeddings = torch.stack([embedder.embed_author(dataset.verdictToAuthor[dataset.idToVerdict[index.item()]]) for index in verdicts_index]).to(DEVICE)
                output = model(batch, authors_embeddings)
            else:
                print("not using embeddings")
                output = model(batch)
            
            loss = loss_fn(output, labels, samples_per_class_train, loss_type=loss_type)
            train_loss.append(loss.item())
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        val_metric = evaluate(eval_dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder)
        val_metrics.append(val_metric)
        
        logging.info("Epoch {} **** Loss {} **** Metrics validation: {}".format(epoch, loss, val_metric))
        if val_metric['f1_weighted'] > best_f1:
            best_f1 = val_metric['f1_weighted']
            torch.save(model.state_dict(), checkpoint_dir)  
        
              

    logging.info("Evaluating")
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)
        
    test_metrics = evaluate(test_dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder, return_predictions=True)
    results = test_metrics.pop('results')
    logging.info(test_metrics)
    
    result_logs = {'id': TIMESTAMP}
    result_logs['seed'] = SEED
    result_logs['type'] = f'NO VERDICTS TEXT + SITUATION {args.situation}'
    result_logs['sbert_model'] = args.sbert_model
    result_logs['model_name'] = args.model_name
    result_logs['use_authors_embeddings'] = USE_AUTHORS
    result_logs['authors_embedding_path'] = authors_embedding_path
    result_logs['author_encoder'] = author_encoder
    result_logs['split_type'] = split_type
    result_logs['train_stats'] = train_size_stats
    result_logs['val_stats'] = val_size_stats
    result_logs['test_stats'] = test_size_stats
    result_logs['epochs'] = num_epochs
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['val_metrics'] = val_metrics
    result_logs['results'] = results
    
    
    #res_file = os.path.join(results_dir, TIMESTAMP + ".json")
    # res_file = os.path.join(r'C:\Users\User\PycharmProjects\perspectivism-personalization\results',
    #                               f'{TIMESTAMP}.json')
    res_file = os.path.join(r'results',
                                  f'{TIMESTAMP}.json')
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)
