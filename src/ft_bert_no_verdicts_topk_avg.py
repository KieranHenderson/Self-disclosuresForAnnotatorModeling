from verdict_embedder import VerdictEmbedder

import sys
print(sys.executable)
from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from datasets import DatasetDict, Dataset, Features, Value


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
from constants import *

import matplotlib.pyplot as plt

TIMESTAMP = get_current_timestamp()

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler(os.path.join(os.path.expanduser("~/PycharmProjects/perspectivism-personalization/logs"), f"{TIMESTAMP}.log")),
        logging.FileHandler(os.path.join("logs", f"{TIMESTAMP}.log")),
        logging.StreamHandler()
    ]
)


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


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args, logging)
    path_to_data = args.path_to_data
    dirname = args.dirname
    bert_checkpoint = args.bert_tok
    model_name = args.model_name
    results_dir = args.results_dir
    verdicts_dir = args.verdicts_dir
    graph_dim = args.graph_dim
    checkpoint_dir = os.path.join('results/best_models', f'{TIMESTAMP}_best_model_sampled.pt')
    graph_checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_graphmodel.pt')
    authors_embedding_path = args.authors_embedding_path
    print(f"Authors embedding path: {authors_embedding_path[40:]}")
    USE_AUTHORS = args.use_authors
    author_encoder = args.author_encoder
    social_norm = args.social_norm
    split_type = args.split_type
    dropout_rate = args.dropout_rate

    if USE_AUTHORS:
        assert author_encoder in {'average', 'graph', 'attribution'}
    else:
        assert author_encoder.lower() == 'none' or author_encoder.lower() == 'priming'  or author_encoder.lower() == 'user_id'
    
    logging.info("Device {}".format(DEVICE))

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
        # TODO Check
        train_verdicts, test_verdicts, train_labels, test_labels = train_test_split(verdict_ids, labels, test_size=0.2, random_state=SEED)
        train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(train_verdicts, train_labels, test_size=0.15, random_state=SEED)
    else:
        raise Exception("Split type is wrong, it should be either sit or author")    
   
    
    train_size_stats = "Training Size: {}, NTA labels {}, YTA labels {}".format(len(train_verdicts), train_labels.count(0), train_labels.count(1))
    logging.info(train_size_stats)
    val_size_stats = "Validation Size: {}, NTA labels {}, YTA labels {}".format(len(val_verdicts), val_labels.count(0), val_labels.count(1))
    logging.info(val_size_stats)
    test_size_stats = "Test Size: {}, NTA labels {}, YTA labels {}".format(len(test_verdicts), test_labels.count(0), test_labels.count(1))
    logging.info(test_size_stats)
    
    if USE_AUTHORS and (author_encoder == 'average' or author_encoder == 'attribution'):
        print(f"Loaded authors embeddings from {authors_embedding_path}")
        # embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=args.user_dim)
        embedder = VerdictEmbedder(embeddings_path=authors_embedding_path)
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
                
                raw_dataset['test']['text'].append(situation_title)
                    
                raw_dataset['test']['label'].append(test_labels[i])
                
                raw_dataset['test']['author_node_idx'].append(-1)
                
                assert test_labels[i] == dataset.verdictToLabel[verdict] 
    
    average_results = []
    for i in range(5):

        np.random.seed(SEED+i)
        torch.manual_seed(SEED+i)
        torch.cuda.manual_seed(SEED+i)
        torch.cuda.manual_seed_all(SEED+i)

        if model_name == 'sbert':
            logging.info("Training with SBERT, model name is {}".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
            model = SentBertClassifier(users_layer=USE_AUTHORS, user_dim=args.user_dim, sbert_model=args.sbert_model, sbert_dim=args.sbert_dim, dropout_rate=dropout_rate)
        # elif model_name == 'judge_bert':
        #     logging.info("Training with Judge Bert, model name is {}".format(model_name))
        #     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        #     model = JudgeBert()
        else:
            raise Exception('Wrong model name')
        
        
        model.to(DEVICE)
        
        ds = DatasetDict()

        for split, d in raw_dataset.items():
            ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'), 'text': Value(dtype='string'), 'index': Value(dtype='int64'), 'author_node_idx': Value(dtype='int64')}))
        
        def tokenize_function(example):
            return tokenizer(example["text"], truncation=True)

        logging.info("Tokenizing the dataset")
        tokenized_dataset = ds.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        
        batch_size = args.batch_size

        train_dataloader = DataLoader(
            tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle = True
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator
        )
        
        test_dataloader = DataLoader(
            tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
        )

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

        train_loss = []

        val_metrics = []
        val_accuracies = []
        val_f1_scores = []
        train_losses = []  # This will store average loss per epoch
        epochs = []

        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []  # This will store losses for each batch in the current epoch
                
            for batch in train_dataloader:
                verdicts_index = batch.pop("index")
                author_node_idx = batch.pop("author_node_idx")
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                if USE_AUTHORS and (author_encoder == 'average' or author_encoder == 'attribution'):
                    try:
                        verdict_embeddings = torch.stack([embedder.embed_verdict(dataset.idToVerdict[index.item()]) for index in verdicts_index]).to(DEVICE)
                    except KeyError as e:
                        logging.warning(f"Missing embedding for verdict_id {e}. Skipping this batch.")
                        continue
                    output = model(batch, verdict_embeddings)
                else: 
                    output = model(batch)
                
                loss = loss_fn(output, labels, samples_per_class_train, loss_type=loss_type)
                epoch_losses.append(loss.item())
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            # Calculate and store average epoch loss
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_epoch_loss)
            
            val_metric = evaluate_similar(eval_dataloader, model, embedder, USE_AUTHORS, dataset, author_encoder)
            val_metrics.append(val_metric)

            # Store validation metrics
            val_accuracies.append(val_metric['accuracy'])
            val_f1_scores.append(val_metric['macro'])
            epochs.append(epoch)
            
            logging.info("Epoch {} **** Loss {} **** Metrics validation: {}".format(epoch, avg_epoch_loss, val_metric))
            if val_metric['f1_weighted'] > best_f1:
                best_f1 = val_metric['f1_weighted']
                torch.save(model.state_dict(), checkpoint_dir)
        
        # Plot and save the training results graph
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
        plt.plot(epochs, val_f1_scores, label='Validation F1 Score', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{args.plot_title}')
        plt.legend()
        loss_plot_path = os.path.join('results/graphs', f'{args.plot_title}.png')
        plt.savefig(loss_plot_path)
        plt.close()



        logging.info("Evaluating")
        model.load_state_dict(torch.load(checkpoint_dir))
        model.to(DEVICE)

        test_metrics = evaluate_similar(test_dataloader, model, embedder, USE_AUTHORS, dataset, author_encoder, return_predictions=True)
        average_results.append(test_metrics)

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

        topk_match = re.search(r'topk_(\d+)', authors_embedding_path)
        result_logs['top_k'] = int(topk_match.group(1)) if topk_match else -1

        embed_type_match = re.search(r'user_(\w+)_embeddings', authors_embedding_path)
        result_logs['embedding_type'] = embed_type_match.group(1) if embed_type_match else 'unknown'






        
        
        #res_file = os.path.join(results_dir, TIMESTAMP + ".json")
        # res_file = os.path.join(r'C:\Users\User\PycharmProjects\perspectivism-personalization\results',
        #                               f'{TIMESTAMP}.json')
        res_file = os.path.join(r'results',
                                    f'{TIMESTAMP}.json')
        with open(res_file, mode='w') as f:
            json.dump(result_logs, f, cls=NpEncoder, indent=2)

    # average results
    averaged_results = results.copy()
    for i in range(1, len(average_results)):
        for key in average_results[i].keys():
            averaged_results[key] += average_results[i][key]

    for key in averaged_results.keys():
        averaged_results[key] /= len(average_results)

    
    logging.info("Averaged results: {}".format(averaged_results))