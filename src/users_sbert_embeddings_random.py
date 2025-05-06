import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle as pkl
import sys

import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from dataset import SocialNormDataset
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *

parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

parser.add_argument("--bert_model", dest="bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--dirname", dest="dirname", type=str, required=True, help="Directory containing the json files")
parser.add_argument("--output_dir", dest="output_dir", type=str, required=True)
parser.add_argument("--embed_sentences", dest="embed_sentences", type=str2bool, default=False, help="If set, embed individual sentences instead of full posts")
parser.add_argument("--posts_per_author", dest="posts_per_author", type=int, default=5, help="Number of posts per author to include in the dataset")
parser.add_argument("--output_file_name", dest="output_file_name", type=str, default="user_embeddings")
parser.add_argument("--random_sampling", dest="random_sampling", type=str2bool, default=False, help="If set, use random sampling instead of all posts")
"""
This script creates the average embeddings to encode
the annotators described in section 6.2, especially
using Sentence BERT for Annotators where the embeddings
are averaged at the end to get the final representation.
TODO: Update
"""

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    embed_sentences = args.embed_sentences
    posts_per_author = args.posts_per_author
    output_file = args.output_file_name + '.pkl'
    random_sampling = args.random_sampling

    # sys.stdout = open("console_output.txt", "w")

    # NOTE: social_chemistry ends up getting passed into SocialNormDataset (dataset.py),
    # so I guessed that social_chemistry_clean_with_fulltexts_and_authors is
    # the right file (diff names from data folder, social_chemistry_posts + not gzip)
    # because in SocialNormDataset, social_chemistry uses fulltext, situation & post id
    # columns which social_chemistry_clean_with_fulltexts has
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts')

    # TODO: had to add encoding="utf8" here -- was getting UnicodeDecodeError otherwise
    # NOTE: same thing down here, this is the social_comments_filtered dataset
    # I assume this is the right one (renamed it) b/c it matches with the columns
    with open(path_to_data+'social_norms_clean.csv', encoding="utf8") as file:
        social_comments = pd.read_csv(file)

    # creates the verdicts data structure
    print("Creating verdicts data structure")

    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())

    print(DEVICE)
    print(len(authors))
    print(authors)



    if 'amit' in args.dirname:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
        # results = extract_authors_vocab_AMIT(filenames[0], authors)
    elif 'demographics' in args.dirname:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_demographics)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    else:
        print(f'Processing text files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_notAMIT)(filename, authors) for filename in tqdm(filenames))

    print("Json files processed")
    print("Number of json files: {}".format(len(filenames)))

    # merge results
    print("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    print(len(authors_vocab))

    print("Using {} model".format(args.bert_model))

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)

    user_embeddings = {}

    def extract_batches(seq, batch_size=32):
        n = len(seq) // batch_size
        batches  = []

        for i in range(n):
            batches.append(seq[i * batch_size:(i+1) * batch_size])
        if len(seq) % batch_size != 0:
            batches.append(seq[n * batch_size:])
        return batches


    DEBUG = False
    
    print(authors_vocab.keys())

    if embed_sentences == False:
        print("Embedding posts")
        # NOTE: Process & generate the embeddings for each author for each POST of the author and then average the embeddings together 
        for author, posts in tqdm(authors_vocab.items(), desc="Embedding authors"):
            sys.stdout.flush()
            if len(posts) >= posts_per_author:

                if random_sampling and len(posts) > posts_per_author and posts_per_author > 0:
                    # Randomly sample posts from the author
                    posts = np.random.choice(posts, size=posts_per_author, replace=False)


                processed_texts = [process_tweet(text[0]) for text in posts]

                # Tokenize posts
                batches_text = extract_batches(processed_texts, 64)
                embeddings = []
                encoded_inputs = [tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt') for processed_texts in batches_text]
                

                for encoded_input in encoded_inputs:
                    with torch.no_grad():
                        # Compute token embeddings
                        encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
                        model_output = model(**encoded_input)
                        # Perform pooling
                        post_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                        # Normalize embeddings
                        post_embeddings = F.normalize(post_embeddings, p=2, dim=1)

                        # Average embeddings
                        average = post_embeddings.cpu().mean(axis=0)
                        embeddings.append(average.unsqueeze(0))

                if len(embeddings) > 1:
                    embedding = torch.cat(embeddings)
                    user_embeddings[author] = embedding.mean(axis=0).numpy()
                else:
                    user_embeddings[author] = embeddings[0].squeeze().numpy()

                


                if DEBUG:
                    print(user_embeddings[author], user_embeddings[author].shape)
                    DEBUG = False

    else: # todo fix sampling and post per author 
        print("Embedding sentences")
        # NOTE: Process & generate the embeddings for each author for each SENTENCE of the author and then average the embeddings together 
        for author, posts in tqdm(authors_vocab.items(), desc="Embedding authors"):
            sys.stdout.flush()

            # split the posts into sentences and process them
            all_sentences = []
            for text in posts:
                sentences = sent_tokenize(text[0])  # Split post into sentences
                processed = [process_tweet(sentence) for sentence in sentences]
                all_sentences.extend(processed)
            
            # random sampling 
            if random_sampling and len(all_sentences) > posts_per_author:
                # Randomly sample sentences from the author
                all_sentences = np.random.choice(all_sentences, size=posts_per_author, replace=False)

            # Tokenize sentences and break into batches of 64
            batches_text = extract_batches(all_sentences, 64)
            embeddings = []
            
            for processed_batch in batches_text:
                encoded_input = tokenizer(processed_batch, padding=True, truncation=True, return_tensors='pt')
                encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}

                with torch.no_grad():
                    # Compute token embeddings
                    model_output = model(**encoded_input)
                    # Perform pooling
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                    # Normalize embeddings
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

                    # Average embeddings
                    average = sentence_embeddings.cpu().mean(axis=0)
                    embeddings.append(average.unsqueeze(0))

                   

            if len(embeddings) > 1:
                embedding = torch.cat(embeddings)
                user_embeddings[author] = embedding.mean(axis=0).numpy()
            else:
                user_embeddings[author] = embeddings[0].squeeze().numpy()


            if DEBUG:
                print(user_embeddings[author], user_embeddings[author].shape)
                DEBUG = False



    print("Saving embeddings")
    sys.stdout.close()
    output_file = os.path.join(args.output_dir, output_file)
    with open(output_file, 'wb') as f:
        pkl.dump(user_embeddings, f)

    