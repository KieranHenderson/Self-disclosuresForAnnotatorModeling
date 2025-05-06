import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle as pkl
import sys

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from dataset import SocialNormDataset
from models import MLPAttribution

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *

parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

parser.add_argument("--age_dir", dest="age_dir", default='../data/demographic/age_list.csv', type=str)
parser.add_argument("--gender_dir", dest="gender_dir", default='../data/demographic/gender_list.csv', type=str)
parser.add_argument("--bert_model", dest="bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--age_gender_authors", dest="age_gender_authors", type=str2bool, default=False)
parser.add_argument("--dirname", dest="dirname", type=str, required=True)
parser.add_argument("--output_dir", dest="output_dir", type=str, required=True)

"""
This script creates the average embeddings to encode
the annotators described in section 6.2, especially
using Sentence BERT for Annotators where the embeddings
are averaged at the end to get the final representation.
"""

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    age_gender_authors = False

    sys.stdout = open("console_output.txt", "w")
    
    if age_gender_authors:
        age_df = pd.read_csv(args.age_dir, sep='\t', names=['author', 'subreddit', 'age'])
        gender_df = pd.read_csv(args.gender_dir, sep='\t', names=['author', 'gender'])
        age_authors = set(age_df.author)
        gender_authors = set(gender_df.author)
        authors = gender_authors.intersection(age_authors)
    else:
        # social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')
        #
        # with open(path_to_data+'social_norms_clean.csv') as file:
        #     social_comments = pd.read_csv(file)

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
        dataset = SocialNormDataset(social_comments, social_chemistry)
        authors = set(dataset.authorsToVerdicts.keys())
    print(DEVICE)
    print(len(authors))
    print(authors)

    # NOTE: I am confused here, I don't understand where the json files are
    # supposed to come from, but it seems like we just want to be able to run
    # extract_authors_vocab_amit (utils/read_files.py) where we extract comments
    # made by specific authors, based on the extract_authors_vocab_amit func body,
    # it seems like the json files should have author, body, id, and parent_id fields.
    # These columns correspond to the social_norms.csv file, so I converted that to json
    # Another point of confusion, we're using parallel processing so the code
    # is expecting a substantial amount of JSON files -- I probably did something wrong here
    if 'amit' in args.dirname:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
        # results = extract_authors_vocab_AMIT(filenames[0], authors)
    else:
        print(f'Processing text files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_notAMIT)(filename, authors) for filename in tqdm(filenames))

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


    DEBUG = True

    # NOTE: Process & generate the embeddings for each author
    for author, texts in tqdm(authors_vocab.items(), desc="Embedding authors"):
        sys.stdout.flush()
        processed_texts = [process_tweet(text[0]) for text in texts]
        # Tokenize sentences
        batches_text = extract_batches(processed_texts, 64)
        embeddings = []
        encoded_inputs = [tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt') for processed_texts in batches_text]

        for encoded_input in encoded_inputs:
            with torch.no_grad():
                # Compute token embeddings
                encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
                model_output = model(**encoded_input)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                # Normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


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
    output_file = os.path.join(args.output_dir, 'user_embeddings.pkl')
    with open(output_file, 'wb') as f:
        pkl.dump(user_embeddings, f)

    