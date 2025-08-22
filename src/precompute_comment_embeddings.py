"""
TODO
"""

import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle as pkl
import json
import logging

from sentence_transformers import util
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *

from dataset import SocialNormDataset

# GPU Optimization
torch.backends.cudnn.benchmark = True

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--path_to_data", required=True, type=str)
parser.add_argument("--bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--dirname", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--embed_sentences", type=str2bool, default=False)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--output_file_name", type=str, default="precomputed_embeddings")

if __name__ == '__main__':
    args = parser.parse_args()
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_to_data = args.path_to_data
    output_file = args.output_file_name + '.pkl'
    embed_sentences = args.embed_sentences
    batch_size = args.batch_size

    logging.info("Loading data...")
    social_chemistry = pd.read_csv(path_to_data + 'social_chemistry_clean_with_fulltexts.csv')
    social_comments = pd.read_csv(path_to_data + 'social_norms_clean.csv', encoding="utf8")
    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())

    logging.info(f"Loaded {len(authors)} authors.")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)
    model.eval()

    # filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
    # logging.info(f"Processing {len(filenames)} JSON files from {args.dirname}")
    # results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(f, authors) for f in tqdm(filenames))

    if 'amit' in args.dirname:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    else:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_demographics)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))

    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    def batch_iterable(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    def encode_batch(texts):
        embeddings = []
        for batch in batch_iterable(texts, batch_size):
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                output = model(**encoded)
                pooled = mean_pooling(output, encoded['attention_mask'])
                normed = F.normalize(pooled, p=2, dim=1)
                embeddings.append(normed.cpu())
        return torch.cat(embeddings, dim=0)

    # Compute embeddings
    logging.info("Encoding comments...")
    comment_embeddings = {}
    for author, comment_list in tqdm(authors_vocab.items(), desc="Encoding"):
        for comment_text, comment_id, post_id in comment_list:
            if not comment_text or not post_id:
                continue
            try:
                if embed_sentences:
                    sentences = [process_tweet(s) for s in sent_tokenize(comment_text) if s.strip() != '']
                    if not sentences:
                        continue
                    emb = encode_batch(sentences).mean(dim=0)
                else:
                    emb = encode_batch([process_tweet(comment_text)])[0]

                comment_embeddings[(author, post_id, comment_id)] = emb.numpy()

            except Exception as e:
                logging.warning(f"Failed to embed comment {comment_id}: {e}")
                continue


    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, output_file)
    with open(final_path, 'wb') as f:
        pkl.dump(comment_embeddings, f)

    logging.info(f"Saved comment embeddings to {final_path}")
