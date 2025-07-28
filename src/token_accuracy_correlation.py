import argparse
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import util
import logging
from collections import defaultdict
import json
from joblib import Parallel, delayed
import glob
import random
from scipy.stats import pearsonr

from utils.read_files import *
from utils.utils import *
from constants import *
from dataset import SocialNormDataset
from utils.clusters_utils import *

# GPU optimization
torch.backends.cudnn.benchmark = True 

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_avg_token_count(keys, comment_id_to_text, tokenizer):
    lengths = []
    for author, parent_id, comment_id in keys:
        if comment_id in comment_id_to_text:
            raw_text = comment_id_to_text[comment_id]
            clean_text = process_tweet(raw_text)
            tokens = tokenizer.tokenize(clean_text)
            lengths.append(len(tokens))
    return np.mean(lengths) if lengths else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_json_path', required=True)
    parser.add_argument('--comment_embeddings_path', required=True)
    parser.add_argument('--post_embeddings_path', required=True)
    parser.add_argument('--path_to_data', required=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--bert_model', default='sentence-transformers/all-distilroberta-v1')
    args = parser.parse_args()

    logging.info("Loading tokenizer: %s", args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)

    # Load data
    with open(args.results_json_path, 'r') as f:
        results_json = json.load(f)
    header = results_json["results"][0]
    rows = results_json["results"][1:]

    with open(args.comment_embeddings_path, 'rb') as f:
        comment_embeddings = pkl.load(f)

    with open(args.post_embeddings_path, 'rb') as f:
        post_embeddings = pkl.load(f)

    social_chemistry = pd.read_csv(os.path.join(args.path_to_data, 'social_chemistry_clean_with_fulltexts.csv'))
    social_comments = pd.read_csv(os.path.join(args.path_to_data, 'social_norms_clean.csv'), encoding="utf8")
    dataset = SocialNormDataset(social_comments, social_chemistry)

    # Verdict to Post mapping
    verdict_to_post = {
        verdict_id: post_id
        for post_id, verdicts in dataset.postToVerdicts.items()
        for verdict_id in verdicts
    }

    # Author history from AMIT
    authors = set(dataset.authorsToVerdicts.keys())
    filenames = sorted(glob.glob(os.path.join('data/amit_filtered_history/', '*.json')))
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    comment_id_to_text = {
        comment_id: comment_text
        for comment_list in authors_vocab.values()
        for comment_text, comment_id, post_id in comment_list
        if comment_id
    }

    # Correlation Calculation
    x_vals, y_vals = [], []

    for verdict_id, pred, gold in tqdm(rows, desc="Computing correlation"):
        post_id = verdict_to_post.get(verdict_id)
        if post_id not in post_embeddings:
            continue

        author = dataset.verdictToAuthor.get(verdict_id)
        if not author:
            continue

        post_emb = torch.tensor(post_embeddings[post_id]).unsqueeze(0).to(DEVICE)

        author_keys, author_vecs = [], []
        for key, vec in comment_embeddings.items():
            a, parent_id, comment_id = key
            if a == author and parent_id != post_id:
                author_keys.append(key)
                author_vecs.append(vec)

        if not author_vecs:
            continue

        comment_tensor = torch.tensor(np.stack(author_vecs), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            sims = util.cos_sim(post_emb, comment_tensor).squeeze(0)
        topk_indices = torch.topk(sims, k=min(args.top_k, len(sims))).indices.cpu().numpy()
        selected_keys = [author_keys[i] for i in topk_indices]

        avg_tokens = compute_avg_token_count(selected_keys, comment_id_to_text, tokenizer)
        x_vals.append(avg_tokens)
        y_vals.append(int(pred == gold))

    # Pearson correlation
    corr, pval = pearsonr(x_vals, y_vals)
    print(f"\n🔍 Correlation between avg token count of top-{args.top_k} similar comments and accuracy:")
    print(f"  Pearson r = {corr:.4f}")
    print(f"  p-value   = {pval:.4g}")
