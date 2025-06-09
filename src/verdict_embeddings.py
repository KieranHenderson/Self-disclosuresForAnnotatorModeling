import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
import logging
from collections import defaultdict
import json

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *
from dataset import SocialNormDataset

# GPU Optimization
torch.backends.cudnn.benchmark = True

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--comment_embeddings_path', required=True)
parser.add_argument('--post_embeddings_path', required=True)
parser.add_argument('--path_to_data', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--output_file_name', required=True)
parser.add_argument('--embed_sentences', type=str2bool, default=False)
parser.add_argument('--bert_model', default='sentence-transformers/all-distilroberta-v1')
parser.add_argument('--json_comments_path', required=False, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    comment_emb_path = args.comment_embeddings_path
    post_emb_path = args.post_embeddings_path
    path_to_data = args.path_to_data
    output_dir = args.output_dir
    top_k = args.top_k
    output_file_name = args.output_file_name
    embed_sentences = args.embed_sentences
    bert_model_name = args.bert_model
    json_comments_path = args.json_comments_path

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading precomputed comment embeddings from %s", comment_emb_path)
    with open(comment_emb_path, 'rb') as f:
        comment_embeddings = pkl.load(f)

    if os.path.exists(post_emb_path):
        logging.info("Loading precomputed post embeddings from %s", post_emb_path)
        with open(post_emb_path, 'rb') as f:
            post_embeddings = pkl.load(f)
    else:
        raise FileNotFoundError(f"Post embeddings not found at {post_emb_path}")

    EMBED_DIM = 768

    social_chemistry = pd.read_csv(os.path.join(path_to_data, 'social_chemistry_clean_with_fulltexts.csv'))
    social_comments = pd.read_csv(os.path.join(path_to_data, 'social_norms_clean.csv'), encoding="utf8")
    dataset = SocialNormDataset(social_comments, social_chemistry)

    author_to_comments = defaultdict(list)

    if json_comments_path:
        logging.info("Using JSON file to restrict comment set: %s", json_comments_path)
        with open(json_comments_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        allowed_keys = set()
        for item in tqdm(json_data, desc="Processing JSON comments"):
            author = item.get("author")
            comment_id = item.get("id")
            parent_id = item.get("parent_id")
            if author and comment_id and parent_id:
                key = (author, parent_id, comment_id)
                if key in comment_embeddings:
                    author_to_comments[author].append((parent_id, comment_id, comment_embeddings[key]))
    else:
        logging.info("Using all available comment embeddings")
        for (author, parent_id, comment_id), embedding in comment_embeddings.items():
            author_to_comments[author].append((parent_id, comment_id, embedding))

    verdict_embeddings = {}
    for post_id in tqdm(dataset.postToVerdicts.keys(), desc="Generating verdict embeddings"):
        post_embedding = post_embeddings.get(post_id)
        if post_embedding is None:
            continue
        post_embedding = torch.tensor(post_embedding).unsqueeze(0).to(DEVICE)

        for verdict_id in dataset.postToVerdicts[post_id]:
            author = dataset.verdictToAuthor.get(verdict_id, None)
            if not author:
                continue

            author_other_comments = [
                (parent_id, comment_id, embedding)
                for parent_id, comment_id, embedding in author_to_comments[author]
                if parent_id != post_id
            ]

            if author_other_comments:
                vecs = torch.tensor(np.stack([embedding for _, _, embedding in author_other_comments]), dtype=torch.float32).to(DEVICE)
                similarities = util.cos_sim(post_embedding, vecs).squeeze(0)
                k = min(top_k, len(similarities))
                topk = torch.topk(similarities, k=k)
                selected_vecs = vecs[topk.indices]
                mean_embedding = selected_vecs.mean(dim=0).cpu().numpy()
                verdict_embeddings[verdict_id] = mean_embedding
            else:
                verdict_embeddings[verdict_id] = np.random.normal(0, 0.5, EMBED_DIM)

    logging.info("Saving verdict embeddings to %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file_name + '.pkl'), 'wb') as f:
        pkl.dump(verdict_embeddings, f)

    logging.info(f"Saved {len(verdict_embeddings)} verdict-level user embeddings")
