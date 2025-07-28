import os
import json
import pickle as pkl
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import pandas as pd
from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *
from dataset import SocialNormDataset
from utils.clusters_utils import *

# GPU Optimization
torch.backends.cudnn.benchmark = True

# Set random seeds for reproducibility
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIMENSION = 768

def load_json_comments(folder):
    all_data = {}
    for file in sorted(os.listdir(folder)):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                all_data[file] = json.load(f)
    return all_data


# function to get 1 unpicked-before top match from each json file in comment_sets 
def get_unique_top_comments(comment_sets, post_id, author, comment_embeddings_map, post_embeddings):
    selected_keys = set()
    selected_embeddings = []

    for file_name, comment_list in comment_sets.items():
        candidates = []
        for comment in comment_list:
            comment_id = comment["id"]
            parent_post_id = comment["parent_id"]
            comment_author = comment.get("author")

            if comment_author != author:
                continue
            if parent_post_id == post_id:
                continue
            if (parent_post_id, comment_id) in comment_embeddings_map:
                embedding = comment_embeddings_map[(parent_post_id, comment_id)]
                candidates.append(((parent_post_id, comment_id), embedding))

        if not candidates:
            continue

        comment_keys, embeddings = zip(*candidates)
        embeddings_tensor = torch.tensor(np.stack(embeddings)).to(DEVICE)
        post_embedding_tensor = torch.tensor(post_embeddings[post_id]).unsqueeze(0).to(DEVICE)
        cosine_similarities = util.cos_sim(post_embedding_tensor, embeddings_tensor).squeeze(0)

        sorted_indices = torch.argsort(cosine_similarities, descending=True)
        for index in sorted_indices:
            key = comment_keys[index.item()]
            if key not in selected_keys:
                selected_keys.add(key)
                selected_embeddings.append(embeddings[index.item()])
                break

    return selected_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_embeddings_path', required=True)
    parser.add_argument('--post_embeddings_path', required=True)
    parser.add_argument('--path_to_data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_file_name', required=True)
    parser.add_argument('--bert_model', default='sentence-transformers/all-distilroberta-v1')
    parser.add_argument('--use_manual', type=str2bool, default=False)
    parser.add_argument('--use_clusters', type=str2bool, default=False)
    parser.add_argument('--persona_data_dir', required=True)

    args = parser.parse_args()

    print("Starting diverse verdict embeddings generation...", flush=True)

    logging.info("Loading precomputed comment embeddings from %s", args.comment_embeddings_path)
    with open(args.comment_embeddings_path, 'rb') as f:
        comment_embeddings = pkl.load(f)

    logging.info("Loading precomputed post embeddings from %s", args.post_embeddings_path)
    with open(args.post_embeddings_path, 'rb') as f:
        post_embeddings = pkl.load(f)

    logging.info("Loading Social Chemistry and Social Norms datasets")
    social_chemistry = pd.read_csv(os.path.join(args.path_to_data, 'social_chemistry_clean_with_fulltexts.csv'))
    social_comments = pd.read_csv(os.path.join(args.path_to_data, 'social_norms_clean.csv'), encoding="utf8")
    social_comments = social_comments.sample(10000)  # Sample for faster processing
    dataset = SocialNormDataset(social_comments, social_chemistry)

    logging.info("Loading BERT model: %s", args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)
    model.eval()

    logging.info("Loading persona data from %s", args.persona_data_dir)
    persona_sets = {}
    if args.use_manual:
        logging.info("Including manual persona data")
        persona_sets.update(load_json_comments(os.path.join(args.persona_data_dir, 'manual')))
    if args.use_clusters:
        logging.info("Including cluster-based persona data")
        persona_sets.update(load_json_comments(os.path.join(args.persona_data_dir, 'clusters')))

    logging.info("Preparing comment embedding map")
    comment_embeddings_map = {(parent_id, comment_id): emb for (_, parent_id, comment_id), emb in comment_embeddings.items()}

    verdict_level_embeddings = {}
    for post_id in tqdm(dataset.postToVerdicts.keys(), desc="Generating diverse verdict embeddings"):
        if post_id not in post_embeddings:
            continue

        for verdict_id in dataset.postToVerdicts[post_id]:
            author = dataset.verdictToAuthor.get(verdict_id, None)
            if not author:
                continue

            selected_author_embeddings = get_unique_top_comments(persona_sets, post_id, author, comment_embeddings_map, post_embeddings)
            if selected_author_embeddings:
                verdict_level_embeddings[verdict_id] = np.mean(np.stack(selected_author_embeddings), axis=0)
            else:
                verdict_level_embeddings[verdict_id] = np.random.normal(0, 0.5, EMBEDDING_DIMENSION)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file_name + ".pkl")
    with open(output_path, 'wb') as f:
        pkl.dump(verdict_level_embeddings, f)
    logging.info("Saved %d verdict embeddings to %s", len(verdict_level_embeddings), output_path)
