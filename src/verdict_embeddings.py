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
from joblib import Parallel, delayed
import glob


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

def encode_texts(texts, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(normed.cpu())
        del inputs, outputs, pooled, normed
        torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

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

    EMBED_DIM = 768  # Dimension of the BERT embeddings

    # Load precomputed comment embeddings
    logging.info("Loading precomputed comment embeddings from %s", comment_emb_path)
    with open(comment_emb_path, 'rb') as f:
        comment_embeddings = pkl.load(f)

    # Filter comments based on JSON file (if provided)
    logging.info("Filtering comments based on JSON file (if provided)")
    filtered_comment_keys = set()
    if json_comments_path:
        logging.info("Using JSON file to restrict comment set: %s", json_comments_path)
        with open(json_comments_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in tqdm(json_data, desc="Processing JSON comments"):
            author = item.get("author")
            comment_id = item.get("id")
            parent_id = item.get("parent_id")
            if author and comment_id and parent_id:
                key = (author, parent_id, comment_id)
                if key in comment_embeddings:
                    filtered_comment_keys.add(key)
    else:
        filtered_comment_keys = set(comment_embeddings.keys())

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
    model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)
    model.eval()

    # Load dataset
    social_chemistry = pd.read_csv(os.path.join(path_to_data, 'social_chemistry_clean_with_fulltexts.csv'))
    social_comments = pd.read_csv(os.path.join(path_to_data, 'social_norms_clean.csv'), encoding="utf8")
    dataset = SocialNormDataset(social_comments, social_chemistry)

    # Load or compute post embeddings
    if os.path.exists(post_emb_path):
        logging.info("Loading precomputed post embeddings from %s", post_emb_path)
        with open(post_emb_path, 'rb') as f:
            post_embeddings = pkl.load(f)
    else:
        logging.info("Precomputed post embeddings not found. Generating new ones...")
        post_id_texts = [(post_id, dataset.postIdToText.get(post_id)) for post_id in dataset.postToVerdicts.keys() if dataset.postIdToText.get(post_id)]


        if embed_sentences:
            all_sentences = []
            sentence_map = {}
            for post_id, fulltext in post_id_texts:
                sents = [process_tweet(s) for s in sent_tokenize(fulltext) if s.strip() != '']
                sentence_map[post_id] = (len(all_sentences), len(all_sentences) + len(sents))
                all_sentences.extend(sents)
            all_embeddings = encode_texts(all_sentences)
            post_embeddings = {post_id: all_embeddings[start:end].mean(dim=0).numpy() for post_id, (start, end) in sentence_map.items()}
        else:
            processed_post_contents = [process_tweet(fulltext) for _, fulltext in post_id_texts]
            post_ids = [post_id for post_id, _ in post_id_texts]
            all_embeddings = encode_texts(processed_post_contents)
            post_embeddings = {post_id: all_embeddings[i].numpy() for i, post_id in enumerate(post_ids)}

        os.makedirs(os.path.dirname(post_emb_path), exist_ok=True)
        with open(post_emb_path, 'wb') as f:
            pkl.dump(post_embeddings, f)
        logging.info("Saved post embeddings to %s", post_emb_path)


    # Creating author_vocab from AMIT dataset
    authors = set(dataset.authorsToVerdicts.keys())
    filenames = sorted(glob.glob(os.path.join('data/amit_filtered_history/', '*.json')))
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    # Create a mapping from comment_id to text
    comment_id_to_text = {}
    for author, comment_list in authors_vocab.items():
        for comment_text, comment_id, post_id in comment_list:
            if comment_id:  # optionally ensure not None
                comment_id_to_text[comment_id] = comment_text



    # Build retrieval corpus based on embed_sentences
    author_to_comments = defaultdict(list)
    author_to_sentences = defaultdict(list)

    if embed_sentences:
        logging.info("Embedding author comments at sentence level for retrieval.")

        all_sentences = []
        sentence_map = []

        for (author, parent_id, comment_id) in tqdm(filtered_comment_keys, desc="Preparing sentences"):
            comment_text = comment_id_to_text.get(comment_id, None)
            if not comment_text:
                continue
            sents = [process_tweet(s) for s in sent_tokenize(comment_text) if s.strip()]
            sentence_map.extend([(author, parent_id, comment_id, idx) for idx in range(len(sents))])
            all_sentences.extend(sents)

        sentence_embeddings = []
        batch_size = 512
        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i:i + batch_size]
            sentence_embeddings.extend(encode_texts(batch))

        for idx, (author, parent_id, comment_id, sent_idx) in enumerate(sentence_map):
            emb = sentence_embeddings[idx].numpy()
            author_to_sentences[author].append((parent_id, comment_id, sent_idx, emb))
    else:
        logging.info("Using filtered comments at comment level for retrieval.")
        for (author, parent_id, comment_id) in filtered_comment_keys:
            embedding = comment_embeddings[(author, parent_id, comment_id)]
            author_to_comments[author].append((parent_id, comment_id, embedding))


    # Compute verdict-specific user embeddings
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

            if embed_sentences:
                author_items = [(parent_id, comment_id, sent_idx, emb) for parent_id, comment_id, sent_idx, emb in author_to_sentences[author] if parent_id != post_id]
                vecs = torch.tensor(np.stack([emb for _, _, _, emb in author_items]), dtype=torch.float32).to(DEVICE) if author_items else None
            else:
                author_items = [(parent_id, comment_id, emb) for parent_id, comment_id, emb in author_to_comments[author] if parent_id != post_id]
                vecs = torch.tensor(np.stack([emb for _, _, emb in author_items]), dtype=torch.float32).to(DEVICE) if author_items else None

            if vecs is not None and vecs.shape[0] > 0:
                similarities = util.cos_sim(post_embedding, vecs).squeeze(0)
                k = min(top_k, len(similarities))
                topk = torch.topk(similarities, k=k)
                selected_vecs = vecs[topk.indices]
                mean_embedding = selected_vecs.mean(dim=0).cpu().numpy()
                verdict_embeddings[verdict_id] = mean_embedding
            else:
                verdict_embeddings[verdict_id] = np.random.normal(0, 0.5, EMBED_DIM)

    # Save final verdict-level embeddings
    logging.info("Saving verdict embeddings to %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file_name + '.pkl'), 'wb') as f:
        pkl.dump(verdict_embeddings, f)

    logging.info(f"Saved {len(verdict_embeddings)} verdict-level user embeddings")
