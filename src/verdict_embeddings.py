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

    EMBED_DIM = 768  # Dimension of the BERT embeddings

    # Load precomputed comment embeddings
    logging.info("Loading precomputed comment embeddings from %s", comment_emb_path)
    with open(comment_emb_path, 'rb') as f:
        comment_embeddings = pkl.load(f)

    # Load or compute post embeddings
    if os.path.exists(post_emb_path):
        logging.info("Loading precomputed post embeddings from %s", post_emb_path)
        with open(post_emb_path, 'rb') as f:
            post_embeddings = pkl.load(f)
    else:
        logging.info("Precomputed post embeddings not found. Generating new ones...")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
        model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)
        model.eval()

        def encode_texts(texts, batch_size=4096):
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

        social_chemistry = pd.read_csv(os.path.join(path_to_data, 'social_chemistry_clean_with_fulltexts.csv'))
        social_comments = pd.read_csv(os.path.join(path_to_data, 'social_norms_clean.csv'), encoding="utf8")
        dataset = SocialNormDataset(social_comments, social_chemistry)

        post_id_texts = [(post_id, dataset.postIdToText.get(post_id)) for post_id in dataset.postToVerdicts.keys() if dataset.postIdToText.get(post_id)]
        post_ids, processed_post_contents = zip(*[(post_id, process_tweet(fulltext)) for post_id, fulltext in post_id_texts])

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
            all_embeddings = encode_texts(processed_post_contents)
            post_embeddings = {post_id: all_embeddings[i].numpy() for i, post_id in enumerate(post_ids)}

        os.makedirs(os.path.dirname(post_emb_path), exist_ok=True)
        with open(post_emb_path, 'wb') as f:
            pkl.dump(post_embeddings, f)
        logging.info("Saved post embeddings to %s", post_emb_path)

    # Load dataset again if needed
    if 'dataset' not in locals():
        social_chemistry = pd.read_csv(os.path.join(path_to_data, 'social_chemistry_clean_with_fulltexts.csv'))
        social_comments = pd.read_csv(os.path.join(path_to_data, 'social_norms_clean.csv'), encoding="utf8")
        dataset = SocialNormDataset(social_comments, social_chemistry)

    # Group comments by author
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

            author_other_comments = [(parent_id, comment_id, embedding) for parent_id, comment_id, embedding in author_to_comments[author] if parent_id != post_id]
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

    # Save final verdict-level embeddings
    logging.info("Saving verdict embeddings to %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file_name + '.pkl'), 'wb') as f:
        pkl.dump(verdict_embeddings, f)

    logging.info(f"Saved {len(verdict_embeddings)} verdict-level user embeddings")
