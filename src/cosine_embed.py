import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle as pkl
import sys

from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from dataset import SocialNormDataset
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *

# Enable GPU performance optimization
torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)
parser.add_argument("--bert_model", dest="bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--dirname", dest="dirname", type=str, required=True)
parser.add_argument("--output_dir", dest="output_dir", type=str, required=True)
parser.add_argument("--embed_sentences", dest="embed_sentences", type=str2bool, default=False)
parser.add_argument("--posts_per_author", dest="posts_per_author", type=int, default=5)
parser.add_argument("--output_file_name", dest="output_file_name", type=str, default="user_embeddings_topk")

if __name__ == '__main__':

    args = parser.parse_args()
    path_to_data = args.path_to_data
    posts_per_author = args.posts_per_author
    output_file = args.output_file_name + '.pkl'
    embed_sentences = args.embed_sentences

    social_chemistry = pd.read_pickle(path_to_data + 'social_chemistry_clean_with_fulltexts')
    with open(path_to_data + 'social_norms_clean.csv', encoding="utf8") as file:
        social_comments = pd.read_csv(file)

    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())


    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)
    print(DEVICE)
    model.eval()

    filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames))
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    user_embeddings = {}
    resume_from = 0
    existing_partials = sorted(glob.glob(os.path.join(args.output_dir, f"{args.output_file_name}_partial_*.pkl")))
    if existing_partials:
        latest = existing_partials[-1]
        with open(latest, 'rb') as f:
            user_embeddings = pkl.load(f)
        resume_from = int(latest.split("_partial_")[-1].split(".pkl")[0])

    def extract_batches(seq, batch_size=128):
        n = len(seq) // batch_size
        batches = [seq[i * batch_size:(i+1) * batch_size] for i in range(n)]
        if len(seq) % batch_size != 0:
            batches.append(seq[n * batch_size:])
        return batches

    def encode_texts(texts):
        all_embeddings = []
        for batch in extract_batches(texts, batch_size=128):
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                output = model(**encoded)
                pooled = mean_pooling(output, encoded['attention_mask'])
                normed = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normed.cpu())
        return torch.cat(all_embeddings, dim=0)

    def encode_sentences(texts):
        all_sentences = []
        for text in texts:
            sentences = sent_tokenize(text)
            all_sentences.extend([process_tweet(s) for s in sentences if s.strip() != ''])
        return encode_texts(all_sentences)

    fulltext_map = dataset.postIdToText

    for idx, post_id in enumerate(tqdm(dataset.postToVerdicts.keys(), desc="Processing posts")):
        fulltext = fulltext_map.get(post_id, None)
        if not fulltext:
            continue

        processed_post_text = process_tweet(fulltext)
        if embed_sentences:
            sentences = [process_tweet(s) for s in sent_tokenize(processed_post_text) if s.strip() != '']
            post_emb = encode_texts(sentences).mean(dim=0, keepdim=True)
        else:
            post_emb = encode_texts([processed_post_text])

        for verdict_id in dataset.postToVerdicts[post_id]:
            author = dataset.verdictToAuthor.get(verdict_id, None)
            if not author:
                continue
            author_posts = authors_vocab.get(author, [])

            # exclude current post
            filtered = [(text, parent_id) for text, _, parent_id in author_posts if parent_id != post_id and text.strip() != '']
            if not filtered:
                continue

            if embed_sentences:
                sentences = []
                for text, _ in filtered:
                    sentences.extend([process_tweet(s) for s in sent_tokenize(text) if s.strip() != ''])
                if not sentences:
                    continue
                comment_embs = encode_texts(sentences)
                sims = util.cos_sim(post_emb, comment_embs).squeeze(0)
            else:
                texts = [process_tweet(text) for text, _ in filtered]
                comment_embs = encode_texts(texts)
                sims = util.cos_sim(post_emb, comment_embs).squeeze(0)

            topk_indices = torch.topk(sims, k=min(posts_per_author, len(comment_embs))).indices.tolist()
            selected_embs = comment_embs[topk_indices]
            user_embeddings[verdict_id] = selected_embs.mean(dim=0).numpy()

        if (idx + 1) % 1000 == 0:
            partial_path = os.path.join(args.output_dir, f"{args.output_file_name}_partial_{idx + 1}.pkl")
            with open(partial_path, 'wb') as f:
                pkl.dump(user_embeddings, f)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, output_file), 'wb') as f:
        pkl.dump(user_embeddings, f)
