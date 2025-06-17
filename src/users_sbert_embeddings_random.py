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

import random

import matplotlib.pyplot as plt
import numpy as np

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

def plot_author_distribution(authors_vocab, embed_sentences, output_dir):
    """Plot distribution of authors by number of posts/sentences they have"""
    # Calculate posts/sentences per author
    author_counts = []
    
    for author, posts in authors_vocab.items():
        if not posts:
            continue
            
        if embed_sentences:
            sentences = []
            for text in posts:
                if isinstance(text, (list, tuple)) and len(text) > 0:
                    sentences.extend(sent_tokenize(str(text[0])))
                elif isinstance(text, str):
                    sentences.extend(sent_tokenize(text))
            count = len(sentences)
        else:
            count = len(posts)
        
        if count > 0:
            author_counts.append(count)
    
    print(author_counts[:50])

    if not author_counts:
        print("Warning: No authors with content found - cannot generate distribution plot")
        return
    
    author_counts = np.array(author_counts)
    
    # Get unique content counts and how many authors have each count
    unique_counts, num_authors = np.unique(author_counts, return_counts=True)
    
    # Calculate cumulative distribution
    sorted_counts = np.sort(unique_counts)
    cumulative_counts = [np.sum(author_counts >= c) for c in sorted_counts]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Exact distribution
    ax1.bar(unique_counts, num_authors, edgecolor='white')
    ax1.set_xlabel('Number of ' + ('Sentences' if embed_sentences else 'Posts'))
    ax1.set_ylabel('Number of Authors')
    ax1.set_title('Distribution of Authors by Content Count')
    ax1.set_xlim(0, 5000)

    # Plot 2: Cumulative distribution
    ax2.step(sorted_counts, cumulative_counts, where='post', color='orange')
    ax2.set_xlabel('Minimum Number of ' + ('Sentences' if embed_sentences else 'Posts'))
    ax2.set_ylabel('Number of Authors')
    ax2.set_title('Authors with At Least X Content Items')
    ax2.set_xlim(0, 5000)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'author_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved author distribution plot to {plot_path}")

    # Save data as CSV (clean and binless)
    data_path = os.path.join(output_dir, 'author_distribution.csv')
    pd.DataFrame({
        'content_count': sorted_counts,
        'num_authors': num_authors[np.argsort(unique_counts)],
        'num_authors_at_least': cumulative_counts
    }).to_csv(data_path, index=False)
    print(f"Saved author distribution data to {data_path}")


if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    embed_sentences = args.embed_sentences
    posts_per_author = args.posts_per_author
    output_file = args.output_file_name + '.pkl'
    random_sampling = args.random_sampling

    print("Posts per author: {}".format(posts_per_author))

    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts')
    with open(path_to_data+'social_norms_clean.csv', encoding="utf8") as file:
        social_comments = pd.read_csv(file)

    # creates the verdicts data structure
    print("Creating verdicts data structure")

    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())

    print(DEVICE)
    print(len(authors))



    if 'amit' in args.dirname:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    else:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_demographics)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))

    print("Json files processed")
    print("Number of json files: {}".format(len(filenames)))

    # merge results
    print("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    print(len(authors_vocab))

    plot_author_distribution(authors_vocab, embed_sentences, args.output_dir)

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
    
    if embed_sentences == False:
        print("Embedding posts")
        # NOTE: Process & generate the embeddings for each author for each POST of the author and then average the embeddings together 
        for author, posts in tqdm(authors_vocab.items(), desc="Embedding authors"):
            sys.stdout.flush()
            if len(posts) >= posts_per_author or posts_per_author == -1:

                if random_sampling and len(posts) > posts_per_author and posts_per_author > 0:
                    # Randomly sample posts from the author
                    posts = random.sample(posts, k=posts_per_author)


                processed_texts = [process_tweet(text[0]) for text in posts]

                # Tokenize posts
                batches_text = extract_batches(processed_texts, 64)
                embeddings = []
            
                for batch in batches_text:
                    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                    encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
                    
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                        post_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                        post_embeddings = F.normalize(post_embeddings, p=2, dim=1)
                        embeddings.append(post_embeddings.cpu())
                
                # Concatenate all embeddings before averaging
                all_embeddings = torch.cat(embeddings)
                user_embeddings[author] = all_embeddings.mean(axis=0).numpy()


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
            
            # only process authors with enough sentences
            if len(all_sentences) >= posts_per_author or posts_per_author == -1:
                # random sampling 
                if random_sampling and len(all_sentences) > posts_per_author:
                    # Randomly sample sentences from the author
                    all_sentences = random.sample(all_sentences, k=posts_per_author)

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

    