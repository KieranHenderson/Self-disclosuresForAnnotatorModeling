import os
import glob
import pickle
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from nltk.tokenize import sent_tokenize
import nltk
from torch.cuda.amp import autocast
from torch.multiprocessing import Pool
import torch.multiprocessing as mp

# Disable tokenizer parallelism (causes issues with multiprocessing)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('punkt', quiet=True)

# Local imports
from dataset import SocialNormDataset
from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *

# Enable GPU optimization flags
torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention if available
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    # Argument parsing with H100-optimized defaults
    parser = ArgumentParser()
    parser.add_argument("--path_to_data", required=True, type=str)
    parser.add_argument("--bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
    parser.add_argument("--dirname", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--embed_sentences", type=str2bool, default=False)
    parser.add_argument("--posts_per_author", type=int, default=5)
    parser.add_argument("--output_file_name", type=str, default="user_embeddings_topk")
    parser.add_argument("--batch_size", type=int, default=2048)  # H100 can handle large batches
    parser.add_argument("--encode_batch_size", type=int, default=8192)  # For text processing
    parser.add_argument("--num_workers", type=int, default=min(32, (os.cpu_count() or 1) * 2))
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    args = parser.parse_args()

    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data - optimized with path joining
    data_path = partial(os.path.join, args.path_to_data)
    social_chemistry = pd.read_pickle(data_path('social_chemistry_clean_with_fulltexts'))
    social_comments = pd.read_csv(data_path('social_norms_clean.csv'), encoding="utf8", engine='c')

    # Initialize dataset
    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())

    # Initialize model with FP16 support
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    tokenizer.backend_tokenizer = tokenizer.backend_tokenizer.normalizer = None  # Disable slow tokenizer features
    
    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)
    model.eval()
    model = torch.compile(model)  # Enable PyTorch 2.0 compilation

    # Process author vocab files in parallel with large batch size
    filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        process_fn = partial(extract_authors_vocab_AMIT, authors=authors)
        results = list(tqdm(
            executor.map(process_fn, filenames, chunksize=4),
            total=len(filenames),
            desc="Processing author vocab files"
        ))

    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    # Check for existing partial results
    user_embeddings = {}
    last_checkpoint_embeddings = set()
    existing_partials = sorted(glob.glob(os.path.join(args.output_dir, f"{args.output_file_name}_partial_*.pkl")))
    
    if existing_partials:
        latest = existing_partials[-1]
        with open(latest, 'rb') as f:
            user_embeddings = pickle.load(f)
        last_checkpoint_embeddings = set(user_embeddings.keys())
        resume_from = int(latest.split("_partial_")[-1].split(".pkl")[0])
    else:
        resume_from = 0

    # Pre-process all texts upfront
    def preprocess_all_texts():
        """Cache all processed texts to avoid redundant processing"""
        text_cache = {}
        
        # Process post texts
        for post_id, text in tqdm(dataset.postIdToText.items(), desc="Preprocessing posts"):
            if text and text.strip():
                processed = process_tweet(text)
                if args.embed_sentences:
                    text_cache[post_id] = [process_tweet(s) for s in sent_tokenize(processed) if s.strip()]
                else:
                    text_cache[post_id] = processed
        
        # Process author texts
        for author, posts in tqdm(authors_vocab.items(), desc="Preprocessing author texts"):
            for text, _, parent_id in posts:
                if text and text.strip():
                    key = (author, parent_id)
                    processed = process_tweet(text)
                    if args.embed_sentences:
                        text_cache[key] = [process_tweet(s) for s in sent_tokenize(processed) if s.strip()]
                    else:
                        text_cache[key] = processed
        
        return text_cache

    text_cache = preprocess_all_texts()

    # Optimized encoding functions with FP16 and large batches
    def encode_batch(text_batch):
        """Process a batch of texts with mixed precision"""
        encoded = tokenizer(text_batch, padding=True, truncation=True, 
                          return_tensors='pt', max_length=512).to(DEVICE, non_blocking=True)
        with torch.no_grad(), autocast():
            output = model(**encoded)
            pooled = mean_pooling(output, encoded['attention_mask'])
            return F.normalize(pooled, p=2, dim=1).cpu()

    def encode_texts(texts):
        """Process texts in optimized batches"""
        embeddings = []
        for i in range(0, len(texts), args.encode_batch_size):
            batch = texts[i:i + args.encode_batch_size]
            embeddings.append(encode_batch(batch))
        return torch.cat(embeddings, dim=0)

    # Parallel processing of verdicts
    def process_verdict_batch(batch_args):
        """Process a batch of verdicts in parallel"""
        verdict_batch, post_emb, post_id = batch_args
        batch_results = {}
        
        for verdict_id in verdict_batch:
            author = dataset.verdictToAuthor.get(verdict_id)
            if not author:
                continue
                
            # Get all author posts (excluding current post)
            filtered = []
            for text, _, parent_id in authors_vocab.get(author, []):
                if parent_id == post_id:
                    continue
                cache_key = (author, parent_id)
                if cache_key in text_cache:
                    filtered.append((text_cache[cache_key], parent_id))
            
            if not filtered:
                continue
                
            # Prepare texts for embedding
            if args.embed_sentences:
                sentences = []
                for text, _ in filtered:
                    if isinstance(text, list):  # Already pre-tokenized sentences
                        sentences.extend(text)
                    else:
                        sentences.extend([s for s in sent_tokenize(text) if s.strip()])
                if not sentences:
                    continue
                comment_embs = encode_texts(sentences)
            else:
                texts = [text if isinstance(text, str) else ' '.join(text) for text, _ in filtered]
                comment_embs = encode_texts(texts)
            
            # Compute similarities and get top-k
            with autocast():
                sims = util.cos_sim(post_emb, comment_embs).squeeze(0)
                k = min(args.posts_per_author, len(comment_embs))
                topk_indices = torch.topk(sims, k=k).indices
                batch_results[verdict_id] = comment_embs[topk_indices].mean(dim=0).float().numpy()  # Ensure FP32 output
        
        return batch_results

    # Main processing loop
    fulltext_map = dataset.postIdToText
    post_ids = list(dataset.postToVerdicts.keys())
    
    for idx in tqdm(range(resume_from, len(post_ids)), desc="Processing posts", initial=resume_from):
        post_id = post_ids[idx]
        cached_text = text_cache.get(post_id)
        
        if not cached_text:
            continue
            
        # Encode the post
        if args.embed_sentences:
            if not cached_text:  # No valid sentences
                continue
            post_emb = encode_texts(cached_text).mean(dim=0, keepdim=True)
        else:
            post_emb = encode_texts([cached_text])
        
        # Split verdicts into parallel batches
        verdicts = list(dataset.postToVerdicts[post_id])
        batch_size = max(1, len(verdicts) // args.num_workers)
        verdict_batches = [verdicts[i:i + batch_size] for i in range(0, len(verdicts), batch_size)]
        
        # Process in parallel
        with Pool(processes=min(args.num_workers, len(verdict_batches))) as pool:
            batch_args = [(batch, post_emb, post_id) for batch in verdict_batches]
            for result in pool.imap_unordered(process_verdict_batch, batch_args):
                user_embeddings.update(result)
        
        # Periodic cleanup and checkpointing
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            
        if (idx + 1) % args.checkpoint_interval == 0:
            # Only save new embeddings since last checkpoint
            new_embeddings = {k: v for k, v in user_embeddings.items() 
                            if k not in last_checkpoint_embeddings}
            if new_embeddings:
                checkpoint_path = os.path.join(args.output_dir, f"{args.output_file_name}_partial_{idx + 1}.pkl")
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(new_embeddings, f)
                last_checkpoint_embeddings.update(new_embeddings.keys())

    # Save final results
    output_path = os.path.join(args.output_dir, f"{args.output_file_name}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(user_embeddings, f)

if __name__ == '__main__':
    main()