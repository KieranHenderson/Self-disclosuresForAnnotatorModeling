"""
Report how many authors have at least k comments for various k.

"""



import argparse
import pickle as pkl
from collections import defaultdict
import logging
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_comments_per_author(comment_embeddings):
    """Build a mapping of author → list of comment tuples."""
    author_comment_count = defaultdict(int)
    for (author, parent_id, comment_id) in comment_embeddings.keys():
        author_comment_count[author] += 1
    return author_comment_count

def compute_author_counts_by_threshold(author_comment_count, thresholds):
    """Return how many authors have at least k comments for each k in thresholds."""
    results = {}
    for k in thresholds:
        count = sum(1 for n in author_comment_count.values() if n >= k)
        results[k] = count
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_embeddings_path', required=True)
    parser.add_argument('--thresholds', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30])
    args = parser.parse_args()

    logging.info("Loading comment embeddings from %s", args.comment_embeddings_path)
    with open(args.comment_embeddings_path, 'rb') as f:
        comment_embeddings = pkl.load(f)

    logging.info("Counting comments per author...")
    author_comment_count = count_comments_per_author(comment_embeddings)

    logging.info("Calculating number of authors with at least k comments...")
    result = compute_author_counts_by_threshold(author_comment_count, args.thresholds)

    logging.info("Results:")
    for k in args.thresholds:
        logging.info(f"Authors with ≥{k} comments: {result[k]}")
