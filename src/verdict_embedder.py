"""
Embedder for verdicts using precomputed embeddings.
"""

import pickle
import torch

class VerdictEmbedder:
    def __init__(self, embeddings_path):
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)

    def embed_verdict(self, verdict_id):
        if verdict_id not in self.embeddings:
            print("skipping!!!!")
            raise KeyError(f"Verdict ID '{verdict_id}' not found in embeddings.")
        return torch.tensor(self.embeddings[verdict_id], dtype=torch.float32)
