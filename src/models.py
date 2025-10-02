"""
Definition of models used in training and inference.
"""

from pyexpat import model
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F

class SentBertClassifier(nn.Module):
    def __init__(self, users_layer=False, demo_layer=False, user_dim=768, 
                 num_outputs=2, sbert_dim=384, 
                 sbert_model='sentence-transformers/all-MiniLM-L6-v2', dropout_rate=0.2):
        super().__init__()
        print(f"Initializing with user_layer={users_layer}, demo_layer={demo_layer}")
        self.model = AutoModel.from_pretrained(sbert_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(sbert_dim, sbert_dim//2)
        self.users_layer = users_layer
        self.demo_layer = demo_layer
        
        if users_layer:
            # Handle user embedding transformation
            self.user_linear1 = nn.Linear(user_dim, user_dim//2)  # Changed to use user_dim
            
            if demo_layer:
                self.demo_linear1 = nn.Linear(user_dim, user_dim//2)  # Assuming demo has same dim as user
                comb_in_dim = (sbert_dim//2) + (user_dim//2)*2
            else:
                comb_in_dim = (sbert_dim//2) + (user_dim//2)
                
            self.combine_linear = nn.Linear(comb_in_dim, comb_in_dim//2)
            self.linear2 = nn.Linear(comb_in_dim//2, num_outputs)
        else:
            self.linear2 = nn.Linear(sbert_dim//2, num_outputs)
            
        self.relu = nn.ReLU()
        
    def forward(self, input, users_embeddings=None, demo_embeddings=None):
        # Process text through SBERT
        bert_output = self.model(**input)
        pooled_output = self.mean_pooling(bert_output, input['attention_mask'])
        downsized_output = self.linear1(self.dropout(pooled_output))
        text_output = self.relu(downsized_output)
        
        if self.users_layer:
            # Validate user embeddings
            if users_embeddings is None:
                raise ValueError("users_embeddings required when users_layer=True")
                
            # Process user embeddings
            users_output = self.relu(self.user_linear1(self.dropout(users_embeddings)))
            
            # Process demo embeddings if enabled
            if self.demo_layer:
                if demo_embeddings is None:
                    raise ValueError("demo_embeddings required when demo_layer=True")
                demo_output = self.relu(self.demo_linear1(self.dropout(demo_embeddings)))
                combined = torch.cat([text_output, users_output, demo_output], dim=1)
            else:
                combined = torch.cat([text_output, users_output], dim=1)
            
            # Process combined features
            output = self.relu(self.combine_linear(combined))
        else:
            output = text_output
            
        # Final output
        return self.linear2(self.dropout(output))
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
