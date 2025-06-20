from pyexpat import model
# from turtle import forward
#from importlib_metadata import Deprecated
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F


# class MLPAttribution(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
        
#     def forward(self, input):
#         output = self.relu(self.linear1(input))
#         return self.linear2(F.dropout(output, p=0.2, training=self.training))


# class JudgeBert(nn.Module):
#     def __init__(self , dim=768):
#         super().__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.dropout = nn.Dropout(0.1)
#         self.linear = nn.Linear(dim, 2)
        
    
#     def forward(self, input):
#         bert_output = self.bert(**input)
#         output = self.linear(self.dropout(bert_output.pooler_output))
#         return output
    
    
#     def size(self):
#         return sum(p.numel() for p in self.parameters())


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
        
#     def forward(self, input):
#         output = self.relu(self.linear1(input))
#         return self.linear2(F.dropout(output, p=0.2, training=self.training))


# class SentBertClassifier(nn.Module):
#     def __init__(self, users_layer=False, demo_layer=False, user_dim=768, 
#                  num_outputs=2, sbert_dim=384, 
#                  sbert_model='sentence-transformers/all-MiniLM-L6-v2'):
#         super().__init__()
#         print("Initializing with user layer set to {}".format(users_layer))
#         self.model = AutoModel.from_pretrained(sbert_model)
#         self.dropout = nn.Dropout(0.2)
#         self.linear1 = nn.Linear(sbert_dim, sbert_dim//2)
#         self.users_layer = users_layer
#         self.demo_layer = demo_layer
        
#         if users_layer:
#             if user_dim > 768:
#                 user_out_dim = user_dim
#             else:
#                 user_out_dim = user_dim // 10

#             # NOTE: issue with mat mul, dim is half the size of what
#             # it should be, so CHANGED FROM user_dim to sbert_dim
#             # out = in x W^T + b where W is weight mat size (user_out_dim, sbert_dim)
#             # b is bias vec of size (user_out_dim)
#             self.user_linear1 = nn.Linear(sbert_dim, user_out_dim)
#             self.demo_linear1 = nn.Linear(sbert_dim, user_out_dim)

#             comb_in_dim = sbert_dim//2 + user_out_dim
#             if demo_layer:
#                 comb_in_dim += user_out_dim
#             self.comb_in_dim = comb_in_dim
#             self.combine_linear = nn.Linear(comb_in_dim, comb_in_dim // 2) # Note might need to change dim if using demo embeddings
#             self.linear2 = nn.Linear(comb_in_dim // 2, num_outputs)
#         else:
#             self.linear2 = nn.Linear(sbert_dim//2, num_outputs)      
            
#         self.relu = nn.ReLU()
        
        
#     def forward(self, input, users_embeddings=None, demo_embeddings=None):
#         bert_output = self.model(**input)
#         pooled_output = self.mean_pooling(bert_output, input['attention_mask'])
#         downsized_output = self.linear1(self.dropout(pooled_output))
#         output = self.relu(downsized_output)
        
#         if self.users_layer:
#             # NOTE: dropout regularization (set fraction of input elements to 0 with prob p)
#             # -> activation func (rectified linear unit)
#             # -> user_linear1 = torch.nn.Linear layer where linear transf
#             # applied to users_embeddings: output = user_embeddings x W^T + b
#             users_output =  self.dropout(self.relu(self.user_linear1(users_embeddings)))
#             text_output = self.dropout(output) # TODO: check if this is needed

#             if demo_embeddings is not None:
#                 demo_output = self.dropout(self.relu(self.demo_linear1(demo_embeddings)))
#                 combined = torch.cat([text_output, users_output, demo_output], dim=1)
#             else:
#                 combined = torch.cat([text_output, users_output], dim=1)


#             output = self.relu(self.combine_linear(combined))
            
#         output = self.linear2(self.dropout(output))
#         return output
    
    
#     def size(self):
#         return sum(p.numel() for p in self.parameters())

        
#     #Mean Pooling - Take attention mask into account for correct averaging
#     def mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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


# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout, heads, concat=False):
#         super().__init__()
#         self.dropout = dropout
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=self.dropout, concat=concat)
#         # self.conv1 = GCNConv(in_channels, hidden_channels)
#         if concat:
#             self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
#                                 concat=False, dropout=self.dropout)
#         else:
#             self.conv2 = GATConv(hidden_channels, hidden_channels // 2, heads=1,
#                                 concat=False, dropout=self.dropout)
            
#     def forward(self, x, edge_index):
#         # x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x