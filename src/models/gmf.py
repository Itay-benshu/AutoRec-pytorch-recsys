import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items,
                 embedding_dimensions=8, embeddings_init_std=.1, **kwargs):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dimensions)
        self.item_embeddings = nn.Embedding(n_items, embedding_dimensions)
        self.user_embeddings.weight = nn.Parameter(torch.normal(0, embeddings_init_std,
                                                                self.user_embeddings.weight.shape))
        self.item_embeddings.weight = nn.Parameter(torch.normal(0, embeddings_init_std,
                                                                self.item_embeddings.weight.shape))
            
        self.dense = nn.Linear(embedding_dimensions, 1)
        
    def forward(self, x):
        user_indices, item_indices, _ = x
        user_embeddings = self.user_embeddings(user_indices)
        item_embeddings = self.item_embeddings(item_indices)
        x = torch.mul(user_embeddings, item_embeddings)
        return self.dense(x)[:, 0]