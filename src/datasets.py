import os
import torch
import numpy as np
from scipy.sparse import csr_matrix, vstack as sparse_vstack


# Dataset for GMF
class GMFRatingDataset(torch.utils.data.Dataset):
    def __init__(self, ratings_df, n_users, n_items):
        super().__init__()
        self.user_ids = ratings_df['user_id'].values
        self.item_ids = ratings_df['item_id'].values
        self.ratings = ratings_df['rating'].values
        
    def __getitem__(self, index):
        return (self.user_ids[index],
                self.item_ids[index],
                self.ratings[index])
    
    def __len__(self):
        return self.user_ids.shape[0]
    
    

# Dataset for IAutoRec
class IAutoRecRatingDataset(torch.utils.data.Dataset):
    def __init__(self, ratings_df, n_users, n_items):
        super().__init__()
        self.R = np.zeros((n_items, n_users))
        self.R[ratings_df['item_id'].values, ratings_df['user_id'].values] = ratings_df['rating'].values
        self.mask = (self.R > 0).astype(float)

    def __getitem__(self, index):
        return (self.R[index], self.mask[index])

    def __len__(self):
        return self.R.shape[0]

# Dataset for UAutoRec, implemented by transposing the IAutoRec
class UAutoRecRatingDataset(IAutoRecRatingDataset):
    def __getitem__(self, index):
        return (self.R.T[index], self.mask.T[index])

    def __len__(self):
        return self.R.shape[1]

# Generic function to make a dataloader
def make_dataloader(df, dataset_cls, n_users, n_items, batch_size=128,  **kwargs):
    return torch.utils.data.DataLoader(
        dataset_cls(df, n_users, n_items),
        num_workers=0, pin_memory=True,
        # We didn't end up needing the sparse implementation and so we discarded it for speed
        # collate_fn=sparse_batch_collate,
        batch_size=batch_size,
        shuffle=True
    )

# One of our attempted improvements - using a weighted dataloader to prioritize sampling by frequency
def make_weighted_dataloader(df, dataset_cls, n_users, n_items, batch_size=128, **kwargs):
    item_weights = df.groupby('user_id').count()['item_id'].values
    item_weights = (item_weights) / item_weights.sum()
    return torch.utils.data.DataLoader(
        dataset_cls(df, n_users, n_items),
        pin_memory=True, num_workers=0,
        batch_size=batch_size,
        # collate_fn=sparse_batch_collate,
        sampler=torch.utils.data.WeightedRandomSampler(item_weights, item_weights.shape[0])
    )


# Below lies sparse data loading implementation,
# we didn't end up using since we had enough memory and is significantly slower due to
# the conversion from sparse to dense

# def sparse_batch_collate(batch: list):
#     """
#     Collate function which to transform scipy coo matrix to pytorch sparse tensor
#     """
#     batch_elements = zip(*batch)
#     batch_elements_res = []
#     for elem in batch_elements:
#         if type(elem[0]) == csr_matrix:
#             elem = torch.tensor(sparse_vstack(elem).toarray(), dtype=torch.float32)
#         else:
#             elem = torch.tensor(elem)
#
#         batch_elements_res.append(elem)
#
#
#     return tuple(batch_elements_res)

# class IAutoRecRatingDataset(torch.utils.data.Dataset):
#     def __init__(self, ratings_df, n_users, n_items):
#         super().__init__()
#         self.R = csr_matrix((ratings_df['rating'].values, (ratings_df['item_id'].values, ratings_df['user_id'].values)), shape=(n_items, n_users))
#         self.mask = (self.R > 0).astype(float)
#
#     def __getitem__(self, index):
#         return (self.R[index], self.mask[index])
#
#     def __len__(self):
#         return self.R.shape[0]
#
#
# class UAutoRecRatingDataset(torch.utils.data.Dataset):
#     def __init__(self, ratings_df, n_users, n_items):
#         super().__init__()
#         self.R = csr_matrix((ratings_df['rating'].values, (ratings_df['user_id'].values, ratings_df['item_id'].values)), shape=(n_users, n_items))
#         self.mask = (self.R > 0).astype(float)
#
#     def __getitem__(self, index):
#         return (self.R[index], self.R[index])
#
#     def __len__(self):
#         return self.R.shape[0]
