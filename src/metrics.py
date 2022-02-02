import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack as sparse_vstack


def mean_MRR(eval_df, topk, cutoff=4):
    def single_user_mrr(user_df):
        relevance = (user_df
                     .sort_values('pred_rating', ascending=False)
                     .head(topk)
                     .assign(relevant=lambda tdf: tdf['rating'] >= cutoff)
        )['relevant'].values

        if relevance.max() == 0:
            return 0

        return 1 / (np.where(relevance == 1)[0][0] + 1)

    return eval_df.groupby('user_id').apply(single_user_mrr).mean()



def mean_nDCG(eval_df, topk):
    def DCG(relevance):
        real_topk = min(topk, relevance.shape[0])
        return (relevance[:real_topk] / (np.log(np.arange(2, real_topk + 2)) / np.log(2))).sum()


    def nDCG_single_user(user_df):
        return (DCG(user_df.sort_values(by='pred_rating', ascending=False)['rating'].values)
                / DCG(user_df['rating'].sort_values(ascending=False).values))


    return eval_df.groupby('user_id').apply(nDCG_single_user).mean()


# Abstract evaluator class, each model should implement its own `process_batch` and `prepare_evaluation_dataframe`
class AbstractModelEvaluator:
    def __init__(self):
        self.reset()
        
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                self.process_batch(model, batch, device)

        eval_df = self.prepare_evaluation_dataframe()

        return {
            'RMSE': np.sqrt(((eval_df['rating'] - eval_df['pred_rating']) ** 2).mean()),
            'MRR@5': mean_MRR(eval_df, topk=5, cutoff=cutoff),
            'MRR@10': mean_MRR(eval_df, topk=10, cutoff=cutoff),
            'nDCG@5': mean_nDCG(eval_df, topk=5),
            'nDCG@10': mean_nDCG(eval_df, topk=10)
        }
    
    def reset(self):
        pass
    
    def process_batch(self, model, batch, device):
        raise NotImplementedError('Subclasses must implement process_batch')
        
    def prepare_evaluation_dataframe(self):
        raise NotImplementedError('Subclasses must implement prepare_evaluation_dataframe')


class GMFEvaluator(AbstractModelEvaluator):
    def reset(self):
        self.y = []
        self.y_pred = []
        self.all_user_ids = []
        self.all_item_ids = []
        
    def process_batch(self, model, batch, device):
        user_ids, item_ids, ratings = batch
        self.y.append(ratings)
        self.all_user_ids.append(user_ids)
        self.all_item_ids.append(item_ids)
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device, dtype=torch.float32)
        pred = model((user_ids, item_ids, ratings))
        self.y_pred.append(pred.detach().cpu().numpy())
        
    def prepare_evaluation_dataframe(self):
        y = np.concatenate(self.y)
        y_pred = np.concatenate(self.y_pred)
        all_user_ids = np.concatenate(self.all_user_ids)
        all_item_ids = np.concatenate(self.all_item_ids)
        
        return pd.DataFrame({
            'user_id': all_user_ids,
            'item_id': all_item_ids,
            'rating': y,
            'pred_rating': y_pred
        })

        
class IAutoRecEvaluator(AbstractModelEvaluator):
    def reset(self):
        self.y = None
        self.y_pred = None
    
    def process_batch(self, model, batch, device):
        r, mask = batch
        r_sparse = csr_matrix(r)
        if self.y is None:
            self.y = r_sparse
        else:
            self.y = sparse_vstack([self.y, r_sparse])
            
        r = r.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        pred_sparse = csr_matrix(((model(r) * mask)).detach().cpu().numpy())
        if self.y_pred is None:
            self.y_pred = pred_sparse
        else:
            self.y_pred = sparse_vstack([self.y_pred, pred_sparse])
    
    def prepare_evaluation_dataframe(self, clip=True):
        item_ids, user_ids = self.y.nonzero()
        pred_rating = np.asarray(self.y_pred[item_ids, user_ids])[0]

        if clip:
            pred_rating = np.clip(pred_rating, 1, 5)

        return pd.DataFrame({
            'item_id': item_ids,
            'user_id': user_ids,
            'rating': np.asarray(self.y[item_ids, user_ids])[0],
            'pred_rating': pred_rating
        })

    
class UAutoRecEvaluator(IAutoRecEvaluator):
    def prepare_evaluation_dataframe(self):
        return (super().prepare_evaluation_dataframe()
                .rename(columns={'item_id': 'user_id', 'user_id': 'item_id'}))
