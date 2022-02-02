import torch
import gc
import optuna
from sklearn.model_selection import train_test_split

from src.models.autorec_v2 import AutoRecV2
from src.training import GMFTrainer, AutoRecTrainer
from src.datasets import make_dataloader, make_weighted_dataloader
from src.models.gmf import GMF
from src.models.autorec import AutoRec
from src.datasets import GMFRatingDataset, IAutoRecRatingDataset, UAutoRecRatingDataset


def GMF_initiator(n_users, n_items, **params):
    return GMF(n_users, n_items, **params)


def IAutoRec_initiator(n_users, n_items, **params):
    return AutoRec(n_users, **params)


def UAutoRec_initiator(n_users, n_items, **params):
    return AutoRec(n_items, **params)

def UAutoRec_improved_initiator(n_users, n_items, **params):
    return AutoRecV2(n_items, **params)

# Handy function to get all initiators for a specific model name
def get_model_stack(model_type):
    # Returns (model_initiator, dataset_cls, trainer_cls)
    if model_type.lower() == 'gmf':
        return GMF_initiator, GMFRatingDataset, GMFTrainer
    elif model_type.lower() == 'iautorec':
        return IAutoRec_initiator, IAutoRecRatingDataset, AutoRecTrainer
    elif model_type.lower() == 'uautorec':
        return UAutoRec_initiator, UAutoRecRatingDataset, AutoRecTrainer
    elif model_type.lower() == 'improvedautorec':
        return UAutoRec_improved_initiator, UAutoRecRatingDataset, AutoRecTrainer
    else:
        raise ValueError(f'Unknown model type {model_type}')


def hyperparameter_tune(model_type, param_grid, df, n_trials=1, batch_size=128, verbose=False,
                        enqueue_trials=None,
                        **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, validation_df = train_test_split(df, test_size=.15)
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    
    def objective(trial):
        trial_funcs = {
            int: trial.suggest_int,
            float: trial.suggest_float
        }
        
        params = {
            param_name: trial_funcs[param_type](param_name, *param_range, log=log)
            for (param_name, (param_type, param_range, log))
            in param_grid.items()
        }

        # Initializations
        model_initiator, dataset_cls, trainer_cls = get_model_stack(model_type)
        train_dataloader = make_dataloader(train_df, dataset_cls, n_users, n_items, batch_size=batch_size, **params)
        validation_dataloader = make_dataloader(validation_df, dataset_cls, n_users, n_items, batch_size=batch_size, **params)
        model = model_initiator(n_users, n_items, **params).to(device)
        trainer = trainer_cls(train_dataloader, validation_dataloader, model,
                              device, verbose=verbose, **params, **kwargs)

        # Training
        train_hist, validation_hist, train_rmse_hist, validation_rmse_hist = trainer.train()
        rmse = validation_rmse_hist[-1]
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return rmse
    
    
    study = optuna.create_study(direction='minimize')
    if enqueue_trials is not None:
        for trial in enqueue_trials:
            study.enqueue_trial(trial)
    
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.trials_dataframe()
