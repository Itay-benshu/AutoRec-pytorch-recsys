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

def IAutoRec_improved_initiator(n_users, n_items, **params):
    return AutoRecV2(n_users, **params)

# Handy function to get all initiators for a specific model name
def get_model_stack(model_type):
    # Returns (model_initiator, dataset_cls, trainer_cls)
    if model_type.lower() == 'gmf':
        return GMF_initiator, GMFRatingDataset, GMFTrainer
    elif model_type.lower() == 'iautorec':
        return IAutoRec_initiator, IAutoRecRatingDataset, AutoRecTrainer
    elif model_type.lower() == 'uautorec':
        return UAutoRec_initiator, UAutoRecRatingDataset, AutoRecTrainer
    elif model_type.lower() == 'improvediautorec':
        return IAutoRec_improved_initiator, IAutoRecRatingDataset, AutoRecTrainer
    elif model_type.lower() == 'improveduautorec':
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


gmf_mf_eval = {'RMSE': 0.8659259573941469,
 'MRR@5': 0.6353056207397497,
 'MRR@10': 0.638713024599718,
 'nDCG@5': 0.9289927814968972,
 'nDCG@10': 0.9441377562483331}

gmf_am_eval = {'RMSE': 0.5784171881737233,
 'MRR@5': 0.8382522350613095,
 'MRR@10': 0.8382522350613095,
 'nDCG@5': 0.9971709071975884,
 'nDCG@10': 0.9974267198786393}


uautorec_ml_eval = {'RMSE': 0.8192990078229133,
 'MRR@5': 0.7509637948374125,
 'MRR@10': 0.7518802978784547,
 'nDCG@5': 0.9688686559990156,
 'nDCG@10': 0.972794925508752}

uautorec_am_eval = {'RMSE': 0.756502392462996,
 'MRR@5': 0.8379130004593798,
 'MRR@10': 0.8379262518110177,
 'nDCG@5': 0.9974158297761423,
 'nDCG@10': 0.9976254812831239}

iautorec_ml_eval = {'RMSE': 0.9927281432205904,
 'MRR@5': 0.5662587998659091,
 'MRR@10': 0.572391035976355,
 'nDCG@5': 0.9022123022913984,
 'nDCG@10': 0.9236118942305}

iautorec_am_eval = {'RMSE': 0.5854881584424756,
 'MRR@5': 0.8308208770627935,
 'MRR@10': 0.8309069898145481,
 'nDCG@5': 0.995897327720598,
 'nDCG@10': 0.9962836874344124}

uautorec_v2_ml_eval = {'RMSE': 0.8080586643045211,
 'MRR@5': 0.7638702648340602,
 'MRR@10': 0.7648229517530565,
 'nDCG@5': 0.9713054469641719,
 'nDCG@10': 0.9745145959005442}

uautorec_v2_am_eval = {'RMSE': 0.7488325980190045,
 'MRR@5': 0.8341884872256969,
 'MRR@10': 0.834234551448057,
 'nDCG@5': 0.996717835821169,
 'nDCG@10': 0.9969636630910041}

iautorec_v2_ml_eval = {'RMSE': 0.9389745496089833,
 'MRR@5': 0.6019974298804359,
 'MRR@10': 0.606772371480721,
 'nDCG@5': 0.9218078474967385,
 'nDCG@10': 0.9402874886153664}

iautorec_v2_am_eval = {'RMSE': 0.595808531324841,
 'MRR@5': 0.8403724513233681,
 'MRR@10': 0.8403724513233681,
 'nDCG@5': 0.9979896292687614,
 'nDCG@10': 0.9980563822752868}
