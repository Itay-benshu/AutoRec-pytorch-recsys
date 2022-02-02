if __name__ == "__main__":
 import os
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 import torch
 import time
 from sklearn.model_selection import train_test_split
 import json
 from src.models.gmf import GMF
 from src.models.autorec import AutoRec
 from src.models.autorec_v2 import AutoRecV2
 from src.training import GMFTrainer, AutoRecTrainer
 from src.datasets import make_dataloader, GMFRatingDataset, IAutoRecRatingDataset, UAutoRecRatingDataset, \
  make_weighted_dataloader
 from src.metrics import GMFEvaluator, IAutoRecEvaluator, UAutoRecEvaluator
 from src.hyperparameter_optimization import hyperparameter_tune
 from sklearn.model_selection import KFold
 load_existing = False
 N_REPEATS = 5
 splits_random_state = np.random.RandomState(1000)

 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print('Using device:', device)

 df = pd.read_csv('data/ml_1m_preprocessed.csv')
 print(df.shape)

 n_users = df['user_id'].max() + 1
 n_items = df['item_id'].max() + 1
 test_eval_results = []
 # # for rep in range(N_REPEATS):
 train_val_df, test_df = train_test_split(df, test_size=.1, random_state=splits_random_state)
 train_users = train_val_df['user_id'].unique()
 # val_users = validation_df['user_id'].unique()
 test_users = test_df['user_id'].unique()
 train_items = train_val_df['item_id'].unique()
 # val_items = validation_df['item_id'].unique()
 test_items = test_df['item_id'].unique()
 iautorec_ml_hyperparams = {'hidden_size': 512, 'reg_lambda': 0.1, 'init_std': 0.03, 'batch_size': 128, 'lr':1e-3}
 train_dataloader = make_dataloader(train_val_df, UAutoRecRatingDataset, n_users, n_items, **iautorec_ml_hyperparams)
 # validation_dataloader = make_dataloader(validation_df, IAutoRecRatingDataset, n_users, n_items, batch_size=100)
 test_dataloader = make_dataloader(test_df, UAutoRecRatingDataset, n_users, n_items, **iautorec_ml_hyperparams)

 # TODO: REMOVE
 # from data_preprocessor_tf import *
 # import time
 # import argparse
 #
 # current_time = time.time()
 #
 # parser = argparse.ArgumentParser(description='I-AutoRec ')
 # parser.add_argument('--hidden_neuron', type=int, default=500)
 # parser.add_argument('--lambda_value', type=float, default=0.1)
 #
 # parser.add_argument('--train_epoch', type=int, default=2000)
 # parser.add_argument('--batch_size', type=int, default=100)
 #
 # parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
 # parser.add_argument('--grad_clip', type=bool, default=True)
 # parser.add_argument('--base_lr', type=float, default=1e-3)
 # parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")
 #
 # parser.add_argument('--random_seed', type=int, default=1000)
 # parser.add_argument('--display_step', type=int, default=1)
 #
 # args = parser.parse_args()
 # np.random.seed(args.random_seed)
 #
 # data_name = 'ml-1m';
 # num_users = 6040;
 # num_items = 3952;
 # num_total_ratings = 1000209;
 # train_ratio = 0.9
 # path = "./data_tf/%s" % data_name + "/"
 #
 # result_path = './results/' + data_name + '/' + str(args.random_seed) + '_' + str(args.optimizer_method) + '_' + str(
 #  args.base_lr) + "_" + str(current_time) + "/"
 # R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
 # user_train_set, item_train_set, user_test_set, item_test_set \
 #  = read_rating(path, num_users, num_items, num_total_ratings, 1, 0, train_ratio)

 ## END REMOVE

 if load_existing:
   autorec = torch.load('trained_models/iautorec.pt')
 else:
  autorec = AutoRec(n_items, **iautorec_ml_hyperparams).to(device)
  autorec_trainer = AutoRecTrainer(train_dataloader, test_dataloader, autorec, device, reg=0, gradient_clipping=True, n_epochs=2000, patience=0, verbose=True, **iautorec_ml_hyperparams)
  # autorec_trainer = AutoRecTrainer((train_R, train_mask_R), (test_R, test_mask_R), autorec, device, reg=0, gradient_clipping=True, n_epochs=2000, patience=0, verbose=True, **iautorec_ml_hyperparams)
  thist, vhist, thist_rme, vhist_rme = autorec_trainer.train()
  torch.save(autorec, f'trained_models/iautorec_{time.time()}.pt')

 test_eval_results.append(IAutoRecEvaluator().evaluate(autorec, test_dataloader, device, train_users, train_items, test_users, test_items, cutoff=5))
 print(pd.DataFrame(test_eval_results).mean())

 iautorec_ml_hyperparams = {'hidden_size': 512,
  'reg_lambda': 1,
  'lr': 1e-3,
  'reg': 0,
  'init_std': 0.01}

 # iautorec_ml_hyperparams, iautorec_trials_df = (
 #  hyperparameter_tune('IAutoRec', {
 #   'hidden_size': (int, (64, 512), False),
 #   'reg_lambda': (float, (1e-4, 1e2), False),
 #   'init_std': (float, (0.01, 0.1), True),
 #   'batch_size': (int, (8, 128), False)
 #  }, train_val_df, lr=1e-3, n_trials=100, patience=20, n_epochs=200, verbose=True)
 # )
 #
 # print(iautorec_ml_hyperparams)
 # with open('hyperparams/iautorec.json', 'w') as f:
 #  json.dump(iautorec_ml_hyperparams, f, indent=4)
 #
 # iautorec_trials_df.to_csv('hyperparams/iautorec_trials.csv', index=False)

