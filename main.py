import os
import argparse
import pandas as pd
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils import Setting, models_load
from src.dataloader.autoencoder import AE_DataLoader, AE_DataSet

from train.trainer import train, evaluate, predict



def main():
    ####################### configs
    config_path = './config/autoencoder.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
    
    os.makedirs(name=config['model_path'], exist_ok=True)

    
    ####################### Setting for Log
    Setting.seed_everything(config['seed'])
    setting = Setting()
    
    ######################## DATA LOAD
    ae_dataloader = AE_DataLoader(config)
    print(f'\n--------------- {config['model']} Load Data ---------------')
    if config['model'] in ('AutoEncoder'):
        #data = AE_loader(config)
        data = ae_dataloader.AE_loader()
    elif config['model'] in ('DAE'):
        pass
    elif config['model'] in ('VAE'):
        pass
    else:
        pass
    print(data.head(5))
    
    ######################## Train/Valid Split
    print(f'\n--------------- {config['model']} Train/Valid Split ---------------')
    if config['model'] in ('AutoEncoder'):
        train_dict, valid_dict = ae_dataloader.AE_split()
    elif config['model'] in ('DAE'):
        pass
    elif config['model'] in ('VAE'):
        pass
    else:
        pass

    print(f'n_items for first user to train: {len(train_dict[0])}')
    print(f'n_items for first user to valid: {len(valid_dict[0])}')
    
    ######################## Make DataLoader for pytorch
    print(f'\n--------------- {config['model']} Make DataLoader for pytorch ---------------')
    AE_dataset = AE_DataSet(num_user = ae_dataloader.num_user)
    data_loader = DataLoader(AE_dataset, batch_size = config['batch_size'], shuffle = True, pin_memory = True, num_workers = config['num_workers'])
    
    ######################## MODEL LOAD
    print(f'--------------- {config['model']} MODEL LOAD---------------')
    dims = [ae_dataloader.num_item] + config['dims'] #num_item = 6,807
    model = models_load(config, dims)
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    ######################## TRAIN
    print(f'--------------- {config['model']} TRAINING ---------------')
    best_hit = 0
    for epoch in range(1, config['epochs'] + 1):
        tbar = tqdm(range(1))
        for _ in tbar:
            train_loss = train(model = model, criterion = criterion, optimizer = optimizer, data_loader = data_loader, make_matrix_data_set = ae_dataloader)
            ndcg, hit = evaluate(model = model, data_loader = data_loader, user_train = train_dict, user_valid = valid_dict, make_matrix_data_set = ae_dataloader)

            if best_hit < hit:
                best_hit = hit
                torch.save(model.state_dict(), os.path.join(config['model_path'], config['model']))

            tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')
    
    return
    
    ######################## Make user-item matrix for AE
    print(f'\n--------------- Make user-item matrix for {config['model']} ---------------')
    if config['model'] in ('AutoEncoder'):
        train_mat, valid_mat = ae_dataloader.make_matrix(train), ae_dataloader.make_matrix(valid)
    elif config['model'] in ('DAE'):
        pass
    elif config['model'] in ('VAE'):
        pass
    else:
        pass

    
    
    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model, valid_auc, valid_acc = train(args, model, x_train, y_train, x_valid, y_valid, setting)
    
    if args.feature_selection:
        print(f'--------------- {args.model} FEATURE SELECT ---------------')
        feature_selected_model = models_load(args)
        model, x_valid, x_test = feature_selection(args, model, feature_selected_model, x_train, y_train, x_valid, y_valid, x_test, setting)
        valid_auc, valid_acc = valid(args, model, x_valid, y_valid)
        print(f"VALID AUC : {valid_auc} VALID ACC : {valid_acc}\n")
    else:
        pass
    

    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, x_test)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE PREDICT ---------------')
    with open('record.txt', 'a') as f:
        f.write(f"Tiemstamp:{setting.save_time}, valid auc:{valid_auc}, valid_acc:{valid_acc}\n")
    f.close()
    
    filename = setting.get_submit_filename(args, valid_auc)
    submission = pd.read_csv(args.data_dir + 'sample_submission.csv')
    submission['prediction'] = predicts

    submission.to_csv(filename, index=False)
    print('make csv file !!! ', filename)

if __name__ == "__main__":
    main()