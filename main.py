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

class vae_loss_fuction:
    def __init__(self, recon_x, x, mu, logvar):
        self.recon_x = recon_x
        self.x = x
        self.mu, self.logvar = mu, logvar
        
    def forward():
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))    

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


def main():
    ####################### configs
    print()
    model_name = input('모델을 선택하세요(AE | DAE | Multi-DAE | VAE | Multi-VAE) :')
    #config_path = './config/autoencoder.yaml'
    config_path = './config/' + model_name +'.yaml'
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
    if config['model'] in ('AutoEncoder', 'DAE', 'Multi-DAE', 'VAE', 'Multi-VAE'):
        data = ae_dataloader.AE_loader()
    else:
        raise Exception('check model name on config file ')
    print(data.head(5))
    
    ######################## Train/Valid Split
    print(f'\n--------------- {config['model']} Train/Valid Split ---------------')
    if config['model'] in ('AutoEncoder', 'DAE', 'Multi-DAE', 'VAE', 'Multi-VAE'):
        train_dict, valid_dict = ae_dataloader.AE_split()
    else:
        raise Exception('check model name on config file ')

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
    
    
    ######################## Loss functions
    if config['loss_function'] == 'MSE':
        criterion = nn.MSELoss()
    elif config['loss_function'] == 'BCE':
        criterion = nn.BCELoss()
    elif config['loss_function'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Loss function Error. please check config file")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    ######################## TRAIN
    print(f'--------------- {config['model']} TRAINING ---------------')
    best_hit = 0
    save_time = str(setting.save_time)
    for epoch in range(1, config['epochs'] + 1):
        tbar = tqdm(range(1))
        for _ in tbar:
            train_loss = train(config, model, criterion, optimizer, data_loader, make_matrix_data_set = ae_dataloader)
            ndcg, hit = evaluate(config=config, model = model, data_loader = data_loader, user_train = train_dict, user_valid = valid_dict, make_matrix_data_set = ae_dataloader)

            if best_hit < hit:
                best_hit = hit
                torch.save(model.state_dict(), os.path.join(config['model_path'], save_time + '_' + config['model'] + '.pt'))

            tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')
    
    ######################## INFERENCE
    print(f'--------------- {config['model']} PREDICT ---------------')
    submission_data_loader = DataLoader(AE_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory = True, num_workers = config['num_workers'])
    model.load_state_dict(torch.load(os.path.join(config['model_path'], save_time + '_' + config['model'] + '.pt')))
    
    user2rec_list = predict(config= config, model = model, data_loader = submission_data_loader,user_train = train_dict, user_valid = valid_dict, make_matrix_data_set = ae_dataloader)
    
    
    ######################## SAVE PREDICT
    print(f'\n--------------- SAVE PREDICT ---------------')
    with open('record.txt', 'a') as f:
        f.write(f"Tiemstamp:{setting.save_time}, Recall@10:{best_hit}\n")
    f.close()
    
    predicts = []
    users = [i for i in range(0, ae_dataloader.num_user)]
    for user in users:
        rec_item_list = user2rec_list[user]
        for item in rec_item_list:
            if user==0:
                print(f'user:{user} -> {ae_dataloader.user_decoder[user]}, item:{item} -> {ae_dataloader.item_decoder[item]}')
            predicts.append(
                {   
                    'user' : ae_dataloader.user_decoder[user],
                    'item' : ae_dataloader.item_decoder[item],
                }
            )
            
    filename = setting.get_submit_filename(config, best_hit)
    
    submission = pd.DataFrame(predicts)
    submission.to_csv(filename, index=False)
    print('make csv file !!! ', filename)


if __name__ == "__main__":
    main()