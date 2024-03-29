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
    
    model_path = './saved_models/20240202_135646_AutoEncoder.pt'
    model.load_state_dict(torch.load(model_path))
    
    ndcg, best_hit = evaluate(config, model, data_loader = data_loader, user_train = train_dict, user_valid = valid_dict, make_matrix_data_set = ae_dataloader)

    ######################## INFERENCE
    print(f'--------------- {config['model']} PREDICT ---------------')
    torch.set_printoptions(profile="full") #torch내용 전체 출력
    submission_data_loader = DataLoader(AE_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory = True, num_workers = config['num_workers'])    
    user2rec_list = predict(config, model, data_loader = submission_data_loader,user_train = train_dict, user_valid = valid_dict, make_matrix_data_set = ae_dataloader)
    
    output = input('출력 파일 생성하시겠습니까? (y/n): ')
    if output == 'y':
        ######################## SAVE PREDICT
        print(f'--------------- SAVE PREDICT ---------------')
        submission = []
        users = [i for i in range(0, ae_dataloader.num_user)]
        for user in users:
            rec_item_list = user2rec_list[user]
            for item in rec_item_list:
                if user==0:
                    print(f'user:{user} -> {ae_dataloader.user_decoder[user]}, item:{item} -> {ae_dataloader.item_decoder[item]}')
                submission.append(
                    {   
                        'user' : ae_dataloader.user_decoder[user],
                        'item' : ae_dataloader.item_decoder[item],
                    }
                )
        filename = setting.get_submit_filename(config, best_hit)
        
        submission = pd.DataFrame(submission)
        submission.to_csv(filename, index=False)
        print('make csv file !!! ', filename)
    
if __name__ == "__main__":
    main()