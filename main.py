import os
import argparse
import pandas as pd
import yaml


from src.utils import Setting, models_load
from src.dataloader.autoencoder import AE_DataLoader, AE_DataSet

#from src.train import train, valid, test


def main():
    ####################### args
    config_path = './config/autoencoder.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
    
    
    ####################### Setting for Log
    Setting.seed_everything(config['seed'])
    setting = Setting()
    
    ######################## DATA LOAD
    dataloader = AE_DataLoader(config)
    print(f'\n--------------- {config['model']} Load Data ---------------')
    if config['model'] in ('AutoEncoder'):
        #data = AE_loader(config)
        data = dataloader.AE_loader()
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
        train, valid = dataloader.AE_split()
    elif config['model'] in ('DAE'):
        pass
    elif config['model'] in ('VAE'):
        pass
    else:
        pass

    print(f'n_items for first user to train: {len(train[0])}')
    print(f'n_items for first user to valid: {len(valid[0])}')
    
    return
    
    ######################## Make user-item matrix for AE
    print(f'\n--------------- Make user-item matrix for {config['model']} ---------------')
    if config['model'] in ('AutoEncoder'):
        train_mat, valid_mat = dataloader.make_matrix(train), datalodaer.make_matrix(valid)
    elif config['model'] in ('DAE'):
        pass
    elif config['model'] in ('VAE'):
        pass
    else:
        pass




    ######################## MODEL LOAD
    print(f'--------------- {args.model} MODEL LOAD---------------')
    model = models_load(args)
    
    
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