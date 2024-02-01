import os
import argparse
import pandas as pd
import yaml


from src.utils import Setting, models_load
#from src.data_preprocess.lightgbm_data import lightgbm_dataloader, lightgbm_preprocess_data, lightgbm_datasplit
#from src.train import train, valid, test


def main():
    ####################### args
    config_path = './autoencoder.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
    
    
    ####################### Setting for Log
    Setting.seed_everything(config['seed'])
    setting = Setting()
    
    ######################## DATA LOAD
    print(f'--------------- {config['model']} Load Data ---------------')
    if config['model'] in ('AutoEncoder'):
        data = gbm_dataloader(args)
    elif config['model'] in ('DAE'):
        data = lightgbm_dataloader(args)
    elif config['model'] in ('VAE'):
        data = catboost_dataloader(args)
    else:
        pass
    
    return 
    ######################## DATA PREPROCESS
    print(f'--------------- {args.model} Data PREPROCESSING---------------')
    if args.model in ('XGB'):
        xgb_data = gbm_dataloader(args)
    elif args.model in ('LIGHTGBM'):
        data = lightgbm_preprocess_data(data)
    elif args.model in ('CATBOOST'):
        catboost_data = catboost_dataloader(args)
    else:
        pass
    print('######################## DATA PREPROCESSING DONE !!!')

    
    ######################## Autogluon
    if args.autogluon == True:
        print(f'--------------- {args.autogluon} ---------------')
        args.model = 'Ensemble'
        train_data, label = data[data['answerCode'] != -1], "answerCode"
        
        print(f'--------------- TRAINING ---------------')
        if args.use_cuda_if_available: #GPU
            predictor = TabularPredictor(label=label, eval_metric="roc_auc", problem_type="binary").fit(train_data, presets=["best_quality"], num_gpus=1)
        else: #CPU
            predictor = TabularPredictor(label=label, eval_metric="roc_auc", problem_type="binary").fit(train_data, presets=["best_quality"])
        
        print(f'--------------- PREDICT ---------------')
        test_data = data[data.dataset == 2]
        test_data = test_data[test_data["userID"] != test_data["userID"].shift(-1)]
        test_data = test_data.drop(["answerCode"], axis=1)
        
        predicts_df = predictor.predict_proba(test_data)
        predicts_df['predicts'] = predicts_df[[0, 1]].max(axis=1)
        predicts = predicts_df['predicts']
        
        try:
            output = predictor.evaluate(train_data, silent=True)
            valid_auc = output["roc_auc"]
        except:
            valid_auc = 0.0000
        

    ######################## Train/Valid Split
    else:
        print(f'--------------- {args.model} Train/Valid Split ---------------')
        if args.model in ('XGB'):
            x_train, y_train, x_valid, y_valid = xgb_datasplit(data)
        elif args.model in ('LIGHTGBM'):
            #data = data.select_dtypes(include=['int', 'float', 'bool']) #lightgbm은 int,flot or bool type변수만 받아들임.
            x_train, y_train, x_valid, y_valid = lightgbm_datasplit(args, data)
        elif args.model in ('CATBOOST'):
            x_train, y_train, x_valid, y_valid = catboost_datasplit(data)
        else:
            pass
        x_test = data[data['answerCode'] == -1].drop(["answerCode"], axis=1)
        print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_valid: {x_valid.shape}, y_valid: {y_valid.shape}, test:{x_test.shape}')
    
    
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