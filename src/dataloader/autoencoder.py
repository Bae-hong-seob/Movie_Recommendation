import pandas as pd

def AE_loader(config):
    data_dir = config['data_dir']
    
    data = pd.read_csv(data_dir + 'train_rating.csv')
    return data