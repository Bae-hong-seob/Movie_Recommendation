import pandas as pd
import numpy as np

from collections import defaultdict


def AE_loader(config):
    data_dir = config['data_dir']
    
    df = pd.read_csv(data_dir + '/train/train_ratings.csv')
    # # sort df. if you need(it takes times)
    # df.sort_values(by=['user', 'time'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    
    return df


def generate_encoder_decoder(df, column_name) -> dict:
    """
    user, item 번호에 대해 오름차순으로 index 0 부터 순차적으로 재할당
    Args:
        df : pandas dataframe
        col : column_name

    Returns:
        dict: 생성된 user encoder, decoder
    """

    encoder = {}
    decoder = {}
    ids = sorted(df[column_name].unique())
    print(column_name, ids[:15])
    

    for idx, _id in enumerate(ids):
        encoder[_id] = idx
        decoder[idx] = _id

    return encoder, decoder


def generate_sequence_data(config, df) -> dict:
    """
    sequence_data 생성

    Returns:
        dict: train user sequence / valid user sequence
    """
    users = defaultdict(list)
    user_train = {}
    user_valid = {}
    for user, item, time in zip(df['user_idx'], df['item_idx'], df['time']): #user별 item sequence 제작.
        users[user].append(item)
    
    np.random.seed(config['seed'])
    for user in users:
        user_total = users[user]
        valid = np.random.choice(user_total, size = config['valid_samples'], replace = False).tolist() # 전체 sequence 중 중간중간 예측 문제.
        train = list(set(user_total) - set(valid))

        user_train[user] = train
        user_valid[user] = valid # valid_samples 개수 만큼 검증에 활용 (현재 Task와 가장 유사하게)

    return user_train, user_valid #user별 train, valid에 사용할 item 목록


def MakeMatrixDataSet(config, df):
    """
    MatrixDataSet 생성
    """
    item_encoder, item_decoder = generate_encoder_decoder(df, 'item')
    user_encoder, user_decoder = generate_encoder_decoder(df, 'user')
    num_item, num_user = len(item_encoder), len(user_encoder)

    df['item_idx'] = df['item'].apply(lambda x : item_encoder[x])
    df['user_idx'] = df['user'].apply(lambda x : user_encoder[x])

    user_train, user_valid = generate_sequence_data(config, df)
    
    return user_train, user_valid




def get_train_valid_data(self):
    return self.user_train, self.user_valid

def make_matrix(self, user_list, train = True):
    """
    user_item_dict를 바탕으로 행렬 생성
    """
    mat = torch.zeros(size = (user_list.size(0), self.num_item))
    for idx, user in enumerate(user_list):
        if train:
            mat[idx, self.user_train[user.item()]] = 1
        else:
            mat[idx, self.user_train[user.item()] + self.user_valid[user.item()]] = 1
    return mat
    
def AE_process(config, data):
    pass

def AE_split():
    pass