import pandas as pd
import numpy as np
import torch

from collections import defaultdict


class AE_DataLoader:
    
    def __init__(self, config):
        self.config = config
        self.df = self.AE_loader()
        # # sort df. if you need(it takes times)
        # self.df.sort_values(by=['user', 'time'], inplace=True)
        # self.df.reset_index(drop=True, inplace=True)
        
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder(self.df, 'user')
        self.item_encoder, self.item_decoder = self.generate_encoder_decoder(self.df, 'item')
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['item'].apply(lambda x : self.item_encoder[x])
        self.df['user_idx'] = self.df['user'].apply(lambda x : self.user_encoder[x])
        
        self.user_train, self.user_valid = self.AE_split()
        
            
    def AE_loader(self):
        data_dir = self.config['data_dir']
        
        df = pd.read_csv(data_dir + '/train/train_ratings.csv')
        # # sort df. if you need(it takes times)
        # df.sort_values(by=['user', 'time'], inplace=True)
        # df.reset_index(drop=True, inplace=True)
        
        return df

        
    def generate_encoder_decoder(self, df, column_name) -> dict:
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
        print(f'n_{column_name}:{len(ids)} -> {ids[:20]}')
        

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    
    def AE_split(self) -> dict:
        """
        sequence_data 생성

        Returns:
            train user sequence / valid user sequence
                user_trian / user_valid -> dict
                    - {user1 : [item1, item2, ... ], 
                        user2 : [item10, item 13, ...], 
                        ... 
                        }
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item, time in zip(self.df['user_idx'], self.df['item_idx'], self.df['time']): #user별 item sequence 제작.
            users[user].append(item)
        
        np.random.seed(self.config['seed'])
        for user in users:
            user_total = users[user]
            valid = np.random.choice(user_total, size = self.config['valid_samples'], replace = False).tolist() # 전체 sequence 중 중간중간 예측 문제.
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid # valid_samples 개수 만큼 검증에 활용 (현재 Task와 가장 유사하게)

        return user_train, user_valid #user별 train, valid에 사용할 item 목록


    def make_matrix(self, user_list, train = True):
        """
        Args:
            user_list.shape : [batch_size,1]
                - [[user_number1], [user_number2], ... , [user_number{batch_size}]] 형태로 들어가 있음
                
        Return:
            mat : user-item matrix
        """
        mat = torch.zeros(size = (user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if train:
                mat[idx, self.user_train[user.item()]] = 1
            else:
                mat[idx, self.user_train[user.item()] + self.user_valid[user.item()]] = 1
        return mat




class AE_DataSet:
    def __init__(self, num_user):
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx): 
        user = self.users[idx]
        return torch.LongTensor([user])