import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def top_k_indices(matrix, k):
    # Get the column indices of the top k values for each row
    top_k_indices = np.argsort(matrix, axis=1)[:, -k:]

    return top_k_indices

def get_ndcg(pred_list, true_list):
    ndcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            ndcg += 1 / np.log2(rank + 2)
    return ndcg

# hit == recall == precision
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit


def train(model, criterion, optimizer, data_loader, make_matrix_data_set):

    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(users)
        mat = mat.to(device)
        recon_mat = model(mat)

        optimizer.zero_grad()
        loss = criterion(recon_mat, mat)
        
        loss_val += loss.item()

        loss.backward()
        optimizer.step()
    
    loss_val /= len(data_loader)

    return loss_val

def evaluate(config, model, data_loader, user_train, user_valid, make_matrix_data_set):
    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    with torch.no_grad():
        for users in data_loader:
            '''
                users shape : [batch_size,1]
                    - [[user_number1], [user_number2], ... , [user_number{batch_size}]] 형태로 들어가 있음.
                    - user.item()은 [user_number1]을 user_number(float)으로 꺼내는 작업
            '''
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(device)

            recon_mat = model(mat)
            recon_mat[mat == 1] = -np.inf
            rec_items_per_user = top_k_indices(recon_mat, config['Recall@K'])

            for user, rec in zip(users, rec_items_per_user):
                uv = user_valid[user.item()]
                up = rec.cpu().numpy().tolist()
                NDCG += get_ndcg(pred_list = up, true_list = uv)
                HIT += get_hit(pred_list = up, true_list = uv)

    NDCG /= len(data_loader.dataset)
    HIT /= len(data_loader.dataset)

    return NDCG, HIT

def predict(config, model, data_loader, user_train, user_valid, make_matrix_data_set):
    model.eval()
    
    user2rec_list = {}
    i = 0
    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users, train = False)
            mat = mat.to(device)

            recon_mat = model(mat)
            recon_mat = recon_mat.softmax(dim = 1)
            recon_mat[mat == 1] = -1.
            rec_items_per_user = top_k_indices(recon_mat, config['Recall@K'])
            if i == 0:
                print(f'user:{len(users)}, rec_items_per_user: {rec_items_per_user.shape}')
                print(f'first user recommend items: {rec_items_per_user[0]}')
                i+=1

            for user, rec in zip(users, rec_items_per_user):
                up = rec.cpu().numpy().tolist()
                user2rec_list[user.item()] = up
    
    return user2rec_list