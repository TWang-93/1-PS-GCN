import pandas as pd
import torch
from utils_GCN import GCN_process, model_predict_NIR
print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    num_epochs = 100
    batch_size = 16
    lr = 1e-4
    w = 1e-4
    patience = 10


    data_NIR_path = '../../../data/' + 'data_fp.csv'
    model_pred_path = '../../../pred_data/Result_model_pred_GCN/'

    #-------------Model training--------------------------------------------------
    # GCN_process('GCN', 'label', dataset_path, model_pred_path, num_epochs, batch_size, lr, w, patience, DEVICE)

    #-------------Model prediction--------------------------------------------------
    model_predict_NIR(data_NIR_path, model_pred_path, 'label', 'GCN', DEVICE)


