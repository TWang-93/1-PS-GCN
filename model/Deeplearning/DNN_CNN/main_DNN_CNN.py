import torch
from utils_DNN_CNN import model_predict_DNN_CNN
import pandas as pd
print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    num_epochs = 1000
    batch_size = 16
    lr = 1e-4
    w = 1e-4
    patience = 10


    result_pre_path = '../../../data/'
    model_pred_path1 = '../../../pred_data/Result_model_pred_CNN/'
    model_pred_path2 = '../../../pred_data/Result_model_pred_DNN/'


    print('---------------------------New sample predictions---------------------------------')

    final_pre_result = pd.read_csv(result_pre_path + 'data_fp.csv')
    model_predict_DNN_CNN(final_pre_result, model_pred_path1, 'CNN', DEVICE)
    model_predict_DNN_CNN(final_pre_result, model_pred_path2, 'DNN', DEVICE)

