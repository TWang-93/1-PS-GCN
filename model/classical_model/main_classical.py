import pandas as pd
from utils_classical import model_predict
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    dataset_path = '../../data/'
    model_pred_path = '../../pred_data/Result_model_pred_classical/'



    print('---------------------------New sample predictions---------------------------------')

    final_pre_result = pd.read_csv(dataset_path + 'data_fp.csv')

    model_predict(final_pre_result, model_pred_path, 'RF', 'hyperopt')
    model_predict(final_pre_result, model_pred_path, 'SVC', 'hyperopt')
    model_predict(final_pre_result, model_pred_path, 'KNN', 'hyperopt')



















