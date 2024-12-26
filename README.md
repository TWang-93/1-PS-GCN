# Type I NIR photosensitizer discovered by machine learning for enhanced tumor photodynamic therapy

## 1. Introduction
This is the code for " Type I NIR photosensitizer discovered by machine learning for enhanced tumor photodynamic therapy " paper. We designed a deep learning model named 1-PS-GCN to achieve high-throughput screening of type I photosensitizers.
## 2. Directory Structure
```sh 
├── data/            
├── model/  
├── pred_data/ 
├── screen/        
├── LICENSE.txt        
└── README.md       
```
* The ``data`` folder contains the original datasets used in this project, which can be utilized for model training or prediction.
* The ``model`` folder stores the 1-PS-GCN model and other comparison models used in this study.  
* The ``pred_data`` folder is used to store the model's prediction results and the best-performing models.  
* The ``screen`` folder is specifically used for the gradient screening of type I photosensitizers.  

## 3. Installation & Usage
Here we provide the detailed step-by-step instructions to ensure you can easily install and configure the project.
### 3.1 Prerequisites
The configuration environment requirements for this project are:
- Python version: 3.8+
- Required libraries: numpy, pandas, torch, rdkit, etc. For more details, please see requirements.txt.
### 3.2 Installation Steps
(1) Clone the Repository  
Run the following command to clone the project from GitHub platform to local:
```sh 
git clone https://github.com/TWang-93/1-PS-GCN.git
```
(2) Install Dependencies  
Running this  command for installing dependency: 
```sh 
pip install -r requirements.txt
```
### 3.3 Running the Project
To train the 1-PS-GCN model or predict Type I photosensitizers, simply run the script located at ./model/Deeplearning/I-PS-GCN/main_GCN.py.

## 4. Example
The following are the main functions of the 1-PS-GCN model, which you can find in the path mentioned in Section 3.3. We provide the best trained models that can be loaded directly and used for predictions. To run the model for predictions, simply comment out the model training section and execute the prediction script. If you wish to train a model, the same approach applies.
```shell
import pandas as pd
import torch
from utils_GCN import GCN_process,model_predict_NIR
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
```
## 5. Copyright and License
The project of 1-PS-GCN follows [MIT License](LICENSE). Please read the license carefully before use.

## 6. Contact 
If you encounter any problems or need support when using 1-PS-GCN, you can contact us through the following methods:
- Author: Tong Wang
- Email: wangtong@hnu.edu.cn
- GitHub: TWang-93