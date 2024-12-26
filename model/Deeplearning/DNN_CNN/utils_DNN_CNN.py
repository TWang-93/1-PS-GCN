import pandas as pd
import numpy as np
import time

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import preprocessing

import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from joblib import dump, load


class DNNDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        my_data = self.data[idx, :].clone().detach()
        label = self.labels[idx]
        return my_data, label


class CNNDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        my_data = self.data[idx, :].clone().detach()
        my_data = my_data.unsqueeze(0)
        label = self.labels[idx]
        return my_data, label


class DNNModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(DNNModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(2065, 128)
        self.linear2 = nn.Linear(128, 2)


    def forward(self, x):
        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
                                  nn.MaxPool1d(kernel_size=2))
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                                  nn.MaxPool1d(kernel_size=2))

        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(514 * 32, 128)
        self.linear2 = nn.Linear(128, 2)


    def forward(self, x):
        x2 = F.relu(self.cnn1(x))
        x2 = self.cnn2(x2)
        x = x2.view(x2.size(0), -1)


        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EarlyStopping:
    def __init__(self, model_path, patience, delta=0.001, verbose=True, trace_func=print):
        self.path = model_path
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        self.counter = 0
        self.best_epoch = 0
        self.best_score = None
        self.early_stop = False
        self.best_accuracy = -np.Inf

    def __call__(self, val_accuracy, model, epoch):
        if self.best_score is None:
            self.best_score = val_accuracy
            self.save_checkpoint(val_accuracy, model)
            self.best_epoch = epoch
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.save_checkpoint(val_accuracy, model)
            self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model):
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.best_accuracy:.4f} --> {val_accuracy:.4f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_accuracy = val_accuracy


def calculate_metrics_and_save(predictions, true_labels, pred_labels, output_path, model_name, loader_name):
    loader_AUC = roc_auc_score(true_labels, predictions[:, 1], average='weighted')
    loader_ACC = accuracy_score(true_labels, pred_labels)
    loader_confusion_matrix = confusion_matrix(true_labels, pred_labels)
    loader_classification_report = classification_report(true_labels, pred_labels, output_dict=True)

    print('___________________________')
    print(f'{model_name}_{loader_name}_AUC:', loader_AUC)
    print(f'{model_name}_{loader_name}_ACC:', loader_ACC)
    print(f'{model_name}_{loader_name}_confusion_matrix:\n', loader_confusion_matrix)
    print(f'{model_name}_{loader_name}_classification_report:\n', loader_classification_report)

    metrics = {
        f'{loader_name}_ACC': loader_ACC,
        f'{loader_name}_AUC': loader_AUC,
        f'{loader_name}_confusion_matrix': loader_confusion_matrix,
        f'{loader_name}_classification_report': loader_classification_report}

    with open(output_path + f'{model_name}_metrics.txt', 'a') as file:
        for key, value in metrics.items():
            if key in [f'{loader_name}_confusion_matrix', f'{loader_name}_classification_report']:
                file.write(f"{key}:\n{value}\n")
            else:
                file.write(f"{key}: {value}\n")


def train_with_early_stopping_DNN_CNN(model, train_loader, criterion, optimizer, num_epochs, val_loader, DEVICE, early_stopper, output_path, model_name):
    print('_______________________________________Model start training_________________________________________________')
    model.to(DEVICE)
    train_loss = []
    validation_loss = []
    train_precision = []
    validation_precision = []
    best_epoch = 0

    t1 = time.time()

    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0.0
        correct_train = 0
        total_train = 0
        tp = 0
        fp = 0

        for batch_idx, (feature, label) in enumerate(train_loader):
            features = feature.to(DEVICE)
            label = label.view(-1).long().to(DEVICE)

            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            out_probs = F.softmax(out, dim=1)
            predicted = out_probs.argmax(dim=1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()

            tp += ((predicted == label) & (predicted == 1)).sum().item()
            fp += ((predicted == 1) & (label != predicted)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        train_precision.append(precision)
        train_loss.append(sum_loss / len(train_loader))

        if val_loader is not None:
            model.eval()
            val_sum_loss = 0.0
            tp_val = 0
            fp_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_idx, (feature, label) in enumerate(val_loader):
                    features = feature.to(DEVICE)
                    label = label.view(-1).long().to(DEVICE)

                    out = model(features)

                    val_loss = criterion(out, label)
                    val_sum_loss += val_loss.item()

                    out_probs = F.softmax(out, dim=1)
                    predicted = out_probs.argmax(dim=1)
                    total_val += label.size(0)

                    tp_val += ((predicted == label) & (predicted == 1)).sum().item()
                    fp_val += ((predicted == 1) & (label != predicted)).sum().item()

            validation_precision_value = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
            validation_precision.append(validation_precision_value)
            validation_loss.append(val_sum_loss / len(val_loader))

            print(
                'Epoch: {} - Train Loss: {:.4f}, Train Precision: {:.4f}, Validation Loss: {:.4f}, Validation Precision: {:.4f}'
                .format(epoch, train_loss[-1], train_precision[-1], validation_loss[-1], validation_precision[-1]))

            if early_stopper is not None:
                early_stopper(validation_precision[-1], model, epoch)
                if early_stopper.early_stop:
                    print(f'Early stopping at epoch {epoch + 1}')
                    best_epoch = early_stopper.best_epoch
                    break

    t2 = time.time()
    print('Total time: {:.2f}min'.format((t2 - t1) / 60))
    print(f"Best model was saved at epoch: {best_epoch}")

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, len(train_loss) + 1), train_loss, "r-", label="Training Loss")
    plt.plot(range(1, len(validation_loss) + 1), validation_loss, "b-", label="Validation Loss")
    plt.legend()
    plt.savefig(output_path + f'{model_name}_loss.png')
    plt.close()

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1, len(train_precision) + 1), train_precision, "r-", label="Train precision")
    plt.plot(range(1, len(validation_precision) + 1), validation_precision, "b-", label="Validation precision")
    plt.legend()
    plt.savefig(output_path + f'{model_name}_precision.png')
    plt.close()

    train_loss = pd.DataFrame(train_loss)
    validation_loss = pd.DataFrame(validation_loss)
    train_precision = pd.DataFrame(train_precision)
    validation_precision = pd.DataFrame(validation_precision)
    train_loss.to_csv(output_path + f'{model_name}_train_loss.csv', index=False)
    validation_loss.to_csv(output_path + f'{model_name}_validation_loss.csv', index=False)
    train_precision.to_csv(output_path + f'{model_name}_train_precision.csv', index=False)
    validation_precision.to_csv(output_path + f'{model_name}_validation_precision.csv', index=False)


def validation_DNN_CNN(model, model_path, val_loader, DEVICE):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(val_loader):
            features = feature.to(DEVICE)
            label = label.view(-1).long().to(DEVICE)
            out = model(features)

            out_probs = F.softmax(out, dim=1)  # Calculate class probabilities
            pred = out_probs.argmax(dim=1)  # Get predicted labels

            predictions.extend(out_probs.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            true_labels.extend(label.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    return predictions, true_labels, pred_labels


def DNN_CNN_process(model_name, dataset_split_path, model_pred_path, num_epochs, batch_size, lr, w, patience, DEVICE):
    x_train = pd.read_csv(dataset_split_path + 'X_train.csv')
    x_val = pd.read_csv(dataset_split_path + 'X_val.csv')
    x_test = pd.read_csv(dataset_split_path + 'X_test.csv')
    y_train = pd.read_csv(dataset_split_path + 'y_train.csv')
    y_val = pd.read_csv(dataset_split_path + 'y_val.csv')
    y_test = pd.read_csv(dataset_split_path + 'y_test.csv')


    x_train = x_train.iloc[:, :]
    x_val = x_val.iloc[:, :]
    x_test = x_test.iloc[:, :]
    print('x_train_shape:', x_train.shape)
    print('x_val_shape:', x_val.shape)
    print('x_test_shape:', x_test.shape)
    print('y_train_shape:', y_train.shape)
    print('y_val_shape:', y_val.shape)
    print('y_test_shape:', y_test.shape)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    y_test = y_test.values.ravel()
    dump(scaler, model_pred_path + f'Classification_standard_scaler.joblib')



    print('x_train_shape:', x_train.shape)
    print('y_train_shape:', y_train.shape)
    print('x_val_shape:', x_val.shape)
    print('y_val_shape:', y_val.shape)

    train_loader, val_loader, test_loader = data_convert(x_train, x_test, x_val, y_train, y_test, y_val, batch_size, model_name)

    if model_name == 'CNN':
        model = CNNModel().to(DEVICE)
    else:
        model = DNNModel().to(DEVICE)

    class_weights = torch.tensor([1.0, 3.0])
    print(model)
    loss_function = CrossEntropyLoss(weight=class_weights.to(DEVICE))
    # loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w)

    model_path_save = model_pred_path + f'{model_name}_best_model.pt'
    # early_stopper = EarlyStopping(model_path_save, patience=patience, verbose=True)
    # train_with_early_stopping_DNN_CNN(model, train_loader, loss_function, optimizer, num_epochs, val_loader, DEVICE,
    #                               early_stopper, model_pred_path, model_name)

    predictions_train, true_train, pred_train = validation_DNN_CNN(model, model_path_save, train_loader, DEVICE)
    predictions_val, true_val, pred_val = validation_DNN_CNN(model, model_path_save, val_loader, DEVICE)
    predictions_test, true_test, pred_test = validation_DNN_CNN(model, model_path_save, test_loader, DEVICE)

    print('train prediction probability：', predictions_train.shape)
    print('train predicts class size：', true_train.shape)
    print('train True class size：', pred_train.shape)

    calculate_metrics_and_save(predictions_train, true_train, pred_train, model_pred_path, model_name, 'train')  # 追加方式添加
    calculate_metrics_and_save(predictions_val, true_val, pred_val, model_pred_path, model_name, 'val')
    calculate_metrics_and_save(predictions_test, true_test, pred_test, model_pred_path, model_name, 'test')

    np.savetxt(model_pred_path + f'Y_pred_train_{model_name}.csv', pred_train, delimiter=',')
    np.savetxt(model_pred_path + f'Y_pred_val_{model_name}.csv', pred_val, delimiter=',')
    np.savetxt(model_pred_path + f'Y_pred_test_{model_name}.csv', pred_test, delimiter=',')


def data_convert(x_train, x_test, x_val, y_train, y_test, y_val, batch_size, model_name):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    if model_name == 'CNN':
        train_set = CNNDataset(x_train, y_train)
        val_set = CNNDataset(x_val, y_val)
        test_set = CNNDataset(x_test, y_test)
    else:
        train_set = DNNDataset(x_train, y_train)
        val_set = DNNDataset(x_val, y_val)
        test_set = DNNDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    examples = enumerate(train_loader)
    batch_idx, (data1, labels) = next(examples)
    print(data1.shape)
    return train_loader, val_loader, test_loader

def model_predict_DNN_CNN(final_pre_result, pred_path, model_name, DEVICE):
    scaler = load(pred_path + 'Classification_standard_scaler.joblib')
    data_fingerprints = final_pre_result.iloc[:, 2:]
    y_pre = final_pre_result.iloc[:, 1]

    x_pre = scaler.transform(data_fingerprints)
    x_pre = torch.tensor(x_pre, dtype=torch.float32)
    y_pre = torch.tensor(y_pre, dtype=torch.long)

    if model_name == 'CNN':
        pre_set = CNNDataset(x_pre, y_pre)
    else:
        pre_set = DNNDataset(x_pre, y_pre)
    pre_loader = DataLoader(pre_set, batch_size=len(pre_set), shuffle=False)

    if model_name == 'CNN':
        model = CNNModel().to(DEVICE)
    else:
        model = DNNModel().to(DEVICE)

    model_path_save = pred_path + f'{model_name}_best_model.pt'
    _, true_pre, pred_pre = validation_DNN_CNN(model, model_path_save, pre_loader, DEVICE)

    df2 = pd.DataFrame(final_pre_result, columns=['canonical_smiles'])
    df2['y_pred'] = pred_pre
    df2.to_csv(pred_path + f'Y_pred_NIR_{model_name}.csv', index=False)

    count_1 = np.sum(pred_pre == 1)
    count_0 = np.sum(pred_pre == 0)
    print(f'The number of samples predicted by the {model_name} model to be 1 is: {count_1}')
    print(f'The number of samples predicted by the {model_name} model to be 0 is: {count_0}')
    print('---------------------------------')

