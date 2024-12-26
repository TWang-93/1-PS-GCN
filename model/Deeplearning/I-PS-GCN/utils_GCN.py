import csv
import pandas as pd
import numpy as np
import time

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report

from sklearn import preprocessing

import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss, Linear, ReLU, Dropout
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from rdkit import Chem
from joblib import dump, load



def one_hot_encoding_unk(value, known_list):
    encoding = [0] * (len(known_list) + 1)
    index = known_list.index(value) if value in known_list else -1
    encoding[index] = 1
    return encoding


class featurization_parameters:
    def __init__(self):
        self.max_atomic_num = 100
        self.atom_features = {'atomic_num': list(range(self.max_atomic_num)),
                              'total_degree': [0, 1, 2, 3, 4, 5],
                              'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
                              'total_numHs': [0, 1, 2, 3, 4],
                              'hybridization': [Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2]}

        self.atom_fdim = sum(len(known_list) + 1 for known_list in self.atom_features.values()) + 3
        self.bond_fdim = 6


feature_params = featurization_parameters()


def atom_features(atom: Chem.rdchem.Atom):
    if atom is None:
        atom_feature_vector  = [0] * feature_params.atom_fdim
    else:
        atom_feature_vector  = one_hot_encoding_unk(atom.GetAtomicNum() - 1, feature_params.atom_features['atomic_num']) + \
            one_hot_encoding_unk(atom.GetTotalDegree(), feature_params.atom_features['total_degree']) + \
            one_hot_encoding_unk(atom.GetFormalCharge(), feature_params.atom_features['formal_charge']) + \
            one_hot_encoding_unk(int(atom.GetTotalNumHs()), feature_params.atom_features['total_numHs']) + \
            one_hot_encoding_unk(int(atom.GetHybridization()), feature_params.atom_features['hybridization']) + \
            [1 if atom.IsInRing()else 0]+ \
            [1 if atom.GetIsAromatic() else 0]+\
            [atom.GetMass() * 0.01]
    return atom_feature_vector


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        bond_feature_vector  = [0] * feature_params.bond_fdim
    else:
        bt = bond.GetBondType()
        bond_feature_vector  = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return bond_feature_vector


def process_single_smiles(data_row, label_name):
    smiles = data_row['canonical_smiles']
    mol = Chem.MolFromSmiles(smiles)

    xs = []
    for atom in mol.GetAtoms():
        x = atom_features(atom)
        xs.append(x)
    x = torch.tensor(xs)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs).view(-1, 6)

    y_values = torch.tensor(int(data_row[f'{label_name}']))
    y = y_values.reshape(1, -1)

    mol_fingerprints = data_row.iloc[2:].values.astype(float)
    mol_fingerprints = mol_fingerprints.reshape(1, -1)
    mol_fingerprints = torch.tensor(mol_fingerprints, dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, mol_fingerprints=mol_fingerprints, smiles=smiles)
    return data


def smiles_data_process(dataset, label_name):
    processed_data = []
    for index, row in dataset.iterrows():
        processed = process_single_smiles(row, label_name)
        processed_data.append(processed)
    return processed_data


def validation_GCN(model, model_path, val_loader, DEVICE):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(DEVICE)
            label = data.y.view(-1).long().to(DEVICE)
            out = model(data)

            out_probs = F.softmax(out, dim=1)  # Calculate class probabilities
            pred = out_probs.argmax(dim=1)  # Get predicted labels

            predictions.extend(out_probs.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            true_labels.extend(label.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    return predictions, true_labels, pred_labels


def calculate_metrics_and_save(predictions, true_labels, pred_labels, output_path, model_name, loader_name):
    loader_AUC = roc_auc_score(true_labels, predictions[:,1], average='weighted')
    loader_ACC = accuracy_score(true_labels, pred_labels)
    loader_confusion_matrix = confusion_matrix(true_labels, pred_labels)
    loader_classification_report = classification_report(true_labels, pred_labels, output_dict=True) #['weighted avg']

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


class PTGCN(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(PTGCN, self).__init__()
        self.in_channels = 131
        self.out_channels = 64
        self.conv1 = GCNConv(self.in_channels, self.out_channels)
        self.conv2 = GCNConv(self.out_channels, self.out_channels)


        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(self.out_channels + 2065, 128)
        self.linear2 = nn.Linear(128, 2)


    def forward(self, data):
        x, edge_index, edge_attr, batch_index, mol_fingerprints = data.x, data.edge_index, data.edge_attr, data.batch, data.mol_fingerprints

        x1 = F.relu(self.conv1(x.float(), edge_index))
        x1 = self.conv2(x1.float(), edge_index)
        x1 = gap(x1, batch_index)
        x = torch.cat([x1, mol_fingerprints.reshape(data.num_graphs, 2065)], dim=1)

        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def GCN_process(model_name, label_name, dataset_split_path, model_pred_path, num_epochs, batch_size, lr, w, patience, DEVICE):
    x_train = pd.read_csv(dataset_split_path + 'X_train.csv')  # , dtype='float32'
    x_val = pd.read_csv(dataset_split_path + 'X_val.csv')
    x_test = pd.read_csv(dataset_split_path + 'X_test.csv')
    y_train = pd.read_csv(dataset_split_path + 'y_train.csv')
    y_val = pd.read_csv(dataset_split_path + 'y_val.csv')
    y_test = pd.read_csv(dataset_split_path + 'y_test.csv')
    print('x_train_shape:', x_train.shape)
    print('x_val_shape:', x_val.shape)
    print('x_test_shape:', x_test.shape)
    print('y_train_shape:', y_train.shape)
    print('y_val_shape:', y_val.shape)
    print('y_test_shape:', y_test.shape)
    print('_______________________________________________________')

    scaler = preprocessing.StandardScaler().fit(x_train.iloc[:, 1:])
    x_train_descriptors = scaler.transform(x_train.iloc[:, 1:])
    x_val_descriptors = scaler.transform(x_val.iloc[:, 1:])
    x_test_descriptors = scaler.transform(x_test.iloc[:, 1:])
    dump(scaler, model_pred_path + './Classification_standard_scaler.joblib')

    x_train_descriptors = pd.DataFrame(x_train_descriptors)
    x_val_descriptors = pd.DataFrame(x_val_descriptors)
    x_test_descriptors = pd.DataFrame(x_test_descriptors)

    combined_data_train = pd.concat([x_train.iloc[:, 0], y_train, x_train_descriptors], axis=1)
    combined_data_val = pd.concat([x_val.iloc[:, 0], y_val, x_val_descriptors], axis=1)
    combined_data_test = pd.concat([x_test.iloc[:, 0], y_test, x_test_descriptors], axis=1)
    print('combined_data_train_shape:', combined_data_train.shape)
    print('combined_data_val_shape:', combined_data_val.shape)
    print('combined_data_test_shape:', combined_data_test.shape)
    print('_______________________________________________________')

    train_processed_data = smiles_data_process(combined_data_train, label_name)
    val_processed_data = smiles_data_process(combined_data_val, label_name)
    test_processed_data = smiles_data_process(combined_data_test, label_name)

    train_loader = DataLoader(train_processed_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_processed_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_processed_data, batch_size, shuffle=False)

    model = PTGCN().to(DEVICE)
    class_weights = torch.tensor([1.0, 3.0])
    print(model)
    loss_function = CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w)

    model_path_save = model_pred_path + f'{model_name}_best_model.pt'
    # early_stopper = EarlyStopping(model_path_save, patience=patience, verbose=True)
    # train_with_early_stopping_GCN(model, train_loader, loss_function, optimizer, num_epochs, val_loader, DEVICE,
    #                           early_stopper, model_pred_path, model_name)

    predictions_train, true_train, pred_train = validation_GCN(model, model_path_save, train_loader, DEVICE)
    predictions_val, true_val, pred_val = validation_GCN(model, model_path_save, val_loader, DEVICE)
    predictions_test, true_test, pred_test = validation_GCN(model, model_path_save, test_loader, DEVICE)

    print('train predicted probability size: ', predictions_train.shape)
    print('train predicts class size: ', true_train.shape)
    print('train true class size: ', pred_train.shape)

    calculate_metrics_and_save(predictions_train, true_train, pred_train, model_pred_path, model_name, 'train') # 追加方式添加
    calculate_metrics_and_save(predictions_val, true_val, pred_val, model_pred_path, model_name, 'val')
    calculate_metrics_and_save(predictions_test, true_test, pred_test, model_pred_path, model_name, 'test')

    np.savetxt(model_pred_path + f'Y_pred_train_{model_name}.csv', pred_train, delimiter=',')
    np.savetxt(model_pred_path + f'Y_pred_val_{model_name}.csv', pred_val, delimiter=',')
    np.savetxt(model_pred_path + f'Y_pred_test_{model_name}.csv', pred_test, delimiter=',')


def model_predict_NIR(final_pre_path: object, pred_path: object, label_name: object, model_name: object, DEVICE: object) -> object:
    scaler = load(pred_path + 'Classification_standard_scaler.joblib')
    model_path_save = pred_path + f'{model_name}_best_model.pt'

    x_pre = pd.read_csv(final_pre_path, dtype={'canonical_smiles': str})
    print('1:',x_pre['canonical_smiles'].dtype)
    x_pre_descriptors = scaler.transform(x_pre.iloc[:, 2:])
    x_pre_descriptors = pd.DataFrame(x_pre_descriptors)
    combined_data_pre = pd.concat([x_pre.iloc[:, [0, 1]], x_pre_descriptors], axis=1)
    print('2:',x_pre['canonical_smiles'].dtype)
    pre_processed_data = smiles_data_process(combined_data_pre, label_name)
    pre_loader = DataLoader(pre_processed_data, 32, shuffle=False)

    model = PTGCN().to(DEVICE)
    _, true_pre, pred_pre = validation_GCN(model, model_path_save, pre_loader, DEVICE)
    y_pred = np.array(pred_pre)

    df2 = pd.DataFrame(combined_data_pre, columns=['canonical_smiles'])
    df2['y_pred'] = y_pred
    df2.to_csv(pred_path + f'Y_pred_NIR_{model_name}.csv', index=False)

    count_1 = np.sum(y_pred == 1)
    count_0 = np.sum(y_pred == 0)
    print(f'The number of samples predicted by the {model_name} model to be 1 is: {count_1}')
    print(f'The number of samples predicted by the {model_name} model to be 0 is: {count_0}')
    print('---------------------------------')


