#!/usr/bin/env python
# coding: utf-8
import os
import torch
import random
import logging
import Sampling
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from Model_Architecture import TransformerFeatureExtractor, MLP_Predictor
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score



##############################################
#              Hyperparameters               #
##############################################
Drug = "Bortezomib".upper()

# File Path
Bulk_label_path = "Data/Cell_Line_Label.csv"
Bulk_data_path = "Data/Cell_Line_Pathway_Active_Score.csv"
Sc_data_path = "Data/GSE114461/Pathway_Active_Score.csv"

# Transformer
T_sample = "UPsampling"                        # sampling strategy: NOsampling, DOWNsampling, UPsampling, SMOTEsampling.
T_batch_size = 150                             # the number of data samples included in each batch
T_lr = 1e-4                                    # learning rate of Transformer
T_dropout = 0                                  # dropout
T_weight_decay = 0                             # weight decay
T_embedding_dim = 1316                         # embedded dimension
T_Bottleneck_dim = 512                         # extracted dimension
T_heads = 4                                    # the number of parallel attention calculations
T_layers = 1                                   # the number of layers in the Transformer encoder

# Predictor
P_lr = 1e-4                                    # learning rate of Predictor
P_dropout = 0.3                                # dropout of Predictor
P_weight_decay = 0                             # weight decay of Predictor
P_input_dim = 256                              # input dimension of Predictor
P_hidden_dim = [64, 32]                        # hidden dimension of Predictor
P_output_dim = 2                               # output dimension of Predictor

# Training Configuration
patience = 10                                  # the period of early stopping
lr_patience = 5                                # the period of learning rate adjustment
num_epochs = 1000                              # the number of iterations
train_test_size = 0.2                          # Proportion of the dataset reserved for testing
train_valid_size = 0.25                        # Proportion of the training set reserved for validation

# General Settings
seed = 24   
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional GPU warning
if device.type == "cpu":
    print("Warning: CUDA device not available, using CPU.")



##############################################
#             Generate Save Path             #
##############################################
directories = ["Result/Bulk_Model", "Result/Bulk_Result", "Result/Sc_Result"]
for path in directories:
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Output directory '{path}' is created!")
        else:
            print(f"Output directory '{path}' already exists!")
    except OSError as e:
        print(f"An error occurred while creating directory '{path}': {e}")
        
# Set up logging configuration
logging.basicConfig(filename="Result/Output.log", level=logging.INFO)



##############################################
#                 Read Data                  #
##############################################
# Read Bulk Data
Bulk_data = pd.read_csv(Bulk_data_path, index_col=0)
Bulk_label = pd.DataFrame(pd.read_csv(Bulk_label_path, index_col=0)[Drug])
Bulk_label[Drug] = Bulk_label[Drug].map({'sensitive': 1, 'resistant': 0})

# Read Sc data
Sc_data = pd.read_csv(Sc_data_path, index_col=0)

# intersection
common = Bulk_data.columns.intersection(Sc_data.columns)
Bulk_data = Bulk_data[common]
Sc_data = Sc_data[common]
Sc_names = Sc_data.index

# MinMaxScale
mmscaler = preprocessing.MinMaxScaler()
Bulk_data = mmscaler.fit_transform(Bulk_data)
Sc_data = mmscaler.fit_transform(Sc_data)



##############################################
#             Divide Bulk Datase             #
##############################################
X_Bulk_train_valid, X_Bulk_test, y_Bulk_train_valid, y_Bulk_test = train_test_split(
    Bulk_data, Bulk_label, test_size=train_test_size, random_state=seed)
X_Bulk_train, X_Bulk_valid, y_Bulk_train, y_Bulk_valid = train_test_split(
    X_Bulk_train_valid, y_Bulk_train_valid, test_size=train_valid_size, random_state=seed)

if T_sample == "NOsampling":
    X_Bulk_train_sample, y_Bulk_train_sample = Sampling.no_sampling(X_Bulk_train, y_Bulk_train)

if T_sample == "DOWNsampling":
    X_Bulk_train_sample, y_Bulk_train_sample = Sampling.down_sampling(X_Bulk_train, y_Bulk_train)

if T_sample == "UPsampling":
    X_Bulk_train_sample, y_Bulk_train_sample = Sampling.up_sampling(X_Bulk_train, y_Bulk_train)

if T_sample == "SMOTEsampling":
    X_Bulk_train_sample, y_Bulk_train_sample = Sampling.smote_sampling(X_Bulk_train, y_Bulk_train)

# Convert to Tensor
Bulk_data_tensor = torch.tensor(np.array(Bulk_data), dtype=torch.float32).to(device)
Bulk_label_tensor = torch.tensor(np.array(Bulk_label), dtype=torch.float32).to(device)
Sc_data_tensor = torch.tensor(np.array(Sc_data), dtype=torch.float32).to(device)

X_Bulk_train_tensor = torch.tensor(np.array(X_Bulk_train_sample), dtype=torch.float32).to(device)
X_Bulk_valid_tensor = torch.tensor(np.array(X_Bulk_valid), dtype=torch.float32).to(device)
X_Bulk_test_tensor  = torch.tensor(np.array(X_Bulk_test), dtype=torch.float32).to(device)

y_Bulk_train_tensor = torch.tensor(np.array(y_Bulk_train_sample), dtype=torch.float32).to(device)
y_Bulk_valid_tensor = torch.tensor(np.array(y_Bulk_valid), dtype=torch.float32).to(device)
y_Bulk_test_tensor  = torch.tensor(np.array(y_Bulk_test), dtype=torch.float32).to(device)

# Creating Datasets and DataLoaders
train_Bulk_dataset = TensorDataset(X_Bulk_train_tensor, y_Bulk_train_tensor)
valid_Bulk_dataset = TensorDataset(X_Bulk_valid_tensor, y_Bulk_valid_tensor)

train_Bulk_loader = DataLoader(train_Bulk_dataset, batch_size=T_batch_size, shuffle=True)
valid_Bulk_loader = DataLoader(valid_Bulk_dataset, batch_size=T_batch_size, shuffle=False)

dataloaders_Bulk = {"train":train_Bulk_loader, "val":valid_Bulk_loader}

# Initialize Transformer model
Bulk_Transformer = TransformerFeatureExtractor(Bulk_data.shape[1],
                                                T_embedding_dim,
                                                T_Bottleneck_dim,
                                                T_heads, 
                                                T_layers, 
                                                T_dropout)
Bulk_Transformer.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Bulk_Transformer.parameters(), lr=T_lr, weight_decay=T_weight_decay)

# Early Stopping
best_val_loss = np.inf
counter = 0
train_loss_all, val_loss_all = [], []

# Train Transformer Model
for epoch in range(num_epochs):
    Bulk_Transformer.train()
    train_loss = 0.0
    for x, y in train_Bulk_loader:
        optimizer.zero_grad()
        _, output = Bulk_Transformer(x) 
        loss = criterion(output, x)
        loss.backward() 
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_Bulk_loader.dataset)
    train_loss_all.append(train_loss)
    
    Bulk_Transformer.eval()
    val_loss = 0.0
    for x, y in valid_Bulk_loader:
        _, output = Bulk_Transformer(x) 
        loss = criterion(output, x)
        val_loss += loss.item() * x.size(0)
    val_loss /= len(valid_Bulk_loader.dataset)
    val_loss_all.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Check whether the performance of the model has improved, and if not, save the model and exit.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save model weights
        torch.save(Bulk_Transformer, "Result/Bulk_Model/Transformer.pth")
        counter = 0
    else:
        counter = counter + 1
    
    # whether to stop training.
    if counter >= patience:
        print("Early Stopping!")
        break

plt.clf()
plt.title("train loss")
plt.plot(train_loss_all)
plt.savefig("Result/Bulk_Result/Trasnformer_train_loss.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.clf()
plt.title("val loss")
plt.plot(val_loss_all)
plt.savefig("Result/Bulk_Result/Trasnformer_val_loss.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

# Read a pre-trained model
Bulk_Transformer = torch.load("Result/Bulk_Model/Transformer.pth")
Bulk_Transformer.eval()

# Define Predictor
Bulk_Predictor = MLP_Predictor(T_Bottleneck_dim, P_input_dim, P_hidden_dim, P_output_dim, P_dropout)
Bulk_Predictor.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer and add regularization.
parameters = itertools.chain(Bulk_Transformer.parameters(), Bulk_Predictor.parameters())
optimizer = Adam(parameters, lr=P_lr, weight_decay=P_weight_decay)

# Early Stopping
best_val_loss = float("inf")
counter = 0
val_loss_all = []

# Adjust the learning rate of model training
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=lr_patience, verbose=True)

# Train Predictor
for epoch in range(num_epochs):
    Bulk_Transformer.train()
    Bulk_Predictor.train()
    train_loss = 0.0
    for x, y in train_Bulk_loader:
        y = y.long()
        y = y.squeeze()
        optimizer.zero_grad()
        feature, output = Bulk_Transformer(x) 
        outputs = Bulk_Predictor(feature)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_Bulk_loader.dataset)
    
    Bulk_Transformer.eval()
    Bulk_Predictor.eval()
    val_loss = 0.0
    y_pred = []
    for x, y in valid_Bulk_loader:
        y = y.long()
        y = y.squeeze()
        feature, output = Bulk_Transformer(x) 
        outputs = Bulk_Predictor(feature)
        loss = criterion(outputs, y)
        val_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        y_pred.extend(preds.cpu().flatten())
    val_loss /= len(valid_Bulk_loader.dataset)
    accuracy = accuracy_score(y_Bulk_valid, y_pred)
    val_loss_all.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Check whether the performance of the model has improved, and if not, save the model and exit.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save model weights
        torch.save(Bulk_Predictor, "result/Bulk_Model/Predictor.pth")
        torch.save(Bulk_Transformer, "result/Bulk_Model/Transformer.pth")
        counter = 0
    else:
        counter += 1
    
    # whether to stop training.
    if counter >= patience:
        print("Early stopping!")
        break

plt.clf()
plt.title("val loss")
plt.plot(val_loss_all)
plt.savefig("Result/Bulk_Result/Predictor_val_loss.png", dpi=300, bbox_inches="tight", pad_inches=0.1)



##############################################
#             Evaluate Bulk Data             #
##############################################
Bulk_Transformer = torch.load("Result/Bulk_Model/Transformer.pth")
Bulk_Transformer.to(device)
Bulk_Transformer.eval()

Bulk_Predictor = torch.load("Result/Bulk_Model/Predictor.pth")
Bulk_Predictor.to(device)
Bulk_Predictor.eval()

# Evaluate Bulk Train Dataset
feature, output = Bulk_Transformer(X_Bulk_train_tensor)
outputs = Bulk_Predictor(feature)
y_pred = outputs.argmax(dim=1).cpu().numpy()
Y = y_Bulk_train_tensor.cpu().numpy()
Accuracy = accuracy_score(Y, y_pred)
print("Bulk_Train_Accuracy = ", Accuracy)
logging.info("Bulk_Train_Accuracy = %s", Accuracy)

# Evaluate Valid Dataset
feature, output = Bulk_Transformer(X_Bulk_valid_tensor)
outputs = Bulk_Predictor(feature)
y_pred = outputs.argmax(dim=1).cpu().numpy()
Y = y_Bulk_valid_tensor.cpu().numpy()
Accuracy = accuracy_score(Y, y_pred)
print("Bulk_Valid_Accuracy = ", Accuracy)
logging.info("Bulk_Valid_Accuracy = %s", Accuracy)

# Evaluate test dataset
feature, output = Bulk_Transformer(X_Bulk_test_tensor)
outputs = Bulk_Predictor(feature)
y_pred = outputs.argmax(dim=1).cpu().numpy()
Y = y_Bulk_test_tensor.cpu().numpy()
sen_score = outputs[:,1].detach().cpu().numpy()

Accuracy = accuracy_score(Y, y_pred)
AUC = roc_auc_score(np.array(Y), np.array(sen_score))
AP = average_precision_score(np.array(Y), np.array(sen_score))

report_dict = classification_report(Y, y_pred, output_dict=True)
report_dict = pd.DataFrame(report_dict)
report_dict.loc["Overall performance"] = [""] * len(report_dict.columns)
report_dict.loc["AUC"] = AUC
report_dict.loc["AP"] = AP
report_dict.to_csv("Result/Bulk_Result/result.csv")

Recall = report_dict["weighted avg"]["recall"]
Precision = report_dict["weighted avg"]["precision"]
f1 = report_dict["weighted avg"]["f1-score"]

print("Evaluate Bulk test dataset")
print("Accuracy = ", round(Accuracy, 3))
print("AUC = ", round(AUC, 3))
print("AP = ", round(AP, 3))
print("Recall = ", round(Recall, 3))
print("Precision = ", round(Precision, 3))
print("F1 score:", round(f1, 3))
logging.info("Evaluate Bulk test dataset")
logging.info("Accuracy = %s", round(Accuracy, 3))
logging.info("AUC = %s", round(AUC, 3))
logging.info("AP = %s", round(AP, 3))
logging.info("Recall = %s", round(Recall, 3))
logging.info("Precision = %s", round(Precision, 3))
logging.info("F1 score = %s", round(f1, 3))



##############################################
#            Predict Target Label            #
##############################################
logging.info("Start：Predict Target Label！")
feature, output = Bulk_Transformer(Sc_data_tensor)
outputs = Bulk_Predictor(feature)
sensitive_score = outputs[:,1].detach().cpu().numpy()
sensitive_score = (sensitive_score - min(sensitive_score)) / (max(sensitive_score) - min(sensitive_score))
y_pred = pd.DataFrame((sensitive_score > 0.5).astype(int))

# Save
sensitive_score = pd.DataFrame(sensitive_score)
sensitive_score.index = Sc_names
sensitive_score.to_csv('Result/Sc_Result/sensitive_score.csv')
y_pred.to_csv("Result/Sc_Result/Predict.csv")
logging.info("Done：Predict Target Label！")


















