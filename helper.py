import os 
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys


def train(model, dataloader, optimizer, criterion, device, epochs, num_class):
    castauto = False
    model.train()
    histroy = []

    for i in range(epochs):
        epoch_loss = 0.0
        for X_batch, length_batch, lstm_batch, y_batch in tqdm(dataloader, desc=f"Epoch {i + 1}/{epochs}", leave=False,
                                                               ncols=80):
            lstm_batch = lstm_batch.to(device)

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if length_batch.dim() > 1:
                length_batch = length_batch.squeeze(-1)
            length_batch = length_batch.to(torch.int64)
            preds = model(lstm_batch, length_batch, X_batch)
            loss = criterion(preds, y_batch.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        histroy.append(avg_loss)
        print(f"Epoch {i + 1}/{epochs} - Loss: {avg_loss:.4f}")
    return model, histroy

# %%
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, length_batch, lstm_batch, y_batch in dataloader:
            lstm_batch = lstm_batch.to(device)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if length_batch.dim() > 1:
                length_batch = length_batch.squeeze(-1)
            length_batch = length_batch.to(torch.int64)

            # Forward pass
            logits = model(lstm_batch, length_batch, X_batch)  # (B, num_classes)
            probs = torch.softmax(logits, dim=1)  # (B, num_classes)

            all_probs.append(probs.detach().cpu())
            all_labels.append(y_batch.detach().cpu())

        # Sanity check: empty batches
        if len(all_labels) == 0 or len(all_probs) == 0:
            print("⚠️ Validation produced no batches. Returning score = 0.0")
            return model, None, None

    # Concatenate across batches
    y_true = torch.cat(all_labels, dim=0).numpy()  # shape: (N,)
    y_score = torch.cat(all_probs, dim=0).numpy()  # shape: (N, num_classes)

    return model, y_true, y_score


def train_re(model, dataloader, optimizer, criterion, device,epochs):
    castauto=False
    model.train()
    histroy = []
    for i in range(epochs):
        epoch_loss = 0.0
        for X_batch,length_batch,lstm_batch, y_batch in tqdm(dataloader, desc=f"Epoch {i+1}/{epochs}", leave=False,ncols=80):
            lstm_batch = lstm_batch.to(device)

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if length_batch.dim() > 1:
                    length_batch = length_batch.squeeze(-1)
            length_batch = length_batch.to(torch.int64)
            preds = model(lstm_batch,length_batch,X_batch)
            loss = criterion(preds, y_batch.long())
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        histroy.append(avg_loss)
        print(f"Epoch {i + 1}/{epochs} - Loss: {avg_loss:.4f}")
    return model,histroy
#%%
def evaluate_re(model, dataloader, device):
    model.eval()
    all_labels = []
    all_pos_scores = []
    with torch.no_grad():
        for X_batch,length_batch,lstm_batch, y_batch in  dataloader:
            lstm_batch = lstm_batch.to(device)
            X_batch = X_batch.to(device)
            if length_batch.dim() > 1:
                length_batch = length_batch.squeeze(-1)
            length_batch = length_batch.to(torch.int64)
            logits = model(lstm_batch,length_batch, X_batch)              # (B, 2), raw logits
            probs_pos = torch.softmax(logits, dim=1)[:, 1]          # (B,), P(class=1
            all_pos_scores.append(probs_pos.detach().cpu())
            all_labels.append(y_batch.detach().cpu())


        if len(all_labels) == 0 or len(all_pos_scores) == 0:
            print("⚠️ Validation produced no batches. Returning score = 0.0")
            return 0.0

    y_true = torch.cat(all_labels,dim=0).numpy()                      # shape (N,)
    y_score = torch.cat(all_pos_scores,dim=0).numpy()

    return model,y_true,y_score
