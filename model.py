# %%
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




# %%
import random


class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout=0.3):
        super(LSTMStack, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(packed)
        #return hn[-1]  # (batch_size, hidden_size)  <-- 2D
        # Use the last hidden state from the top layer
        lstm_out = hn[-1]  # shape: (batch_size, hidden_size)
         # Apply LayerNorm and dropout
        lstm_out = self.layernorm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        return lstm_out  # (B, hidden_size)
        
class CNNExtractor(nn.Module):
    def __init__(self, input_size, cnn_channels, kernel_size, dropout):
        super(CNNExtractor, self).__init__()
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Dropout(dropout)
        )
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x expected as (B, S, F); Conv1d wants (B, F, S)
        x = x.permute(0, 2, 1)  # (B, F, S)
        x = self.cnn(x)  # (B, C, S)
        # Collapse the time dimension so we return 2D features
        # Global average and max pooling
        avg_pool = self.global_avg(x)  # (B, C, 1)
        max_pool = self.global_max(x)  # (B, C, 1)

        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2C, 1)
        return combined.squeeze(-1)  # (B, 2 * cnn_channels)


class DiagnosisModel(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_channels, kernel_size, num_layers, dropout, num_class=2):
        super(DiagnosisModel, self).__init__()
        self.lstm = LSTMStack(input_size, hidden_size, num_layers,dropout)
        self.cnn = CNNExtractor(input_size, cnn_channels, kernel_size, dropout)

        in_features = hidden_size + 2 * cnn_channels  # match what we actually concatenate
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 64),
            nn.Dropout(dropout / 3),
            nn.ReLU(),
            nn.Linear(64, num_class),  # 2 classe
        )

    def forward(self, lstm_input, lengths, cnn_input):
        lstm_features = self.lstm(lstm_input, lengths)  # (B, hidden_size)
        cnn_features = self.cnn(cnn_input)  # (B, cnn_channels)

        # Both branches are 2D now, so concatenation works
        combined = torch.cat([lstm_features, cnn_features], dim=1)  # (B, hidden_size + cnn_channels)
        return self.mlp(combined)  # (batch_size, 2)



class LSTMStack_s(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout_prob,num_classes=2):
        super(LSTMStack_s, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, x,lengths,x2):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(packed)
        last_hidden = hn[-1]  # (batch_size, hidden_size)  <-- 2D
        normed = self.layernorm(last_hidden)
        dropped = self.dropout(normed)
        out = self.classifier(dropped)  # shape: (batch_size, num_classes)

        return out



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        
        """
        alpha: Optional weight for each class (list, Tensor) or scalar for binary
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: [B, C] raw logits
        targets: [B] class indices
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.view(-1, 1)

        log_pt = log_probs.gather(1, targets)
        pt = probs.gather(1, targets)

        if self.alpha is not None:
            at = self.alpha.to(inputs.device).gather(0, targets.squeeze())
            log_pt = log_pt * at.view(-1, 1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  
