import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import pandas as pd

class LSTM(nn.Module):
    def __init__(self, hidden_size: int, input_size: int = 3, dropout_rate: float = 0.5, 
                 bidirectional: bool = False, context_window_size: int = 480, pred_size: int = 12):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.input_size = input_size  # Number of features (3 for hourly dataset)
        self.num_layers = 5
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.context_window_size = context_window_size  # Explicitly set this parameter
        self.pred_size = pred_size  # Explicitly set prediction window size

        # LSTM layer - expects input shape [batch, seq_len, features]
        self.lstm = nn.LSTM(input_size=self.input_size, 
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers, 
                           bias=True, 
                           batch_first=True,
                           dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
                           bidirectional=self.bidirectional)
        
        # Fully connected layers for prediction
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=128)
        self.fc_final = nn.Linear(in_features=128, out_features=self.pred_size)  # Output size matches pred_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, features]
                where seq_len should match self.context_window_size
            
        Returns:
            Predictions tensor of shape [batch_size, pred_size]
        """
        # Verify input dimensions match expected context window size
        batch_size, seq_len, features = x.shape
        assert seq_len == self.context_window_size, f"Expected sequence length {self.context_window_size}, got {seq_len}"
        assert features == self.input_size, f"Expected {self.input_size} features, got {features}"
        
        # Pass through LSTM - get the output from the last time step
        output, (h_n, c_n) = self.lstm(x)
        
        # Use the final hidden state from the last layer
        if self.bidirectional:
            # For bidirectional, concatenate the last hidden state from both directions
            h_n_final = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        else:
            # For unidirectional, just use the last hidden state
            h_n_final = h_n[-1, :, :]
        
        # Apply dropout and fully connected layers
        x = self.dropout(h_n_final)
        x = torch.relu(self.fc(x))
        pred = self.fc_final(x)
        
        return pred
