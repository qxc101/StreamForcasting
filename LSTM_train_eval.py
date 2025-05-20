import numpy as np
import torch
import torch.nn as nn
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
from permetrics.regression import RegressionMetric
from tqdm import tqdm

from datasets.preprocessing import create_timeseries_sequences, nse, data_creation
from datasets.postprocessing import (
    calculate_metrics_for_flow_category,
    plot_prediction_comparison, 
    plot_high_low_flow_comparison, 
    plot_detailed_prediction_results, 
    plot_flow_duration_curve
    )
from models.lstm import LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TimeSeriesNormalizer class for data standardization
class TimeSeriesNormalizer:
    def __init__(self, epsilon=1e-5):
        """
        Initialize the normalizer with a small constant to avoid division by zero.

        Args:
            epsilon (float): A small constant to ensure numerical stability.
        """
        self.epsilon = epsilon

    def normalize(self, x):
        """
        Normalize the input time series along a specified dimension.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            tuple: A tuple containing the normalized tensor and the means and standard deviations used for normalization.
        """
        # For [batch, time_steps, features] format
        means = x.mean(1, keepdim=True).detach()
        x_normalized = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.epsilon).detach()
        x_normalized = x_normalized / stdev

        return x_normalized, means, stdev

    def denormalize(self, x_normalized, means, stdev):
        """
        Denormalize the input time series using the provided means and standard deviations.

        Args:
            x_normalized (torch.Tensor): The normalized tensor to be denormalized.
            means (torch.Tensor): The means used during the normalization process.
            stdev (torch.Tensor): The standard deviations used during the normalization process.

        Returns:
            torch.Tensor: The denormalized tensor.
        """
        x_denormalized = x_normalized * stdev + means
        return x_denormalized

def training(epochs, loss_function, optimizer, model, train_dataloader, val_dataloader, path):
    """Training function for LSTM model"""
    train_loss_vals = []
    val_loss_vals = []
    best_val_loss = float('inf')
    model.to(device)
    
    # Create normalizer for data standardization
    normalizer = TimeSeriesNormalizer()
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_bar:
            optimizer.zero_grad()
            
            # LSTM needs data in format [batch, seq_len, features]
            # Our data is in [batch, features, time_steps]
            train_input = batch[:, :, :model.context_window_size].permute(0, 2, 1).float().to(device)
            train_real_vals = batch[:, -1, model.context_window_size:].float().to(device)
            
            # Normalize the input data
            train_input, means, stdev = normalizer.normalize(train_input)
            
            if torch.isnan(train_real_vals).all():
                continue

            output = model(train_input)        
            batch_loss = loss_function(output, train_real_vals)
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()
            train_bar.set_postfix(loss=f"{batch_loss.item():.6f}")

        train_loss = total_train_loss/len(train_dataloader)

        model.eval()
        total_val_loss = 0
        all_val_outputs_np = [] 
        all_val_real_vals_np = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                val_input = batch[:, :, :model.context_window_size].permute(0, 2, 1).float().to(device)
                val_real_vals = batch[:, -1, model.context_window_size:].float().to(device)
                
                # Normalize validation data
                val_input, val_means, val_stdev = normalizer.normalize(val_input)
                
                if torch.isnan(val_real_vals).all:
                    continue
                val_output = model(val_input)
                
                val_loss = loss_function(val_output, val_real_vals)
                total_val_loss += val_loss.item()

                all_val_outputs_np.append(val_output.cpu().numpy())
                all_val_real_vals_np.append(val_real_vals.cpu().numpy())

        val_loss = total_val_loss/len(val_dataloader)
        val_outputs_epoch_all = np.concatenate(all_val_outputs_np, axis=0)
        val_real_vals_epoch_all = np.concatenate(all_val_real_vals_np, axis=0)

        epoch_NSE = nse(val_outputs_epoch_all, val_real_vals_epoch_all)

        train_loss_vals.append(train_loss)
        val_loss_vals.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_nse = epoch_NSE
            torch.save(model.state_dict(), path)

        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | NSE: {epoch_NSE:.6f}')
    return model

def evaluation(model, test_dataloader):
    """Evaluation function for LSTM model"""
    model.eval()
    all_outputs = []
    all_real_vals = []
    
    # Create normalizer for data standardization
    normalizer = TimeSeriesNormalizer()
    
    with torch.no_grad():
        for batch in test_dataloader:
            test_input = batch[:, :, :model.context_window_size].permute(0, 2, 1).float().to(device)
            test_real_vals = batch[:, -1, model.context_window_size:].to(device)
            if torch.isnan(test_real_vals).all():
                continue
            # Normalize test data
            test_input, test_means, test_stdev = normalizer.normalize(test_input)
            
            test_output = model(test_input)
            
            all_outputs.append(test_output.detach().cpu())
            all_real_vals.append(test_real_vals.detach().cpu())
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_real_vals = torch.cat(all_real_vals, dim=0).numpy()
    
    return all_outputs, all_real_vals


def mse_masked(y_pred,y_true):
    """计算带有NaN值的MSE损失"""
    mask = ~torch.isnan(y_true)
    num_valid = torch.sum(mask)

    if num_valid > 0:
        # 只在非NaN位置计算平方误差
        squared_error = torch.where(mask, (y_pred - y_true) ** 2, torch.zeros_like(y_true))
        mse_loss = torch.sum(squared_error) / num_valid
    else:
        mse_loss = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)

    return mse_loss


# Main execution block
results = []
results_high = []
results_low = []
locations = []
x = pd.read_csv("datasets/stream_flow.csv", index_col=0)

# Sample basin IDs - can be expanded as needed
ids = [8175000, 3015500, 2472000]

for basin_id in ids:
    print(f"Processing basin {basin_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locations.append(basin_id)
    certain_basins = x[x['basin'] == basin_id]
    basin = certain_basins[['Dayl(s)','PRCP(mm/day)','SRAD(W/m2)','SWE(mm)','Tmax(C)','Vp(Pa)','QObs(mm/d)']]

    batch_size = 64
    time_series_size = 365
    pred_size = 3
    num_channels = 7
    path = 'results/path_for_best_model_weights.pth'

    train_dataloader, val_dataloader, test_dataloader = data_creation(basin, time_series_size, pred_size, 
                                                                     batch_size, batch_size, batch_size)
    
    # Initialize LSTM model
    model = LSTM(hidden_size=128, input_size=num_channels, dropout_rate=0.2, bidirectional=False)
    model.fc_final = nn.Linear(in_features=128, out_features=pred_size)
    model.context_window_size = time_series_size

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = training(50, loss_function, optimizer, model, train_dataloader, val_dataloader, path)

    # Load best model
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        continue

    # Evaluate model - note we're using the 'evaluation' function (fixed from original)
    predicted_vals, real_vals = evaluation(model, test_dataloader)

    # Visualization
    plot_prediction_comparison(real_vals, predicted_vals, basin_id, modelname='LSTM')
    plot_high_low_flow_comparison(real_vals, predicted_vals, basin_id, modelname='LSTM')
    plot_detailed_prediction_results(real_vals, predicted_vals, basin_id, modelname='LSTM')
    plot_flow_duration_curve(real_vals, predicted_vals, basin_id, modelname='LSTM')

    # Calculate metrics for different flow categories
    metrics_all = calculate_metrics_for_flow_category(real_vals, predicted_vals)
    metrics_high = calculate_metrics_for_flow_category(real_vals, predicted_vals, (None, 90))
    metrics_low = calculate_metrics_for_flow_category(real_vals, predicted_vals, (10, None))
    
    results.append(metrics_all)
    results_high.append(metrics_high)
    results_low.append(metrics_low)
    
    print(f"Basin {basin_id} - All flows: KGE={metrics_all[0]:.4f}, NSE={metrics_all[1]:.4f}, MSE={metrics_all[2]:.4f}, RMSE={metrics_all[3]:.4f}")
    print(f"Basin {basin_id} - High flows: KGE={metrics_high[0]:.4f}, NSE={metrics_high[1]:.4f}, MSE={metrics_high[2]:.4f}, RMSE={metrics_high[3]:.4f}")
    print(f"Basin {basin_id} - Low flows: KGE={metrics_low[0]:.4f}, NSE={metrics_low[1]:.4f}, MSE={metrics_low[2]:.4f}, RMSE={metrics_low[3]:.4f}")
    

# Create DataFrames for results
lstm_results = pd.DataFrame(results, columns=['KGE', 'NSE', 'MSE', 'RMSE'], index=locations)
lstm_high_results = pd.DataFrame(results_high, columns=['KGE', 'NSE', 'MSE', 'RMSE'], index=locations)
lstm_low_results = pd.DataFrame(results_low, columns=['KGE', 'NSE', 'MSE', 'RMSE'], index=locations)

# Save results to CSV
lstm_results.to_csv('results/lstm_results.csv')
lstm_high_results.to_csv('results/lstm_high_results.csv')
lstm_low_results.to_csv('results/lstm_low_results.csv')

# Print summary statistics
print("\n=== LSTM Results Summary ===")
print("All Data:")
print(lstm_results.describe())

print("\nHigh Flow Data:")
print(lstm_high_results.describe())

print("\nLow Flow Data:")
print(lstm_low_results.describe())

# Find best and worst basins
best_nse_basin = lstm_results['NSE'].idxmax()
worst_nse_basin = lstm_results['NSE'].idxmin()
print(f"\nBasin with highest NSE: {best_nse_basin} (NSE = {lstm_results['NSE'].max():.4f})")
print(f"Basin with lowest NSE: {worst_nse_basin} (NSE = {lstm_results['NSE'].min():.4f})")