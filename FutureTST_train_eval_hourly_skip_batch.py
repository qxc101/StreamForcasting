import numpy as np
import torch
import torch.nn as nn
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import geopandas as gpd
from copy import deepcopy
from tqdm import tqdm

from datasets.preprocessing import create_timeseries_sequences, nse, data_creation
from datasets.postprocessing import (
    calculate_metrics_for_flow_category,
    plot_prediction_comparison, 
    plot_high_low_flow_comparison, 
    plot_detailed_prediction_results, 
    plot_flow_duration_curve
    )
from models.futureTST import FutureTST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def training(epochs, loss_function, optimizer, model, train_dataloader, val_dataloader, path):
    train_loss_vals = []
    val_loss_vals = []
    best_val_loss = float('inf')
    model.to(device)
    best_model = None
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_bar:
            optimizer.zero_grad()
            # (Pdb) batch[0, -1, model.context_window_size:]
            # tensor([41.0003, 41.0003, 41.0003, 41.0003, 42.0245, 43.0133, 43.0133, 43.0133,
            #         41.0003, 38.9874, 38.4930, 37.9986])
            # (Pdb) batch[0, 0, model.context_window_size:]
            # tensor([1961.1799, 1961.1799, 1961.1799, 1961.1799, 1961.1851, 1961.1899,
            #         1961.1899, 1961.1899, 1961.1801, 1961.1700, 1961.1650, 1961.1600])
            # (Pdb) batch[0, 1, model.context_window_size:]
            # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            # (Pdb) batch[0, 2, model.context_window_size:]
            # tensor([41.0003, 41.0003, 41.0003, 41.0003, 42.0245, 43.0133, 43.0133, 43.0133,
            #         41.0003, 38.9874, 38.4930, 37.9986])
            # (Pdb) batch[0, 2, model.context_window_size:]

            train_real_vals = batch[:, -1, model.context_window_size:].float().to(device)
            if -999.0 in train_real_vals:
                print("Found -999.0 in train_real_vals")
                continue
            if -999.0 in train_real_vals:
                import pdb; pdb.set_trace()
            output = model(batch.float().to(device)).squeeze(1)
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
        NSE = 0
        #val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
        with torch.no_grad():
            for batch in val_dataloader:
                val_output = model(batch.float().to(device)).squeeze(1)
                val_real_vals = batch[:, -1, model.context_window_size:].float().to(device) 

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
    model.eval()
    all_outputs = []
    all_real_vals = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Check for -999.0 values in the batch
            if -999.0 in batch:
                print("Found -999.0 in batch")
                continue
            if -999.0 in batch:
                import pdb; pdb.set_trace()
            test_output = model(batch.float().to(device)).squeeze(1)
            test_real_vals = batch[:, -1, model.context_window_size:].to(device)
            
            all_outputs.append(test_output.detach().cpu())
            all_real_vals.append(test_real_vals.detach().cpu())
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_real_vals = torch.cat(all_real_vals, dim=0).numpy()
    
    return all_outputs, all_real_vals



results = []
results_high = []
results_low = []
locations = []
x = pd.read_csv("datasets/SMFV2_Data.csv",index_col=0)

ids = [1]


for basin_id in ids:
    print(f"Processing basin {basin_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locations.append(basin_id)
    certain_basins = x
    #basin = certain_basins[['Dayl(s)','PRCP(mm/day)','SRAD(W/m2)','SWE(mm)','Tmax(C)','Vp(Pa)','QObs(mm/d)']]
    basin = certain_basins[["HG (FT)","MAP (IN)","QR (CFS)"]]


    batch_size = 256
    time_series_size = 300
    pred_size = 12
    num_channels = 3
    path = 'results/FutureTST_hourly_best.pth'
    print(certain_basins.shape)
    train_dataloader,val_dataloader,test_dataloader = data_creation(basin,time_series_size,pred_size,batch_size,batch_size,batch_size) 
    #Past 365 Days, pred_size Day Future, Batch Sizes for train, val, and test

    model = FutureTST(context_window_size=time_series_size,patch_size=16, stride_len=8, d_model=256,
                  num_transformer_layers=2, mlp_size=128, num_heads=8, mlp_dropout=0.2,
                  pred_size=pred_size, embedding_dropout=0.1,input_channels=num_channels)
    


    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = training(1,loss_function,optimizer,model,train_dataloader,val_dataloader,path)


    # # Load best model
    # try:
    #     checkpoint = torch.load(path)
    #     model.load_state_dict(checkpoint)
    #     print(f"Model loaded from {path}")
    # except Exception as e:
    #     print(f"Error loading model from {path}: {e}")
    #     continue


    # Evaluate model
    predicted_vals, real_vals = evaluation(model, test_dataloader)


    # # visualization
    # plot_prediction_comparison(real_vals, predicted_vals, basin_id, modelname='FutureTST')
    
    # # high_low_flow_comparison
    # plot_high_low_flow_comparison(real_vals, predicted_vals, basin_id, modelname='FutureTST')

    # # detailed
    # plot_detailed_prediction_results(real_vals, predicted_vals, basin_id, modelname='FutureTST')

    # # flow_duration_curve
    # plot_flow_duration_curve(real_vals, predicted_vals, basin_id, modelname='FutureTST')



    # Calculate metrics for different flow categories
    metrics_all = calculate_metrics_for_flow_category(real_vals, predicted_vals)
    metrics_high = calculate_metrics_for_flow_category(real_vals, predicted_vals, (None, 90))
    metrics_low = calculate_metrics_for_flow_category(real_vals, predicted_vals, (10, None))
    
    results.append(metrics_all)
    results_high.append(metrics_high)
    results_low.append(metrics_low)
    
    print(f"Basin {basin_id} - All flows: KGE={metrics_all[0]:.4f}, NSE={metrics_all[1]:.4f}, MSE={metrics_all[2]:.4f}, RMSE={metrics_all[3]:.4f}")
    print(f"Basin {basin_id} - High flows: KGE={metrics_high[0]:.4f}, NSE={metrics_high[1]:.4f}, MSE={metrics_high[2]:.4f}, RMSE={metrics_all[3]:.4f}")
    print(f"Basin {basin_id} - Low flows: KGE={metrics_low[0]:.4f}, NSE={metrics_low[1]:.4f}, MSE={metrics_low[2]:.4f}, RMSE={metrics_all[3]:.4f}")
    


# Create DataFrames for results
one_day_results = pd.DataFrame(results, columns=['KGE', 'NSE', 'MSE', 'RMSE'], index=locations)
one_day_high_results = pd.DataFrame(results_high, columns=['KGE', 'NSE', 'MSE', 'RMSE'], index=locations)
one_day_low_results = pd.DataFrame(results_low, columns=['KGE', 'NSE', 'MSE', 'RMSE'], index=locations)

# Save results to CSV
one_day_results.to_csv('results/futuretst_results.csv')
one_day_high_results.to_csv('results/futuretst_high_results.csv')
one_day_low_results.to_csv('results/futuretst_low_results.csv')


# Print summary statistics
print("\n=== Results Summary ===")
print("All Data:")
print(one_day_results.describe())

print("\nHigh Flow Data:")
print(one_day_high_results.describe())

print("\nLow Flow Data:")
print(one_day_low_results.describe())

# Find best and worst basins
best_nse_basin = one_day_results['NSE'].idxmax()
worst_nse_basin = one_day_results['NSE'].idxmin()
print(f"\nBasin with highest NSE: {best_nse_basin} (NSE = {one_day_results['NSE'].max():.4f})")
print(f"Basin with lowest NSE: {worst_nse_basin} (NSE = {one_day_results['NSE'].min():.4f})")


