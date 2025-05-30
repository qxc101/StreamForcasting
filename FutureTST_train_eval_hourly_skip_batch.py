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
import argparse
from datasets.preprocessing import create_timeseries_sequences, nse, data_creation
from datasets.postprocessing import (
    calculate_metrics_for_flow_category,
    plot_prediction_comparison, 
    plot_high_low_flow_comparison, 
    plot_detailed_prediction_results, 
    plot_flow_duration_curve,
    plot_nse_of_pred_time_step,
    plot_ob_vs_pred_time_step,
    plot_kge_of_pred_time_step,
    plot_detailed_prediction_results_multistep,
    plot_r2_of_pred_time_step
    )
from models.futureTST import FutureTST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def training(epochs, loss_function, optimizer, model, train_dataloader, val_dataloader, path, patience=100):
    train_loss_vals = []
    val_loss_vals = []
    best_val_loss = float('inf')
    model.to(device)
    best_model = None
    wait = 0  
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_bar:
            optimizer.zero_grad()

            valid_indices = []
            for i in range(batch.shape[0]):
                if -999.0 not in batch[i]:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                print("All sequences in batch contain -999.0, skipping batch")
                continue
                
            # Create a new batch with only valid sequences
            batch = batch[valid_indices]
            train_real_vals = batch[:, -1, model.context_window_size:].float().to(device)
            output = model(batch.float().to(device)).squeeze(1)
            batch_loss = loss_function(output, train_real_vals)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
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
                valid_indices = []
                for i in range(batch.shape[0]):
                    if -999.0 not in batch[i]:
                        valid_indices.append(i)
                
                if len(valid_indices) == 0:
                    print("All sequences in batch contain -999.0, skipping batch")
                    continue
                    
                # Create a new batch with only valid sequences
                batch = batch[valid_indices]
            
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
            print(f"Best model found at epoch {best_epoch} with val loss: {best_val_loss:.6f} and NSE: {epoch_NSE:.6f}")
            torch.save(model.state_dict(), path)
            wait = 0                              
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val = {best_val_loss:.6f})")
                
                break
        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | NSE: {epoch_NSE:.6f}')
    return model

def evaluation(model, test_dataloader):
    model.eval()
    all_outputs = []
    all_real_vals = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Check for -999.0 values in the batch
            valid_indices = []
            for i in range(batch.shape[0]):
                if -999.0 not in batch[i]:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                print("All sequences in batch contain -999.0, skipping batch")
                continue
            batch = batch[valid_indices]
            
            test_output = model(batch.float().to(device)).squeeze(1)
            test_real_vals = batch[:, -1, model.context_window_size:].to(device)
            
            all_outputs.append(test_output.detach().cpu())
            all_real_vals.append(test_real_vals.detach().cpu())
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_real_vals = torch.cat(all_real_vals, dim=0).numpy()
    # import pdb; pdb.set_trace()
    return all_outputs, all_real_vals


if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Train and evaluate FutureTST model for stream forecasting')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training, validation, and testing')
    parser.add_argument('--time_series_size', type=int, default=144, help='Size of the time series context window')
    parser.add_argument('--pred_size', type=int, default=24, help='Size of the prediction window')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--model_path', type=str, default='results/FutureTST_hourly_best_basin1_pred12_nse0837.pth', 
                        help='Path to save/load the model')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patches for the model')
    parser.add_argument('--stride_len', type=int, default=16, help='Stride length for patching')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--mlp_size', type=int, default=128, help='Size of MLP layer')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='Dropout rate for MLP')
    parser.add_argument('--embedding_dropout', type=float, default=0, help='Dropout rate for embeddings')
    parser.add_argument('--decay', '-dc', action='store_true', default=False, help='If we are investigating decay rate')
    parser.add_argument('--eval_only', '-eo', action='store_true', default=False, help='If we are only evaluating the model')
    parser.add_argument('--finetune', "-ft", action='store_true', default=False, help='If we doing fine-tuning')
    parser.add_argument('--ft_model_path', type=str, default='results/FutureTST_hourly_best_basin1_pred12_nse0837.pth')
    parser.add_argument('--epoch', type=int, default=50, help='training epochs')
    parser.add_argument('--sixhourly', "-sh", action='store_true', default=False, help='If we doing 6 hourly data')
    args = parser.parse_args()
    
    if args.sixhourly:
        print(f"Loading dataset from 'datasets/SMFV2_Data_withbasin_6hourly.csv'")
        x = pd.read_csv("datasets/SMFV2_Data_withbasin_6hourly.csv", index_col=0)
    else:
        # Load the dataset
        print(f"Loading dataset from 'datasets/SMFV2_Data_withbasin.csv'")
        x = pd.read_csv("datasets/SMFV2_Data_withbasin.csv",index_col=0)

    ids = [1]
    if args.decay:
        # decay_pred_sizes = [6, 12, 24, 36, 48]
        # decay_time_series_sizes = [24, 24, 48, 72, 96]
        decay_pred_sizes = [36, 48]
        decay_time_series_sizes = [72, 96]
    else:
        decay_pred_sizes = [args.pred_size]
        decay_time_series_sizes = [args.time_series_size]

    for decay_pred_size, decay_time_series_size in zip(decay_pred_sizes, decay_time_series_sizes):
        results = []
        results_high = []
        results_low = []
        locations = []
        for basin_id in ids:
            print(f"Processing basin {basin_id}")
            if args.sixhourly:
                model_path = f"results/FutureTST_hourly_best_basin{basin_id}_pred{decay_pred_size}_6hourly.pth"
            else:
                model_path = f"results/FutureTST_hourly_best_basin{basin_id}_pred{decay_pred_size}.pth"

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            locations.append(basin_id)
            certain_basins = x[x['basin'] == basin_id]
            #basin = certain_basins[['Dayl(s)','PRCP(mm/day)','SRAD(W/m2)','SWE(mm)','Tmax(C)','Vp(Pa)','QObs(mm/d)']]
            basin = certain_basins[["HG (FT)","MAP (IN)","QR (CFS)"]]

            # Use command line arguments
            batch_size = args.batch_size
            time_series_size = decay_time_series_size
            pred_size = decay_pred_size
            print(f"Prediction size: {pred_size}, time series size: {time_series_size}")
            num_channels = args.num_channels
            
            print(certain_basins.shape)
            train_dataloader,val_dataloader,test_dataloader = data_creation(basin,time_series_size,pred_size,batch_size,batch_size,batch_size) 
            #Past 365 Days, pred_size Day Future, Batch Sizes for train, val, and test

            model = FutureTST(context_window_size=time_series_size,
                        patch_size=args.patch_size, 
                        stride_len=args.stride_len, 
                        d_model=args.d_model,
                        num_transformer_layers=args.num_transformer_layers, 
                        mlp_size=args.mlp_size, 
                        num_heads=args.num_heads, 
                        mlp_dropout=args.mlp_dropout,
                        pred_size=pred_size, 
                        embedding_dropout=args.embedding_dropout,
                        input_channels=num_channels)

            loss_function = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            if args.eval_only == False and not args.finetune:
                model = training(args.epoch,loss_function,optimizer,model,train_dataloader,val_dataloader,model_path)
            if args.finetune:
                ft_train_dataloader= data_creation(basin,time_series_size,pred_size,batch_size,batch_size,batch_size, ft_percentile=90) 

                # Load the fine-tuning model
                print(f"Loading fine-tuning model from {args.ft_model_path}")
                checkpoint = torch.load(args.ft_model_path, map_location=device)
                model.load_state_dict(checkpoint)

                # Freeze specific layers in the FutureTST model
                for name, param in model.named_parameters():
                    if "normalizer" in name or "extract_patches" in name or "positional_encoding" in name:
                        # Freeze normalizer, patch extraction, and positional encoding
                        param.requires_grad = False
                    elif "transformer.encoder.layers.0" in name or "transformer.encoder.layers.1" in name:
                        # Freeze the first two layers of the transformer encoder
                        param.requires_grad = False
                    elif "transformer.decoder.layers.0" in name:
                        # Freeze the first layer of the transformer decoder
                        param.requires_grad = False
                    else:
                        # Keep other layers trainable
                        param.requires_grad = True

                # Verify which layers are frozen
                print("Trainable parameters:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"  {name}")

                model.to(device)
                print("Fine-tuning model loaded successfully.")

                loss_function = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
                model_path = args.ft_model_path.replace('.pth', '_ft.pth')
                model = training(25,loss_function,optimizer,model,ft_train_dataloader,val_dataloader,model_path)


            # Load best model
            print("-"*100)
            print("Starting evaluation...")
            checkpoint = torch.load(model_path, map_location=device)
            # checkpoint = torch.load("results/FutureTST_hourly_best_basin1_pred12_nse0837.pth")
            model.load_state_dict(checkpoint)
            model.to(device)
            print(f"Model loaded from {model_path}")

            predicted_vals, real_vals = evaluation(model, train_dataloader)

            # Evaluate model
            if args.sixhourly:
                print(f"Evaluating model on test data with 6-hourly intervals")
                

                # # visualization
                plot_prediction_comparison(real_vals, predicted_vals, basin_id, modelname='FutureTST', save_path='plots_6hourly/comparison')
                
                # # high_low_flow_comparison
                plot_high_low_flow_comparison(real_vals, predicted_vals, basin_id, modelname='FutureTST', save_path='plots_6hourly/highlowcomparision')

                # # detailed
                plot_detailed_prediction_results(real_vals, predicted_vals, basin_id, modelname='FutureTST',save_dir ='plots_6hourly/detailed')

                # # flow_duration_curve
                plot_flow_duration_curve(real_vals, predicted_vals, basin_id, modelname='FutureTST', save_dir='plots_6hourly/fdc')

                ts_nse = plot_nse_of_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', save_dir='plots_6hourly/performance/')
                ts_kge = plot_kge_of_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', save_dir='plots_6hourly/performance/')
                ts_r2 = plot_r2_of_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', save_dir='plots_6hourly/performance/')
                plot_detailed_prediction_results_multistep(real_vals, predicted_vals, basin_id, modelname='FutureTST', 
                                                           save_dir='plots_6hourly/performance/')

                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=None, end=None, 
                                          save_dir='plots_6hourly/performance/', plot_selected_steps=True)
                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=None, end=int(2160/6), 
                                          save_dir='plots_6hourly/performance/')
                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=None, end=int(720/6), 
                                          plot_selected_steps=True, 
                                          save_dir='plots_6hourly/performance/')
                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=int(550/6), end=int(700/6), 
                                          save_dir='plots_6hourly/performance/')

            else:
                # # visualization
                plot_prediction_comparison(real_vals, predicted_vals, basin_id, modelname='FutureTST')
                
                # # high_low_flow_comparison
                plot_high_low_flow_comparison(real_vals, predicted_vals, basin_id, modelname='FutureTST')

                # # detailed
                plot_detailed_prediction_results(real_vals, predicted_vals, basin_id, modelname='FutureTST')

                # # flow_duration_curve
                plot_flow_duration_curve(real_vals, predicted_vals, basin_id, modelname='FutureTST')

                ts_nse = plot_nse_of_pred_time_step(real_vals, predicted_vals, modelname='FutureTST')
                ts_kge = plot_kge_of_pred_time_step(real_vals, predicted_vals, modelname='FutureTST')
                ts_r2 = plot_r2_of_pred_time_step(real_vals, predicted_vals, modelname='FutureTST')
                plot_detailed_prediction_results_multistep(real_vals, predicted_vals, basin_id, modelname='FutureTST')

                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=None, end=None, plot_selected_steps=True)
                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=None, end=2160)
                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=None, end=720, plot_selected_steps=True)
                plot_ob_vs_pred_time_step(real_vals, predicted_vals, modelname='FutureTST', start=550, end=700)

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
        if args.sixhourly:
            one_day_results.to_csv(f'results/futuretst_hourly_results_pred{decay_pred_size}_6hourly.csv')
            one_day_high_results.to_csv(f'results/futuretst_hourly_high_results_pred{decay_pred_size}_6hourly.csv')
            one_day_low_results.to_csv(f'results/futuretst_hourly_low_results_pred{decay_pred_size}_6hourly.csv')
        else:
            one_day_results.to_csv(f'results/futuretst_hourly_results_pred{decay_pred_size}.csv')
            one_day_high_results.to_csv(f'results/futuretst_hourly_high_results_pred{decay_pred_size}.csv')
            one_day_low_results.to_csv(f'results/futuretst_hourly_low_results_pred{decay_pred_size}.csv')


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


