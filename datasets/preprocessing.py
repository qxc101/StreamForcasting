import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_timeseries_sequences(df, prev_time_steps, future_time_steps):
    data = df.values
    num_rows, num_columns = data.shape
    total_steps = prev_time_steps + future_time_steps

    sequences = []
    for start in range(num_rows - total_steps + 1):
        end = start + total_steps
        sequence = data[start:end].T
        sequences.append(sequence)

    sequences = np.array(sequences)
    return sequences

def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def data_creation(basin_data,prev_time_steps, future_time_steps, train_batch_size, val_batch_size, test_batch_size, ft_percentile=None):
    # min_vals = basin_data.min(axis=0)
    # max_vals = basin_data.max(axis=0)
    # basin_data = (basin_data - min_vals) / (max_vals - min_vals)
    

    total_time_steps = basin_data.shape[0]
    train_size = int(total_time_steps * 0.8)
    val_size = int(total_time_steps * 0.1)
    #test_size = total_time_steps - train_size - val_size  # Use remainder to avoid rounding issues
    
    train_basin = basin_data[:train_size]
    val_basin = basin_data[train_size:train_size + val_size]
    test_basin = basin_data[train_size + val_size:]


    train_basin_sequences = create_timeseries_sequences(train_basin, prev_time_steps, future_time_steps)
    val_basin_sequences = create_timeseries_sequences(val_basin, prev_time_steps, future_time_steps)
    test_basin_sequences = create_timeseries_sequences(test_basin, prev_time_steps, future_time_steps)
    
    # Combine sequences from all basins
    #train_sequences.append(train_basin_sequences)
    #val_sequences.append(val_basin_sequences)
    #test_sequences.append(test_basin_sequences)
    #train_sequences = np.concatenate(train_sequences, axis=0)
    #val_sequences = np.concatenate(val_sequences, axis=0)
    #test_sequences = np.concatenate(test_sequences, axis=0)

    train_sequences = torch.from_numpy(train_basin_sequences).float()
    val_sequences = torch.from_numpy(val_basin_sequences).float()
    test_sequences = torch.from_numpy(test_basin_sequences).float()

    #print("Shape of train sequences:", train_sequences.size())
    #print("Shape of validation sequences:", val_sequences.size())
    #print("Shape of test sequences:", test_sequences.size())

    train_dataset = TimeSeriesDataset(train_sequences)
    val_dataset = TimeSeriesDataset(val_sequences)
    test_dataset = TimeSeriesDataset(test_sequences)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if ft_percentile == None:
        return train_dataloader, val_dataloader, test_dataloader
    else:
        last_feature_values = train_basin_sequences[:, -1, -future_time_steps:].flatten() 
        threshold = np.percentile(last_feature_values, ft_percentile) 
        mask = np.any(train_basin_sequences[:, -1, -future_time_steps:] >= threshold, axis=1)  
        
        print(f"Threshold for feature values at percentile {ft_percentile}: {threshold}")

        ft_train_basin_sequences = train_basin_sequences[mask] 
        ft_train_dataset = TimeSeriesDataset(ft_train_basin_sequences)
        ft_train_dataloader = DataLoader(ft_train_dataset, batch_size=train_batch_size, shuffle=True)
        # import pdb; pdb.set_trace()
        return ft_train_dataloader
