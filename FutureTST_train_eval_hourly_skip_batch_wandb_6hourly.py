#!/usr/bin/env python
# ---------------------------------------------------------------
#  FutureTST  ── wandb hyper-parameter sweep  +  Early Stopping
# ---------------------------------------------------------------
import os, math, argparse, yaml, numpy as np, pandas as pd, torch
import torch.nn as nn
import wandb
from copy import deepcopy
from tqdm import tqdm

from datasets.preprocessing import create_timeseries_sequences, nse, data_creation
from datasets.postprocessing import calculate_metrics_for_flow_category
from models.futureTST import FutureTST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------- training loop w/ Early Stopping ----
def training(epochs, patience, loss_fn, optim, model, train_loader, val_loader):
    best_state, best_val = None, float("inf")
    wait = 0                                     # epochs since last improvement

    for epoch in range(epochs):
        # ----- TRAIN --------------------------------------------------
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optim.zero_grad()
            valid_indices = []
            for i in range(batch.shape[0]):
                if -999.0 not in batch[i]:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                print("All sequences in batch contain -999.0, skipping batch")
                continue
                
            # Create a new batch with only valid sequences
            batch = batch[valid_indices]
            y_true = batch[:, -1, model.context_window_size:].float().to(DEVICE)
            y_pred = model(batch.float().to(DEVICE)).squeeze(1)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optim.step()
            tr_loss += loss.item()
            wandb.log({"train_loss": loss.item()}, commit=False)
        tr_loss /= len(train_loader)

        # ----- VALIDATE ----------------------------------------------
        model.eval()
        val_loss, outs, reals = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                valid_indices = []
                valid_indices = []
                for i in range(batch.shape[0]):
                    if -999.0 not in batch[i]:
                        valid_indices.append(i)
                
                if len(valid_indices) == 0:
                    print("All sequences in batch contain -999.0, skipping batch")
                    continue
                    
                # Create a new batch with only valid sequences
                batch = batch[valid_indices]
                
                y_true = batch[:, -1, model.context_window_size:].float().to(DEVICE)
                y_pred = model(batch.float().to(DEVICE)).squeeze(1)
                loss = loss_fn(y_pred, y_true)
                val_loss += loss.item()
                outs.append(y_pred.cpu().numpy())
                reals.append(y_true.cpu().numpy())
            val_loss /= len(val_loader)
            outs, reals = np.concatenate(outs), np.concatenate(reals)
            val_nse = nse(outs, reals)

            wandb.log({"val_loss": val_loss, "val_NSE": val_nse})

        # ----- EARLY STOP LOGIC --------------------------------------
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0                              # reset patience counter
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val = {best_val:.6f})")
                wandb.log({"early_stop_epoch": epoch+1})
                break

        print(
            f"Epoch {epoch+1}: train={tr_loss:.6f} | val={val_loss:.6f} | NSE={val_nse:.4f}"
        )

    return best_state, best_val

# ---------------------------- evaluation -----------------------------
def evaluation(model, test_loader):
    model.eval()
    outs, reals = [], []
    with torch.no_grad():
        for batch in test_loader:
            valid_indices = []
            for i in range(batch.shape[0]):
                if -999.0 not in batch[i]:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                print("All sequences in batch contain -999.0, skipping batch")
                continue
                
            # Create a new batch with only valid sequences
            batch = batch[valid_indices]
            y_true = batch[:, -1, model.context_window_size:].float().to(DEVICE)
            y_pred = model(batch.float().to(DEVICE)).squeeze(1)
            outs.append(y_pred.cpu())
            reals.append(y_true.cpu())
    outs, reals = torch.cat(outs).numpy(), torch.cat(reals).numpy()
    return outs, reals

# ---------------------------- sweep training fn ----------------------
def sweep_train(config=None):
    with wandb.init(config=config):
        cfg = wandb.config

        df = pd.read_csv("datasets/SMFV2_Data_withbasin_6hourly.csv", index_col=0)
        basin_df = df[["HG (FT)", "MAP (IN)", "QR (CFS)"]]

        train_dl, val_dl, test_dl = data_creation(
            basin_df,
            cfg.context_window_size,
            cfg.pred_size,
            cfg.batch_size,
            cfg.batch_size,
            cfg.batch_size,
        )

        model = FutureTST(
            context_window_size=cfg.context_window_size,
            patch_size=cfg.patch_size,
            stride_len=cfg.stride_len,
            d_model=cfg.d_model,
            num_transformer_layers=cfg.num_transformer_layers,
            mlp_size=cfg.mlp_size,
            num_heads=cfg.num_heads,
            mlp_dropout=cfg.mlp_dropout,
            pred_size=cfg.pred_size,
            embedding_dropout=cfg.embedding_dropout,
            input_channels=3,
        ).to(DEVICE)

        loss_fn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        wandb.watch(model, log="gradients", log_freq=100)

        best_state, _ = training(
            cfg.epochs, cfg.patience, loss_fn, optim, model, train_dl, val_dl
        )
        model.load_state_dict(best_state)
        preds, truths = evaluation(model, test_dl)
        test_nse = nse(preds, truths)
        wandb.log({"test_NSE": test_nse})

# ---------------------------- sweep definition -----------------------
SWEEP_DICT = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "context_window_size": {"values": [48, 72, 96, 120, 144, 168, 192, 216, 240]},
        "patch_size": {"values": [16, 32, 64]},
        "stride_len": {"values": [4, 8, 16]},
        "d_model": {"values": [128, 256, 512]},
        "num_transformer_layers": {"values": [2, 8]},
        "mlp_size": {"values": [64, 128, 256]},
        "num_heads": {"values": [8, 16, 32]},
        "mlp_dropout": {"values": [0.0, 0.1, 0.2]},
        "embedding_dropout": {"values": [0.0, 0.1, 0.2]},
        "lr": {"values": [1e-3, 1e-4, 1e-5, 1e-6]},
        "batch_size": {"values": [512]},
        "epochs": {"value": 200},         # maximum length
        "patience": {"value": 50},        # Early Stopping patience
        "pred_size": {"value": 28},
    },
}

# ---------------------------- main entry -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="FutureTST_sweep", help="wandb project name")
    ap.add_argument("--count", type=int, default=200, help="number of sweep trials")
    args = ap.parse_args()
    sweep_id = wandb.sweep(SWEEP_DICT, project=args.project)
    print(f"Created sweep {sweep_id}")
    wandb.agent(sweep_id, function=sweep_train, count=args.count)
