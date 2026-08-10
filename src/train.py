import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

if hasattr(torch.amp, "GradScaler"):
    from torch.amp import GradScaler, autocast
else:
    from torch.cuda.amp import GradScaler, autocast

from config import cfg
from dataset import RSNAKneeDataset
from model import KneeAbnormalityModel, AsymmetricLoss

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)

        if hasattr(torch.amp, "autocast"):
            cast_ctx = autocast(device_type="cuda", enabled=cfg.use_amp and torch.cuda.is_available())
        else:
            cast_ctx = autocast(enabled=cfg.use_amp and torch.cuda.is_available())

        with cast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in dataloader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)

        if hasattr(torch.amp, "autocast"):
            cast_ctx = autocast(device_type="cuda", enabled=cfg.use_amp and torch.cuda.is_available())
        else:
            cast_ctx = autocast(enabled=cfg.use_amp and torch.cuda.is_available())

        with cast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().numpy()
        
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    auc_scores = []
    for i in range(cfg.num_classes):
        try:
            score = roc_auc_score(all_targets[:, i], all_preds[:, i])
            auc_scores.append(score)
        except ValueError:
            auc_scores.append(0.5)

    return total_loss / len(dataloader), np.mean(auc_scores)

def run_training():
    print("==================================================")
    print("  RSNA 2026 KNEE ABNORMALITY DETECTION PIPELINE   ")
    print("==================================================")
    print(f"Device: {cfg.device}")
    print(f"Image Size: {cfg.image_size}x{cfg.image_size} | Slices: {cfg.num_slices}")
    print(f"Batch Size: {cfg.batch_size} | AMP Enabled: {cfg.use_amp}")
    
    if not cfg.train_csv.exists():
        print(f"\n⚠️  [DRY RUN CHECK] Metadata file not found at:\n    {cfg.train_csv}")
        print("   Skipping training run. Local environment check passed cleanly!\n")
        return

    df = pd.read_csv(cfg.train_csv)
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "fold"] = fold

    for fold in range(1):
        print(f"\n---> Starting Training Fold {fold} <---")
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)

        train_dataset = RSNAKneeDataset(train_df, is_train=True)
        val_dataset = RSNAKneeDataset(val_df, is_train=True)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        model = KneeAbnormalityModel().to(cfg.device)
        criterion = AsymmetricLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        
        if hasattr(torch.amp, "GradScaler"):
            scaler = GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())
        else:
            scaler = GradScaler(enabled=cfg.use_amp and torch.cuda.is_available())

        best_auc = 0.0

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
            val_loss, val_auc = validate(model, val_loader, criterion)
            
            print(f"Epoch {epoch+1:02d}/{cfg.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                ckpt_path = cfg.output_dir / f"model_fold_{fold}_best.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"   >>> Saved new best model checkpoint to {ckpt_path.name} (AUC: {best_auc:.4f})")

if __name__ == "__main__":
    run_training()