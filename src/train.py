import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import cfg
from dataset import RSNARadiologyDataset
from model import RSNA2D5Model, AsymmetricLoss

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / cfg.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.accum_steps

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
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

    macro_auc = np.mean(auc_scores)
    return total_loss / len(dataloader), macro_auc

def run_training():
    df = pd.read_csv(cfg.train_csv)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "fold"] = fold

    for fold in range(5):
        if cfg.debug and fold > 0:
            break

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)

        train_dataset = RSNARadiologyDataset(train_df, is_train=True)
        val_dataset = RSNARadiologyDataset(val_df, is_train=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        model = RSNA2D5Model().to(cfg.device)
        criterion = AsymmetricLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        scaler = GradScaler(enabled=cfg.use_amp)

        best_auc = 0.0

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
            val_loss, val_auc = validate(model, val_loader, criterion)

            if val_auc > best_auc:
                best_auc = val_auc
                checkpoint_path = cfg.output_dir / f"model_fold_{fold}_best.pth"
                torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    run_training()