import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from nn_nkf import NNAKF
from data_loader import get_ngsim_dataloader, generate_synthetic_trajectories

np.random.seed(0)
torch.manual_seed(0)

# Hyperparameters for training enhancements
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic trajectories.
    Returns pairs of (measurements, full states).
    """
    def __init__(self, states, measurements):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.measurements = torch.tensor(measurements, dtype=torch.float32)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.measurements[idx], self.states[idx]


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # Unpack batch: synthetic yields (meas, states), real yields meas only
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            meas, true_states = batch
            supervise_full = True
        else:
            meas = batch
            supervise_full = False

        meas = meas.to(device).float()
        optimizer.zero_grad()
        est_states = model(meas)

        if supervise_full:
            true_states = true_states.to(device).float()
            loss = criterion(est_states, true_states)
        else:
            # supervise only positions when true states unavailable
            pred_pos = est_states[..., :config.MEASUREMENT_DIM]
            loss = criterion(pred_pos, meas)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        batch_loss = loss.detach().cpu().item()
        total_loss += batch_loss * meas.size(0)

    return total_loss / len(dataloader.dataset)


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                meas, true_states = batch
                supervise_full = True
            else:
                meas = batch
                supervise_full = False

            meas = meas.to(device).float()
            est_states = model(meas)

            if supervise_full:
                true_states = true_states.to(device).float()
                loss = criterion(est_states, true_states)
            else:
                pred_pos = est_states[..., :config.MEASUREMENT_DIM]
                loss = criterion(pred_pos, meas)

            batch_loss = loss.detach().cpu().item()
            total_loss += batch_loss * meas.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Train NNAKF on synthetic or NGSIM data')
    parser.add_argument('--use_synthetic', action='store_true', help='use synthetic trajectories')
    parser.add_argument('--seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE, help='learning rate')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='number of epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Prepare data loaders
    if args.use_synthetic:
        states, measurements = generate_synthetic_trajectories(
            num_trajectories=5000,
            seq_length=args.seq_length
        )
        split = int(0.8 * states.shape[0])
        train_ds = SyntheticDataset(states[:split], measurements[:split])
        val_ds = SyntheticDataset(states[split:], measurements[split:])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    else:
        # NGSIM positions only. Without splits [OLD]
        ''' train_loader = get_ngsim_dataloader(
            str(config.NGSIM_RAW_CSV), args.seq_length, args.batch_size
        )
        val_loader = get_ngsim_dataloader(
            str(config.NGSIM_RAW_CSV), args.seq_length, args.batch_size
        )  # consider splitting by Vehicle_ID for real validation
        '''
        # With splits [NEW]
        from data_loader import get_ngsim_dataloaders
        train_loader, val_loader = get_ngsim_dataloaders(
            str(config.NGSIM_RAW_CSV),
            args.seq_length,
            args.batch_size,
            sample_rate=10,
            train_frac=0.8,
            seed=config.RANDOM_SEED
        )
        
    # Model, optimizer, scheduler, loss
    device = config.DEVICE
    model = NNAKF().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {current_lr:.1e}")
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            ckpt_path = os.path.join(args.output_dir, 'best_nnakf.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Validation hasn't improved for {EARLY_STOPPING_PATIENCE} epochs. Early stopping.")
                break

    writer.close()

if __name__ == '__main__':
    main()
