import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config


def generate_synthetic_trajectories(num_trajectories, seq_length, dt=config.DT):
    """
    Generate synthetic trajectories with occasional random acceleration.
    Returns states and noisy measurements.
    """
    states = np.zeros((num_trajectories, seq_length, config.STATE_DIM))
    measurements = np.zeros((num_trajectories, seq_length, config.MEASUREMENT_DIM))
    for i in range(num_trajectories):
        x, y = 0.0, 0.0
        vx, vy = np.random.uniform(-5, 5, size=2)
        for t in range(seq_length):
            if np.random.rand() < 0.05:
                ax, ay = np.random.randn(2) * config.PROCESS_NOISE_STD
            else:
                ax, ay = 0.0, 0.0
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            states[i, t] = [x, y, vx, vy]
            measurements[i, t] = [
                x + np.random.randn() * config.MEAS_NOISE_STD,
                y + np.random.randn() * config.MEAS_NOISE_STD
            ]
    return states, measurements


def stream_ngsim_csv(csv_path, chunk_size=1_000_000, sample_every_n=10):
    """
    Stream CSV in chunks, downsampling frames, yielding minimal columns.
    """
    cols = ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y']
    try:
        reader = pd.read_csv(csv_path, usecols=cols, chunksize=chunk_size)
    except ValueError:
        raise RuntimeError(f"Missing required columns: {cols}")
    for chunk in reader:
        chunk = chunk[chunk['Frame_ID'] % sample_every_n == 0]
        yield chunk


class NGSIMDataset(Dataset):
    """
    Dataset for sliding-window NGSIM sequences, split by vehicle IDs.
    """
    def __init__(self, csv_path, seq_length, sample_rate=10, vehicle_ids=None):
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.vehicle_ids = set(vehicle_ids) if vehicle_ids is not None else None
        seqs = []
        for df in stream_ngsim_csv(csv_path, chunk_size=1_000_000, sample_every_n=self.sample_rate):
            for vid, group in df.groupby('Vehicle_ID'):
                if self.vehicle_ids and vid not in self.vehicle_ids:
                    continue
                arr = group.sort_values('Frame_ID')[['Local_X','Local_Y']].to_numpy()
                for i in range(len(arr) - seq_length):
                    seqs.append(arr[i:i+seq_length])
        self.data = np.stack(seqs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_ngsim_dataloaders(
    csv_path, seq_length, batch_size,
    sample_rate=10, train_frac=0.8, seed=config.RANDOM_SEED
):
    """
    Split Vehicle_IDs into train/val and return DataLoaders.
    """
    ids = set()
    for chunk in pd.read_csv(csv_path, usecols=['Vehicle_ID'], chunksize=1_000_000):
        ids.update(chunk['Vehicle_ID'].unique())
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    split_idx = int(train_frac * len(ids))
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]

    train_ds = NGSIMDataset(csv_path, seq_length, sample_rate, train_ids)
    val_ds   = NGSIMDataset(csv_path, seq_length, sample_rate, val_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_ngsim_dataloader(csv_path, seq_length, batch_size, sample_rate=10):
    """
    Legacy loader without split.
    """
    ds = NGSIMDataset(csv_path, seq_length, sample_rate)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)