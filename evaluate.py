import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from nn_nkf import NNAKF
from kf import KalmanFilter
from data_loader import get_ngsim_dataloader, generate_synthetic_trajectories

# Use checkpoint path from config (fallback to default if not set)
CHECKPOINT_PATH = getattr(config, 'CHECKPOINT_PATH', 'checkpoints/best_nnakf.pth')


def compute_rmse(est, true):
    """
    Compute RMSE per state dimension
    """
    mse = ((est - true) ** 2).mean(axis=(0, 1))
    return np.sqrt(mse)


def evaluate_on_synthetic(model, use_kf, seq_length, device):
    # generate a small synthetic test set
    states, measurements = generate_synthetic_trajectories(200, seq_length)
    meas_t = torch.tensor(measurements, dtype=torch.float32).to(device)
    true_t = torch.tensor(states, dtype=torch.float32).to(device)

    if use_kf:
        kf = KalmanFilter()
        kf_est_list = []
        z0, z1 = seq[0], seq[1]
        vx0 = (z1[0] - z0[0]) / config.DT
        vy0 = (z1[1] - z0[1]) / config.DT
        kf.x = np.array([z1[0], z1[1], vx0, vy0])
        for seq in measurements:
            # For synthetic, initial state at origin is correct
            kf.x = np.zeros(config.STATE_DIM)
            kf.P = np.eye(config.STATE_DIM)
            est_seq = []
            for z in seq:
                kf.predict()
                x_upd, _, _ = kf.update(z)
                est_seq.append(x_upd)
            kf_est_list.append(est_seq)
        nn_est = torch.tensor(np.array(kf_est_list), dtype=torch.float32).to(device)
    else:
        with torch.no_grad():
            nn_est = model(meas_t)

    rmse = compute_rmse(nn_est.cpu().numpy(), true_t.cpu().numpy())
    print(f"RMSE per state [x, y, vx, vy]: {rmse}")


def evaluate_on_ngsim(model, use_kf, csv_path, seq_length, batch_size, device):
    loader = get_ngsim_dataloader(csv_path, seq_length, batch_size)

    all_true = []
    all_nn = []
    all_kf = []

    model.eval()
    kf = KalmanFilter()

    with torch.no_grad():
        for batch in loader:
            meas = batch.to(device).float()  # (B, T, 2)
            true_pos = meas.cpu().numpy()    # ground truth positions

            # NN-AKF estimates
            nn_out = model(meas)[..., :config.MEASUREMENT_DIM].cpu().numpy()
            all_nn.append(nn_out)

            # KF baseline estimates
            kf_batch = []
            for seq in meas.cpu().numpy():
                # Initialize filter state to first measurement (zero velocity)
                z0 = seq[0]
                kf.x = np.array([z0[0], z0[1], 0.0, 0.0])
                kf.P = np.eye(config.STATE_DIM)
                est_seq = [kf.x[:config.MEASUREMENT_DIM].copy()]
                for z in seq[1:]:
                    kf.predict()
                    x_upd, _, _ = kf.update(z)
                    est_seq.append(x_upd[:config.MEASUREMENT_DIM])
                kf_batch.append(est_seq)
            all_kf.append(np.array(kf_batch))
            all_true.append(true_pos)

    all_true = np.concatenate(all_true, axis=0)
    all_nn = np.concatenate(all_nn, axis=0)
    all_kf = np.concatenate(all_kf, axis=0)

    rmse_nn = np.sqrt(((all_nn - all_true) ** 2).mean())
    rmse_kf = np.sqrt(((all_kf - all_true) ** 2).mean())

    print(f"NGSIM Position RMSE — NNAKF: {rmse_nn:.4f}, KF: {rmse_kf:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate NNAKF or KF')
    parser.add_argument('--use_synthetic', action='store_true', help='evaluate on synthetic data')
    parser.add_argument('--use_kf', action='store_true', help='evaluate baseline KF only')
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--csv_path', type=str, default=str(config.NGSIM_RAW_CSV))
    args = parser.parse_args()

    device = config.DEVICE

    # load model
    model = NNAKF().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    if args.use_synthetic:
        evaluate_on_synthetic(model, args.use_kf, args.seq_length, device)
    else:
        evaluate_on_ngsim(model, args.use_kf, args.csv_path, args.seq_length, args.batch_size, device)


if __name__ == '__main__':
    main()
