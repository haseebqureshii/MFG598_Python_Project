import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from nn_nkf import NNAKF
from kf import KalmanFilter
from data_loader import generate_synthetic_trajectories, get_ngsim_dataloader

np.random.seed(0)
torch.manual_seed(0)

# Configuration for checkpoint and plots directory
CHECKPOINT_PATH = getattr(config, 'CHECKPOINT_PATH', 'checkpoints/best_nnakf.pth')
PLOTS_DIR = getattr(config, 'PLOTS_DIR', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_trajectories(true, est_kf, est_nn, idx=0):
    """
    Plot and save 2D trajectories for one sequence index.
    Arguments:
        true: np.ndarray, shape (1, T, 2)
        est_kf: np.ndarray, shape (1, T, 2)
        est_nn: np.ndarray, shape (1, T, 2)
        idx: index within the first dim (use 0 since we pass a single sequence)
    """
    plt.figure()
    plt.plot(true[idx, :, 0], true[idx, :, 1], label='True', linewidth=2)
    plt.plot(est_kf[idx, :, 0], est_kf[idx, :, 1], label='KF', linestyle='--')
    plt.plot(est_nn[idx, :, 0], est_nn[idx, :, 1], label='NNAKF', linestyle=':')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.title(f'2D Trajectory Comparison (idx={idx})')
    filepath = os.path.join(PLOTS_DIR, f'trajectory_idx{idx}.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved trajectory plot to {filepath}")


def plot_noise_adaptation(measurements, model, idx=0):
    """
    Plot and save adaptive Q weights σ_i over time for one sequence.
    Arguments:
        measurements: Tensor of shape (1, T, meas_dim)
        model: NNAKF instance
        idx: index within the batch (use 0 since single sequence)
    """
    device = config.DEVICE
    measurements = measurements.to(device).float()
    model.eval()

    sigmas = []
    hidden = None
    B, T, _ = measurements.shape
    F = model.F.to(device)
    H = model.H.to(device)
    Q0 = model.Q0.to(device)
    R = model.R.to(device)
    Q_tilde = model.Q_tilde

    x = torch.zeros(B, model.state_dim, device=device)
    P = torch.eye(model.state_dim, device=device).unsqueeze(0).repeat(B, 1, 1)

    for t in range(T):
        z = measurements[:, t].to(device)
        # Predict
        x_pred = (F @ x.unsqueeze(-1)).squeeze(-1)
        P_pred0 = F @ P @ F.t() + Q0

        # Innovation
        innovation = z - (H @ x_pred.unsqueeze(-1)).squeeze(-1)
        S0 = H @ P_pred0 @ H.t() + R
        In = innovation.pow(2) / torch.diagonal(S0, dim1=-2, dim2=-1)

        # RNN -> sigma
        out, hidden = model.lstm(In.unsqueeze(1), hidden)
        sigma = model.sigmoid(model.fc(out.squeeze(1)))  # (B, N)
        sigmas.append(sigma.cpu().detach().numpy())

        # Update state and covariance (KF update omitted for plotting)
        x = x_pred
        P = P_pred0

    sigmas = np.stack(sigmas, axis=1)  # (B, T, N)

    plt.figure()
    for i in range(sigmas.shape[2]):
        plt.plot(sigmas[idx, :, i], label=f'sigma_{i}')
    plt.xlabel('Time step')
    plt.ylabel('Sigma weight')
    plt.title(f'Adaptive Noise Weights over Time (idx={idx})')
    filepath = os.path.join(PLOTS_DIR, f'noise_weights_idx{idx}.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved noise adaptation plot to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Visualization for KF vs NNAKF')
    parser.add_argument('--use_synthetic', action='store_true', help='use synthetic data')
    parser.add_argument('--seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('--idx', type=int, default=0, help='sequence index to plot')
    args = parser.parse_args()

    device = config.DEVICE
    model = NNAKF().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    seq_idx = args.idx

    if args.use_synthetic:
        # Generate synthetic data
        states, meas = generate_synthetic_trajectories(100, args.seq_length)
        # Compute KF estimates for all trajectories
        est_kf_all = []
        kf = KalmanFilter()
        for seq in meas:
            kf.x = np.zeros(config.STATE_DIM)
            kf.P = np.eye(config.STATE_DIM)
            traj = []
            for z in seq:
                kf.predict()
                x_upd, _, _ = kf.update(z)
                traj.append(x_upd[:config.MEASUREMENT_DIM])
            est_kf_all.append(traj)
        est_kf_all = np.array(est_kf_all)  # (100, T, 2)

        # Compute NNAKF estimates
        with torch.no_grad():
            est_nn_all = model(torch.tensor(meas, dtype=torch.float32).to(device))
            est_nn_all = est_nn_all.cpu().numpy()[..., :config.MEASUREMENT_DIM]  # (100, T, 2)

        # Select single sequence
        true_seq = states[seq_idx:seq_idx+1, :, :config.MEASUREMENT_DIM]    # (1, T, 2)
        kf_seq   = est_kf_all[seq_idx:seq_idx+1]                             # (1, T, 2)
        nn_seq   = est_nn_all[seq_idx:seq_idx+1]                             # (1, T, 2)

        # Plot and save
        plot_trajectories(true_seq, kf_seq, nn_seq, idx=0)
        meas_seq = torch.tensor(meas[seq_idx:seq_idx+1], dtype=torch.float32)
        plot_noise_adaptation(meas_seq, model, idx=0)

    else:
        # Load one batch of NGSIM data (batch_size=1)
        loader = get_ngsim_dataloader(str(config.NGSIM_RAW_CSV), args.seq_length, batch_size=1)
        batch = next(iter(loader))  # shape (1, T, 2)

        # KF estimates
        meas_np = batch.numpy()[0]  # (T, 2)
        est_kf = []
        kf = KalmanFilter()
        for z in meas_np:
            kf.predict()
            x_upd, _, _ = kf.update(z)
            est_kf.append(x_upd[:config.MEASUREMENT_DIM])
        est_kf = np.array(est_kf)[None, :, :]  # (1, T, 2)

        # NNAKF estimates
        with torch.no_grad():
            est_nn = model(batch.to(device).float())
            est_nn = est_nn.cpu().numpy()[..., :config.MEASUREMENT_DIM]  # (1, T, 2)

        # True positions
        true = np.array(meas_np)[None, :, :]  # (1, T, 2)

        # Plot and save
        plot_trajectories(true, est_kf, est_nn, idx=0)
        plot_noise_adaptation(batch.to(device), model, idx=0)


if __name__ == '__main__':
    main()
