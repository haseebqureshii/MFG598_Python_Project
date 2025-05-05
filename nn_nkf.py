import torch
import torch.nn as nn
import config

class NNAKF(nn.Module):
    """
    Neural Network Adapted Kalman Filter (NNAKF).
    Wraps a differentiable Kalman filter and an LSTM-based RNN to adapt the process noise covariance Q̃ per Eq. (12).
    """
    def __init__(self, state_dim=config.STATE_DIM,
                 meas_dim=config.MEASUREMENT_DIM,
                 rnn_hidden=config.RNN_HIDDEN_SIZE,
                 rnn_layers=config.RNN_NUM_LAYERS,
                 adapt_count=10):
        super().__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.N = adapt_count  # number of additional noise matrices

        # Constant-velocity model matrices
        dt = config.DT
        self.F = torch.tensor([  # state transition
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=torch.float32)
        self.H = torch.tensor([  # observe position only
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=torch.float32)

        # Base noise covariances
        q0 = config.PROCESS_NOISE_STD ** 2
        self.register_buffer('Q0', torch.eye(state_dim) * q0)
        r0 = config.MEAS_NOISE_STD ** 2
        self.register_buffer('R', torch.eye(meas_dim) * r0)

        # Trainable additional noise matrices Q_i
        # Initialized small to perturb around Q0
        init = torch.eye(state_dim).unsqueeze(0).repeat(self.N, 1, 1) * 1e-3
        self.Q_tilde = nn.Parameter(init)

        # RNN to generate σ_i from normalized innovations In,j (Eq. 11)
        self.lstm = nn.LSTM(input_size=meas_dim,
                            hidden_size=rnn_hidden,
                            num_layers=rnn_layers,
                            batch_first=True)
        self.fc = nn.Linear(rnn_hidden, self.N)
        self.sigmoid = nn.Sigmoid()

    def forward(self, measurements):
        """
        Run NNAKF over a batch of measurement sequences.
        Inputs:
            measurements: Tensor of shape (B, T, meas_dim)
        Returns:
            states: Tensor of shape (B, T, state_dim)
        """
        B, T, _ = measurements.shape
        device = measurements.device
        F = self.F.to(device)
        H = self.H.to(device)
        Q0 = self.Q0.to(device)
        R = self.R.to(device)
        Q_tilde = self.Q_tilde  # already registered parameter

        # initialize state and covariance
        x = torch.zeros(B, self.state_dim, device=device)
        P = torch.eye(self.state_dim, device=device).unsqueeze(0).repeat(B, 1, 1)

        outputs = []
        hidden = None  # LSTM hidden state

        for t in range(T):
            z = measurements[:, t]               # (B, meas_dim)

            # 1) Predict with base noise Q0
            x_pred = (F @ x.unsqueeze(-1)).squeeze(-1)       # (B, state_dim)
            P_pred0 = F @ P @ F.t() + Q0                     # (B, state_dim, state_dim)

            # 2) Compute normalized squared innovations In,j (Eq. 11)
            innovation = z - (H @ x_pred.unsqueeze(-1)).squeeze(-1)
            S0 = H @ P_pred0 @ H.t() + R
            In = innovation.pow(2) / torch.diagonal(S0, dim1=-2, dim2=-1)

            # 3) RNN → σ coefficients (batch, N)
            out, hidden = self.lstm(In.unsqueeze(1), hidden)
            sigma = self.sigmoid(self.fc(out.squeeze(1)))    # (B, N)

            # 4) Build adaptive Q (Eq. 12)
            # Q_t = Q0 + sum_i σ_i Q̃_i
            Q_adapt = Q0.unsqueeze(0) + torch.einsum('bn,nij->bij', sigma, Q_tilde)

            # 5) Update predicted covariance
            P_pred = P_pred0 - Q0 + Q_adapt

            # 6) Kalman update step
            S = H @ P_pred @ H.t() + R
            K = P_pred @ H.t() @ torch.inverse(S)
            x = x_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
            I = torch.eye(self.state_dim, device=device).unsqueeze(0).repeat(B, 1, 1)
            P = (I - K @ H) @ P_pred

            outputs.append(x.unsqueeze(1))

        return torch.cat(outputs, dim=1)
