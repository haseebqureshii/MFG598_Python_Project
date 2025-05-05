# kf.py

import numpy as np
import config

class KalmanFilter:
    """
    Discrete-time Linear Kalman Filter with fixed process and measurement noise.
    Implements Eqs. (1)–(7) from the reference.
    """
    def __init__(self):
        # time step
        dt = config.DT

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Observation matrix (we measure positions only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Fixed process noise covariance Q = q * I
        q = config.PROCESS_NOISE_STD ** 2
        self.Q = q * np.eye(config.STATE_DIM)

        # Fixed measurement noise covariance R = r * I
        r = config.MEAS_NOISE_STD ** 2
        self.R = r * np.eye(config.MEASUREMENT_DIM)

        # Initial state estimate (zeros)
        self.x = np.zeros((config.STATE_DIM,))

        # Initial estimate covariance (identity)
        self.P = np.eye(config.STATE_DIM)

    def predict(self):
        """
        Predict the next state and covariance:
        x_k+1|k = F x_k|k
        P_k+1|k = F P_k|k F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Update state estimate with measurement z:
        y = z - H x  (innovation)
        S = H P H^T + R  (innovation covariance)
        K = P H^T S^-1  (Kalman gain)
        x = x + K y
        P = (I - K H) P
        """
        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy(), self.P.copy(), K
