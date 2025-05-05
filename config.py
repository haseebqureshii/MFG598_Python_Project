from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

NGSIM_RAW_CSV = DATA_DIR / 'Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250502.csv'
NGSIM_H5 = DATA_DIR / 'ngsim_processed.h5'

STATE_DIM = 4           # [x, y, vx, vy]
MEASUREMENT_DIM = 2     # only [x, y]
DT = 0.1                # time step (s)

PROCESS_NOISE_STD = 0.1
MEAS_NOISE_STD = 1.0

RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 2

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50

# checkpoint and plot dirs
CHECKPOINT_PATH = BASE_DIR / 'checkpoints' / 'best_nnakf.pth'
PLOTS_DIR = BASE_DIR / 'plots'

# reproducibility seed
RANDOM_SEED = 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')