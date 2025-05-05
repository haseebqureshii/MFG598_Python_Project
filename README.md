# Neural Network–Adapted Kalman Filter (NNAKF)

A PyTorch implementation of the Neural Network–Adapted Kalman Filter (NNAKF) for position tracking, evaluated on both synthetic maneuvers and the NGSIM Vehicle Trajectories dataset.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Configuration](#configuration)
7. [Data Preparation](#data-preparation)
8. [Usage](#usage)

   * [Synthetic Training & Evaluation](#synthetic-training--evaluation)
   * [Real-Data Training & Evaluation](#real-data-training--evaluation)
   * [Visualization](#visualization)
9. [Hyperparameters & Options](#hyperparameters--options)
10. [Results Summary](#results-summary)
11. [Future Work](#future-work)
12. [License](#license)

---

## Project Description

This repository demonstrates how to adapt the process noise covariance $Q$ of a discrete-time Kalman filter using a recurrent neural network (LSTM) that monitors the filter’s innovations. By learning when to “loosen” or “tighten” the filter’s trust in the motion model, the NNAKF achieves significantly lower position error on both synthetic maneuvers and real traffic data.

## Features

* **Synthetic module**: replicates toy 3D maneuver simulations from the original paper
* **NGSIM loader**: streams and down-samples 1.4 GB of real vehicle trajectories, with an 80/20 train/val split by *Vehicle\_ID*
* **Modular codebase**: separate files for configuration, data loading, KF baseline (`kf.py`), NNAKF model (`nn_nkf.py`), training (`train.py`), evaluation (`evaluate.py`), and visualization (`visualise.py`)
* **Training enhancements**: early stopping, learning-rate scheduling, gradient clipping, weight decay

## Prerequisites

* Python 3.9–3.12
* [PyTorch](https://pytorch.org/) ≥ 2.0
* `numpy`, `pandas`, `matplotlib`, `tensorboard`, `filterpy` (for baseline KF) 

All dependencies are listed in `requirements.txt`.

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/nnakf_project.git
cd nnakf_project

# 2. Create and activate a venv (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt
```

## Project Structure

```
nnakf_project/
├── config.py            # global settings & hyperparams
├── data_loader.py       # synthetic & NGSIM loaders, ID-based splits
├── kf.py                # fixed-Q KalmanFilter class
├── nn_nkf.py            # PyTorch implementation of NNAKF
├── train.py             # training loop (synthetic or real)
├── evaluate.py          # computes RMSE for KF vs. NNAKF
├── visualise.py         # saves trajectory & σ-weight plots
├── requirements.txt     # exact package versions
└── README.md            # this document
```

## Configuration

All hyperparameters and paths live in `config.py`:

* `STATE_DIM`, `MEASUREMENT_DIM`, `DT`
* Noise STDs: `PROCESS_NOISE_STD`, `MEAS_NOISE_STD`
* RNN dims: `RNN_HIDDEN_SIZE`, `RNN_NUM_LAYERS`
* Training: `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`
* Paths: `NGSIM_RAW_CSV`, `CHECKPOINT_PATH`, `PLOTS_DIR`
* Random seed: `RANDOM_SEED`
* Device: `DEVICE`

## Data Preparation

1. **NGSIM**: download CSV from [https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
2. Place it at `data/Vehicle_Trajectories.csv`
3. The code will automatically down-sample and split by `Vehicle_ID`.

## Usage

### Synthetic Training & Evaluation

```bash
# Train on synthetic data
python train.py --use_synthetic

# Evaluate on synthetic test set
python evaluate.py --use_synthetic

# Visualize one trajectory and σ-weights
python visualise.py --use_synthetic --idx 0
```

### Real-Data Training & Evaluation

```bash
# Train on NGSIM data (splits by Vehicle_ID)
python train.py

# Evaluate on NGSIM
python evaluate.py

# Visualize real trajectory and σ-weights (idx 0)
python visualise.py
```

### Visualization

* Saved plots appear in the `plots/` directory:

  * `trajectory_idx{idx}.png`
  * `noise_weights_idx{idx}.png`

## Hyperparameters & Options

* `--seq_length`: length of sliding window (default 50)
* `--batch_size`: training batch size (default 32)
* `--learning_rate`: optimizer LR (default 1e-3)
* `--epochs`: max epochs (default 50)
* `--output_dir`: where to save `best_nnakf.pth` and logs
* `BATCH_SIZE`, `EPOCHS` etc. can be overridden in `config.py`

## Results Summary

* **Synthetic**: RMSE per state $[x,y,v_x,v_y]$ dropped from \~0.58 to \~0.38 after synthetic training enhancements.
* **Real NGSIM**: position RMSE dropped from \~0.58 m (KF) to \~0.49 m (NNAKF); baseline KF with default init gave \~273 m error, demonstrating the need for proper initialization or measurement tuning.

## Future Work

* Incorporate `v_Vel` (and/or `v_Accel`) channels for richer measurements.
* Explore alternative RNNs (GRU, attention) or sequence models.
* Containerize with Docker for reproducibility.
* Add unit tests and continuous integration.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
