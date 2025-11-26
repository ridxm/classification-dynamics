# Classification Dynamics

A professional, config-driven codebase for training classifiers to estimate regions of attraction (ROA) for various dynamical systems. This codebase has been restructured to follow modern best practices with Hydra configuration management, PyTorch Lightning training orchestration, and a modular architecture.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Design](#architecture--design)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Notebooks](#notebooks)
  - [Configuration](#configuration)
- [System-Specific Details](#system-specific-details)
- [Data Storage](#data-storage)
- [Adding a New System](#adding-a-new-system)

## Overview

This codebase provides a unified framework for training binary classifiers that predict ROA membership for trajectories from various dynamical systems. The restructuring introduced:

- **Hydra Configuration Management**: Hierarchical, composable YAML configuration files with dependency injection
- **PyTorch Lightning**: Unified training orchestration with automatic checkpointing and logging
- **Modular Architecture**: Separated concerns for data loading, models, systems, and training
- **Config-Driven Notebooks**: All parameters in YAML files, zero hardcoded values
- **Single Training Script**: One universal script works for all systems via config selection

## Key Features

### 1. Universal Training Script
One training script handles all systems:
```bash
python src/training/train.py --config-name=train_mountain_car
python src/training/train.py --config-name=train_pendulum_lqr
python src/training/train.py --config-name=train_cartpole_pybullet
```

### 2. Config-Driven Everything
- All parameters in YAML files
- Easy to reproduce experiments
- Version control friendly
- No hardcoded values in code or notebooks

### 3. Dependency Injection
- Components instantiated from config via `_target_` pattern
- Zero hardcoded class paths
- Easy to swap implementations

### 4. Shared Data Storage
- All data in shared network directory
- No local data duplication
- Direct access from all modules

## Architecture & Design

### Design Principles

1. **Separation of Concerns**: Data, models, systems, and training are separate modules
2. **DRY (Don't Repeat Yourself)**: Shared base classes eliminate code duplication
3. **Configuration as Code**: All parameters in YAML, not in code
4. **Dependency Injection**: Components instantiated from config, not hardcoded

### Base Classes

#### `BaseSystem` (`src/systems/base.py`)
Abstract base class for all dynamical systems:
- `state_dim`: State dimension
- `action_dim`: Action dimension  
- `state_bounds`: State space bounds

#### `BaseClassifierDataModule` (`src/data/base_datamodule.py`)
PyTorch Lightning DataModule with shared logic:
- Train/val/test splits
- DataLoader creation with weighted sampling
- Class weight calculation
- System-specific modules only override `load_data_file()`

#### `ClassifierModule` (`src/training/classifier_module.py`)
PyTorch Lightning module wrapper:
- Implements `training_step()` and `validation_step()`
- Handles optimizer configuration
- Uses torchmetrics for evaluation

### Configuration Hierarchy

Configs use Hydra's composition pattern:

```yaml
# configs/train_mountain_car.yaml
defaults:
  - system: mountain_car
  - model: classifier_medium
  - data: mountain_car_data
  - optimizer: adam
  - device: gpu0

name: mountain_car_classifier
batch_size: 256
base_lr: 1e-3
```

Main configs compose from component configs using `defaults:`. Variables are interpolated with `${variable}`. All instantiable components use `_target_` pattern for dependency injection.

### Data Module Pattern

Each system has a data module that:
1. Inherits from `BaseClassifierDataModule`
2. Implements `load_data_file()` with system-specific logic
3. Handles any necessary transformations (e.g., theta → sin/cos)
4. Returns normalized data with feature_max for test normalization

### Training Module Pattern

The `ClassifierModule`:
- Wraps the classifier model
- Implements training/validation loops
- Handles optimizer configuration
- Uses torchmetrics for evaluation metrics

## Directory Structure

```
classification_dynamics/
├── configs/                    # Hydra configuration files
│   ├── train_*.yaml           # Main training configs (one per system)
│   ├── system/                # System definitions
│   │   ├── mountain_car.yaml
│   │   ├── pendulum_lqr.yaml
│   │   ├── pendulum_cartesian.yaml
│   │   ├── humanoid.yaml
│   │   └── cartpole_pybullet.yaml
│   ├── model/                 # Model architectures
│   │   ├── classifier_small.yaml
│   │   ├── classifier_medium.yaml
│   │   └── classifier_large.yaml
│   ├── data/                  # Data loading configs
│   │   ├── mountain_car_data.yaml
│   │   ├── pendulum_lqr_data.yaml
│   │   ├── pendulum_cartesian_data.yaml
│   │   ├── humanoid_data.yaml
│   │   └── cartpole_pybullet_data.yaml
│   ├── notebook/              # Notebook visualization configs
│   │   ├── analysis_defaults.yaml
│   │   ├── mountain_car_viz.yaml
│   │   ├── pendulum_lqr_viz.yaml
│   │   ├── pendulum_cartesian_viz.yaml
│   │   ├── humanoid_viz.yaml
│   │   └── cartpole_pybullet_viz.yaml
│   ├── optimizer/             # Optimizer configs
│   │   ├── adam.yaml
│   │   └── adamw.yaml
│   └── device/                 # Device configs
│       ├── gpu0.yaml
│       ├── gpu1.yaml
│       └── cpu.yaml
├── src/                        # Source code
│   ├── systems/               # System definitions
│   │   ├── base.py
│   │   ├── mountain_car.py
│   │   ├── pendulum_lqr.py
│   │   ├── pendulum_cartesian.py
│   │   ├── humanoid.py
│   │   └── cartpole_pybullet.py
│   ├── models/                # Model architectures
│   │   ├── mlp.py
│   │   └── classifier.py
│   ├── data/                   # Data loading modules
│   │   ├── base_datamodule.py
│   │   ├── mountain_car_data.py
│   │   ├── pendulum_lqr_data.py
│   │   ├── pendulum_cartesian_data.py
│   │   ├── humanoid_data.py
│   │   └── cartpole_pybullet_data.py
│   ├── training/               # Training scripts
│   │   ├── train.py           # Universal training script
│   │   └── classifier_module.py
│   ├── evaluation/            # Evaluation utilities
│   ├── visualization/         # Visualization utilities
│   └── utils/                 # Utility functions
│       └── notebook_config.py  # NotebookConfig utility
├── notebooks/                 # Jupyter notebooks
│   ├── mountain_car/
│   │   └── model.ipynb
│   ├── pendulum_lqr/
│   │   └── model.ipynb
│   ├── pendulum_cartesian/
│   │   └── model.ipynb
│   ├── humanoid/
│   │   └── model.ipynb
│   └── cartpole_pybullet/
│       └── model.ipynb
├── outputs/                   # Training outputs (created by Hydra)
│   └── <experiment_name>/
│       └── <timestamp>/
│           ├── checkpoints/
│           │   └── best.ckpt
│           └── .hydra/
│               └── config.yaml
├── scripts/                   # Utility scripts
│   ├── archive_old_folders.sh
│   ├── migrate_checkpoints.sh
│   └── migrate_data.sh
├── archive/                   # Archived old code (for reference)
├── tests/                     # Test suite
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

### Install Package

```bash
# From project root
pip install -e .
```

### Install Dependencies Directly

```bash
pip install torch pytorch-lightning hydra-core omegaconf numpy scikit-learn matplotlib torchmetrics pyyaml
```

## Usage

### Training

#### Basic Usage

Train a model for a specific system:

```bash
# Mountain Car
python src/training/train.py --config-name=train_mountain_car

# Pendulum LQR
python src/training/train.py --config-name=train_pendulum_lqr

# Pendulum Cartesian
python src/training/train.py --config-name=train_pendulum_cartesian

# Humanoid
python src/training/train.py --config-name=train_humanoid

# CartPole PyBullet
python src/training/train.py --config-name=train_cartpole_pybullet
```

#### Override Parameters

Override any parameter from the command line:

```bash
# Change batch size
python src/training/train.py --config-name=train_mountain_car batch_size=512

# Change learning rate
python src/training/train.py --config-name=train_mountain_car base_lr=1e-4

# Use CPU instead of GPU
python src/training/train.py --config-name=train_mountain_car device=cpu

# Change model architecture
python src/training/train.py --config-name=train_mountain_car model=classifier_large

# Reduce epochs for testing
python src/training/train.py --config-name=train_mountain_car trainer.max_epochs=2
```

#### Training Outputs

Training outputs are automatically saved to:
```
outputs/<experiment_name>/<timestamp>/
├── checkpoints/
│   └── best.ckpt         # Best model checkpoint
└── .hydra/
    └── config.yaml       # Full resolved configuration
```

The full resolved configuration is saved in `.hydra/config.yaml` for reproducibility.

### Notebooks

All notebooks use the `NotebookConfig` utility to load parameters from YAML files. This ensures zero hardcoded values.

#### Loading Configuration

```python
import sys
sys.path.append('../..')
from src.utils.notebook_config import NotebookConfig

# Load ALL parameters from config file
cfg = NotebookConfig('mountain_car')  # or 'pendulum_lqr', 'cartpole_pybullet', etc.

# Access parameters with dot notation
checkpoint_path = cfg['data.checkpoint_path']
figsize = cfg['viz.figsize']
grid_resolution = cfg['viz.phase_space.grid_resolution']
batch_size = cfg['training.batch_size']
```

#### Example: Complete Notebook Workflow

```python
# 1. Load config
cfg = NotebookConfig('mountain_car')

# 2. Load data using config
from src.data.mountain_car_data import MountainCarDataModule

datamodule = MountainCarDataModule(
    data_file=cfg['data.roa_labels_file'],
    data_dir=cfg['data.data_dir'],
    train_size=cfg['data.train_size'],
    val_size=cfg['data.val_size'],
    batch_size=cfg['training.batch_size'],
    num_workers=0,
    use_weighted_sampler=True,
    pin_memory=False
)
datamodule.setup()

# 3. Create model using config
from src.models.classifier import Classifier

model = Classifier(
    input_dim=cfg['model.input_dim'],
    hidden_dims=cfg['model.hidden_dims'],
    output_dim=cfg['model.output_dim'],
    dropout=cfg['model.dropout'],
    use_batch_norm=True
)

# 4. Visualize using config
import matplotlib.pyplot as plt
figsize = tuple(cfg['viz.figsize'])
dpi = cfg['viz.dpi']
plt.figure(figsize=figsize, dpi=dpi)
```

**Key Benefit**: To change any parameter, edit the YAML config file, not the notebook!

### Configuration

#### Main Training Config

Example: `configs/train_mountain_car.yaml`

```yaml
# @package _global_

defaults:
  - system: mountain_car
  - model: classifier_medium
  - data: mountain_car_data
  - optimizer: adam
  - device: gpu0
  - _self_

# Experiment name
name: mountain_car_classifier
seed: 42

# Training parameters
batch_size: 256
base_lr: 1e-3

# Model configuration
model:
  _target_: src.models.classifier.Classifier
  input_dim: ${system.state_dim}
  hidden_dims: [128, 256, 128, 64]
  output_dim: 2
  dropout: 0.3
  use_batch_norm: true

# Optimizer (partial - will be completed with model params)
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: ${base_lr}

# PyTorch Lightning Trainer
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  accelerator: ${device.accelerator}
  devices: ${device.devices}
  enable_progress_bar: true
  enable_model_summary: true
  log_every_n_steps: 50
  val_check_interval: 1.0
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val_loss
      mode: min
      save_top_k: 1
      filename: best
      save_last: true
    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val_loss
      mode: min
      patience: 10
      verbose: true

# Training module
training_module:
  _target_: src.training.classifier_module.ClassifierModule

# Hydra output directory
hydra:
  run:
    dir: outputs/${name}/${now:%Y-%m-%d_%H-%M-%S}
```

#### System Config

Example: `configs/system/mountain_car.yaml`

```yaml
# @package _global_.system
_target_: src.systems.mountain_car.MountainCarSystem
state_dim: 2
action_dim: 1
```

#### Data Config

Example: `configs/data/mountain_car_data.yaml`

```yaml
_target_: src.data.mountain_car_data.MountainCarDataModule
data_file: data/mountain_car/roa_labels.txt  # Ignored if data_dir is set
data_dir: /common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008
batch_size: ${batch_size}
num_workers: 4
train_size: 1000
val_size: 500
use_weighted_sampler: true
pin_memory: true
```

#### Notebook Config

Example: `configs/notebook/mountain_car_viz.yaml`

```yaml
# ============================================================================
# DATA PATHS (all in shared directory)
# ============================================================================
data:
  data_dir: /common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008
  trajectories_dir: /common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008/trajectories
  roa_labels_file: /common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008/roa_labels.txt
  shuffled_indices_file: /common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008/shuffled_indices.txt
  dataset_description_file: /common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008/dataset_description.json
  checkpoint_path: outputs/mountain_car_classifier/best.ckpt
  train_size: 1000
  val_size: 500

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
training:
  batch_size: 512
  learning_rate: 0.001
  num_epochs: 50

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
model:
  input_dim: 2
  hidden_dims: [128, 256, 128, 64]
  output_dim: 2
  dropout: 0.3

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
viz:
  figsize: [12, 8]
  dpi: 150
  phase_space:
    x_lim: [-2.0, 1.0]
    y_lim: [-0.1, 0.1]
    grid_resolution: 100
    x_label: "Position"
    y_label: "Velocity"
    colormap: viridis

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
analysis:
  num_samples: 1000
  batch_size: 256
  confidence_threshold: 0.1
```

## System-Specific Details

### Mountain Car
- **State dimension**: 2 (position, velocity)
- **Model input**: 2 features
- **Data**: `mountain_car_power_0p0008`
- **Notebook**: `notebooks/mountain_car/model.ipynb`

### Pendulum LQR
- **State dimension**: 2 (theta, theta_dot)
- **Model input**: 3 features (after sin/cos transformation: [sin(theta), cos(theta), theta_dot])
- **Transformation**: `theta` → `[sin(theta), cos(theta)]`
- **Data**: `pendulum_lqr_50k`
- **Notebook**: `notebooks/pendulum_lqr/model.ipynb`

### Pendulum Cartesian
- **State dimension**: 4 (x, y, x_dot, y_dot)
- **Model input**: 4 features
- **Data**: `pendulum_cartesian_50k`
- **Notebook**: `notebooks/pendulum_cartesian/model.ipynb`

### Humanoid
- **State dimension**: 67
- **Model input**: 67 features
- **Data**: `humanoid_get_up`
- **Notebook**: `notebooks/humanoid/model.ipynb`

### CartPole PyBullet
- **State dimension**: 4 (x, theta, x_dot, theta_dot)
- **Model input**: 5 features (after transformation: [x, sin(theta), cos(theta), x_dot, theta_dot])
- **Transformation**: `theta` → `[sin(theta), cos(theta)]`
- **Normalization**: Only features [0, 3, 4] (x, x_dot, theta_dot) are normalized, NOT sin/cos
- **Data**: `cartpole_pybullet`
- **Notebook**: `notebooks/cartpole_pybullet/model.ipynb`
- **Key technique**: Selective normalization preserves sin/cos bounds

## Data Storage

**Important**: All data files are stored in a shared network directory:
```
/common/users/shared/pracsys/genMoPlan/data_trajectories
```

No local data storage is needed. All data modules and configs are configured to read directly from the shared directory:

- `mountain_car_power_0p0008` - Mountain Car data
- `pendulum_lqr_50k` - Pendulum LQR data
- `pendulum_cartesian_50k` - Pendulum Cartesian data
- `humanoid_get_up` - Humanoid data
- `cartpole_pybullet` - CartPole PyBullet data

The `data/` directory in the project is not used for data storage. All data paths in configs point to the shared directory.

## Adding a New System

To add a new system, follow these steps:

### 1. Create System Class

`src/systems/<system>.py`:
```python
from .base import BaseSystem

class NewSystem(BaseSystem):
    @property
    def state_dim(self) -> int:
        return <state_dimension>
    
    @property
    def action_dim(self) -> int:
        return <action_dimension>
    
    @property
    def state_bounds(self) -> List[Tuple[float, float]]:
        return [<bounds>]
```

### 2. Create Data Module

`src/data/<system>_data.py`:
```python
from .base_datamodule import BaseClassifierDataModule

class NewSystemDataModule(BaseClassifierDataModule):
    def load_data_file(self, data_file, train_size, val_size):
        # Implement system-specific data loading
        # Handle any transformations
        # Return (X_train, y_train, X_val, y_val, feature_max)
        pass
```

### 3. Create Config Files

- `configs/system/<system>.yaml` - System definition
- `configs/data/<system>_data.yaml` - Data loading config
- `configs/train_<system>.yaml` - Main training config
- `configs/notebook/<system>_viz.yaml` - Notebook visualization config

### 4. Create Notebook

`notebooks/<system>/model.ipynb` - Use existing notebooks as templates

### 5. Test

```bash
# Test training
python src/training/train.py --config-name=train_<system>

# Test notebook
# Open notebook and verify it loads config correctly
```

## Key Design Decisions

### Why Hydra?
- Hierarchical configuration composition
- Command-line overrides
- Automatic config saving for reproducibility
- Dependency injection via `_target_` pattern

### Why PyTorch Lightning?
- Automatic checkpointing
- Built-in logging
- Device management
- Training loop abstraction

### Why Config-Driven Notebooks?
- Single source of truth for parameters
- Easy to reproduce experiments
- No hardcoded values
- Easy to share configurations

### Why Shared Data Storage?
- No data duplication
- Single source of truth
- Easy to update data
- Consistent across all systems

## Troubleshooting

### Import Errors
Make sure you're running from the project root:
```bash
cd /common/home/rm1838/Documents/classification_dynamics
python src/training/train.py --config-name=train_mountain_car
```

### Config Path Errors
The training script uses relative paths. If you get config path errors:
- Make sure you're running from project root
- Or update the `config_path` in `train.py` to use absolute path

### Data Path Errors
If data loading fails:
- Check that data files exist in the shared directory
- Verify paths in config files
- Update `data_dir` in data configs if needed

### Model Dimension Mismatch
If you get dimension errors:
- Check `input_dim` in model config matches transformed feature dimension
- For systems with transformations (pendulum_lqr, cartpole_pybullet), model input is different from raw state dimension

## Archive

Old code from before restructuring has been moved to `archive/` for reference:
- `archive/mountain-car/` - Original Mountain Car code
- `archive/pendulum-lqr/` - Original Pendulum LQR code
- `archive/pendulum-cartesian/` - Original Pendulum Cartesian code
- `archive/Humanoid/` - Original Humanoid code
- `archive/cartpole-pybullet/` - Original CartPole PyBullet code

These are preserved for reference but are not part of the active codebase.

## License

[Add your license here]
