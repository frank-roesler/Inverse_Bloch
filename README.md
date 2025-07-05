# Inverse Bloch Simulation and Training Suite

This repository provides tools for training, simulating, and analyzing RF pulse design and Bloch simulations, with a focus on neural network-based approaches for MRI applications.

## Project Structure

- **forward.py**: Run inference with a trained model, generate and plot results.
- **backward.py**: Main training script for neural network models.
- **params.py**: Central configuration for training and simulation parameters.
- **tests.py**: (Commented) Example/test code for model and simulation routines.
- **requirements.txt**: Python dependencies.
- **README.md**: Project overview and instructions (this file).

### Submodules
- **utils_train/**: Training utilities, model definitions, logging, and plotting during training.
- **utils_bloch/**: Bloch simulation routines, batch processing, and related utilities.
- **utils_infer/**: Inference and analysis utilities, including plotting and parameter export.
- **data/**: Input data and literature references.
- **results/**: Output folders for training logs, plots, and exported parameters.

## Main Features

- **Training**: Train neural network models to generate RF pulses and gradients for slice-selective excitation using `backward.py`.
- **Inference**: Run trained models and visualize results with `forward.py`.
- **Simulation**: Bloch simulation routines for evaluating pulse performance.
- **Logging & Export**: Automatic logging of training progress, model parameters, and export to CSV/JSON.
- **Plotting**: Visualization of magnetization, pulse, gradient, and loss curves.

## Usage

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Training
Edit `params.py` to set your desired parameters, then run:
```bash
python backward.py
```

### 3. Inference & Plotting
Run inference and generate plots:
```bash
python forward.py
```

### 4. Export Parameters
Export model and training parameters to CSV:
```python
from utils_infer.export_param_csv import export_param_csv
export_param_csv(input_path, output_path)
```

## Data & Results
- Place input `.mat` files in the `data/` directory.
- Training and inference results are saved in timestamped folders under `results/`.

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- (Other dependencies in `requirements.txt`)

## Author
Frank Rösler

## License
[Specify your license here]

---
For more details, see code comments and docstrings in each module.
