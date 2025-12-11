# Hyperspherical Variational Auto-Encoders

PyTorch implementation of Hyperspherical Variational Auto-Encoders (S-VAE) for weather data.

## Environment Setup

### Requirements

- Python >= 3.6
- PyTorch >= 0.4.1
- NumPy
- SciPy
- xarray
- zarr
- tqdm

### Installation

```bash
# Install the package
python setup.py install

# Or install dependencies manually
pip install torch numpy scipy xarray zarr tqdm
```

## Quick Start

### Training

#### Using Shell Script (Recommended)

```bash
bash examples/run_svae_train.sh
```

#### Using Python Script

```bash
python examples/train_weather_svae_improved.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --output-dir outputs/svae_improved \
    --epochs 600 \
    --batch-size 32 \
    --latent-channels 4 \
    --hidden-dims 64 128 256 512 \
    --distribution normal \
    --lr 1e-4 \
    --kl-weight 1e-6 \
    --use-residual \
    --use-advanced-loss \
    --save-model
```

### Testing

```bash
python examples/test_weather_svae.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --model-path outputs/svae_improved/best_model.pth \
    --variable 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --output-dir outputs/svae_test
```

## Parameters

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data-path` | str | required | Path to zarr data |
| `--variable` | str | `2m_temperature` | Variable name |
| `--time-slice` | str | None | Time slice, format: `2020-01-01:2020-12-31` |
| `--levels` | int list | None | Levels for multi-level variables |
| `--train-split` | float | 0.8 | Training set ratio |
| `--augment` | flag | False | Enable data augmentation |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--latent-channels` | int | 4 | Latent space channels |
| `--hidden-dims` | int list | `64 128 256 512` | Encoder/decoder hidden dimensions |
| `--distribution` | str | `normal` | Distribution type: `normal` or `vmf` |
| `--use-residual` | flag | True | Use residual connections |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 100 | Number of epochs |
| `--batch-size` | int | 32 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--device` | str | None | Device (auto-select if not specified) |
| `--lr-scheduler` | str | `plateau` | LR scheduler: `cosine`, `plateau`, `step`, `exponential` |
| `--lr-scheduler-params` | str | None | LR scheduler parameters (JSON) |
| `--kl-weight` | float | 1e-6 | KL divergence weight |
| `--kl-annealing` | flag | False | Enable KL annealing |
| `--use-advanced-loss` | flag | True | Use advanced loss function |
| `--grad-loss-weight` | float | 0.1 | Gradient loss weight |
| `--perceptual-weight` | float | 1.0 | Perceptual loss weight |

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-dir` | str | `outputs/svae_improved` | Output directory |
| `--save-model` | flag | False | Save model |
| `--resume` | str | None | Resume from checkpoint |

## Examples

### Standard VAE (Gaussian Distribution)

```bash
python examples/train_weather_svae_improved.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --distribution normal \
    --latent-channels 4 \
    --hidden-dims 64 128 256 512 \
    --use-residual \
    --use-advanced-loss \
    --lr 1e-4 \
    --kl-weight 1e-6 \
    --lr-scheduler plateau \
    --lr-scheduler-params '{"mode":"min","factor":0.5,"patience":10,"min_lr":1e-6}' \
    --epochs 600 \
    --batch-size 32 \
    --output-dir outputs/svae_normal \
    --save-model
```

### S-VAE (Hyperspherical Distribution)

```bash
python examples/train_weather_svae_improved.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --distribution vmf \
    --latent-channels 4 \
    --hidden-dims 64 128 256 512 \
    --use-residual \
    --use-advanced-loss \
    --lr 1e-4 \
    --kl-weight 1e-6 \
    --lr-scheduler plateau \
    --lr-scheduler-params '{"mode":"min","factor":0.5,"patience":10,"min_lr":1e-6}' \
    --epochs 600 \
    --batch-size 32 \
    --output-dir outputs/svae_vmf \
    --save-model
```

## Output Files

After training, the following files are generated in `--output-dir`:

- `config.json`: Training configuration and normalization statistics
- `best_model.pth`: Best model checkpoint (based on test reconstruction loss)
- `checkpoint_latest.pth`: Latest checkpoint for resuming training
- `train_history.json`: Training history

## Project Structure

```
s-vae-pytorch/
├── hyperspherical_vae/          # Core library
│   ├── distributions/           # Von Mises-Fisher and Hyperspherical Uniform distributions
│   └── ops/                     # Low-level operations
├── examples/                    # Example scripts
│   ├── train_weather_svae_improved.py  # Training script
│   ├── test_weather_svae.py           # Testing script
│   └── run_svae_train.sh              # Training shell script
└── setup.py                     # Installation script
```

## Citation

If you use this library in your research, please cite:

```
@article{s-vae18,
  title={Hyperspherical Variational Auto-Encoders},
  author={Davidson, Tim R. and Falorsi, Luca and De Cao, Nicola and Kipf, Thomas and Tomczak, Jakub M.},
  journal={34th Conference on Uncertainty in Artificial Intelligence (UAI-18)},
  year={2018}
}
```

## License

MIT
