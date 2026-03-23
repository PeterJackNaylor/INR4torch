# INR4torch

A PyTorch framework for Physics-Informed Neural Networks (PINNs) using
Implicit Neural Representations (INRs).

## Features

- **Multiple architectures**: SIREN, Random Fourier Features (RFF),
  WIRE (complex Gabor wavelets), Multiplicative Filter Networks (MFN),
  and Kolmogorov-Arnold Networks (KAN)
- **Advanced training**: temporal causality weighting, self-adapting
  loss balancing, ReLoBRaLo, cosine annealing, gradient clipping
- **Mixed-precision training**: automatic FP16/BF16 with GradScaler
- **Hyperparameter search**: Optuna integration with pruning
- **Modular design**: extend `DataPlaceholder` for your data,
  extend `DensityEstimator` for your PDE losses

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install .
```

## Quick Start

```python
import pinns
from pinns.models import INR

# Load configuration
hp = pinns.read_yaml('default-parameters.yml')

# Define your dataset function
def dataset_fn(hp, gpu=False):
    train_data = MyDataLoader(train_path, gpu=gpu, bs=hp.losses['mse']['bs'])
    test_data = MyDataLoader(test_path, gpu=gpu, bs=hp.losses['mse']['bs'],
                             need_target=True)
    return train_data, test_data

# Train
NN, hp = pinns.train(hp, MyPDEEstimator, dataset_fn, INR, gpu=True)
```

## Architecture

```
pinns/
├── __init__.py           # Public API
├── models.py             # INR model (top-level factory)
├── model_utils.py        # Layer implementations (SIREN, RFF, WIRE, MFN, KAN, ...)
├── data_loader.py        # DataPlaceholder base class
├── density_estimation.py # DensityEstimator training engine
├── training.py           # train() orchestration function
├── diff_operators.py     # gradient, hessian, laplace, divergence, jacobian
├── pde_utils.py          # gen_uniform sampling
├── parser.py             # YAML config loading (AttrDict, read_yaml)
└── kan_utils.py          # KAN layer implementation
```

## Extending for Your PDE

### 1. Create a DataLoader

Subclass `DataPlaceholder` and override `setup_data()`:

```python
from pinns import DataPlaceholder

class MyData(DataPlaceholder):
    def setup_data(self, pc):
        samples = pc[:, :2]   # (t, x)
        targets = pc[:, 2:3]  # u
        self.input_size = 2
        self.output_size = 1
        self.test = False
        return samples.astype(np.float32), targets.astype(np.float32)
```

### 2. Create a PDE Loss

Subclass `DensityEstimator` and add loss methods:

```python
from pinns import DensityEstimator, gen_uniform, gradient

class MyPDE(DensityEstimator):
    def pde(self, zhat, z, **kwargs):
        bs = self.hp.losses['pde']['bs']
        t = gen_uniform(bs, self.device, start=-1, end=1)
        x = gen_uniform(bs, self.device, start=-1, end=1)
        t.requires_grad_(True)
        x.requires_grad_(True)
        u = self.model(t, x)
        du_dt = gradient(u, t)
        du_dx = gradient(u, x)
        residual = du_dt + du_dx  # advection equation
        return self.L2(residual)
```

### 3. Configure and Train

```yaml
# my_config.yml
max_iters: 5000
model:
  name: SIREN
  hidden_nlayers: 4
  hidden_width: 256
losses:
  mse:
    bs: 64
    report: True
    loss_balancing: True
  pde:
    bs: 4096
    method: pde
    penalty: L2
    loss_balancing: True
```

## Examples

The `example/` directory contains complete implementations:

- **advection/**: 1D advection equation with periodic boundaries
- **QG/**: 2D quasi-geostrophic equations (ocean dynamics)
- **Hydro/**: Hydrological water balance conservation

## Supported Models

| Model | Description | Reference |
|-------|-------------|-----------|
| SIREN | Sinusoidal activations | Sitzmann et al. (NeurIPS 2020) |
| RFF | Random Fourier Features | Tancik et al. (NeurIPS 2020) |
| WIRE | Complex Gabor wavelets | Saragadam et al. (CVPR 2023) |
| MFN | Multiplicative Filter Networks | Fathony et al. (ICLR 2021) |
| KAN | Kolmogorov-Arnold Networks | Liu et al. (2024) |

## Benchmark Results

Best score obtained on Advection with RFF (4 layers, 256 neurons): 3.8227 x 10^-2

Best score obtained on Advection with KAN (1 layer, 32 neurons): 8.8881 x 10^-3

## License

MIT License -- see [LICENSE](LICENSE).
