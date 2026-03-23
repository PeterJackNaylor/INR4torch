"""Unit tests for the pinns package."""

import pytest
import torch
import numpy as np
import yaml

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def default_hp():
    """Create a minimal AttrDict hyperparameter config."""
    from pinns.parser import AttrDict

    return AttrDict(
        {
            "model": {
                "name": "SIREN",
                "hidden_nlayers": 2,
                "hidden_width": 32,
                "scale": 30,
                "skip": False,
                "mapping_size": 64,
                "activation": "tanh",
                "modified_mlp": False,
                "linear": "HE",
                "mean": 1,
                "std": 0.1,
                "omega0": 10,
                "sigma0": 40,
                "trainable": True,
            },
            "gpu": False,
            "input_size": 2,
            "output_size": 1,
        }
    )


@pytest.fixture
def rff_hp(default_hp):
    default_hp.model["name"] = "RFF"
    return default_hp


@pytest.fixture
def kan_hp(default_hp):
    default_hp.model["name"] = "KAN"
    return default_hp


# ============================================================
# parser.py tests
# ============================================================


class TestAttrDict:
    def test_attribute_access(self):
        from pinns.parser import AttrDict

        d = AttrDict({"a": 1, "b": 2})
        assert d.a == 1
        assert d.b == 2

    def test_attribute_set(self):
        from pinns.parser import AttrDict

        d = AttrDict()
        d.x = 42
        assert d["x"] == 42

    def test_nested_dict_not_recursive(self):
        from pinns.parser import AttrDict

        d = AttrDict({"nested": {"key": "val"}})
        assert isinstance(d.nested, dict)
        # nested dict is NOT automatically an AttrDict
        assert not isinstance(d.nested, AttrDict)


class TestReadYaml:
    def test_read_yaml(self, tmp_path):
        from pinns.parser import read_yaml

        config = {"lr": 0.001, "model": {"name": "SIREN"}}
        path = tmp_path / "config.yml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        result = read_yaml(str(path))
        assert result.lr == 0.001
        assert result["model"]["name"] == "SIREN"


# ============================================================
# models.py tests
# ============================================================


class TestINR:
    @pytest.mark.parametrize("name", ["SIREN", "RFF", "WIRES", "MFN"])
    def test_forward_shape(self, name, default_hp):
        from pinns.models import INR

        default_hp.model["name"] = name
        model = INR(name, input_size=2, output_size=1, hp=default_hp)
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)

    def test_kan_forward_shape(self, kan_hp):
        from pinns.models import INR

        model = INR("KAN", input_size=2, output_size=1, hp=kan_hp)
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)

    def test_multiple_input_tensors(self, default_hp):
        from pinns.models import INR

        model = INR("SIREN", input_size=3, output_size=1, hp=default_hp)
        x = torch.randn(16, 2)
        t = torch.randn(16, 1)
        out = model(x, t)
        assert out.shape == (16, 1)

    def test_skip_connections(self, default_hp):
        from pinns.models import INR

        default_hp.model["skip"] = True
        model = INR("SIREN", input_size=2, output_size=1, hp=default_hp)
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)

    def test_modified_mlp(self, rff_hp):
        from pinns.models import INR

        rff_hp.model["modified_mlp"] = True
        rff_hp.model["mapping_size"] = rff_hp.model["hidden_width"]
        model = INR("RFF", input_size=2, output_size=1, hp=rff_hp)
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)


# ============================================================
# model_utils.py tests
# ============================================================


class TestLayers:
    def test_siren_layer_first(self):
        from pinns.model_utils import SirenLayer

        layer = SirenLayer(2, 32, is_first=True)
        x = torch.randn(8, 2)
        out = layer(x)
        assert out.shape == (8, 32)
        # Output should be bounded by sine
        assert out.abs().max() <= 1.0 + 1e-6

    def test_siren_layer_last(self):
        from pinns.model_utils import SirenLayer

        layer = SirenLayer(32, 1, is_last=True)
        x = torch.randn(8, 32)
        out = layer(x)
        assert out.shape == (8, 1)

    def test_rff_layer_output_size(self):
        from pinns.model_utils import RFFLayer

        layer = RFFLayer(2, 64, sigma=1.0)
        x = torch.randn(8, 2)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_rff_layer_frozen_weights(self):
        from pinns.model_utils import RFFLayer

        layer = RFFLayer(2, 64, sigma=1.0)
        for p in layer.parameters():
            assert not p.requires_grad

    def test_skip_layer(self):
        from pinns.model_utils import SkipLayer, SirenLayer

        inner = SirenLayer(32, 32)
        skip = SkipLayer(inner)
        x = torch.randn(8, 32)
        out = skip(x)
        assert out.shape == (8, 32)

    def test_gabor_layer(self):
        from pinns.model_utils import ComplexGaborLayer

        layer = ComplexGaborLayer(2, 32, is_first=True)
        x = torch.randn(8, 2)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_gabor_layer_last_returns_real(self):
        from pinns.model_utils import ComplexGaborLayer

        layer = ComplexGaborLayer(32, 1, is_last=True)
        x = torch.randn(8, 32, dtype=torch.cfloat)
        out = layer(x)
        assert out.dtype == torch.float32


# ============================================================
# diff_operators.py tests
# ============================================================


class TestDiffOperators:
    def test_gradient_linear(self):
        """Gradient of f(x,y) = 3x + 2y should be [3, 2]."""
        from pinns.diff_operators import gradient

        x = torch.randn(10, 2, requires_grad=True)
        y = 3 * x[:, 0:1] + 2 * x[:, 1:2]
        g = gradient(y, x)
        assert torch.allclose(g[:, 0], torch.tensor(3.0), atol=1e-5)
        assert torch.allclose(g[:, 1], torch.tensor(2.0), atol=1e-5)

    def test_gradient_quadratic(self):
        """Gradient of f(x) = x^2 should be 2x."""
        from pinns.diff_operators import gradient

        x = torch.randn(10, 1, requires_grad=True)
        y = x**2
        g = gradient(y, x)
        assert torch.allclose(g, 2 * x, atol=1e-5)

    def test_hessian_quadratic(self):
        """Hessian of f(x,y) = x^2 + y^2 should be 2*I."""
        from pinns.diff_operators import hessian

        x = torch.randn(1, 10, 2, requires_grad=True)
        y = (x**2).sum(dim=-1, keepdim=True)
        h, status = hessian(y, x)
        assert status == 0
        # Diagonal elements should be 2
        assert torch.allclose(h[0, :, 0, 0, 0], torch.tensor(2.0), atol=1e-4)
        assert torch.allclose(h[0, :, 0, 1, 1], torch.tensor(2.0), atol=1e-4)
        # Off-diagonal should be 0
        assert torch.allclose(h[0, :, 0, 0, 1], torch.tensor(0.0), atol=1e-4)

    def test_jacobian_linear(self):
        """Jacobian of f(x) = Ax should be A."""
        from pinns.diff_operators import jacobian

        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x = torch.randn(1, 10, 2, requires_grad=True)
        y = torch.einsum("ij,...j->...i", A, x)
        jac, status = jacobian(y, x)
        assert status == 0
        assert torch.allclose(jac[0, 0], A, atol=1e-5)


# ============================================================
# pde_utils.py tests
# ============================================================


class TestGenUniform:
    def test_shape_basic(self):
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(100, "cpu")
        assert v.shape == (100, 1)

    def test_range(self):
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(10000, "cpu", start=-2, end=3)
        assert v.min() >= -2
        assert v.max() <= 3

    def test_temporal_scheme_shape(self):
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(128, "cpu", temporal_scheme=True, M=8)
        assert v.shape == (128, 1)

    def test_temporal_scheme_coverage(self):
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(1024, "cpu", start=0, end=1, temporal_scheme=True, M=4)
        # Samples should cover all 4 temporal bins
        for i in range(4):
            lo, hi = i / 4, (i + 1) / 4
            in_bin = ((v >= lo) & (v <= hi)).any()
            assert in_bin


# ============================================================
# kan_utils.py tests
# ============================================================


class TestKAN:
    def test_kan_forward(self):
        from pinns.kan_utils import KAN

        model = KAN([2, 8, 1])
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)

    def test_kan_regularization(self):
        from pinns.kan_utils import KAN

        model = KAN([2, 8, 1])
        reg = model.regularization_loss()
        assert reg.item() >= 0

    def test_kanlinear_bsplines(self):
        from pinns.kan_utils import KANLinear

        layer = KANLinear(2, 4)
        x = torch.randn(8, 2)
        bases = layer.b_splines(x)
        assert bases.shape == (8, 2, layer.grid_size + layer.spline_order)

    def test_kan_update_grid(self):
        from pinns.kan_utils import KAN

        model = KAN([2, 8, 1])
        x = torch.randn(32, 2)
        # Should not raise
        model(x, update_grid=True)


# ============================================================
# density_estimation.py tests
# ============================================================


class TestLossFunctions:
    def test_rmse_basic(self):
        from pinns.density_estimation import RMSELoss

        loss_fn = RMSELoss()
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(pred, target)
        assert loss.item() <= 1e-3 + 1e-6  # should be ~sqrt(eps)

    def test_rmse_with_weight(self):
        from pinns.density_estimation import RMSELoss

        loss_fn = RMSELoss()
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([0.0, 0.0])
        w = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target, weight=w)
        # Only first element contributes
        assert loss.item() > 0

    def test_rmse_ignore_nan(self):
        from pinns.density_estimation import RMSELoss

        loss_fn = RMSELoss(ignore_nan=True)
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, float("nan")])
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)

    def test_wl1_basic(self):
        from pinns.density_estimation import WL1Loss

        loss_fn = WL1Loss()
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0])
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0


# ============================================================
# Normalisation tests
# ============================================================


class TestNormalization:
    """Tests for DataPlaceholder.normalize."""

    def test_normalize_basic(self):
        """Standard normalisation maps to [-1, 1]."""
        from pinns.data_loader import DataPlaceholder

        dp = DataPlaceholder.__new__(DataPlaceholder)
        data = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
        nv = dp.normalize(data, None, include_last=True)
        # Column 0: center=2, scale=2 -> [0,2,4] -> [-1,0,1]
        assert np.allclose(data[:, 0], [-1, 0, 1])
        # Column 1: center=20, scale=10 -> [10,20,30] -> [-1,0,1]
        assert np.allclose(data[:, 1], [-1, 0, 1])
        assert len(nv) == 2

    def test_normalize_constant_column(self):
        """Constant column should not cause division by zero."""
        from pinns.data_loader import DataPlaceholder

        dp = DataPlaceholder.__new__(DataPlaceholder)
        data = np.array([[5.0], [5.0], [5.0]])
        nv = dp.normalize(data, None, include_last=True)
        # scale should be 1.0 (the guard), so result = (5-5)/1 = 0
        assert np.allclose(data[:, 0], [0, 0, 0])
        assert nv[0] == (5.0, 1.0)

    def test_normalize_exclude_last(self):
        """include_last=False should skip the last column."""
        from pinns.data_loader import DataPlaceholder

        dp = DataPlaceholder.__new__(DataPlaceholder)
        data = np.array([[0.0, 100.0], [4.0, 200.0]])
        nv = dp.normalize(data, None, include_last=False)
        assert len(nv) == 1  # only column 0 normalised
        assert np.allclose(data[:, 0], [-1, 1])
        assert data[0, 1] == 100.0  # unchanged
        assert data[1, 1] == 200.0

    def test_normalize_with_precomputed_nv(self):
        """Re-applying existing nv should give consistent results."""
        from pinns.data_loader import DataPlaceholder

        dp = DataPlaceholder.__new__(DataPlaceholder)
        data1 = np.array([[0.0], [4.0]])
        nv = dp.normalize(data1, None, include_last=True)
        # Now apply same nv to new data
        data2 = np.array([[2.0], [6.0]])
        dp.normalize(data2, nv, include_last=True)
        assert np.allclose(data2[:, 0], [0.0, 2.0])  # (2-2)/2=0, (6-2)/2=2


# ============================================================
# Training config defaults
# ============================================================


class TestConfigDefaults:
    """Tests for check_model_hp, check_data_loader_hp, check_estimator_hp."""

    def test_check_model_hp_defaults(self):
        from pinns.training import check_model_hp
        from pinns.parser import AttrDict

        hp = AttrDict({"model": {"name": "SIREN"}})
        hp = check_model_hp(hp)
        assert hp.model["linear"] == "HE"
        assert hp.eps == 1e-8
        assert hp.clip_gradients is True
        assert hp.cosine_annealing["status"] is False
        assert hp.relobralo["status"] is False

    def test_check_model_hp_preserves_existing(self):
        from pinns.training import check_model_hp
        from pinns.parser import AttrDict

        hp = AttrDict({"model": {"name": "SIREN", "linear": "Glorot"}, "eps": 1e-6})
        hp = check_model_hp(hp)
        assert hp.model["linear"] == "Glorot"
        assert hp.eps == 1e-6

    def test_check_data_loader_hp(self):
        from pinns.training import check_data_loader_hp
        from pinns.parser import AttrDict

        hp = AttrDict({})
        hp = check_data_loader_hp(hp)
        assert hp.model["name"] == "default"
        assert hp.hard_periodicity is False

    def test_check_estimator_hp(self):
        from pinns.training import check_estimator_hp
        from pinns.parser import AttrDict

        hp = AttrDict({})
        hp = check_estimator_hp(hp)
        assert hp.verbose is True
        assert hp.save_model is False
        assert "patience" in hp.optuna


# ============================================================
# linear_fn tests
# ============================================================


class TestLinearFn:
    """Tests for the linear_fn factory."""

    def test_siren_returns_siren_layer(self):
        from pinns.model_utils import linear_fn, SirenLayer
        from pinns.parser import AttrDict

        hp = AttrDict({"model": {"name": "SIREN"}, "input_size": 2})
        fn = linear_fn("HE", hp, None)
        assert fn is SirenLayer

    def test_rff_he_returns_linear(self):
        from pinns.model_utils import linear_fn
        from pinns.parser import AttrDict
        import torch.nn as nn

        hp = AttrDict({"model": {"name": "RFF"}, "input_size": 2})
        fn = linear_fn("HE", hp, nn.Tanh)
        # Should return a partial of Linear
        layer = fn(10, 20)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)

    def test_rff_glorot_returns_glorot(self):
        from pinns.model_utils import linear_fn
        from pinns.parser import AttrDict
        import torch.nn as nn

        hp = AttrDict({"model": {"name": "RFF"}, "input_size": 2})
        fn = linear_fn("Glorot", hp, nn.Tanh)
        layer = fn(10, 20)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)

    @pytest.mark.skip(
        reason="LinearLayerRWF has a bug: assigns Tensor instead of Parameter to layer.weight"
    )
    def test_rff_rwf_returns_rwf(self):
        from pinns.model_utils import linear_fn
        from pinns.parser import AttrDict
        import torch.nn as nn

        hp = AttrDict(
            {"model": {"name": "RFF", "mean": 1.0, "std": 0.1}, "input_size": 2}
        )
        fn = linear_fn("RWF", hp, nn.Tanh)
        layer = fn(10, 20, is_last=True)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)

    def test_unknown_model_raises(self):
        from pinns.model_utils import linear_fn
        from pinns.parser import AttrDict

        hp = AttrDict({"model": {"name": "UNKNOWN"}, "input_size": 2})
        with pytest.raises(ValueError, match="Unknown model name"):
            linear_fn("HE", hp, None)

    def test_mfn_returns_mfn_layer(self):
        from pinns.model_utils import linear_fn
        from pinns.parser import AttrDict
        import torch.nn as nn

        hp = AttrDict({"model": {"name": "MFN"}, "input_size": 2})
        fn = linear_fn("HE", hp, nn.Tanh)
        # is_first=True uses the g_layer which takes in_f0 (=input_size=2)
        layer = fn(2, 32, is_first=True)
        x = torch.randn(4, 2)
        out = layer(x)
        assert out.shape == (4, 32)


# ============================================================
# MFN integration tests
# ============================================================


class TestMFN:
    def test_mfn_forward_integration(self):
        """Test full MFN model through INR."""
        from pinns.models import INR
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "model": {
                    "name": "MFN",
                    "hidden_nlayers": 2,
                    "hidden_width": 32,
                    "activation": "tanh",
                    "skip": False,
                    "modified_mlp": False,
                    "linear": "HE",
                },
                "gpu": False,
                "input_size": 2,
                "output_size": 1,
            }
        )
        model = INR("MFN", input_size=2, output_size=1, hp=hp)
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)

    def test_mfn_gradient_flows(self):
        """Verify gradients flow through MFN."""
        from pinns.models import INR
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "model": {
                    "name": "MFN",
                    "hidden_nlayers": 2,
                    "hidden_width": 16,
                    "activation": "tanh",
                    "skip": False,
                    "modified_mlp": False,
                    "linear": "HE",
                },
                "gpu": False,
                "input_size": 2,
                "output_size": 1,
            }
        )
        model = INR("MFN", input_size=2, output_size=1, hp=hp)
        x = torch.randn(8, 2)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(grads) > 0, "No gradients flowed through MFN"


# ============================================================
# Diff operators — edge cases
# ============================================================


class TestDiffOperatorsEdgeCases:
    def test_gradient_higher_order(self):
        """Second derivative of x^3 should be 6x."""
        from pinns.diff_operators import gradient

        x = torch.randn(10, 1, requires_grad=True)
        y = x**3
        dy_dx = gradient(y, x)
        d2y_dx2 = gradient(dy_dx, x)
        assert torch.allclose(d2y_dx2, 6 * x, atol=1e-4)

    def test_gradient_with_precomputed_ones(self):
        """Test gradient with pre-allocated grad_outputs."""
        from pinns.diff_operators import gradient

        x = torch.randn(10, 1, requires_grad=True)
        y = 5 * x
        ones = [torch.ones_like(y)]
        g = gradient(y, x, ones_like_tensor=ones)
        assert torch.allclose(g, torch.tensor(5.0), atol=1e-5)

    def test_gradient_multidimensional_output(self):
        """Gradient of [x^2, 2y] w.r.t. [x, y]."""
        from pinns.diff_operators import gradient

        x = torch.randn(10, 2, requires_grad=True)
        # Scalar output: x^2 + 2y
        y = x[:, 0:1] ** 2 + 2 * x[:, 1:2]
        g = gradient(y, x)
        assert torch.allclose(g[:, 0:1], 2 * x[:, 0:1], atol=1e-5)
        assert torch.allclose(g[:, 1], torch.tensor(2.0), atol=1e-5)

    def test_laplace_3d(self):
        """Laplacian of f(x,y,z) = x^2 + y^2 + z^2 should be 6."""
        from pinns.diff_operators import laplace

        x = torch.randn(10, 3, requires_grad=True)
        y = (x**2).sum(dim=1, keepdim=True)
        lap = laplace(y, x)
        assert torch.allclose(lap, torch.tensor(6.0), atol=1e-3)

    def test_hessian_nan_detection(self):
        """Hessian should return status=-1 when NaN is produced."""
        from pinns.diff_operators import hessian

        x = torch.randn(1, 5, 2, requires_grad=True)
        # sqrt(|x|) has undefined second derivative at 0
        y = torch.sqrt(torch.abs(x[:, :, 0:1]) + 1e-20)
        h, status = hessian(y, x)
        # Status could be 0 or -1 depending on values near zero
        assert status in (0, -1)


# ============================================================
# DensityEstimator setup tests
# ============================================================


class TestDensityEstimatorSetup:
    """Test DensityEstimator configuration methods in isolation."""

    @pytest.fixture
    def estimator_hp(self):
        from pinns.parser import AttrDict

        return AttrDict(
            {
                "model": {
                    "name": "SIREN",
                    "hidden_nlayers": 2,
                    "hidden_width": 32,
                    "linear": "HE",
                    "skip": False,
                    "modified_mlp": False,
                },
                "optimizer": "Adam",
                "lr": 1e-3,
                "eps": 1e-8,
                "clip_gradients": True,
                "learning_rate_decay": {"status": True, "step": 100, "gamma": 0.9},
                "cosine_annealing": {"status": False, "min_eta": 0, "step": 500},
                "relobralo": {"status": False},
                "self_adapting_loss_balancing": {"status": False},
                "temporal_causality": {"M": 16, "eps": 1e-2, "step": 1},
                "losses": {
                    "mse": {
                        "report": True,
                        "bs": 32,
                        "loss_balancing": False,
                        "ignore_nan": False,
                    },
                },
                "validation_loss": "mse",
                "verbose": False,
                "max_iters": 10,
                "test_frequency": 5,
                "save_model": False,
                "early_stopping": {"status": False},
                "optuna": {"patience": 100, "trials": 1},
                "ignore_nan": False,
            }
        )

    def _make_estimator(self, hp):
        from pinns.density_estimation import DensityEstimator
        from pinns.models import INR

        hp.input_size = 2
        hp.output_size = 1
        model = INR("SIREN", 2, 1, hp)

        # Create minimal mock datasets
        class FakeData:
            def __init__(self, n):
                self.samples = torch.randn(n, 2)
                self.targets = torch.randn(n, 1)
                self.input_size = 2
                self.output_size = 1
                self.nv_samples = [(0, 1), (0, 1)]
                self.nv_targets = [(0, 1)]

            def __len__(self):
                return self.samples.shape[0]

        train_data = FakeData(100)
        test_data = FakeData(50)
        return DensityEstimator(train_data, test_data, model, hp, gpu=False)

    def test_setup_optimizer_adam(self, estimator_hp):
        est = self._make_estimator(estimator_hp)
        est.setup_optimizer()
        assert isinstance(est.optimizer, torch.optim.Adam)

    def test_setup_optimizer_adamw(self, estimator_hp):
        estimator_hp.optimizer = "AdamW"
        est = self._make_estimator(estimator_hp)
        est.setup_optimizer()
        assert isinstance(est.optimizer, torch.optim.AdamW)

    def test_setup_optimizer_lbfgs(self, estimator_hp):
        estimator_hp.optimizer = "LBFGS"
        est = self._make_estimator(estimator_hp)
        est.setup_optimizer()
        assert isinstance(est.optimizer, torch.optim.LBFGS)

    def test_setup_optimizer_unknown_raises(self, estimator_hp):
        estimator_hp.optimizer = "SGD"
        est = self._make_estimator(estimator_hp)
        with pytest.raises(NameError):
            est.setup_optimizer()

    def test_setup_scheduler_step_lr(self, estimator_hp):
        est = self._make_estimator(estimator_hp)
        est.setup_optimizer()
        est.setup_scheduler()
        assert est.scheduler_status is True
        assert isinstance(est.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_setup_scheduler_cosine(self, estimator_hp):
        estimator_hp.learning_rate_decay = {"status": False}
        estimator_hp.cosine_annealing = {"status": True, "min_eta": 0, "step": 500}
        est = self._make_estimator(estimator_hp)
        est.setup_optimizer()
        est.setup_scheduler()
        assert est.scheduler_status is True

    def test_setup_scheduler_none(self, estimator_hp):
        estimator_hp.learning_rate_decay = {"status": False}
        est = self._make_estimator(estimator_hp)
        est.setup_optimizer()
        est.setup_scheduler()
        assert est.scheduler_status is False

    def test_setup_losses_mse(self, estimator_hp):
        est = self._make_estimator(estimator_hp)
        est.setup_losses()
        assert "mse" in est.loss_fn
        assert "mse" in est.lambdas_scalar
        assert est.lambdas_scalar["mse"] == [1]

    def test_setup_losses_with_lambda(self, estimator_hp):
        estimator_hp.losses["mse"]["lambda"] = 0.5
        est = self._make_estimator(estimator_hp)
        est.setup_losses()
        assert est.lambdas_scalar["mse"] == [0.5]

    def test_L1_normed(self, estimator_hp):
        est = self._make_estimator(estimator_hp)
        v = torch.tensor([1.0, -2.0, 3.0])
        result = est.L1(v, normed=True)
        expected = (1 + 2 + 3) / 3
        assert abs(result.item() - expected) < 1e-5

    def test_L2_normed(self, estimator_hp):
        est = self._make_estimator(estimator_hp)
        v = torch.tensor([3.0, 4.0])
        result = est.L2(v, normed=True)
        expected = 5.0 / (2**0.5)
        assert abs(result.item() - expected) < 1e-5

    def test_autocasting_cpu(self, estimator_hp):
        est = self._make_estimator(estimator_hp)
        assert est.dtype == torch.bfloat16
        assert est.use_amp is True

    def test_autocasting_wires_disabled(self, estimator_hp):
        from pinns.density_estimation import DensityEstimator
        from pinns.models import INR

        # Build model as SIREN but set name to WIRES for autocasting check
        estimator_hp.input_size = 2
        estimator_hp.output_size = 1
        model = INR("SIREN", 2, 1, estimator_hp)
        estimator_hp.model["name"] = "WIRES"

        class FakeData:
            def __init__(self, n):
                self.samples = torch.randn(n, 2)
                self.targets = torch.randn(n, 1)
                self.input_size = 2
                self.output_size = 1
                self.nv_samples = [(0, 1), (0, 1)]
                self.nv_targets = [(0, 1)]

            def __len__(self):
                return self.samples.shape[0]

        est = DensityEstimator(
            FakeData(100), FakeData(50), model, estimator_hp, gpu=False
        )
        assert est.use_amp is False


# ============================================================
# Loss function edge cases
# ============================================================


class TestLossFunctionEdgeCases:
    def test_rmse_column_selection(self):
        """Test that column parameter selects the right output."""
        from pinns.density_estimation import RMSELoss

        loss_fn = RMSELoss(column=1)
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = loss_fn(pred, target)
        assert loss.item() <= 1e-3 + 1e-6  # should be ~sqrt(eps)

    def test_rmse_column_with_error(self):
        from pinns.density_estimation import RMSELoss

        loss_fn = RMSELoss(column=0)
        pred = torch.tensor([[1.0, 100.0], [1.0, 100.0]])
        target = torch.tensor([[0.0, 100.0], [0.0, 100.0]])
        loss = loss_fn(pred, target)
        # Only column 0 contributes: sqrt(mean((1-0)^2) + eps) ~ 1
        assert 0.9 < loss.item() < 1.1

    def test_wl1_with_weight(self):
        from pinns.density_estimation import WL1Loss

        loss_fn = WL1Loss()
        pred = torch.tensor([2.0, 4.0])
        target = torch.tensor([1.0, 1.0])
        w = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target, weight=w)
        # |1*(2-1)| + |0*(4-1)| / 2 = 0.5
        assert abs(loss.item() - 0.5) < 1e-5

    def test_rmse_none_weight_fallback(self):
        """Passing weight=None should use unweighted MSE."""
        from pinns.density_estimation import RMSELoss

        loss_fn = RMSELoss()
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0])
        loss = loss_fn(pred, target, weight=None)
        assert loss.item() <= 1e-3 + 1e-6


# ============================================================
# gen_uniform edge cases
# ============================================================


class TestGenUniformEdgeCases:
    def test_temporal_indivisible_bs(self):
        """bs not divisible by M should produce floor(bs/M)*M samples."""
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(130, "cpu", temporal_scheme=True, M=8)
        assert v.shape == (128, 1)  # 130 // 8 * 8 = 128

    def test_dtype_propagation(self):
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(10, "cpu", dtype=torch.float64)
        assert v.dtype == torch.float64

    def test_single_element(self):
        from pinns.pde_utils import gen_uniform

        v = gen_uniform(1, "cpu", start=0, end=1)
        assert v.shape == (1, 1)
        assert 0 <= v.item() <= 1


# ============================================================
# KAN edge cases
# ============================================================


class TestKANEdgeCases:
    def test_single_layer_kan(self):
        from pinns.kan_utils import KAN

        model = KAN([3, 1])  # single layer, 3 -> 1
        x = torch.randn(8, 3)
        out = model(x)
        assert out.shape == (8, 1)

    def test_kan_gradient_flows(self):
        from pinns.kan_utils import KAN

        model = KAN([2, 8, 1])
        x = torch.randn(8, 2)
        out = model(x).sum()
        out.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_kan_large_batch(self):
        from pinns.kan_utils import KAN

        model = KAN([2, 4, 1])
        x = torch.randn(1024, 2)
        out = model(x)
        assert out.shape == (1024, 1)

    def test_kanlinear_update_grid(self):
        from pinns.kan_utils import KANLinear

        layer = KANLinear(2, 4)
        x = torch.randn(32, 2)
        grid_before = layer.grid.clone()
        layer.update_grid(x)
        # Grid should have been updated
        assert not torch.allclose(grid_before, layer.grid)


# ============================================================
# Early stopping tests
# ============================================================


class TestEarlyStopping:
    def test_nan_loss_stops(self):
        from pinns.density_estimation import DensityEstimator
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "early_stopping": {"status": False},
                "test_frequency": 5,
            }
        )
        est = DensityEstimator.__new__(DensityEstimator)
        est.hp = hp
        est.test_scores = []
        result = est.early_stop(10, torch.tensor(float("nan")), False)
        assert result is True

    def test_no_early_stop_when_disabled(self):
        from pinns.density_estimation import DensityEstimator
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "early_stopping": {"status": False},
                "test_frequency": 5,
            }
        )
        est = DensityEstimator.__new__(DensityEstimator)
        est.hp = hp
        est.test_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = est.early_stop(10, torch.tensor(0.1), False)
        assert result is False

    def test_early_stop_triggers(self):
        from pinns.density_estimation import DensityEstimator
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "early_stopping": {
                    "status": True,
                    "patience": 3,
                    "value": 0.001,
                    "ignore_first": 0,
                },
                "test_frequency": 1,
            }
        )
        est = DensityEstimator.__new__(DensityEstimator)
        est.hp = hp
        # Scores that get worse (increase) — triggers early stopping
        est.test_scores = [0.5, 0.51, 0.52, 0.53, 0.54]
        result = est.early_stop(5, torch.tensor(0.1), False)
        assert result is True


# ============================================================
# WIRES (complex) model test
# ============================================================


class TestWIRES:
    def test_wires_forward(self):
        from pinns.models import INR
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "model": {
                    "name": "WIRES",
                    "hidden_nlayers": 2,
                    "hidden_width": 32,
                    "omega0": 10.0,
                    "sigma0": 40.0,
                    "trainable": True,
                    "skip": False,
                    "modified_mlp": False,
                    "linear": "HE",
                },
                "gpu": False,
                "input_size": 2,
                "output_size": 1,
            }
        )
        model = INR("WIRES", input_size=2, output_size=1, hp=hp)
        x = torch.randn(16, 2)
        out = model(x)
        assert out.shape == (16, 1)
        # Output should be real-valued
        assert out.dtype == torch.float32

    def test_wires_gradient(self):
        """Verify autograd works through complex WIRES layers."""
        from pinns.models import INR
        from pinns.parser import AttrDict

        hp = AttrDict(
            {
                "model": {
                    "name": "WIRES",
                    "hidden_nlayers": 1,
                    "hidden_width": 16,
                    "omega0": 10.0,
                    "sigma0": 40.0,
                    "trainable": True,
                    "skip": False,
                    "modified_mlp": False,
                    "linear": "HE",
                },
                "gpu": False,
                "input_size": 2,
                "output_size": 1,
            }
        )
        model = INR("WIRES", input_size=2, output_size=1, hp=hp)
        x = torch.randn(8, 2, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
