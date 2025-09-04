import numpy as np
import torch
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)

from CCMetrics.CC_base import CCBaseMetric, CCDiceMetric

# Globally disable gradient computation for this entire module
torch.set_grad_enabled(False)

try:
    import cupy as cp

    cp.ones(3)  # Test if CuPy is properly installed and can access GPU
except ImportError:
    raise ImportError(
        "CuPy is required for CCBaseMetricGPU. Please install CuPy to use this feature."
    )

assert (
    cp.cuda.is_available()
), "CUDA is not available. Please ensure you have a compatible GPU and CUDA installed."
from CCMetrics.space_separation_on_gpu import compute_voronoi_regions_fast_on_gpu


class CCBaseMetricGPU(CCBaseMetric):
    """
    CCBaseMetricGPU is a class that represents the base metric for connected components on GPU.
    The computation of the Metric stays within Monai and the CPU, but preprocessing is done on GPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xp = cp
        self.backend = "cupy"
        self.space_separation = compute_voronoi_regions_fast_on_gpu

    def _verify_and_convert(self, y_pred, y):
        # Automatically convert to cupy if numpy is given
        if isinstance(y_pred, np.ndarray):
            y_pred = cp.asarray(y_pred)
        if isinstance(y, np.ndarray):
            y = cp.asarray(y)

        # Automatically convert to cupy if torch is given
        if isinstance(y_pred, torch.Tensor):
            if y_pred.is_cuda:
                y_pred = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y_pred))
            else:
                y_pred = cp.asarray(y_pred.detach().numpy())
        if isinstance(y, torch.Tensor):
            if y.is_cuda:
                y = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y))
            else:
                y = cp.asarray(y.detach().numpy())

        assert isinstance(
            y_pred, cp.ndarray
        ), f"y_pred must be a cupy array, numpy array or torch tensor. Got {type(y_pred)}"
        assert isinstance(
            y, cp.ndarray
        ), f"y must be a cupy array, numpy array or torch tensor. Got {type(y)}"

        # Check conditions
        assert (
            len(y_pred.shape) == 5
        ), "Input shape is not correct. Expected shape: (B,C,D,H,W) as input y_pred"
        assert (
            len(y.shape) == 5
        ), "Input shape is not correct. Expected shape: (B,C,D,H,W) as input y"
        assert (
            y_pred.shape == y.shape
        ), f"Input shapes do not match. Got {y_pred.shape} and {y.shape}"
        assert (
            y_pred.shape[1] == 2
        ), f"Expected two classes in the input. Got {y_pred.shape[1]}"
        assert y.shape[1] == 2, f"Expected two classes in the input. Got {y.shape[1]}"
        assert (
            y_pred.shape[0] == 1
        ), f"Currently only a batch size of 1 is supported. Got {y_pred.shape[0]} in y_pred"
        assert (
            y.shape[0] == 1
        ), f"Currently only a batch size of 1 is supported. Got {y.shape[0]} in y"

        # --- Force consistent dtype once ---
        target_dtype = cp.float64
        y_pred = y_pred.astype(target_dtype, copy=False)
        y = y.astype(target_dtype, copy=False)

        return y_pred, y

    def _convert_to_target(self, y_pred, y):
        # Convert back to torch and move to CPU
        y_pred = torch.from_dlpack(cp.asarray(y_pred).toDlpack()).cpu()
        y = torch.from_dlpack(cp.asarray(y).toDlpack()).cpu()

        return y_pred, y


class CCDiceMetricGPU(CCDiceMetric, CCBaseMetricGPU):
    """
    Uses CCDiceMetric.__call__ (bincount path) unchanged,
    while taking _verify_and_convert/xp/space_separation from CCBaseMetricGPU.
    """

    def __init__(self, *args, **kwargs):
        # Explicitly call the GPU base init instead of CCDiceMetric.__init__
        CCBaseMetricGPU.__init__(
            self,
            DiceMetric,
            metric_best_score=1.0,
            metric_worst_score=0.0,
            **kwargs,
        )


class CCHausdorffDistanceMetricGPU(CCBaseMetricGPU):
    """
    CCHausdorffDistanceMetric is a class that represents the Hausdorff distance metric for connected components on GPU.
    It inherits from the CCBaseMetricGPU class.
    """

    def __init__(self, *args, **kwargs):
        # Explicitly call the GPU base init instead of CCDiceMetric.__init__
        super().__init__(
            HausdorffDistanceMetric,
            *args,
            metric_best_score=0.0,
            metric_worst_score=50.0,
            **kwargs,
        )


class CCHausdorffDistance95MetricGPU(CCBaseMetricGPU):
    """
    A class representing a metric for calculating the 95th percentile Hausdorff distance for connected components on GPU.
    It inherits from the CCBaseMetricGPU class.
    """

    def __init__(self, *args, **kwargs):
        # Explicitly call the GPU base init instead of CCDiceMetric.__init__
        super().__init__(
            HausdorffDistanceMetric,
            *args,
            metric_best_score=0.0,
            percentile=95,
            metric_worst_score=50.0,
            **kwargs,
        )


class CCSurfaceDistanceMetricGPU(CCBaseMetricGPU):
    """
    A class representing a metric for calculating the SurfaceDistance metric for connected components on GPU.
    It inherits from the CCBaseMetricGPU class.
    """

    def __init__(self, *args, **kwargs):
        # Explicitly call the GPU base init instead of CCDiceMetric.__init__
        super().__init__(
            SurfaceDistanceMetric,
            *args,
            metric_best_score=0.0,
            metric_worst_score=50.0,
            **kwargs,
        )


class CCSurfaceDiceMetricGPU(CCBaseMetricGPU):
    """
    A class representing a metric for calculating the SurfaceDiceMetric metric for connected components on GPU.
    It inherits from the CCBaseMetricGPU class.
    """

    def __init__(self, *args, **kwargs):
        # Explicitly call the GPU base init instead of CCDiceMetric.__init__
        super().__init__(
            SurfaceDiceMetric,
            *args,
            metric_best_score=1.0,
            metric_worst_score=0.0,
            **kwargs,
        )
