import os
import os as _os
import shutil
import warnings as _warnings
from typing import Dict, List, Tuple

import pytest
import torch
from monai.metrics import DiceMetric
from torch import Tensor

from CCMetrics import CCDiceMetric, CCHausdorffDistance95Metric, CCSurfaceDiceMetric
from CCMetrics.CC_base import CCBaseMetric

# Detect CuPy/CUDA availability without importing CCMetrics.CC_base_gpu (which asserts)
try:
    import cupy as _cp  # type: ignore

    HAS_CUPY_CUDA = bool(_cp.cuda.is_available())
except Exception:
    HAS_CUPY_CUDA = False


@pytest.fixture(
    params=[
        # Simple case
        {"size": (64, 64, 64), "offset": (1, 1, 1)},
        # Smaller case
        {"size": (32, 32, 32), "offset": (0, 0, 0)},
    ]
)
def sample_data(request) -> Tuple[Tensor, Tensor]:
    """
    Create sample data for testing with different sizes and offsets.

    Args:
        request: Pytest request object containing parameters

    Returns:
        Tuple of (ground truth, prediction) tensors
    """
    size = request.param["size"]
    offset = request.param["offset"]

    y = torch.zeros((1, 2, *size))
    y_hat = torch.zeros((1, 2, *size))

    # Create a simple cube in ground truth
    y[0, 1, 20:25, 20:25, 20:25] = 1
    y[0, 0] = 1 - y[0, 1]

    # Create a slightly offset cube in prediction
    ox, oy, oz = offset
    y_hat[0, 1, 20 + ox : 25 + ox, 20 + oy : 25 + oy, 20 + oz : 25 + oz] = 1
    y_hat[0, 0] = 1 - y_hat[0, 1]

    return y, y_hat


@pytest.fixture(scope="module")
def cache_dir() -> str:
    """
    Create and manage a cache directory for testing.
    Uses a test-specific directory to avoid conflicts with existing caches.

    Returns:
        Path to cache directory
    """
    cache = ".test_metrics_cache"  # Use a specific test cache directory
    try:
        os.makedirs(cache, exist_ok=True)
        yield cache
    finally:
        if os.path.exists(cache):
            try:
                shutil.rmtree(cache)
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not remove test cache directory: {e}")


def test_cc_dice_metric_validity(
    sample_data: Tuple[Tensor, Tensor], cache_dir: str
) -> None:
    """
    Test the Connected Components Dice metric.

    Args:
        sample_data: Tuple of ground truth and predicted tensors
        cache_dir: Directory for caching results
    """
    y, y_hat = sample_data
    metric = CCDiceMetric(use_caching=True, caching_dir=cache_dir)
    metric(y_pred=y_hat, y=y)
    result = metric.cc_aggregate().mean().item()

    assert isinstance(result, float), "Dice score should be a float"
    assert 0 <= result <= 1, "Dice score should be between 0 and 1"


def test_cc_surface_dice_metric_validity(
    sample_data: Tuple[Tensor, Tensor], cache_dir: str
) -> None:
    """
    Test the Connected Components Surface Dice metric.

    Args:
        sample_data: Tuple of ground truth and predicted tensors
        cache_dir: Directory for caching results
    """
    y, y_hat = sample_data
    metric = CCSurfaceDiceMetric(
        use_caching=True, caching_dir=cache_dir, class_thresholds=[1]
    )
    metric(y_pred=y_hat, y=y)
    result = metric.cc_aggregate().mean().item()

    assert isinstance(result, float), "Surface Dice score should be a float"
    assert 0 <= result <= 1, "Surface Dice score should be between 0 and 1"


def test_cc_hausdorff_metric_validity(
    sample_data: Tuple[Tensor, Tensor], cache_dir: str
) -> None:
    """
    Test the Connected Components Hausdorff Distance metric.

    Args:
        sample_data: Tuple of ground truth and predicted tensors
        cache_dir: Directory for caching results
    """
    y, y_hat = sample_data
    metric = CCHausdorffDistance95Metric(
        use_caching=True, caching_dir=cache_dir, metric_worst_score=30
    )
    metric(y_pred=y_hat, y=y)
    result = metric.cc_aggregate().mean().item()

    assert isinstance(result, float), "Hausdorff distance should be a float"
    assert result >= 0, "Hausdorff distance should be non-negative"
    assert result <= 30, "Hausdorff distance should not exceed worst score"


def test_empty_input(cache_dir: str) -> None:
    """
    Test metrics with empty input tensors.

    Args:
        cache_dir: Directory for caching results
    """
    empty_y = torch.zeros((1, 2, 16, 16, 16))
    empty_y_hat = torch.zeros((1, 2, 16, 16, 16))

    metrics: Dict = {
        "dice": CCDiceMetric(use_caching=True, caching_dir=cache_dir),
        "surface_dice": CCSurfaceDiceMetric(
            use_caching=True, caching_dir=cache_dir, class_thresholds=[1]
        ),
        "hd95": CCHausdorffDistance95Metric(
            use_caching=True, caching_dir=cache_dir, metric_worst_score=30
        ),
    }

    for name, metric in metrics.items():
        metric(y_pred=empty_y_hat, y=empty_y)
        result = metric.cc_aggregate().mean().item()
        assert isinstance(result, float), f"{name} should handle empty input"


@pytest.fixture
def multi_patient_data() -> Tuple[List[Tensor], List[Tensor]]:
    """
    Create test data for multiple patients, each with different numbers of components.
    Returns ground truth and prediction tensors for two patients:
    - Patient 1: Two components
    - Patient 2: One component

    Note: Each patient is processed individually due to batch size=1 limitation.

    Returns:
        Tuple of (ground truth list, prediction list) where each list contains
        tensors for individual patients
    """
    shape = (1, 2, 32, 32, 32)

    # Patient 1 data
    y1 = torch.zeros(shape)
    y_hat1 = torch.zeros(shape)

    # Two components for patient 1
    y1[0, 1, 5:10, 5:10, 5:10] = 1  # First component
    y1[0, 1, 20:25, 20:25, 20:25] = 1  # Second component
    y1[0, 0] = 1 - y1[0, 1]

    y_hat1[0, 1, 6:11, 6:11, 6:11] = 1  # Offset first component
    y_hat1[0, 1, 21:26, 21:26, 21:26] = 1  # Offset second component
    y_hat1[0, 0] = 1 - y_hat1[0, 1]

    # Patient 2 data
    y2 = torch.zeros(shape)
    y_hat2 = torch.zeros(shape)

    # One component for patient 2
    y2[0, 1, 15:20, 15:20, 15:20] = 1
    y2[0, 0] = 1 - y2[0, 1]

    y_hat2[0, 1, 16:21, 16:21, 16:21] = 1
    y_hat2[0, 0] = 1 - y_hat2[0, 1]

    return [y1, y2], [y_hat1, y_hat2]


def test_cc_dice_aggregation_modes(
    cache_dir: str, multi_patient_data: Tuple[List[Tensor], List[Tensor]]
) -> None:
    """
    Test CCDiceMetric aggregation modes and numerical correctness.

    Tests two key aspects:
    1. Aggregation behavior: 'patient' vs 'overall' reduction modes
    2. Numerical correctness: Dice scores for 5x5x5 cubes with 1-voxel offset
       Expected score = 2|X∩Y|/(|X|+|Y|) = 2(64)/(125+125) = 0.512
    """
    y_list, y_hat_list = multi_patient_data
    expected_dice = 0.512  # theoretical value for 1-voxel offset
    tolerance = 0.01

    # Test both aggregation modes with same data
    metric_patient = CCDiceMetric(
        use_caching=True, caching_dir=cache_dir, cc_reduction="patient"
    )
    metric_overall = CCDiceMetric(
        use_caching=True, caching_dir=cache_dir, cc_reduction="overall"
    )

    for y, y_hat in zip(y_list, y_hat_list):
        metric_patient(y_pred=y_hat, y=y)
        metric_overall(y_pred=y_hat, y=y)

    # Get per-patient scores
    patient_scores = metric_patient.cc_aggregate().tolist()

    # Get per-component scores
    component_scores = metric_overall.cc_aggregate().tolist()

    print(f"Patient-level scores: {patient_scores}")
    print(f"Component-level scores: {component_scores}")

    # Test 1: Verify aggregation behavior
    assert len(patient_scores) == 2, "Should have one score per patient"
    assert len(component_scores) == 3, "Should have one score per component"

    # Test 2: Verify numerical correctness
    assert all(
        abs(score - expected_dice) < tolerance
        for score in patient_scores + component_scores
    ), "All scores should match theoretical value"


def _import_gpu_dice():
    """Helper to import GPU Dice metric lazily when CUDA is available."""
    if not HAS_CUPY_CUDA:
        pytest.skip("CuPy/CUDA not available; skipping GPU tests")
    # Import only when CUDA is present to avoid ImportError/asserts
    from CCMetrics.CC_base_gpu import CCDiceMetricGPU  # type: ignore

    return CCDiceMetricGPU


def test_gpu_dice_matches_cpu_single(sample_data: Tuple[Tensor, Tensor]) -> None:
    """
    Ensure GPU CCDiceMetric computes the same as CPU CCDiceMetric on a single sample.
    Skips if CuPy/CUDA are not available.
    """
    CCDiceMetricGPU = _import_gpu_dice()

    y, y_hat = sample_data
    cpu = CCDiceMetric(cc_reduction="patient")
    gpu = CCDiceMetricGPU(cc_reduction="patient")

    cpu(y_pred=y_hat, y=y)
    gpu(y_pred=y_hat, y=y)

    cpu_val = float(cpu.cc_aggregate().mean().item())
    gpu_val = float(gpu.cc_aggregate().mean().item())
    assert abs(cpu_val - gpu_val) < 1e-6


def test_gpu_dice_matches_cpu_multi(
    multi_patient_data: Tuple[List[Tensor], List[Tensor]]
) -> None:
    """
    Ensure GPU CCDiceMetric matches CPU for both 'patient' and 'overall' aggregations.
    Skips if CuPy/CUDA are not available.
    """
    CCDiceMetricGPU = _import_gpu_dice()

    y_list, y_hat_list = multi_patient_data

    cpu_patient = CCDiceMetric(cc_reduction="patient")
    gpu_patient = CCDiceMetricGPU(cc_reduction="patient")

    cpu_overall = CCDiceMetric(cc_reduction="overall")
    gpu_overall = CCDiceMetricGPU(cc_reduction="overall")

    for y, y_hat in zip(y_list, y_hat_list):
        cpu_patient(y_pred=y_hat, y=y)
        gpu_patient(y_pred=y_hat, y=y)
        cpu_overall(y_pred=y_hat, y=y)
        gpu_overall(y_pred=y_hat, y=y)

    cp_p = cpu_patient.cc_aggregate().detach().cpu().numpy()
    gp_p = gpu_patient.cc_aggregate().detach().cpu().numpy()
    assert cp_p.shape == gp_p.shape
    assert ((cp_p - gp_p) ** 2).sum() < 1e-10

    cp_o = cpu_overall.cc_aggregate().detach().cpu().numpy()
    gp_o = gpu_overall.cc_aggregate().detach().cpu().numpy()
    assert cp_o.shape == gp_o.shape
    assert ((cp_o - gp_o) ** 2).sum() < 1e-10


def test_dice_consistency_cpu_single(sample_data: Tuple[Tensor, Tensor]) -> None:
    """
    Compare CCDiceMetric (vectorized bincount) with CCBaseMetric(DiceMetric)
    on a single sample to ensure both implementations agree.
    """
    y, y_hat = sample_data

    cc_fast = CCDiceMetric(cc_reduction="patient")
    cc_ref = CCBaseMetric(
        DiceMetric,
        cc_reduction="patient",
        metric_best_score=1.0,
        metric_worst_score=0.0,
    )

    cc_fast(y_pred=y_hat, y=y)
    cc_ref(y_pred=y_hat, y=y)

    fast_val = cc_fast.cc_aggregate().detach().cpu().numpy()
    ref_val = cc_ref.cc_aggregate().detach().cpu().numpy()

    assert fast_val.shape == ref_val.shape == (1,)
    assert abs(float(fast_val[0]) - float(ref_val[0])) < 1e-6


def test_dice_consistency_cpu_multi(
    multi_patient_data: Tuple[List[Tensor], List[Tensor]]
) -> None:
    """
    Compare CCDiceMetric and CCBaseMetric(DiceMetric) across multiple patients for
    both aggregation modes: 'patient' and 'overall'.
    """
    y_list, y_hat_list = multi_patient_data

    # Patient aggregation
    fast_patient = CCDiceMetric(cc_reduction="patient")
    ref_patient = CCBaseMetric(
        DiceMetric,
        cc_reduction="patient",
        metric_best_score=1.0,
        metric_worst_score=0.0,
    )

    # Overall aggregation
    fast_overall = CCDiceMetric(cc_reduction="overall")
    ref_overall = CCBaseMetric(
        DiceMetric,
        cc_reduction="overall",
        metric_best_score=1.0,
        metric_worst_score=0.0,
    )

    for y, y_hat in zip(y_list, y_hat_list):
        fast_patient(y_pred=y_hat, y=y)
        ref_patient(y_pred=y_hat, y=y)
        fast_overall(y_pred=y_hat, y=y)
        ref_overall(y_pred=y_hat, y=y)

    # Compare patient-wise results
    fp = fast_patient.cc_aggregate().detach().cpu().numpy()
    rp = ref_patient.cc_aggregate().detach().cpu().numpy()
    assert fp.shape == rp.shape
    assert ((fp - rp) ** 2).sum() < 1e-10

    # Compare overall (per-component) results
    fo = fast_overall.cc_aggregate().detach().cpu().numpy()
    ro = ref_overall.cc_aggregate().detach().cpu().numpy()
    assert fo.shape == ro.shape
    # Order of components should match because both use the same space separation
    assert ((fo - ro) ** 2).sum() < 1e-10
