# CC-Metrics
## Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks

[![Paper](https://img.shields.io/badge/PDF-Paper-green.svg)](https://arxiv.org/pdf/2410.18684) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![YouTube](https://img.shields.io/badge/YouTube-Video-red.svg)](https://www.youtube.com/watch?v=VBiXteZkSHs) [![PyPI Downloads](https://static.pepy.tech/badge/ccmetrics)](https://pepy.tech/projects/ccmetrics)


## Description
Traditional metrics often fail to adequately capture the performance of models in multi-instance segmentation scenarios, particularly when dealing with heterogeneous structures of varying sizes. CC-Metrics addresses this by:

1. Identifying individual connected components in ground-truth labels
2. Creating Voronoi regions around each component to define its territory
3. Mapping predictions within each Voronoi region to the corresponding ground-truth component
4. Computing standard metrics on these mapped regions for more granular assessment

### Recent News from new release (v0.0.3)
The latest version includes performance improvements:
- **Faster Voronoi computation**: New `compute_voronoi_regions_fast` implementation using single EDT computation
- **Memory-efficient processing**: Automatic cropping to regions of interest reduces memory footprint
- **Caching no longer needed**: Fast computation is now efficient enough that caching is typically unnecessary

Below is an example visualization of the Voronoi-based mapping process:

![CC-Metrics Workflow](resources/title_fig.jpg)

For more details, you can read the full paper [here](https://arxiv.org/pdf/2410.18684).


## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [How to Use CC-Metrics](#how-to-use-cc-metrics)
    - [Basic Usage](#basic-usage)
    - [Supported Metrics](#supported-metrics)
    - [Metric Aggregation](#metric-aggregation)
    - [Caching Mechanism](#caching-mechanism)
    - [Advanced Examples](#advanced-examples)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

You may install CC-Metrics simply by running

```bash
pip install CCMetrics
```

Another option is the editable installation directly from the source code:

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- MONAI 0.9+

```bash
git clone https://github.com/alexanderjaus/CC-Metrics.git
cd CC-Metrics
pip install -e .
```

### GPU Acceleration

CC-Metrics includes optional GPU-accelerated preprocessing for the Voronoi space separation to speed up large-volume evaluations.

- Requirements: NVIDIA GPU, CUDA 12.x, a CUDA-enabled PyTorch, and a matching CuPy build (`cupy-cuda12x`).
- Install from PyPI with extras: `pip install "CCMetrics[gpu]"` (or `CCMetrics[all]`).
- Install from source with extras: `pip install -e .[gpu]`.

Usage example:

```python
from CCMetrics import CCDiceMetricGPU
cc_dice_gpu = CCDiceMetricGPU(cc_reduction="patient", use_caching=False)
```

Notes on CPU/GPU parity: CPU and GPU paths share the same algorithm (single-EDT Voronoi + MONAI metrics). Minor numerical differences can occur due to library tie-breaking when assigning equidistant background voxels to components. In our checks across 441 volumes, aggregated CC‑Dice differences were very small (patient-wise mean diff ~1e-5; per-component mean diff ~5e-5), with rare worst-case patient differences up to ~3.5e-3. If exact agreement is required, run the CPU metric path.

## How to Use CC-Metrics

CC-Metrics defines wrappers around MONAI's Cumulative metrics to enable per-component evaluation.

### Basic Usage

Here's a simple example using the CCDiceMetric:

```python
from CCMetrics import CCDiceMetric
import torch

# Create the metric with desired parameters
cc_dice = CCDiceMetric(
    cc_reduction="patient",  # Aggregation mode
    use_caching=False        # Recommended: use fast computation (v0.0.3+)
)

# Create sample prediction and ground truth tensors
# Tensors must be in shape (B, C, D, H, W) where:
# B = batch size (currently only B=1 is supported)
# C = number of channels (must be 2: background and foreground)
# D, H, W = depth, height, width of the volumetric data
y = torch.zeros((1, 2, 64, 64, 64))
y_hat = torch.zeros((1, 2, 64, 64, 64))

# Create two ground truth components
y[0, 1, 20:25, 20:25, 20:25] = 1  # Component 1
y[0, 1, 40:45, 40:45, 40:45] = 1  # Component 2
y[0, 0] = 1 - y[0, 1]  # Background

# Create prediction (slightly offset from ground truth)
y_hat[0, 1, 21:26, 21:26, 21:26] = 1  # Prediction for component 1
y_hat[0, 1, 41:46, 39:44, 41:46] = 1  # Prediction for component 2
y_hat[0, 0] = 1 - y_hat[0, 1]  # Background

# Compute the metric
cc_dice(y_pred=y_hat, y=y)

# Get the results
patient_wise_results = cc_dice.cc_aggregate()
#tensor([0.5120])

print(f"CC-Dice score: {patient_wise_results.mean().item()}")

# You can change the scheme during aggregation
component_wise_results = cc_dice.cc_aggregate(mode="overall")
#tensor([0.5120, 0.5120])
```

### Supported Metrics

CC-Metrics includes the following metrics, all derived from MONAI:

- **CCDiceMetric**: Component-wise Dice coefficient
  ```python
  CCDiceMetric()
  ```

- **CCHausdorffDistanceMetric**: Component-wise Hausdorff distance
  ```python
  CCHausdorffDistanceMetric(metric_worst_score=30)
  ```

- **CCHausdorffDistance95Metric**: Component-wise 95th percentile Hausdorff distance
  ```python
  CCHausdorffDistance95Metric(metric_worst_score=30)
  ```

- **CCSurfaceDistanceMetric**: Component-wise average surface distance
  ```python
  CCSurfaceDistanceMetric(metric_worst_score=30)
  ```

- **CCSurfaceDiceMetric**: Component-wise Surface Dice score
  ```python
  CCSurfaceDiceMetric(class_thresholds=[1])
  ```
  This class needs the additional parameter class_thresholds, a list of class-specific thresholds. The thresholds relate to the acceptable amount of deviation in the segmentation boundary in pixels. Each threshold needs to be a finite, non-negative number. More details [here](https://docs.monai.io/en/stable/metrics.html#monai.metrics.SurfaceDiceMetric)


### Metric Aggregation

The `CCBaseMetric` class supports two types of metric aggregation modes:

1. **Patient-Level Aggregation (`patient`)**:
   - Computes the mean metric score for each patient by aggregating all connected components within the patient
   - Returns a list of mean scores, one for each patient
   - Useful when you want to evaluate performance on a per-patient basis

2. **Overall Aggregation (`overall`)**:
   - Treats all connected components across all patients equally
   - Aggregates the metric scores for all components into a single list
   - Useful when you want to evaluate performance across all components regardless of patient boundaries

The aggregation mode can be specified using the `cc_aggregate` method, with the default mode being `patient`.

```python
# Patient-level aggregation (default)
patient_results = cc_dice.cc_aggregate(mode="patient")

# Overall aggregation
overall_results = cc_dice.cc_aggregate(mode="overall")
```

### Caching Mechanism

CC-Metrics requires the computation of a generalized Voronoi diagram which serves as the mapping mechanism between predictions and ground-truth. As the separation of the image space only depends on the ground-truth, the mapping can be cached and reused between intermediate evaluations or across metrics.

#### Recommended Approach (v0.0.3+)

**We now recommend disabling caching** and relying on the fast computation instead. The new `compute_voronoi_regions_fast` function is so efficient that the overhead of caching often outweighs the benefits:

```python
cc_dice = CCDiceMetric(
    cc_reduction="patient",
    use_caching=False  # Recommended: use fast computation instead
)
```

#### Performance Comparison

Starting with v0.0.3, the Voronoi computation has been significantly optimized. The new `compute_voronoi_regions_fast` function provides:
- Faster computation through single EDT operations
- Reduced memory overhead
- Better scalability for large volumes with many components
- Improved flexibility

#### Legacy Caching Support

For backward compatibility, caching is still supported but generally not recommended:

```python
# Legacy approach - not recommended for most use cases
cc_dice = CCDiceMetric(use_caching=True, caching_dir="/path/to/cache")
```

### Advanced Examples

#### Performance Optimizations

CC-Metrics v0.0.3+ includes several performance improvements that make it more efficient for large-scale evaluations:

**Automatic Region Cropping**: The library now automatically crops to the minimal bounding box containing each connected component and its Voronoi region, significantly reducing memory usage and computation time for sparse segmentations.

**Improved Voronoi Computation**: The new `compute_voronoi_regions_fast` function uses a more efficient single EDT (Euclidean Distance Transform) approach, eliminating the need for KDTree-based computations.

**Memory Management**: Enhanced garbage collection and tensor operation optimization reduce memory leaks and improve performance for batch processing.

#### Evaluating Multiple Metrics on the Same Data

```python
from CCMetrics import CCDiceMetric, CCSurfaceDiceMetric, CCHausdorffDistance95Metric
import torch

# Create sample data
y = torch.zeros((1, 2, 64, 64, 64))
y_hat = torch.zeros((1, 2, 64, 64, 64))

# Set up components (simplified example)
y[0, 1, 20:25, 20:25, 20:25] = 1
y[0, 0] = 1 - y[0, 1]
y_hat[0, 1, 21:26, 21:26, 21:26] = 1
y_hat[0, 0] = 1 - y_hat[0, 1]

# Define shared settings (caching no longer recommended)
use_fast_computation = True

# Initialize metrics
metrics = {
    "dice": CCDiceMetric(use_caching=False),
    "surface_dice": CCSurfaceDiceMetric(use_caching=False, class_thresholds=[1]),
    "hd95": CCHausdorffDistance95Metric(use_caching=False, metric_worst_score=30)
}

# Compute all metrics
results = {}
for name, metric in metrics.items():
    metric(y_pred=y_hat, y=y)
    results[name] = metric.cc_aggregate().mean().item()

print(f"Results: {results}")
```

## FAQ

### Q: What's new in version 2.0?

A: Version 2.0 includes significant performance improvements:
- New optimized Voronoi computation algorithm (`compute_voronoi_regions_fast`)
- Automatic region cropping to reduce memory usage
- Enhanced memory management and tensor operations
- Overall 2-5x speedup in typical use cases compared to previous version
- **Caching is no longer recommended** - the fast computation is now efficient enough that caching overhead often exceeds benefits

### Q: Should I use caching?

A: **No, we recommend disabling caching**. The new `compute_voronoi_regions_fast` function is so efficient that the I/O overhead of caching typically outweighs any performance benefits. Simply set `use_caching=False` when initializing metrics.

### Q: Why use CC-Metrics instead of traditional metrics?

A: Traditional metrics like Dice can be misleading in multi-instance segmentation tasks. CC-Metrics provides a more granular assessment of performance by evaluating each component separately, making it particularly valuable for medical imaging tasks with multiple structures of varying sizes.

### Q: How does CC-Metrics handle false negatives (ground truth components with no matching predictions)?

A: CC-Metrics assigns the worst score to false negative regions, ensuring they appropriately penalize the overall performance score.

### Q: How does CC-Metrics handle false positives (predicted components with no matching ground truth)?

A: CC-Metrics evaluates locally thus positive predictions reduce the scores in the region into which they fall.

### Q: Is multi-class segmentation supported?

A: Currently, CC-Metrics only supports binary segmentation (background and foreground). Multi-class support is planned for future releases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you make use of this project in your work, please cite the CC-Metrics paper:

```bibtex
@article{jaus2024every,
  title={Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks},
  author={Jaus, Alexander and Seibold, Constantin Marc and Rei{\ss}, Simon and Marinov, Zdravko and Li, Keyi and Ye, Zeling and Krieg, Stefan and Kleesiek, Jens and Stiefelhagen, Rainer},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={3904--3912},
  year={2025}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
