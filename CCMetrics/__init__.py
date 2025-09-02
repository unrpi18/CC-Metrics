from CCMetrics.CC_base import (
    CCBaseMetric,
    CCDiceMetric,
    CCHausdorffDistance95Metric,
    CCHausdorffDistanceMetric,
    CCSurfaceDiceMetric,
    CCSurfaceDistanceMetric,
)

try:
    from CCMetrics.CC_base_gpu import (
        CCBaseMetricGPU,
        CCDiceMetricGPU,
        CCHausdorffDistance95MetricGPU,
        CCHausdorffDistanceMetricGPU,
        CCSurfaceDiceMetricGPU,
        CCSurfaceDistanceMetricGPU,
    )
except ImportError:
    # CuPy not available, GPU functionality will not be available
    (
        CCBaseMetricGPU,
        CCDiceMetricGPU,
        CCHausdorffDistanceMetricGPU,
        CCHausdorffDistance95MetricGPU,
        CCSurfaceDistanceMetricGPU,
        CCSurfaceDiceMetricGPU,
    ) = (None, None, None, None, None, None)
