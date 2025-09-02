import gc
import hashlib
import os

import numpy as np
import torch
from monai.metrics import (
    Cumulative,
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)

from CCMetrics.space_separation import compute_voronoi_regions_fast

# Globally disable gradient computation for this entire module
torch.set_grad_enabled(False)


class CCBaseMetric:

    def __init__(
        self,
        BaseMetric: Cumulative,
        *args,
        use_caching=False,
        caching_dir=".cache",
        metric_best_score=None,
        metric_worst_score=None,
        **kwargs,
    ):
        """
        Initializes a CC_base object.

        Args:
            BaseMetric (Cumulative): The base Monai metric to be used.
            *args: Variable length argument list, passed to the Monai metric.
            use_caching (bool, optional): Flag to enable caching. Defaults to True.
            caching_dir (str, optional): Directory to store the cache. Defaults to ".cache".
            metric_best_score: The best score for the metric. Must be defined.
            metric_worst_score: The worst score for the metric. Must be defined.
            **kwargs: Arbitrary keyword arguments, passed to the Monai metric.

        Raises:
            AssertionError: If metric_best_score or metric_worst_score is not defined.

        """
        assert metric_best_score is not None, "Best score must be defined"
        assert metric_worst_score is not None, "Worst score must be defined"
        self.metric_perfect_score = metric_best_score
        self.metric_worst_score = metric_worst_score
        self.buffer_collection = []
        if kwargs.get("include_background", False):
            raise ValueError("Background class is not supported")
        else:
            kwargs["include_background"] = False

        if kwargs.get("cc_reduction", None):
            assert kwargs["cc_reduction"] in [
                "patient",
                "overall",
            ], f"Unknown aggregation function {kwargs['cc_reduction']}"
            self.cc_reduction = kwargs["cc_reduction"]
            del kwargs["cc_reduction"]
        else:
            self.cc_reduction = "patient"

        self.base_metric = BaseMetric(*args, **kwargs)
        self.use_caching = use_caching
        self.caching_dir = caching_dir
        if self.use_caching and not os.path.exists(self.caching_dir):
            os.makedirs(self.caching_dir)

        # Set cpu backend
        self.xp = np
        self.backend = "numpy"
        self.space_separation = compute_voronoi_regions_fast

    def _verify_and_convert(self, y_pred, y):

        # Automatically convert to numy
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        assert isinstance(
            y_pred, self.xp.ndarray
        ), f"Input is not a numpy array. Got {type(y_pred)}"
        assert isinstance(
            y, self.xp.ndarray
        ), f"Input is not a numpy array. Got {type(y)}"

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

        return y_pred, y

    def _convert_to_target(self, y_pred, y):
        if type(y_pred) == self.xp.ndarray:
            y_pred = torch.from_numpy(y_pred)
        if type(y) == self.xp.ndarray:
            y = torch.from_numpy(y)
        return y_pred, y

    def __call__(self, y_pred, y):
        """
        Calculates the metric for the predicted and ground truth tensors.

        Args:
            y_pred (numpy.ndarray or torch.Tensor): The predicted tensor.
            y (numpy.ndarray or torch.Tensor): The ground truth tensor.

        Raises:
            AssertionError: If the input shapes or conditions are not correct.

        Returns:
            None
        """
        y_pred, y = self._verify_and_convert(y_pred, y)

        # Compute argmax
        pred_helper = y_pred.argmax(1)
        label_helper = y.argmax(1)

        # Check if pure background class
        if label_helper[0].sum() == 0:
            if pred_helper[0].sum() == 0:
                # Case perfect prediction: No foreground class present in prediction
                self.buffer_collection.append(torch.tensor([self.metric_perfect_score]))
            else:
                # Case worst prediction: Predicted Foreground class but no GT
                self.buffer_collection.append(torch.tensor([self.metric_worst_score]))
            return

        cc_assignment = self.space_separation(label_helper[0])

        missed_components = 0

        for cc_id in self.xp.unique(cc_assignment):
            cc_mask = cc_assignment == cc_id
            min_corner_idx = self.xp.argwhere(cc_mask).min(axis=0)
            max_corner_idx = self.xp.argwhere(cc_mask).max(axis=0)

            # Cut out the region of interest
            crop_pred = pred_helper[0][
                min_corner_idx[0] : max_corner_idx[0] + 1,
                min_corner_idx[1] : max_corner_idx[1] + 1,
                min_corner_idx[2] : max_corner_idx[2] + 1,
            ]
            crop_label = label_helper[0][
                min_corner_idx[0] : max_corner_idx[0] + 1,
                min_corner_idx[1] : max_corner_idx[1] + 1,
                min_corner_idx[2] : max_corner_idx[2] + 1,
            ]
            pred_masked = (
                crop_pred
                * cc_mask[
                    min_corner_idx[0] : max_corner_idx[0] + 1,
                    min_corner_idx[1] : max_corner_idx[1] + 1,
                    min_corner_idx[2] : max_corner_idx[2] + 1,
                ]
            )
            label_masked = (
                crop_label
                * cc_mask[
                    min_corner_idx[0] : max_corner_idx[0] + 1,
                    min_corner_idx[1] : max_corner_idx[1] + 1,
                    min_corner_idx[2] : max_corner_idx[2] + 1,
                ]
            )

            if pred_masked.sum() == 0:
                missed_components += 1

            # Remap metrics back to one-hot encoding
            # pred_onehot = F.one_hot(pred_masked, num_classes=2).permute(3, 0, 1, 2)
            # label_onehot = F.one_hot(label_masked, num_classes=2).permute(3, 0, 1, 2)
            pred_onehot = self.xp.moveaxis(self.xp.eye(2)[pred_masked], -1, 0)
            label_onehot = self.xp.moveaxis(self.xp.eye(2)[label_masked], -1, 0)

            pred_onehot, label_onehot = self._convert_to_target(
                pred_onehot[self.xp.newaxis], label_onehot[self.xp.newaxis]
            )

            self.base_metric(y_pred=pred_onehot, y=label_onehot)

            del crop_pred, crop_label, pred_masked, label_masked
            del cc_mask
            gc.collect()
        del pred_helper
        del label_helper

        # Get metric buffer and reset it
        metric_buffer = self.base_metric.get_buffer()
        self.buffer_collection.append(metric_buffer)
        self.base_metric.reset()

    def cc_aggregate(self, mode=None):
        """
        Aggregates the buffer collection based on the specified mode.

        Args:
            mode (str, optional): The aggregation mode. Can be "patient" or "overall".
                If not provided, the default mode specified in self.cc_reduction will be used.

        Returns:
            torch.Tensor: The aggregated result based on the specified mode.

        Raises:
            AssertionError: If an unknown aggregation function is provided.

        """
        if mode is None:
            mode = self.cc_reduction
        assert mode in ["patient", "overall"], f"Unknown aggregation function {mode}"
        cleaned_buffer = [
            torch.where(
                torch.isinf(x),
                torch.tensor(
                    self.metric_worst_score, dtype=torch.float32, device=x.device
                ),
                x,
            )
            for x in self.buffer_collection
        ]
        cleaned_buffer = [
            torch.where(
                torch.isnan(x),
                torch.tensor(
                    self.metric_worst_score, dtype=torch.float32, device=x.device
                ),
                x,
            )
            for x in cleaned_buffer
        ]
        cleaned_buffer = [x.reshape(-1, 1) for x in cleaned_buffer]
        if mode == "patient":
            # Aggregate per patient and return list of means
            return torch.stack([x.mean() for x in cleaned_buffer])
        elif mode == "overall":
            # Aggregate overall. All components are considered as equal. Return full list
            return torch.concatenate(cleaned_buffer).squeeze()

    def get_buffer(self):
        """
        Returns the buffer collection.
        """
        return self.buffer_collection

    def reset(self):
        """
        Resets the buffer collection.
        """
        self.buffer_collection = []

    def cache_datapoint(self, y):
        """
        Caches the datapoint if caching is enabled.

        Args:
            y (torch.Tensor): The input tensor.

        Raises:
            ValueError: If caching is disabled.

        Returns:
            None
        """
        if self.use_caching:
            # Handle data input
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            assert isinstance(
                y, np.ndarray
            ), "Input is not a numpy array or torch tensor. Caching is not possible"
            assert (
                len(y.shape) == 3
            ), "Input shape is not correct. Expected shape: (D,H,W) as input y"

            gt_fingerprint = hashlib.md5(y.tobytes()).hexdigest()
            target_path = f"{os.path.join(self.caching_dir, gt_fingerprint)}.npy"
            if os.path.exists(target_path):
                return
            cc_assignment = self.space_separation(y)
            np.save(target_path, cc_assignment)
        else:
            raise ValueError("Caching is disabled")


# Define used metrics from the paper <https://arxiv.org/pdf/2410.18684>
# For unbound metrics, the worst score is set to None and should be handled by the user, as it is infinite


class CCDiceMetric(CCBaseMetric):
    """
    CCDiceMetric is a class that represents the Dice metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            DiceMetric, *args, metric_best_score=1.0, metric_worst_score=0.0, **kwargs
        )

    def __call__(self, y_pred, y):
        """
        Calculates the Dice metric for the predicted and ground truth tensors.
        Args:
            y_pred (numpy.ndarray or torch.Tensor): The predicted tensor.
            y (numpy.ndarray or torch.Tensor): The ground truth tensor.
        Raises:
            AssertionError: If the input shapes or conditions are not correct.
        Returns:
            None
        """
        y_pred, y = self._verify_and_convert(y_pred, y)

        pred_helper = y_pred.argmax(1)
        label_helper = y.argmax(1)
        if label_helper[0].sum() == 0:
            if pred_helper[0].sum() == 0:
                # Case perfect prediction: No foreground class present in prediction
                self.buffer_collection.append(torch.tensor([self.metric_perfect_score]))
            else:
                # Case worst prediction: Predicted Foreground class but no GT
                self.buffer_collection.append(torch.tensor([self.metric_worst_score]))
            return
        cc_assignment = self.space_separation(label_helper[0])

        uniq, inv = self.xp.unique(cc_assignment.ravel(), return_inverse=True)

        nof_components = uniq.size

        code = label_helper.ravel() << 1 | pred_helper.ravel()
        idx = inv << 2 | code
        hist = self.xp.bincount(idx, minlength=nof_components * 4).reshape(-1, 4)
        TN, FP, FN, TP = hist[:, 0], hist[:, 1], hist[:, 2], hist[:, 3]
        denom = 2 * TP + FP + FN
        dice_scores = self.xp.where(denom > 0, (2 * TP) / denom, 1.0)
        dice_scores = (
            torch.from_numpy(self.xp.asnumpy(dice_scores))
            if self.backend == "cupy"
            else torch.from_numpy(dice_scores)
        )
        self.buffer_collection.append(dice_scores.unsqueeze(-1))


class CCHausdorffDistanceMetric(CCBaseMetric):
    """
    CCHausdorffDistanceMetric is a class that represents the Hausdorff distance metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            HausdorffDistanceMetric, *args, metric_best_score=0.0, **kwargs
        )


class CCHausdorffDistance95Metric(CCBaseMetric):
    """
    A class representing a metric for calculating the 95th percentile Hausdorff distance for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            HausdorffDistanceMetric,
            *args,
            metric_best_score=0.0,
            percentile=95,
            **kwargs,
        )


class CCSurfaceDistanceMetric(CCBaseMetric):
    """
    A class representing a metric for calculating the SurfaceDistance metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(SurfaceDistanceMetric, *args, metric_best_score=0.0, **kwargs)


class CCSurfaceDiceMetric(CCBaseMetric):
    """
    A class representing a metric for calculating the SurfaceDiceMetric metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            SurfaceDiceMetric,
            *args,
            metric_best_score=1.0,
            metric_worst_score=0.0,
            **kwargs,
        )
