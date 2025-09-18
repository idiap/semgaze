#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import torch
import torchmetrics as tm
from torchmetrics.functional.classification.auroc import binary_auroc

from semgaze.utils.common import generate_binary_gaze_heatmap, is_point_in_box


class Distance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_point_pred: torch.Tensor,
        gaze_point_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        mask = inout_gt == 1
        if mask.any():
            self.sum_dist += (gaze_point_gt[mask] - gaze_point_pred[mask]).pow(2).sum(1).sqrt().sum()
            self.num_obs += mask.sum()

    def compute(self):
        if self.num_obs != 0:
            dist = self.sum_dist / self.num_obs  # type: ignore
        else:
            dist = torch.tensor(-1000.0, device=self.device)
        return dist


class GFTestDistance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist_to_avg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_avg_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_min_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, gaze_point_pred: torch.Tensor, gaze_point_gt: torch.Tensor):
        for k, (gp_pred, gp_gt) in enumerate(zip(gaze_point_pred, gaze_point_gt)):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points

            # Compute average gaze point
            gp_gt_avg = gp_gt.mean(0)
            # Compute distance from pred to avg gt point
            self.sum_dist_to_avg += (gp_gt_avg - gp_pred).pow(2).sum().sqrt()
            # Compute avg distance between pred and gt points
            self.sum_avg_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().mean()
            # Compute min distance between pred and gt points
            self.sum_min_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().min()
        self.num_obs += len(gaze_point_pred)

    def compute(self):
        dist_to_avg = self.sum_dist_to_avg / self.num_obs
        avg_dist = self.sum_avg_dist / self.num_obs
        min_dist = self.sum_min_dist / self.num_obs
        return dist_to_avg, avg_dist, min_dist


class GFTestAUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes AUC for GazeFollow Test set. The AUC is computed for each image in the batch, after resizing the predicted
        heatmap to the original size of the image. The ground-truth binary heatmap is generated from the ground-truth gaze
        point(s) in the original image size. At the end, the mean is returned.
        """

        super().__init__()
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_pt: torch.Tensor,
    ):
        size = gaze_heatmap_pred.shape[1:]  # (b, h, w) >> (h, w)
        for hm_pred, gp_gt in zip(gaze_heatmap_pred, gaze_pt):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points
            hm_gt_binary = generate_binary_gaze_heatmap(gp_gt, size=size)
            self.sum_auc += binary_auroc(hm_pred, hm_gt_binary)
        self.num_obs += len(gaze_heatmap_pred)

    def compute(self):
        auc = self.sum_auc / self.num_obs
        return auc


class GazeAccuracy(tm.Metric):
    higher_is_better = True
    full_state_update: bool = False
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gaze_point_pred: torch.Tensor, gaze_bbox_gt: torch.Tensor):
        isin = is_point_in_box(gaze_point_pred, gaze_bbox_gt).diag()
        self.correct += isin.sum()
        self.total += gaze_point_pred.size(0)

    def compute(self):
        return self.correct.float() / self.total

    
class MultiAccuracy(tm.Metric):
    higher_is_better = True
    full_state_update: bool = False
    
    def __init__(self, top_k: int = 1, ignore_index = None):
        super().__init__()
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).any(dim=1)
            preds = preds[mask]
            target = target[mask]
        
        # Get the top k predictions
        top_k_preds = preds.topk(self.top_k, dim=1)[1]
        
        # Check if any of the top k predictions match any of the target classes
        target = target.unsqueeze(1) # Expand dims of target for broadcasting
        correct = torch.any(top_k_preds.unsqueeze(2) == target, dim=1).any(dim=1).sum()
        self.correct += correct
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total