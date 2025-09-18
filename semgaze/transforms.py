#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math
import torch
import torchvision.transforms.functional as TF

from semgaze.utils.common import get_img_size, pair


# ============================================================================= #
#                                    TRANSFORMS                                 #
# ============================================================================= #
class RandomHeadBboxJitter(object):
    """
    Applies random jittering to the coordinates of the head bounding boxes.
    """

    def __init__(self, p=1.0, tr=(-0.1, 0.1)):
        """
        Args:
            p (float, optional): probability of jittering the box. Defaults to 1.0.
            tr (tuple, optional): factor by which to increase the box in all directions. Defaults to (-0.1, 0.1).
        """
        self.p = p
        self.tr = tr if isinstance(tr, tuple) else (-abs(tr), abs(tr))

    def __call__(self, head_bboxes, img_w, img_h):
        if torch.rand(1) <= self.p:
            ws, hs = (
                head_bboxes[:, [2]] - head_bboxes[:, [0]],
                head_bboxes[:, [3]] - head_bboxes[:, [1]],
            )
            jitter = torch.empty((len(head_bboxes), 4)).uniform_(self.tr[0], self.tr[1])
            head_bboxes = head_bboxes + torch.cat([-ws, -hs, ws, hs], dim=1) * jitter
            head_bboxes[:, [0, 2]] = head_bboxes[:, [0, 2]].clip(0.0, img_w)
            head_bboxes[:, [1, 3]] = head_bboxes[:, [1, 3]].clip(0.0, img_h)

        return head_bboxes


class Resize(object):
    """
    Resizes the input image to the desired size.
    """

    def __init__(self, img_size, head_size):
        assert isinstance(img_size, (int, tuple)), f"img_size needs to be either an int or tuple. Found {img_size} instead."
        assert isinstance(head_size, tuple), f"head_size needs to be a tuple. Found {head_size} instead."
        self.img_size = pair(img_size)
        self.head_size = head_size

    def __call__(self, sample):
        num_heads = len(sample["heads"])

        # Resize Image
        sample["image"] = TF.resize(sample["image"], self.img_size, antialias=True)  # type: ignore
        
        # Resize Heads
        for k in range(num_heads):
            sample["heads"][k] = TF.resize(sample["heads"][k], self.head_size, antialias=True)

        return sample


class RandomHorizontalFlip(object):
    """
    Flips the input image horizontally.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) <= self.p:
            num_heads = len(sample["heads"])

            # Flip Image
            sample["image"] = TF.hflip(sample["image"])  # type: ignore

            # Flip Heads
            for k in range(num_heads):
                sample["heads"][k] = TF.hflip(sample["heads"][k])

            # Flip Head Bboxes
            sample["head_bboxes"][:, [0, 2]] = 1.0 - sample["head_bboxes"][:, [2, 0]]
            
            # Flip Gaze Points
            if sample["inout"] == 1.:
                if sample["gaze_pt"].ndim == 2:
                    mask = sample["gaze_pt"][:, 0] != -1.0
                    sample["gaze_pt"][mask, 0] = 1.0 - sample["gaze_pt"][mask, 0]
                else:
                    sample["gaze_pt"][0] = 1.0 - sample["gaze_pt"][0]

        return sample


class ColorJitter(object):
    """
    Applies random colors transformations to the input (ie. brightness,
    contrast, saturation and hue).
    """

    def __init__(self, brightness, contrast, saturation, hue, p=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) <= self.p:
            num_heads = len(sample["heads"])

            # Sample color transformation factors and order
            brightness_factor = None if self.brightness is None else torch.rand(1).uniform_(*self.brightness)
            contrast_factor = None if self.contrast is None else torch.rand(1).uniform_(*self.contrast)
            saturation_factor = None if self.saturation is None else torch.rand(1).uniform_(*self.saturation)
            hue_factor = None if self.hue is None else torch.rand(1).uniform_(*self.hue)

            fn_indices = torch.randperm(4)
            for fn_id in fn_indices:
                if fn_id == 0 and brightness_factor is not None:
                    sample["image"] = TF.adjust_brightness(sample["image"], brightness_factor)
                    for k in range(num_heads):
                        sample["heads"][k] = TF.adjust_brightness(sample["heads"][k], brightness_factor)

                elif fn_id == 1 and contrast_factor is not None:
                    sample["image"] = TF.adjust_contrast(sample["image"], contrast_factor)
                    for k in range(num_heads):
                        sample["heads"][k] = TF.adjust_contrast(sample["heads"][k], contrast_factor)

                elif fn_id == 2 and saturation_factor is not None:
                    sample["image"] = TF.adjust_saturation(sample["image"], saturation_factor)
                    for k in range(num_heads):
                        sample["heads"][k] = TF.adjust_saturation(sample["heads"][k], saturation_factor)

                elif fn_id == 3 and hue_factor is not None:
                    sample["image"] = TF.adjust_hue(sample["image"], hue_factor)
                    for k in range(num_heads):
                        sample["heads"][k] = TF.adjust_hue(sample["heads"][k], hue_factor)

        return sample


class Normalize(object):
    def __init__(
        self,
        img_mean=[0.44232, 0.40506, 0.36457],
        img_std=[0.28674, 0.27776, 0.27995],
    ):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, sample):
        sample["image"] = TF.normalize(sample["image"], self.img_mean, self.img_std)
        sample["heads"] = TF.normalize(sample["heads"], self.img_mean, self.img_std)
        return sample


class ToTensor(object):
    """
    Convert inputs to tensors.
    """

    def __call__(self, sample):
        num_heads = len(sample["heads"])
        
        # Convert image
        sample["image"] = TF.to_tensor(sample["image"])

        # Convert heads
        for k in range(num_heads):
            sample["heads"][k] = TF.to_tensor(sample["heads"][k])
        sample["heads"] = torch.stack(sample["heads"], dim=0)
        return sample

class RandomCropSafeGaze(object):
    """
    Randomly crops the input image while ensuring the gaze target and the head bounding box
    remain within the crop. The crop is also chosen such that it respects the given aspect
    ratio (or the aspect ratio of the image if None).
    """

    def __init__(self, aspect=None, p=1.0, p_safe=1.0):
        self.aspect = aspect
        self.p = p
        self.p_safe = p_safe

    def __call__(self, sample):
        if torch.rand(1) <= self.p:
            num_heads = len(sample["heads"])
            img_w, img_h = sample["image"].size

            # Convert all coordinates to pixels upfront
            head_bboxes_px = sample["head_bboxes"] * torch.tensor([[img_w, img_h]]).repeat(1, 2)
            
            # Get gaze point in pixels
            if torch.rand(1) <= self.p_safe:
                gaze_pt_px = sample["gaze_pt"] * torch.tensor([img_w, img_h])
            else:
                gaze_pt_px = sample["head_bboxes"][-1, :2] * torch.tensor([img_w, img_h])
            
            # Get crop parameters in pixels
            crop_xmin, crop_ymin, crop_w, crop_h = self._get_random_crop_bbox_px(
                img_w, img_h, head_bboxes_px, gaze_pt_px
            )
            
            # Ensure crop stays within image bounds
            crop_w = min(crop_w, img_w - crop_xmin)
            crop_h = min(crop_h, img_h - crop_ymin)
            
            # Crop Image
            sample["image"] = TF.crop(sample["image"], crop_ymin, crop_xmin, crop_h, crop_w)

            # Convert Head Bboxes (from pixels to normalized in new crop)
            head_bboxes_cropped_px = head_bboxes_px - torch.tensor([[crop_xmin, crop_ymin]]).repeat(1, 2)
            sample["head_bboxes"] = head_bboxes_cropped_px / torch.tensor([[crop_w, crop_h]]).repeat(1, 2)
            
            # Check if gaze point is outside crop (using pixel coordinates for precision)
            gaze_outside_crop = (sample["inout"] == 1.) and (sample["gaze_pt"].ndim == 1) and \
                               (gaze_pt_px[0] < crop_xmin or gaze_pt_px[0] >= crop_xmin + crop_w or
                                gaze_pt_px[1] < crop_ymin or gaze_pt_px[1] >= crop_ymin + crop_h)
            
            if gaze_outside_crop:
                print('INSIDE IF BLOCK TO CONVERT INOUT FROM 1 TO 0')
                print(f"gaze_pt_px: {gaze_pt_px}")
                print(f"gaze_pt_px: {sample['head_bboxes']}")
                print(f"(img_w, img_h): ({img_w}, {img_h})")
                print(f"(crop_xmin, crop_ymin): ({crop_xmin}, {crop_ymin})")
                print(f"(crop_w, crop_h): ({crop_w}, {crop_h})")
                
                sample["inout"] = 1. - sample["inout"]
                sample["gaze_pt"] = torch.tensor([-1., -1.], dtype=torch.float)

            # Convert Gaze Points (only if still inside)
            if sample["inout"] == 1.:
                if sample["gaze_pt"].ndim == 2:  # gazefollow test set has multiple annotations
                    mask = sample["gaze_pt"][:, 0] != -1.0
                    gaze_pts_px = sample["gaze_pt"][mask] * torch.tensor([[img_w, img_h]])
                    gaze_pts_cropped_px = gaze_pts_px - torch.tensor([[crop_xmin, crop_ymin]])
                    sample["gaze_pt"][mask] = gaze_pts_cropped_px / torch.tensor([[crop_w, crop_h]])
                else:
                    gaze_pt_cropped_px = gaze_pt_px - torch.tensor([crop_xmin, crop_ymin])
                    sample["gaze_pt"] = gaze_pt_cropped_px / torch.tensor([crop_w, crop_h])
                
        return sample

    def _get_random_crop_bbox_px(self, img_w, img_h, head_bboxes_px, gaze_pt_px):
        """
        Computes the parameters of a random crop that maintains the aspect ratio, and includes
        the gaze point and head bounding box. All coordinates are in pixels.
        """

        # Compute aspect ratio
        aspect = img_w / img_h if self.aspect is None else self.aspect

        # Create safe zone coordinates in pixels
        coords_px = torch.concat([head_bboxes_px, gaze_pt_px.repeat(2).unsqueeze(0)], dim=0)
        zone_xmin = coords_px[:, 0].min().item()
        zone_ymin = coords_px[:, 1].min().item()
        zone_xmax = coords_px[:, 2].max().item()
        zone_ymax = coords_px[:, 3].max().item()

        # Expand the "safe" zone a bit
        zone_xmin, zone_ymin, zone_xmax, zone_ymax = self._expand_px(
            zone_xmin, zone_ymin, zone_xmax, zone_ymax, img_w, img_h
        )
        zone_w = zone_xmax - zone_xmin
        zone_h = zone_ymax - zone_ymin

        # Randomly select a crop size
        if zone_w >= zone_h * aspect:
            crop_w = torch.rand(1).uniform_(zone_w, img_w).item()
            crop_h = crop_w / aspect
        else:
            crop_h = torch.rand(1).uniform_(zone_h, img_h).item()
            crop_w = crop_h * aspect

        # Find min and max possible positions for top-left point
        xmin = max(zone_xmax - crop_w, 0)
        xmax = min(zone_xmin, max(img_w - crop_w, 0))
        ymin = max(zone_ymax - crop_h, 0)
        ymax = min(zone_ymin, max(img_h - crop_h, 0))

        # Randomly select a top left point
        crop_xmin = torch.rand(1).uniform_(xmin, xmax).item() if xmin <= xmax else 0.0
        crop_ymin = torch.rand(1).uniform_(ymin, ymax).item() if ymin <= ymax else 0.0

        # Convert to integers for actual cropping
        crop_xmin = int(round(crop_xmin))
        crop_ymin = int(round(crop_ymin))
        crop_w = int(round(crop_w))
        crop_h = int(round(crop_h))

        return crop_xmin, crop_ymin, crop_w, crop_h

    def _expand_px(self, xmin, ymin, xmax, ymax, img_w, img_h, k=0.05):
        """Expand bbox while ensuring it stays within image. All coordinates in pixels."""
        w, h = abs(xmax - xmin), abs(ymax - ymin)
        xmin = max(xmin - k * w, 0.0)
        ymin = max(ymin - k * h, 0.0)
        xmax = min(xmax + k * w, img_w)
        ymax = min(ymax + k * h, img_h)
        return xmin, ymin, xmax, ymax

class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
