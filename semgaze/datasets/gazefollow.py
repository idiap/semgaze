#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
import json
from typing import Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from semgaze.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from semgaze.utils.common import pair, expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, square_bbox



IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# ============================================================================= #
#                               GAZEFOLLOW DATASET                              #
# ============================================================================= #
class GazeFollowDataset(Dataset):
    def __init__(
        self,
        root,
        root_project,
        root_heads,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 1,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        assert (num_people == -1) or (num_people > 0), f"Expected `num_people` to be strictly positive or `-1`, but received {num_people} instead."
        assert 0 <= head_thr <= 1, f"Expected `head_thr` to be in [0, 1]. Received {head_thr} instead."

        self.root = root
        self.root_project = root_project
        self.root_heads = root_heads
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.annotations, self.vocab2id = self.load_annotations()

    def load_annotations(self) -> pd.DataFrame:
        annotations = pd.DataFrame()
        if self.split == "test":
            columns = ["path", "id", "body_x", "body_y", "body_w", "body_h", "eye_x", "eye_y", "gaze_x", "gaze_y", 
                       "head_xmin", "head_ymin", "head_xmax", "head_ymax", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root, "test_annotations_release.txt"),
                sep=",",
                names=columns,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Add inout col for consistency (ie. missing from test set)
            annotations["inout"] = 1
            # Each test image is annotated by multiple people (around 10 on avg.)
            self.image_paths = annotations.path.unique().tolist()
            self.length = len(self.image_paths)

        elif self.split in ("train", "val"):
            columns = ["path", "id", "body_x", "body_y", "body_w", "body_h", "eye_x", "eye_y", "gaze_x", "gaze_y", 
                       "head_xmin", "head_ymin", "head_xmax", "head_ymax", "inout", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root_project, f"data/gazefollow/{self.split}_annotations_new.txt"), # reprocessed train/val head bboxes
                sep=",",
                names=columns,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Clean annotations (e.g. remove invalid ones)
            annotations = self._clean_annotations(annotations)
            self.length = len(annotations)
            
        #  Load Gaze Labels and merge with annotations
        df_label = pd.read_csv(os.path.join(self.root_project, f"data/gazefollow/gaze-labels-{self.split}.csv"))
        merge_on = ["path", "id"] if self.split in ["train", "val"] else ["path"]
        annotations = pd.merge(annotations, df_label, how="left", on=merge_on)
        
        # Each test image is annotated by multiple people (around 10 on avg.)
        self.image_paths = sorted(annotations.path.unique())
        self.length = len(self.image_paths) if self.split == "test" else len(annotations)
        
        # Load vocab2id
        with open(os.path.join(self.root_project, 'data/gazefollow/vocab2id.json'), 'r') as f:
            vocab2id = json.load(f)

        return annotations, vocab2id


    def _clean_annotations(self, annotations):
        # Only keep "in" and "out". (-1 is invalid)
        annotations = annotations[annotations.inout != -1]
        # Discard instances where max in bbox coordinates is smaller than min
        annotations = annotations[annotations.head_xmin < annotations.head_xmax]
        annotations = annotations[annotations.head_ymin < annotations.head_ymax]
        return annotations.reset_index(drop=True)

    def __getitem__(self, index: int) -> Dict:
        if self.split in ("train", "val"): 
            item = self.annotations.iloc[index]
            gaze_pt = torch.tensor([item["gaze_x"], item["gaze_y"]], dtype=torch.float)
            gaze_label = item["gaze_pseudo_label"]
            gaze_labels = [gaze_label]
            gaze_label_id = torch.tensor(item["label_id"])
            gaze_label_ids = torch.tensor([gaze_label_id])
            idx = item["id"]
        elif self.split == "test":
            image_path = self.image_paths[index]
            p_annotations = self.annotations[self.annotations.path == image_path]
            gaze_pt = torch.from_numpy(p_annotations[["gaze_x", "gaze_y"]].values).float()
            p = 20 - len(gaze_pt)
            gaze_pt = F.pad(gaze_pt, (0, 0, 0, p), value=-1.0) # Pad to have same length for batching
            idx = p_annotations["id"].values.tolist() + [-1] * p 
            item = p_annotations.iloc[0]
            
            gaze_label = item.gaze_gt_label
            gaze_labels = item.gaze_gt_labels # string in the form class1-class2-...
            gaze_label_id = torch.tensor(item.test_label_id) # GT test vocab
            gaze_label_ids = torch.tensor([gaze_label_id])
            if gaze_label_id != -1:
                gaze_label_ids = torch.tensor([self.vocab2id[label] for label in gaze_labels.split('-')])
            l = 5 - len(gaze_label_ids)
            gaze_label_ids = F.pad(gaze_label_ids, (0, l), value=-1) # pad to 5 for batching
            

        # eyes_pt = torch.tensor([item["eye_x"], item["eye_y"]], dtype=torch.float) # not used
        inout = torch.tensor(item["inout"], dtype=torch.float)
        path = item["path"]
        split, partition, img_name = item["path"].split('/')
        basename, ext = os.path.splitext(img_name)

        # Load image
        image = Image.open(os.path.join(self.root, item["path"])).convert("RGB")
        img_w, img_h = image.size
        
        # Load target head bbox
        target_head_bbox = item[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
        target_head_bbox = torch.from_numpy(target_head_bbox.values.astype(np.float32)).unsqueeze(0)
        target_head_bbox = expand_bbox(target_head_bbox, img_w, img_h, k=0.1) # annotated boxes are a bit tight

        # Load context head bboxes (if n > 1)
        context_head_bboxes = torch.zeros((0, 4))
        if (self.num_people == -1) or (self.num_people > 1):
            det_file = f"{split}/{partition}/{basename}-head-detections.npy"
            detections = np.load(os.path.join(self.root_heads, det_file))

            # Process context head bboxes
            if len(detections) > 0:
                scores = torch.tensor(detections[:, -1])
                context_head_bboxes = torch.tensor(detections[(scores >= self.head_thr).tolist(), :-1])
                ious = box_iou(context_head_bboxes, target_head_bbox).flatten()
                context_head_bboxes = context_head_bboxes[ious <= 0.5]

            # Shuffle context people and keep the first `num_people - 1` indices
            if self.split == "train":
                perm_indices = torch.randperm(context_head_bboxes.size(0))
                context_head_bboxes = context_head_bboxes[perm_indices]
            num_context_heads = len(context_head_bboxes)
            num_keep = num_context_heads if self.num_people == -1 else self.num_people - 1
            context_head_bboxes = context_head_bboxes[:num_keep]

        # Concatenate main head bbox with others and apply jitter
        head_bboxes = torch.concat([context_head_bboxes, target_head_bbox], dim=0).to(torch.float)
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract Heads (negative values add padding)
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))  # type:ignore

        # Normalize Head Bboxes and clip to [0, 1]
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        head_bboxes = torch.clamp(head_bboxes, min=0.0, max=1.0)
        
        # Load gaze label embeddings
        if pd.isnull(gaze_label):
            gaze_label = gaze_labels = ""
            gaze_label_emb = torch.zeros(512)
        else:
            label_emb_path = os.path.join(self.root_project, f"data/gazefollow/label-embeds/{gaze_label}-emb.pt")
            gaze_label_emb = torch.load(label_emb_path, weights_only=False)
            gaze_label_emb = F.normalize(gaze_label_emb, p=2, dim=-1)
        
        # Build Sample
        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pt": gaze_pt,
            "gaze_label": gaze_label,
            "gaze_label_id": gaze_label_id,
            "gaze_labels": gaze_labels,
            "gaze_label_ids": gaze_label_ids,
            "gaze_label_emb": gaze_label_emb,
            "inout": inout,
            "id": idx,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": path,
        }

        # Transform
        if self.transform:
            sample = self.transform(sample)
            
        # Pad missing people (ie. heads + bboxes)
        num_heads = len(head_bboxes)
        num_missing_heads = self.num_people - num_heads if self.num_people != -1 else 0
        if num_missing_heads > 0:
            pad = (0, 0, num_missing_heads, 0)
            sample["head_bboxes"] = F.pad(sample["head_bboxes"], pad, mode="constant", value=0.)
            if isinstance(sample["heads"], torch.Tensor):
                pad = (0, 0, 0, 0, 0, 0, num_missing_heads, 0)
                sample["heads"] = F.pad(sample["heads"], pad, mode="constant", value=0.)
            else:
                sample["heads"] = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * num_missing_heads + heads

        # Compute head centers
        sample["head_centers"] = torch.hstack(
            [
                (sample["head_bboxes"][:, [0]] + sample["head_bboxes"][:, [2]]) / 2,
                (sample["head_bboxes"][:, [1]] + sample["head_bboxes"][:, [3]]) / 2,
            ]
        )
        
        # Generate gaze heatmap
        if sample["inout"] == 1.0:
            sample["gaze_heatmap"] = generate_gaze_heatmap(sample["gaze_pt"], sigma=self.heatmap_sigma, size=self.heatmap_size)    
        else:
            sample["gaze_heatmap"] = torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float)
        
        # Compute gaze vector (only for target person)
        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][-1]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)
        
        # Generate head mask
        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample

    def __len__(self):
        return self.length


# ============================================================================= #
#                             GAZEFOLLOW DATAMODULE                             #
# ============================================================================= #
class GazeFollowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_project: str,
        root_heads: str,
        batch_size: Union[int, dict] = 32,
        image_size: Union[int, tuple[int, int]] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: Union[int, tuple[int, int]] = 64,
        num_people: dict = {"train": 1, "val": 1, "test": 1},
        return_head_mask: bool = False,
    ):
        super().__init__()
        self.root = root
        self.root_project = root_project
        self.root_heads = root_heads
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.batch_size = {stage: batch_size for stage in ["train", "val", "test"]} if isinstance(batch_size, int) else batch_size
        self.return_head_mask = return_head_mask
        
    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=0.8, p_safe=1.0),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.0, 1.5), hue=None, p=0.8),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.train_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "train",
                train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "test",
                test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return dataloader
