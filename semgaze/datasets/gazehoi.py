#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
import sys
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

from semgaze.transforms import (ColorJitter, Compose, Normalize, 
                                RandomCropSafeGaze, RandomHeadBboxJitter, 
                                RandomHorizontalFlip, Resize, ToTensor)

from semgaze.utils.common import pair, expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, square_bbox



IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# ============================================================================= #
#                                 GazeHOI DATASET                               #
# ============================================================================= #
class GazeHOIDataset(Dataset):
    def __init__(        
        self,
        root,
        root_project,
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

        assert split in ("train", "val", "test", "all"), f"Expected `split` to be one of [`train`, `val`, `test`, `all`] but received `{split}` instead."
        assert (num_people == -1) or (num_people > 0), f"Expected `num_people` to be strictly positive or `-1`, but received {num_people} instead."

        self.root = root
        self.root_project = root_project
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.annotations, self.vocab2id = self.load_annotations()
        self.gaze_predictions = pd.read_csv(os.path.join(self.root_project, "data/gazehoi/gaze-predictions.csv"))
        self.key2indices = self.gaze_predictions.groupby(["path", "pid"]).indices

        
    def load_annotations(self):
        # Load dataframe with annotations
        #annotations = pd.read_csv(os.path.join(self.root, 'annotations', f'{self.split}-annotations.csv'))
        annotations_file = os.path.join(self.root_project, f"data/gazehoi/{self.split}-annotations.csv")
        annotations = pd.read_csv(annotations_file)
        
        # Filter out annotations where the head bbox is not available (e.g. not visible)
        cond = (annotations.h_xmin == -1.)
        annotations = annotations[~cond].reset_index(drop=True)  
        
        # Load/Build vocabulary mapping
        with open(os.path.join(self.root_project, 'data/gazehoi/vocab2id.json'), 'r') as f:
            vocab2id = json.load(f)
        
        return annotations, vocab2id
        
        
    def __getitem__(self, index: int) -> Dict:
        
        item = self.annotations.iloc[index] # get the annotation row from the dataframe
        file_name = item['file_name']
        basename, ext = os.path.splitext(file_name)

        # Load image
        img_path = os.path.join(self.root, 'images', file_name)
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size
        
        # Load object bbox
        obj_bbox = item[['o_xmin', 'o_ymin', 'o_xmax', 'o_ymax']]
        obj_bbox = torch.from_numpy(obj_bbox.values.astype(np.float32))
        obj_w, obj_h = obj_bbox[2] - obj_bbox[0], obj_bbox[3] - obj_bbox[1]
        obj_bbox /= torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)

        # Load pid, inout, gaze point and gaze label
        pid = item["pair_id"]
        inout = torch.tensor(1., dtype=torch.float)
        gaze_label = item["object"]
        gaze_label_id = torch.tensor(self.vocab2id[item.object])
        gaze_labels = "-".join(eval(item["aux_object"]))
        
        gaze_pt = torch.tensor([item["oc_x"] / img_w, item["oc_y"] / img_h], dtype=torch.float)
        if self.split == "train":
            pred_id = self.key2indices[(item.file_name, item.pair_id)]
            assert len(pred_id) == 1, "Groupby is not working properly"
            item_pred = self.gaze_predictions.iloc[pred_id].squeeze()
            assert item_pred.path == item.file_name, "Found mismatch in files between ground-truth and predictions."
            gaze_pt_pred = torch.tensor([item_pred["gp_pred_x"], item_pred["gp_pred_y"]], dtype=torch.float)
            if (obj_bbox[0] <= gaze_pt_pred[0] <= obj_bbox[2]) and (obj_bbox[1] <= gaze_pt_pred[1] <= obj_bbox[3]) and \
            max(obj_w, obj_h) / min(img_w, img_h) >= 0.3: 
                gaze_pt = gaze_pt_pred

        # Load head bboxes
        ## For target person
        target_head_bbox = item[['h_xmin', 'h_ymin', 'h_xmax', 'h_ymax']]
        target_head_bbox = torch.from_numpy(target_head_bbox.values.astype(np.float32)).unsqueeze(0)
        target_head_bbox = expand_bbox(target_head_bbox, img_w, img_h, k=0.1)
        target_head_center =  torch.stack([target_head_bbox[0, [0, 2]].mean(), target_head_bbox[0, [1, 3]].mean()])
        target_head_center /= torch.tensor([img_w, img_h], dtype=torch.float)
        
        ## For context people (ie. detected w/ Yolo)
        context_head_bboxes = torch.zeros((0, 4))
        if (self.num_people == -1) or (self.num_people > 1):
            det_file = f"{basename}-head-detections.npy"
            detections = np.load(os.path.join(self.root, "head-detections", det_file))

            # Process context head bboxes
            if len(detections) > 0:
                scores = torch.tensor(detections[:, -1])
                context_head_bboxes = torch.tensor(detections[(scores >= self.head_thr).tolist(), :-1])
                context_head_bboxes = expand_bbox(context_head_bboxes, img_w, img_h, k=0.1)
                ious = box_iou(context_head_bboxes, target_head_bbox).flatten()
                context_head_bboxes = context_head_bboxes[ious <= 0.9] # all head bboxes are detections in vcoco -> remove target

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
        
        # Load gaze labels
        label_emb_path = os.path.join(self.root_project, f"data/gazehoi/label-embeds/{gaze_label}-emb.pt")
        gaze_label_emb = torch.load(label_emb_path, weights_only=True)
        gaze_label_emb = F.normalize(gaze_label_emb, p=2, dim=-1)
    
        # Build Sample
        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pt": gaze_pt,
            "gaze_label": gaze_label,
            "gaze_label_id": gaze_label_id,
            "gaze_label_emb": gaze_label_emb,
            "gaze_labels": gaze_labels,
            "obj_bbox": obj_bbox,
            "inout": inout,
            "id": pid,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": file_name,
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
        try:
            sample["gaze_heatmap"] = generate_gaze_heatmap(sample["gaze_pt"], sigma=self.heatmap_sigma, size=self.heatmap_size)
        except Exception as e:
            print(f"Path: {sample['path']}")
            print(f"Gaze point: {sample['gaze_pt']}")
            print(f"Error generating gaze heatmap: {e}")
            import sys
            sys.stdout.flush()  # 👈 force the buffer to flush
            raise e
        
        # Compute gaze vec
        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][-1]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)
        
        # Generate head mask
        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample

    def __len__(self):
        return len(self.annotations)


# ============================================================================= #
#                               GazeHOI DATAMODULE                              #
# ============================================================================= #
class GazeHOIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_project: str,
        image_size: Union[int, tuple[int, int]] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 1,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
        batch_size: Union[int, dict] = 32,
    ):  
        
        super().__init__()
        self.root = root
        self.root_project = root_project
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.batch_size = {stage: batch_size for stage in ["train", "val", "test"]} if isinstance(batch_size, int) else batch_size


    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=0.8),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.train_dataset = GazeHOIDataset(
                root=self.root,
                root_project=self.root_project,
                split="train",
                transform=train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                head_thr=self.head_thr,
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeHOIDataset(                
                root=self.root,
                root_project=self.root_project,
                split="val",
                transform=val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                head_thr=self.head_thr,
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
            self.val_dataset = GazeHOIDataset(
                root=self.root,
                root_project=self.root_project,
                split="val",
                transform=val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                head_thr=self.head_thr,
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
            self.test_dataset = GazeHOIDataset(
                root=self.root,
                root_project=self.root_project,
                split="test",
                transform=test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size, 
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                head_thr=self.head_thr,
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )
        return dataloader




    
