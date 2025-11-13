# -*- coding: utf-8 -*-
#
# Segmentation Dataset
#
# For an example of the required dataset structure, check out the example dataset
# in the test_database folder
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import csv
from pathlib import Path
from typing import Callable, Union, Tuple, List

import albumentations as A
import numpy as np
import torch
import torchstain
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
import tqdm
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import center_of_mass
from cellvit.training.utils.tools import get_bounding_box

class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        filelist_path: Union[Path, str] = None,
        # transforms: Callable = A.Compose(
        #     [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        # ),
        transforms: bool = False,
        normalize_stains: bool = False,
    ) -> None:
        """Segmentation Dataset for Cell Segmentation

        For an example of the required dataset structure, check out the example dataset
        in the test_database folder

        Args:
            dataset_path (Union[Path, str]): Path to the dataset parent folder
            split (str): Split of the dataset (train, test)
            filelist_path (Union[Path, str], optional): Path to a filelist (csv) to retrieve just a subset of images to use.
                Otherwise, all images from split are used. Defaults to None.
            transforms (Callable, optional): Transformations. Defaults to A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]).
            normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        """
        super().__init__()
        self.transforms = transforms
        self.normalize_stains = normalize_stains
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split / "images"
        self.annotation_path = self.dataset_path / self.split / "labels"

        self.images = [
            f
            for f in sorted(self.image_path.glob("*"))
            if f.suffix in [".png", ".jpg", ".jpeg"]
        ]

        if filelist_path is not None:
            selected_files = []
            with open(filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    selected_files.append(row[0])
            self.images = [f for f in self.images if f.stem in selected_files]

        self.annotations = []
        for img_path in self.images:
            img_name = img_path.stem
            self.annotations.append(self.annotation_path / f"{img_name}.npy")

        self.cache_images = {}
        self.cache_annotations = {}


    @staticmethod
    def clean_invalid_cells(inst_map, type_map):
        """
        Removes cells from inst_map that have no valid counterparts in type_map.
        
        Args:
            inst_map (np.ndarray): Instance map with unique cell IDs
            type_map (np.ndarray): Type map with cell class numbers
        
        Returns:
            tuple: (cleaned_inst_map, invalid_cell_ids)
                - cleaned_inst_map: inst_map with invalid cells removed
                - invalid_cell_ids: list of cell IDs that were removed
        """
        cleaned_inst_map = inst_map.copy()
        
        for inst_id in np.unique(inst_map):
            if inst_id == 0:  # Skip background
                continue
                
            # Get the cell type values for this instance
            cell_type = type_map[inst_map == inst_id]
            cell_type_nonzero = cell_type[cell_type != 0]
            
            # If no non-zero type values exist, this cell is invalid
            if len(cell_type_nonzero) == 0:
                cleaned_inst_map[inst_map == inst_id] = 0
        
        return cleaned_inst_map

    def cache_dataset(self) -> None:
        """Cache the dataset in memory"""
        for img_path, annot_path in tqdm.tqdm(
            zip(self.images, self.annotations), total=len(self.images)
        ):
            img = Image.open(img_path)
            img = img.convert("RGB")
            self.cache_images[img_path.stem] = img
            # if img.size != (256, 256):
            #     self.logger.info(f"Error: Image {img_path} has size {img.size}, expected (256, 256)")

            annotation = np.load(annot_path, allow_pickle=True)
            inst_map = annotation.item().get("inst_map")
            inst_map = inst_map.astype(np.uint32)
            type_map = annotation.item().get("type_map")
            type_map = type_map.astype(np.uint32)

            # if inst_map.shape != (256, 256):
            #     self.logger.info(f"Error: inst_map in {annot_path} has shape {inst_map.shape}, expected (256, 256)")
            # if type_map.shape != (256, 256):
            #     self.logger.info(f"Error: type_map in {annot_path} has shape {type_map.shape}, expected (256, 256)")

            # mask = np.stack([inst_map, type_map], axis=-1)
            inst_map = self.clean_invalid_cells(inst_map, type_map)
            stacked_mask = np.stack((inst_map, type_map), axis=-1)
            stacked_mask = np.array(stacked_mask, dtype=np.int64)

            # cache_annotation = {}
            # cache_annotation["inst_map"] = inst_map
            # cache_annotation["type_map"] = type_map
            self.cache_annotations[img_path.stem] = stacked_mask

            

            # cell_annot = []
            # for inst_id in np.unique(inst_map):
            #     if inst_id == 0:
            #         continue
            #     inst_mask = inst_map == inst_id
            #     inst_mask = inst_mask.astype(np.uint8)
            #     y, x = center_of_mass(inst_mask)

            #     cell_type = type_map[inst_map == inst_id]  # mask
            #     cell_type = cell_type[cell_type != 0]
            #     type_ids, counts = np.unique(cell_type, return_counts=True)
            #     cell_annot.append(
            #         (
            #             int(np.round(x)),
            #             int(np.round(y)),
            #             int(type_ids[np.argmax(counts)]),
            #         )
            #     )


            # self.cache_annotations[img_path.stem] = cell_annot
    
    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map


    # def get_zipped_item(img_path):
        
    #     img = Image.open(img_path)
    #     img = img.convert("RGB")

    #     img_name = img_path.stem
    #     annot_path = self.annotation_path / f"{img_name}.npy"

    #     annotation = np.load(annot_path, allow_pickle=True)
    #     inst_map = annotation.item().get("inst_map")
    #     inst_map = inst_map.astype(np.uint32)
    #     type_map = annotation.item().get("type_map")
    #     type_map = type_map.astype(np.uint32)

    #     mask = np.stack([inst_map, type_map], axis=-1)

    #     return img, mask




    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, list, list, str]:
        """Get item from dataset

        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, list, list, str]:
            * Image
            * List of detections
            * List of types
            * Name of the Patch
        """
        img_path = self.images[index]
        img_name = img_path.stem
        
        # img, mask = self.get_zipped_item(img_path)


        img = self.cache_images[img_name]
        stacked_mask = self.cache_annotations[img_name]
        # detections = [(int(x), int(y)) for x, y, _ in cell_annot]
        # types = [int(int(t) - 1) for _, _, t in cell_annot]

        # detections = [(int(x), int(y)) for x, y, _ in cell_annot]
        # types = [int(int(t) - 1) for _, _, t in cell_annot]
        # print(detections)
        # inst_map = annotation["inst_map"]
        # inst_map = inst_map.astype(np.uint32)
        # type_map = annotation["type_map"]
        # type_map = type_map.astype(np.uint32)

        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))

        img = np.array(img).astype(np.uint8)

        stacked_mask = np.array(stacked_mask, dtype=np.int64)

        if self.transforms:
            # transformed = self.transforms(image=img, keypoints=detections)
            # img = transformed["image"]
            # detections = transformed["keypoints"]
            # types = [types[idx] for idx, _ in enumerate(detections)]

            transformed = self.transforms(image=img, mask=stacked_mask)
            img = transformed["image"]
            stacked_mask = transformed["mask"]
            inst_map = stacked_mask[:, :, 0].clone()
            type_map = stacked_mask[:, :, 1].clone()
        else:
            inst_map = stacked_mask[:, :, 0].copy()
            type_map = stacked_mask[:, :, 1].copy()


        
        np_map = np.array(inst_map, np.uint8)
        np_map[np_map > 0] = 1
        
        # Generate hv_map using the instance map
        hv_map = SegmentationDataset.gen_instance_hv_map(np.array(inst_map, dtype=np.int64))

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }



        # return img, detections, types, img_name
        return img, masks, img_name

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, List[list], List[list], List[str]]:
        """Create a custom batch

        Needed to unpack List of tuples with dictionaries and array

        Args:
            batch (List[Tuple]): Input batch consisting of a list of tuples (patch, cell_coordinates, cell_types, patch_names)

        Returns:
            Tuple[torch.Tensor, List[list], List[list], List[str]]:
                * patches with shape [batch_size, 3, patch_size, patch_size]
                * List of detections, each entry is a list with one entry for each ground truth cell
                * list of types, each entry is the cell type for each ground truth cell
                * list of patch names
        """
        # imgs, detections_list, types_list, names = zip(*batch)


        imgs, masks, names = zip(*batch)

        imgs = torch.stack(imgs)
        batched_masks = {
            "nuclei_binary_map": torch.stack([m["nuclei_binary_map"] for m in masks]),
            "nuclei_type_map": torch.stack([m["nuclei_type_map"] for m in masks]),
            "hv_map": torch.stack([m["hv_map"] for m in masks]),
            "instance_map": torch.stack([m["instance_map"] for m in masks]),
        }
        # return imgs, list(detections_list), list(types_list), list(names)
        return imgs, batched_masks, list(names)
