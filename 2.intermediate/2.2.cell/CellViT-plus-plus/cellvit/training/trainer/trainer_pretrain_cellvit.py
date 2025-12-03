# -*- coding: utf-8 -*-
# CellViT Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_trainer import BaseTrainer
from cellvit.training.utils.metrics import get_fast_pq, remap_label
from cellvit.training.utils.tools import AverageMeter
from cellvit.utils.tools import remove_small_objects
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes, measurements
from skimage.segmentation import watershed
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index

from cellvit.training.utils.matching_metrics import matching
from collections import defaultdict

class CellViTPretrainer(BaseTrainer):
    """CellViT trainer class

    Args:
        model (CellViT): CellViT model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
            {
                "nuclei_binary_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "hv_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
            }
            Required Keys are:
                * nuclei_binary_map
                * hv_map
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        log_images (bool, optional): If images should be logged to WandB. Defaults to False.
        magnification (int, optional): Image magnification. Please select either 40 or 20. Defaults to 40.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
    """

    def __init__(
        self,
        model: CellViT,
        loss_fn_dict: dict,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        num_classes: int,
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        log_images: bool = False,
        magnification: int = 40,
        mixed_precision: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            accum_iter=1,
            log_images=log_images,
            mixed_precision=mixed_precision,
        )
        self.loss_fn_dict = loss_fn_dict
        self.num_classes = num_classes
        self.magnification = magnification

        # setup logging objects
        self.loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss", ":.4f")}
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(
                    f"{branch}_{loss_name}", ":.4f"
                )

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, unfreeze_decoder_epoch: int = 10, unfreeze_all_epoch: int = 50
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_dataloader (DataLoader): Train dataloader
            unfreeze_epoch (int, optional): Epoch to unfreeze layers
        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """

        self.model.train()
        # if epoch >= unfreeze_epoch:
        #     self.model.unfreeze_encoder()
        if epoch == unfreeze_decoder_epoch:
            self.model.unfreeze_all()
            self.model.freeze_encoder()
            self.logger.info("Decoder unfrozen")
        elif epoch == unfreeze_all_epoch:
            self.model.unfreeze_encoder()
            self.logger.info("Encoder unfrozen")

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        binary_f1s = []
        binary_accs = []
        class_metrics_aggregated = defaultdict(list)

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()

        train_loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True)

        for batch_idx, batch in train_loop:
            batch_metrics, batch_metrics_class = self.train_step(batch, batch_idx, len(train_dataloader), epoch)
            binary_dice_scores = (
                binary_dice_scores + batch_metrics["binary_dice_scores"]
            )
            binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
            )
            binary_f1s = (
                binary_f1s + batch_metrics["binary_f1s"]
            )
            binary_accs = ( 
                binary_accs + batch_metrics["binary_accs"]
            )
            pq_scores = pq_scores + batch_metrics["pq_scores"]
            for metric_name, metric_value in batch_metrics_class.items():
                class_metrics_aggregated[f"{metric_name}"].append(metric_value)


            train_loop.set_postfix(
                {
                    "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                    "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                }
            )
        
        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        binary_f1s = np.array(binary_f1s)
        binary_accs = np.array(binary_accs)

        class_scalar_metrics = {}
        for metric_key, metric_list in class_metrics_aggregated.items():
            metric_array = np.array(metric_list)
            class_scalar_metrics[f"{metric_key}-Mean/Train"] = np.nanmean(metric_array)
        class_scalar_metrics["pq_class_mean/Train"] = np.mean([value for key, value in class_scalar_metrics.items() if key.startswith("pq_class_")])              

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-Cell-Dice-Mean/Train": np.nanmean(binary_dice_scores),
            "Binary-Cell-Jacard-Mean/Train": np.nanmean(binary_jaccard_scores),
            f"bPQ/Train": np.nanmean(pq_scores),
            "Binary-Cell-F1-Mean/Train": np.nanmean(binary_f1s),
            "Binary-Cell-Acc-Mean/Train": np.nanmean(binary_accs),
            **class_scalar_metrics
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[f"{branch}_{loss_name}/Train"] = self.loss_avg_tracker[
                    f"{branch}_{loss_name}"
                ].avg

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"bPQ-Score: {np.nanmean(pq_scores):.4f} - "
            f"Binary-Cell-F1: {np.nanmean(binary_f1s):.4f} - "
            f"Binary-Cell-accuracy: {np.nanmean(binary_accs):.4f}"
        )

        if class_scalar_metrics:
            # Collect all class metrics for the current mode
            class_metrics_list = []
            for metric_key, metric_value in sorted(class_scalar_metrics.items()):
                class_metrics_list.append(f"{metric_key}: {metric_value:.4f}")
            class_metrics_str = " - ".join(class_metrics_list)
            self.logger.info(f"{class_metrics_str}")

        return scalar_metrics, None

    def train_step(
        self,
        batch: object,
        batch_idx: int,
        num_batches: int,
        epoch: int,
    ) -> Tuple[dict, Union[plt.Figure, None]]:
        """Training step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            num_batches (int): Total number of batches in epoch

        Returns:
            Tuple[dict, UNone]]:
                * Batch-Metrics: dictionary with the following keys:
                * Example prediction image, here None
        """
        # unpack batch
        imgs = batch[0].to(self.device)  # imgs shape: (batch_size, 3, H, W)
        masks = batch[1]  # dict: keys: "instance_map", "nuclei_binary_map", "hv_map"
        # self.logger.info(masks[0])

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)
                # self.logger.info(predictions_)

                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                # batched_masks = {
                #     "nuclei_binary_map": torch.stack([m["nuclei_binary_map"] for m in masks]),
                #     "hv_map": torch.stack([m["hv_map"] for m in masks]),
                #     "instance_map": torch.stack([m["instance_map"] for m in masks]),
                #     }
                gt = self.unpack_masks(masks=masks)

                # calculate loss
                total_loss = self.calculate_loss(predictions, gt)
                # self.logger.info(f"Total loss: {total_loss}")
                # backward pass
                self.scaler.scale(total_loss).backward()
                if (
                    ((batch_idx + 1) % self.accum_iter == 0)
                    or ((batch_idx + 1) == num_batches)
                    or (self.accum_iter == 1)
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
        else:
            predictions_ = self.model.forward(imgs)
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks)

            # calculate loss
            total_loss = self.calculate_loss(predictions, gt)

            total_loss.backward()
            if (
                ((batch_idx + 1) % self.accum_iter == 0)
                or ((batch_idx + 1) == num_batches)
                or (self.accum_iter == 1)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

        batch_metrics, batch_metrics_class = self.calculate_step_metric_train(predictions, gt)

        # if batch_idx == num_batches - 1:  # End of epoch
        #     self.save_combined_image(imgs, masks, predictions, epoch)

        return batch_metrics, batch_metrics_class

    def save_combined_image(self, imgs, masks, predictions, epoch):
        """Save combined image of input, mask, and prediction."""
        import os
        from PIL import Image
        # Convert tensors to numpy arrays
        img = imgs[0].cpu().permute(1, 2, 0).numpy()
        mask = masks["instance_map"].cpu().numpy()
        # prediction = predictions.cpu().numpy() 

        save_folder = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.1.nucleus/cellvit/images/"
        # Ensure the directory exists
        os.makedirs(save_folder, exist_ok=True)

        # Save image as PNG
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        # img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil.save(os.path.join(save_folder, f"epoch_{epoch}_img.png"))

        # Save mask and prediction as numpy arrays
        np.save(os.path.join(save_folder, f"epoch_{epoch}_mask.npy"), mask)
        np.save(os.path.join(save_folder, f"epoch_{epoch}_prediction.npy"), predictions)
        # # Create a figure with subplots
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(img)
        # axs[0].set_title("Input Image")
        # axs[0].axis("off")

        # axs[1].imshow(mask, cmap="gray")
        # axs[1].set_title("Mask")
        # axs[1].axis("off")

        # axs[2].imshow(prediction, cmap="gray")
        # axs[2].set_title("Prediction")
        # axs[2].axis("off")

        # # Save the figure
        # os.makedirs("epoch_images", exist_ok=True)
        # plt.savefig(f"/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.1.nucleus/cellvit/images/epoch_{epoch}.png")
        # plt.close()

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader, mode: str = "val"
    ) -> Tuple[dict, dict, float]:
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Tuple[dict, dict, float]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
                * Early stopping metric
        """
        self.model.eval()

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        # cell_type_pq_scores = []
        binary_f1s = []
        binary_accs = []
        class_metrics_aggregated = defaultdict(list)


        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()

        val_loop = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader),disable=True)

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                batch_metrics, batch_metrics_class = self.validation_step(batch, batch_idx)
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )
                binary_f1s = (
                    binary_f1s + batch_metrics["binary_f1s"]
                )
                binary_accs = ( 
                    binary_accs + batch_metrics["binary_accs"]
                )
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                # cell_type_pq_scores = (
                #     cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
                # )

                for metric_name, metric_value in batch_metrics_class.items():
                    class_metrics_aggregated[f"{metric_name}"].append(metric_value)


                val_loop.set_postfix(
                    {
                        "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                        "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                    }
                )

        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        binary_f1s = np.array(binary_f1s)
        binary_accs = np.array(binary_accs)

        class_scalar_metrics = {}
        for metric_key, metric_list in class_metrics_aggregated.items():
            metric_array = np.array(metric_list)
            class_scalar_metrics[f"{metric_key}-Mean/{mode}"] = np.nanmean(metric_array)
        class_scalar_metrics[f"pq_class_mean/{mode}"] = np.mean([value for key, value in class_scalar_metrics.items() if key.startswith("pq_class_")])  


        scalar_metrics = {
            f"Loss/{mode}": self.loss_avg_tracker["Total_Loss"].avg,
            f"Binary-Cell-Dice-Mean/{mode}": np.nanmean(binary_dice_scores),
            f"Binary-Cell-Jacard-Mean/{mode}": np.nanmean(binary_jaccard_scores),
            f"bPQ/{mode}": np.nanmean(pq_scores),
            f"Binary-Cell-F1-Mean/{mode}": np.nanmean(binary_f1s),
            f"Binary-Cell-Acc-Mean/{mode}": np.nanmean(binary_accs),
            # f"mPQ/{mode}": np.nanmean(
            #     [np.nanmean(pq) for pq in cell_type_pq_scores]
            # ),
            **class_scalar_metrics
        }
        if mode == "val":
            for branch, loss_fns in self.loss_fn_dict.items():
                for loss_name in loss_fns:
                    scalar_metrics[
                        f"{branch}_{loss_name}/Validation"
                    ] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        self.logger.info(
            f"{mode} epoch stats: {'': <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"bPQ-Score: {np.nanmean(pq_scores):.4f} - "
            f"Binary-Cell-F1: {np.nanmean(binary_f1s):.4f} - "
            f"Binary-Cell-accuracy: {np.nanmean(binary_accs):.4f}"
        )
            # f"mPQ-Score: {scalar_metrics[f'mPQ/{mode}']:.4f}"

        if class_scalar_metrics:
            # Collect all class metrics for the current mode
            class_metrics_list = []
            for metric_key, metric_value in sorted(class_scalar_metrics.items()):
                class_metrics_list.append(f"{metric_key}: {metric_value:.4f}")
            class_metrics_str = " - ".join(class_metrics_list)
            self.logger.info(f"{class_metrics_str}")
        

        return scalar_metrics, None, class_scalar_metrics[f"pq_class_mean/{mode}"]  #np.nanmean(pq_scores)  #scalar_metrics[f'Loss/{mode}']

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
    ):
        """Validation step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            return_example_images (bool): If an example preciction image should be returned

        Returns:
            Tuple[dict, None]]:
                * Batch-Metrics: dictionary, structure not fixed yet
                * Example prediction image, here None
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0].to(self.device)
        masks = batch[1]

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)
                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks)
                # calculate loss
                _ = self.calculate_loss(predictions, gt)

        else:
            predictions_ = self.model.forward(imgs)
            # reshaping and postprocessing
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks)
            # calculate loss
            _ = self.calculate_loss(predictions, gt)

        # get metrics for this batch
        batch_metrics, batch_metrics_class = self.calculate_step_metric_validation(predictions, gt)

        return batch_metrics, batch_metrics_class

    # def unpack_predictions(self, predictions: dict) -> dict:
    #     """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

    #     Args:
    #         predictions (dict): Dictionary with the following keys:
    #             * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, 2, H, W)
    #             * hv_map: Logit output for hv-prediction. Shape: (batch_size, 2, H, W)

    #     Returns:
    #         dict: Processed network output. Keys are:
    #             * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
    #             * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
    #             * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
    #                 Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
    #     """
    #     predictions["nuclei_binary_map"] = F.softmax(
    #         predictions["nuclei_binary_map"], dim=1
    #     ) # shape: (batch_size, 2, H, W)
    #     predictions["nuclei_type_map"] = F.softmax(
    #         predictions["nuclei_type_map"], dim=1
    #     )  # shape: (batch_size, num_nuclei_classes, H, W)

    #     # convert to instance preds
    #     instance_predictions = []
    #     predictions_ = predictions.copy()
    #     predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
    #         0, 2, 3, 1
    #     )
    #     predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)
    #     for i in range(predictions_["nuclei_binary_map"].shape[0]):
    #         pred_inst = np.concatenate(
    #             [
    #                 torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
    #                 .detach()
    #                 .cpu()[..., None]
    #                 .numpy(),
    #                 predictions_["hv_map"][i].detach().cpu().numpy(),
    #             ],
    #             axis=-1,
    #         )
    #         pred_inst = np.squeeze(pred_inst)
    #         pred_inst = self.proc_np_hv(pred_inst)
    #         instance_predictions.append(pred_inst)

    #     predictions["instance_map"] = torch.Tensor(np.stack(instance_predictions))

    #     return predictions

    def unpack_predictions(self, predictions: dict) -> dict:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, 2, H, W)
                * hv_map: Logit output for hv-prediction. Shape: (batch_size, 2, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)

        Returns:
            DataclassHVStorage: Processed network output
        """
        predictions["tissue_types"] = predictions["tissue_types"].to(self.device)
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = self.model.calculate_instance_map(
            predictions, self.magnification
        )  # shape: (batch_size, H, W)
        predictions["instance_types_nuclei"] = self.model.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"]
        ).to(
            self.device
        )  # shape: (batch_size, num_nuclei_classes, H, W)

        if "regression_map" not in predictions.keys():
            predictions["regression_map"] = None
            
        return predictions

    def unpack_masks(self, masks: dict) -> dict:
        """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

        Args:
            masks (dict): Required keys are:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map (torch.Tensor): Binary nuclei segmentations. Shape: (batch_size, H, W)
                * hv_map (torch.Tensor): HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            dict:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        """
        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(
            torch.float32
        )  # background, nuclei
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        # print(nuclei_type_maps)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_classes
        ).type(
            torch.float32
        )  # background + nuclei types

        # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, num_nuclei_classes)
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, 2)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, H, W, 2)
            "instance_map": masks["instance_map"].to(self.device),
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class 
        }
        # gt_nuclei_binary_map_onehot = (
        #     F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        # ).type(torch.float32)

        # gt = {
        #     "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(self.device),
        #     "hv_map": masks["hv_map"].to(self.device),
        #     "instance_map": masks["instance_map"].to(self.device),
        # }


        return gt

    def calculate_loss(self, predictions: dict, gt: dict) -> torch.Tensor:
        """Calculate the loss

        Args:
            predictions (dict): Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            gt (dict): Ground-truth:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            torch.Tensor: Loss
        """

        total_loss = 0

        for branch, gt_value in gt.items():
            if branch in [
                "instance_map",
            ]:
                continue
            if branch not in self.loss_fn_dict:
                continue
            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=predictions[branch],
                        target=gt_value,
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                else:
                    loss_value = loss_fn(input=predictions[branch], target=gt_value)
                total_loss = total_loss + weight * loss_value
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss

    def calculate_step_metric_train(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (dict): Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            gt (dict): Ground-truth:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores
        """
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        # predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].detach().cpu()
        # gt["nuclei_binary_map"] = gt["nuclei_binary_map"].detach().cpu()


        binary_dice_scores = []
        binary_jaccard_scores = []
        f1_scores = []
        accs = []
        # cell_type_pq_scores = []
        pq_scores = []

        batch_metrics_class = {}

        for i in range(gt["nuclei_binary_map"].shape[0]):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

            stats = matching(target_binary_map, pred_binary_map, thresh=0.5)
            f1_scores.append(float(stats.f1))
            accs.append(float(stats.accuracy))

            # pq values
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)

            # pq values per class (skip background)
            # nuclei_type_pq = []
            for j in range(0, self.num_classes):
                pred_nuclei_instance_class = remap_label(
                    predictions["instance_types_nuclei"][i][j, ...]
                )
                target_nuclei_instance_class = remap_label(
                    gt["instance_types_nuclei"][i][j, ...]
                )

                # if ground truth is empty, skip from calculation
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                else:
                    [_, _, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class,
                        target_nuclei_instance_class,
                        match_iou=0.5,
                    )
                    stats_class = matching(target_nuclei_instance_class, pred_nuclei_instance_class, thresh=0.5)
                    batch_metrics_class[f"pq_class_{j}"] = pq_tmp
                    batch_metrics_class[f"f1_class_{j}"] = stats_class.f1
                    batch_metrics_class[f"acc_class_{j}"] = stats_class.accuracy
                # nuclei_type_pq.append(pq_tmp)

            # cell_type_pq_scores.append(nuclei_type_pq)

        batch_metrics_binary = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "binary_f1s": f1_scores,
            "binary_accs": accs,
            "pq_scores": pq_scores,
            # "cell_type_pq_scores": cell_type_pq_scores,
        }

        return batch_metrics_binary, batch_metrics_class

    def calculate_step_metric_validation(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the validation step

        Args:
            predictions (dict): Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            gt (dict): Ground-truth:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores, pq_scores
        """

        # Tissue Tpyes logits to probs and argmax to get class
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        # predictions["nuclei_binary_map"] = predictions["instance_map"].detach().cpu()
        # gt["nuclei_binary_map"] = gt["nuclei_binary_map"].detach().cpu()


        binary_dice_scores = []
        binary_jaccard_scores = []
        f1_scores = []
        accs = []
        # cell_type_pq_scores = []
        pq_scores = []

        batch_metrics_class = {}

        for i in range(gt["nuclei_binary_map"].shape[0]):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

            stats = matching(target_binary_map, pred_binary_map, thresh=0.5)
            f1_scores.append(float(stats.f1))
            accs.append(float(stats.accuracy))
            
            # pq values
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)

            # pq values per class (skip background)
            # nuclei_type_pq = []
            # nuclei_type_f1 = []
            for j in range(0, self.num_classes):
                pred_nuclei_instance_class = remap_label(
                    predictions["instance_types_nuclei"][i][j, ...]
                )
                target_nuclei_instance_class = remap_label(
                    gt["instance_types_nuclei"][i][j, ...]
                )

                # if ground truth is empty, skip from calculation
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                else:
                    [_, _, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class,
                        target_nuclei_instance_class,
                        match_iou=0.5,
                    )
                    stats_class = matching(target_nuclei_instance_class, pred_nuclei_instance_class, thresh=0.5)
                    batch_metrics_class[f"pq_class_{j}"] = pq_tmp
                    batch_metrics_class[f"f1_class_{j}"] = stats_class.f1
                    batch_metrics_class[f"acc_class_{j}"] = stats_class.accuracy
                # nuclei_type_pq.append(pq_tmp)

            # cell_type_pq_scores.append(nuclei_type_pq)

        batch_metrics_binary = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "binary_f1s": f1_scores,
            "binary_accs": accs,
            "pq_scores": pq_scores,
        }

        return batch_metrics_binary, batch_metrics_class

    def proc_np_hv(self, pred_inst: np.ndarray, object_size: int = 10, ksize: int = 21):
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21

        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """

        # Check input types and values
        assert isinstance(pred_inst, np.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 3, "pred_inst must be a 3-dimensional array"
        assert (
            pred_inst.shape[2] == 3
        ), "The last dimension of pred_inst must have a size of 3"
        assert isinstance(object_size, int), "object_size must be an integer"
        assert object_size > 0, "object_size must be greater than 0"
        assert isinstance(ksize, int), "ksize must be an integer"
        assert ksize > 0, "ksize must be greater than 0"

        # ensure dtype and extract individual channels
        pred = np.array(pred_inst, dtype=np.float32)
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        # Normalize the horizontal and vertical direction maps to [0, 1]
        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Apply Sobel filter to the direction maps
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        # Normalize and invert the Sobel filtered images
        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        # Combine the Sobel filtered images
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # Create distance map
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        # Apply all
        overall = np.array(overall >= 0.4, dtype=np.int32)
        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        # Separate instances
        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred
