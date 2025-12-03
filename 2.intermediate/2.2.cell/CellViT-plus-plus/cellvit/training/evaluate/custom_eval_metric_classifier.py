# custom_eval_metric_classifier.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import tqdm
import hashlib
from pathlib import Path
import h5py
from typing import List, Union
import logging

class CM_extractor:
    def __init__(
            self,
            model: torch.nn.Module,
            cellvit_model: torch.nn.Module,
            device: str,
            logger: logging.Logger,
            logdir: Union[Path, str],
            num_classes: int,
            experiment_config: dict,
            mixed_precision: bool = False,
            **kwargs,
        ):
        self.num_classes = num_classes
        self.cellvit_model = cellvit_model
        self.cellvit_model.eval()

        self.cache_cell_dataset = False
        self.cached_dataset = {}
        self.train_dataset_hash: str
        self.val_dataset_hash: str
        self.test_dataset_hash: str
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(experiment_config["random_seed"])

    def _calculate_hashes(self, train_dataloader, val_dataloader, test_dataloader):
        """Calculate hashes for training and validation dataset"""
        conf = self.experiment_config
        if "train_filelist" in conf["data"]:
            if "hash_info" in conf["data"]:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['train_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train_{conf['data']['hash_info']}"
            else:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['train_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train"
        else:
            if "hash_info" in conf["data"]:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train_{conf['data']['hash_info']}"
            else:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train"

        hasher = hashlib.sha256()
        hasher.update(train_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.train_dataset_hash = hash_value

        # validation
        if "val_filelist" in conf["data"]:
            if "hash_info" in conf["data"]:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['val_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val_{conf['data']['hash_info']}"
            else:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['val_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val"
        else:
            if "hash_info" in conf["data"]:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val_{conf['data']['hash_info']}"
            else:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val"

        hasher = hashlib.sha256()
        hasher.update(val_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.val_dataset_hash = hash_value

        # test
        if "test_filelist" in conf["data"]:
            if "hash_info" in conf["data"]:
                test_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['test_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_test_{conf['data']['hash_info']}"
            else:
                test_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['test_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_test"
        else:
            if "hash_info" in conf["data"]:
                test_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_test_{conf['data']['hash_info']}"
            else:
                test_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_test"

        hasher = hashlib.sha256()
        hasher.update(test_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.test_dataset_hash = hash_value

    def _test_cache_exists(self, dataset_part: str) -> bool:
        """Test if cache exists"""
        dataset_path = Path(self.experiment_config["data"]["dataset_path"])
        if dataset_part == "train":
            cache_path = dataset_path / "cache" / f"{self.train_dataset_hash}.h5"
        elif dataset_part == "val":
            cache_path = dataset_path / "cache" / f"{self.val_dataset_hash}.h5"
        elif dataset_part == "test":
            cache_path = dataset_path / "cache" / f"{self.test_dataset_hash}.h5"
        else:
            raise NotImplementedError("Unknown set")
        if cache_path.exists():
            return True
        else:
            return False

    def _load_from_cache(self, dataset_part: str) -> List:
        """Load from cache"""
        dataset_path = Path(self.experiment_config["data"]["dataset_path"])
        if dataset_part == "train":
            cache_path = dataset_path / "cache" / f"{self.train_dataset_hash}.h5"
        elif dataset_part == "val":
            cache_path = dataset_path / "cache" / f"{self.val_dataset_hash}.h5"
        elif dataset_part == "test":
            cache_path = dataset_path / "cache" / f"{self.test_dataset_hash}.h5"

        extracted_cells = []

        f = h5py.File(cache_path, "r")
        # Load data from datasets
        images = f["images"][:]
        coords = f["coords"][:]
        types = f["types"][:]
        tokens = f["tokens"][:]
        # Close the HDF5 file
        f.close()

        for image, coord, cell_type, token in zip(images, coords, types, tokens):
            cell = {
                "image": image.decode("utf-8"),
                "coords": coord[0],
                "type": torch.tensor(cell_type),
                "token": torch.tensor(token).type(torch.float32),
            }
            extracted_cells.append(cell)
        self.logger.info(f"Loaded dataset from cache: {str(cache_path)}")

        return extracted_cells

    def eval(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            **kwargs,):

        val_embedding_dataset = BaseCellEmbeddingDataset(extracted_cells)
        val_embedding_dataloader = DataLoader(
            val_embedding_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            worker_init_fn=BaseExperiment.seed_worker,
        )

    def evaluate_dataset(self, dataset_name, dataloader):
        """
        Evaluate the model for the given dataset and return predictions and ground truth.
        """
        with torch.no_grad():
            # model
            self.model.eval()

            # scores
            predictions = []
            probabilities = []
            gt = []

        # loop
        loop = tqdm.tqdm(
            enumerate(dataloader), total=len(dataloader)
        )
        for batch_idx, batch in loop:
            batch_metrics = self.validation_step(batch, batch_idx)
            predictions.append(batch_metrics["predictions"])
            probabilities.append(batch_metrics["probabilities"])
            gt.append(batch_metrics["gt"])

        predictions = torch.cat(predictions, dim=0).detach().cpu()
        gt = torch.cat(gt, dim=0).detach().cpu()

        plot_cm(predictions, gt, dataset_name,labels)

        return predictions, gt

    def plot_cm(self,predictions, gt, dataset_name,labels):

        cm = confusion_matrix(gt, predictions, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, values_format='.1%')
        plt.title(f'Multiclass Confusion Matrix - {dataset_name}')
        plt.savefig(self.logpath + f'/{dataset_name}_confusion_matrix.png')

    all_predictions = []
    all_gt = []

    all_predictions.append(evaluate_dataset('train', train_dataloader)[0])
    all_gt.append(evaluate_dataset('train', train_dataloader)[1])

    all_predictions.append(evaluate_dataset('val', val_dataloader)[0])
    all_gt.append(evaluate_dataset('val', val_dataloader)[1])

    all_predictions.append(evaluate_dataset('test', test_dataloader)[0])
    all_gt.append(evaluate_dataset('test', test_dataloader)[1])

    all_predictions.append(evaluate_dataset('total', total_dataloader)[0])
    all_gt.append(evaluate_dataset('total', total_dataloader)[1])

    # Concatenate all predictions and ground truth
    all_predictions = torch.cat(all_predictions, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    plot_cm(all_predictions, all_gt, labels, 'total')