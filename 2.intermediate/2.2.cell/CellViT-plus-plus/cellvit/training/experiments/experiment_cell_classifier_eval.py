class ExperimentCellVitClassifier(BaseExperiment):
    """CellVit Experiment Class

    Args:
        default_conf (dict): Default configuration
        checkpoint (Union[str, Path], optional): Checkpoint to use for training. Defaults to None
        just_load_model (bool, optional): If this flag is set and a checkpoint is provided, just the checkpoint of the model is loaded
            and not the checkpoint of the optimizer and scheduler. Usefull to perform pretraining and then finetune from epoch 0.and
            Defaults to false.

    Attributes:
        default_conf (dict): Default configuration
        run_conf (dict): Run configuration
        logger (Logger): Logger
        checkpoint (dict): Checkpoint
        just_load_model (bool): Just load model
        run_name (str): Run name

    Methods:
        run_experiment() -> tuple[Path, dict, nn.Module, dict]:
            Main Experiment Code
        get_loss_fn(weighted_sampling: bool = False, weight_factor: int = 5, weight_list: List[float] = None) -> Callable:
            Return loss function
        get_scheduler(scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:
            Get the learning rate scheduler for CellViT
        get_datasets(dataset: str, train_transforms: Callable = None, val_transforms: Callable = None, normalize_stains_train: bool = False, normalize_stains_val: bool = False, train_filelist: Union[Path, str] = None, val_filelist: Union[Path, str] = None) -> Tuple[Dataset, Dataset]:
            Retrieve training dataset and validation dataset
        get_wandb_init_dict() -> dict:
            Get the wandb init dictionary
        get_transforms(dataset: str, normalize_settings_default: dict, transform_settings: dict, input_shape: int) -> Tuple[Callable, Callable]:
            Get the transforms for the dataset
        get_trainer(dataset: str) -> BaseTrainer:
            Get the trainer for the dataset
        load_cellvit_model(cellvit_path: Union[str, Path]) -> Tuple[nn.Module, dict]:
            Load the CellViT model
        def _get_cellvit_architecture(model_type: Literal, model_config: dict) -> nn.Module:
            Return the trained model for inference
    """

    @staticmethod
    def seed_run(seed: int) -> None:
        """Seed the experiment

        Args:
            seed (int): Seed
        """
        # seeding
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run_experiment(self) -> tuple[Path, dict, nn.Module, dict]:
        """Main Experiment Code"""
        ### Setup
        # close loggers
        self.close_remaining_logger()

        # seeding
        self.seed_run(self.default_conf["random_seed"])

        # get the config for the current run
        self.run_conf = copy.deepcopy(self.default_conf)
        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{self.run_conf['logging']['log_comment']}"

        wandb_run_id = generate_id()
        resume = None
        if self.checkpoint is not None:
            wandb_run_id = self.checkpoint["wandb_id"]
            resume = "must"
            self.run_name = self.checkpoint["run_name"]

        # initialize wandb
        run = wandb.init(
            project=self.run_conf["logging"]["project"],
            tags=self.run_conf["logging"].get("tags", []),
            name=self.run_name,
            notes=self.run_conf["logging"]["notes"],
            dir=self.run_conf["logging"]["wandb_dir"],
            mode=self.run_conf["logging"]["mode"].lower(),
            group=self.run_conf["logging"].get("group", str(uuid.uuid4())),
            allow_val_change=True,
            id=wandb_run_id,
            resume=resume,
            settings=wandb.Settings(start_method="fork"),
        )

        # get ids
        self.run_conf["logging"]["run_id"] = run.id
        self.run_conf["logging"]["wandb_file"] = run.id

        # overwrite configuration with sweep values are leave them as they are
        if self.run_conf["run_sweep"] is True:
            self.run_conf["logging"]["sweep_id"] = run.sweep_id
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"])
                / f"sweep_{run.sweep_id}"
                / f"{self.run_name}_{self.run_conf['logging']['run_id']}"
            )
            self.overwrite_sweep_values(self.run_conf, run.config)
        else:
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"]) / self.run_name
            )

        # update wandb
        wandb.config.update(
            self.run_conf, allow_val_change=True
        )  # this may lead to the problem

        # create output folder, instantiate logger and store config
        self.create_output_dir(self.run_conf["logging"]["log_dir"])
        self.logger = self.instantiate_logger()
        self.logger.info("Instantiated Logger. WandB init and config update finished.")
        self.logger.info(f"Run ist stored here: {self.run_conf['logging']['log_dir']}")
        self.store_config()

        self.logger.info(
            f"Cuda devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        ### Machine Learning
        device = f"cuda:{self.run_conf['gpu']}"
        self.logger.info(f"Using GPU: {device}")
        self.logger.info(f"Using device: {device}")
        # device = 'cpu'
        self.logger.info(self.run_conf["training"].get("weight_list", 5))

        # loss functions
        loss_fn = self.get_loss_fn(
            weighted_sampling=self.run_conf["training"].get("weighted_sampling", False),
            weight_factor=self.run_conf["training"].get("weight_factor", 5),
            weight_list=self.run_conf["training"].get("weight_list", 5),
        )
        self.logger.info("Loss function:")
        self.logger.info(loss_fn)

        # cellvit_model
        cellvit_model, cellvit_run_conf = self.load_cellvit_model(
            self.run_conf["cellvit_path"]
        )
        cellvit_model.to(device)

        embed_dim = BACKBONE_EMBED_DIM[cellvit_run_conf["model"]["backbone"]]
        try:
            embed_dim = self.overwrite_emd_dim()
        except:
            pass


        if not self.run_conf["checkpoint_path"] is not None:
            model = LinearClassifier(
                embed_dim=embed_dim,
                hidden_dim=self.run_conf["model"].get("hidden_dim", 100),
                num_classes=self.run_conf["data"]["num_classes"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
            )
            self.logger.info("Vanilla classifier is used.")
        else:
            self.logger.info(f"Loading classifier checkpoint from {self.run_conf['checkpoint_path']}")

            pretrained_classifier_checkpoint = torch.load(self.run_conf["checkpoint_path"], map_location="cpu")
            pretrained_classifier_run_conf = unflatten_dict(pretrained_classifier_checkpoint["config"], ".")
            pretrained_classifier = LinearClassifier(
                embed_dim=pretrained_classifier_checkpoint["model_state_dict"]["fc1.weight"].shape[1],
                hidden_dim=pretrained_classifier_run_conf["model"].get("hidden_dim", 100),
                num_classes=pretrained_classifier_run_conf["data"]["num_classes"],
                drop_rate=0,
            )
            self.logger.info(pretrained_classifier.load_state_dict(pretrained_classifier_checkpoint["model_state_dict"]))

            model = pretrained_classifier

            if model.fc2.out_features == self.run_conf["data"]["num_classes"]:
                self.logger.info(print("Pretrained model is not altered"))
            else:
                import torch.nn as nn
                new_fc2 = nn.Linear(model.fc2.in_features, self.run_conf["data"]["num_classes"])
                model.fc2 = new_fc2

            

        self.logger.info(f"\n{summary(model, input_size=(1, embed_dim), device='cpu')}")
        model.to(device)

        # optimizer
        optimizer = self.get_optimizer(
            model,
            self.run_conf["training"]["optimizer"],
            self.run_conf["training"]["optimizer_hyperparameter"],
        )

        # scheduler
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_type=self.run_conf["training"]["scheduler"]["scheduler_type"],
        )

        # early stopping (no early stopping for basic setup)
        early_stopping = None
        if "early_stopping_patience" in self.run_conf["training"]:
            if self.run_conf["training"]["early_stopping_patience"] is not None:
                early_stopping = EarlyStopping(
                    patience=self.run_conf["training"]["early_stopping_patience"],
                    strategy="maximize",
                )

        ### Data handling
        train_transforms, val_transforms = self.get_transforms(
            dataset=self.run_conf["data"]["dataset"],
            normalize_settings_default=cellvit_run_conf["transformations"]["normalize"],
            transform_settings=self.run_conf.get("transformations", None),
            input_shape=self.run_conf["data"].get("input_shape", 1024),
        )

        train_dataset, val_dataset, test_dataset = self.get_datasets(
            dataset=self.run_conf["data"]["dataset"],
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            normalize_stains_train=self.run_conf["data"].get(
                "normalize_stains_train", False
            ),
            normalize_stains_val=self.run_conf["data"].get(
                "normalize_stains_val", False
            ),
            train_filelist=self.run_conf["data"].get("train_filelist", None),
            val_filelist=self.run_conf["data"].get("val_filelist", None),
            test_filelist=self.run_conf["data"].get("test_filelist", None),
        )

        # define dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,  # TODO: shift back
            num_workers=8,
            shuffle=False,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
            collate_fn=train_dataset.collate_batch,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,  # TODO: shift back
            num_workers=8,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            collate_fn=val_dataset.collate_batch,
        )

        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=8,  
                num_workers=8,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                collate_fn=test_dataset.collate_batch,
            )
        else:
            test_dataloader = None
            self.logger.info(f"No test dataset provided: {test_dataset}")

    eval_fn = CM_extractor

    cm_extractor = eval_fn(
        model=model,
        cellvit_model=cellvit_model,
        device=device,
        label=run_conf["data"].get("label", None),
    )

    cm_extractor.eval(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        export_folder=self.run_conf["logging"]["log_dir"],
    )