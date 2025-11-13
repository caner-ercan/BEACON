import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from config import CSV_PATH, BASE_PATH, DATA_PATH, PRE_TRAIN_EPOCHS, VAL_STEPS_PER_EPOCH, INIT_LR, TEMPERATURE, ALGORITHM, BATCH_SIZE
from data_loader import get_tif_files, create_datasets
from model import foundationModel, create_contrastive_model, compile_model
from callbacks import create_callbacks
import tensorflow_similarity.losses as tfsim_losses
import tensorflow_addons as tfa
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='path-152x2-remedis-m')
    args = parser.parse_args()
    return args 
args = parse_args()



def main():
    tif_files_train, tif_files_val = get_tif_files(BASE_PATH, CSV_PATH)
    train_ds, val_ds = create_datasets(tif_files_train, tif_files_val)
    PRE_TRAIN_STEPS_PER_EPOCH = len(tif_files_train) // BATCH_SIZE
    args.arch = 'path-152x2-remedis-m'
    remedis_model = foundationModel(arch=args.arch)

    contrastive_model = create_contrastive_model(backbone=remedis_model)
    loss = tfsim_losses.SimCLRLoss(name=ALGORITHM, temperature=TEMPERATURE)
    optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
    contrastive_model = compile_model(contrastive_model, optimizer, loss)
    contrastive_model.summary()

    log_dir = DATA_PATH / args.arch / "models" / "logs" / f"{loss.name}_{time.time()}"
    chkpt_dir = DATA_PATH / args.arch / "models" / "checkpoints" / f"{loss.name}_{time.time()}"
    callbacks = create_callbacks(log_dir, chkpt_dir)

    history = contrastive_model.fit(
        train_ds,
        epochs=PRE_TRAIN_EPOCHS,
        steps_per_epoch=PRE_TRAIN_STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS_PER_EPOCH,
        callbacks=callbacks,
        verbose=1,
    )

    # delete previous model and backbone
    del contrastive_model
    del remedis_model

    remedis_model_2 = foundationModel(arch=args.arch, trainable=True)
    remedis_model_2.summary()

    contrastive_model_step2 = create_contrastive_model(backbone=remedis_model_2)
    optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR / 10)
    contrastive_model_step2 = compile_model(contrastive_model_step2, optimizer, loss)
    contrastive_model_step2.load_weights(chkpt_dir)

    log_dir = f"{log_dir}_step2"
    chkpt_dir = f"{chkpt_dir}_step2"
    callbacks = create_callbacks(log_dir, chkpt_dir)

    history = contrastive_model_step2.fit(
        train_ds,
        epochs=PRE_TRAIN_EPOCHS,
        steps_per_epoch=PRE_TRAIN_STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS_PER_EPOCH,
        callbacks=callbacks,
        verbose=1,
    )

    tf.saved_model.save(contrastive_model_step2, chkpt_dir)

if __name__ == "__main__":
    main()
    print("Done!" + args.arch)
    print("INIT_LR: ", INIT_LR)
    print("BATCH_SIZE: ", BATCH_SIZE)
    print("CHECKPOINT_DIR: ", chkpt_dir)
    print("LOG_DIR: ", log_dir)
    print("DATASET_DIR: ", BASE_PATH)
    print("EPOCHS: ", epochs)



