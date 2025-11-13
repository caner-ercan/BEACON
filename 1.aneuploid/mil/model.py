import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_similarity as tfsim
import tensorflow_similarity.losses as tfsim_losses
import tensorflow_addons as tfa
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten
from config import FOUNDATION_MODEL_BASE, ALGORITHM, TEMPERATURE, INIT_LR

def foundationModel(arch='path-50x1-remedis-s', no_linear=False, noAverage=False, trainable=False):
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    with tf.device(device):
        hub_path = os.path.join(FOUNDATION_MODEL_BASE, arch)
        module_keras = hub.KerasLayer(hub_path, trainable=trainable)
        model = tf.keras.Sequential(module_keras)
        model.build(input_shape=(None, 224, 224, 3))
        if not noAverage:
            model.add(AveragePooling2D(pool_size=(7, 7)))
            model.add(Flatten())
        if not no_linear:
            model.add(Dense(1024, activation=None, use_bias=False))
    return model

def create_contrastive_model(backbone, projector=None, predictor=None):
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    with tf.device(device):
        contrastive_model = tfsim.models.create_contrastive_model(
            backbone=backbone,
            projector=projector,
            predictor=predictor,
            algorithm=ALGORITHM,
            name=ALGORITHM,
        )
    return contrastive_model

def compile_model(model, optimizer, loss):
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    with tf.device(device):
        model.compile(
            optimizer=optimizer,
            loss=loss,
        )
    return model