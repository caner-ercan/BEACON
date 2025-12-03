from http.client import FOUND
import sys
sys.path.insert(0, '/rsrch5/home/trans_mol_path/cercan/.conda/envs/clam_albm_tf/lib/python3.7/site-packages/')
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
import tensorflow_hub as hub
import os
from keras.layers import Flatten, Dense, AveragePooling2D

foundation_model_base = '/rsrch5/home/trans_mol_path/cercan/foundationModels/physionet.org/files/medical-ai-research-foundation/1.0.0/'


# def foundationModel(arch = 'path-50x1-remedis-s',no_linear=False, noAverage = False):
#     device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
#     with tf.device(device):
#         hub_path = os.path.join(foundation_model_base,arch)

#         module_keras=hub.KerasLayer(hub_path, trainable=False)
#         model = tf.keras.Sequential(module_keras)
#         model.build(input_shape=(None, 224,224,3))
#         #model.add(Flatten()) # avrpool
#         if not noAverage:
#             model.add(AveragePooling2D(pool_size=(7, 7))) #TODO change to trainable conv layer
#             model.add(Flatten())
#         if not no_linear:
#             model.add(Dense(1024, activation= None,use_bias=False)) # input_shape=(131072,)

#     return model


def foundationModel(arch = 'path-50x1-remedis-s',no_linear=False, noAverage = False):
    output_base_path = "/rsrch5/home/trans_mol_path/cercan/foundationModels/finetuned/448_ds2/"
    hub_path = os.path.join(output_base_path, arch, "saved_backbone_model")
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    with tf.device(device):
        # hub_path = arch
        module_keras=hub.KerasLayer(hub_path, trainable=False)
        model = tf.keras.Sequential(module_keras)
        model.build(input_shape=(None, 224,224,3))
    return model

# def foundationModel(arch = 'path-50x1-remedis-s'):
#     hub_path = os.path.join(foundation_model_base,arch)
#     module_keras=hub.KerasLayer(hub_path, trainable=False)

#     model = tf.keras.Sequential([
#         #tf.keras.layers.InputLayer(input_shape=(224,224,3)),
#         module_keras,
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1024, activation= None, input_shape=(131072,))
#     ])

#     model.build(input_shape=(None, 224,224,3))
#     model.compile()

#     return model
