import tensorflow as tf
from transformers import TFAutoModelForSemanticSegmentation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os

cache_dir = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/model_cache"

id2label = {
    0: 'background',
    1: 'columnar',
    2: 'stroma',
    3: 'squamous',
}
label2id = {label: id for id, label in id2label.items()}
nClasses = len(id2label)



id2label_tcga = {
    0: 'background',
    1: 'tumor',
    2: 'stroma',
    3: 'inflam',
    4: 'necrosis',
    5: 'fat',
    6: 'parenchyma',
    7: 'blood_vessel',
    8: 'blood'
}
label2id_tcga = {label: id for id, label in id2label_tcga.items()}
num_labels_tcga = len(id2label_tcga)

def create_model(model_checkpoint, learning_rate, weight_path=None ):
    if not weight_path:
        model = TFAutoModelForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=nClasses,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate))
    else:
        model = TFAutoModelForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=num_labels_tcga,
            id2label=id2label_tcga,
            label2id=label2id_tcga,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
        model.load_weights(os.path.join(weight_path, "tf_model.h5"))

        base_model_4c = TFAutoModelForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=nClasses,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
        new_decode = base_model_4c.decode_head
        model.decode_head = new_decode

        model.config.id2label = id2label
        model.config.label2id = label2id
        model.config.num_labels = nClasses

        print("Weights loaded from:", weight_path)

    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model

def train_model(model, train_data_generator, valid_data_generator, steps_per_epoch, epochs, callbacks):
    history = model.fit(
        train_data_generator,
        validation_data=valid_data_generator,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=2
    )
    return history
