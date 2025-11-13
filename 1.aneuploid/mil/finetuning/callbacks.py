import tensorflow as tf
import tensorflow_similarity.callbacks as tfsim_callbacks

def create_callbacks(log_dir, chkpt_dir):
    tbc = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=100,
    )
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpt_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )
    return [tbc, mcp]