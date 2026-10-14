import tensorflow as tf

from src.config import CLASSES, IMG_CHANNELS, IMG_SIZE


def build_transfer_model(learning_rate=0.001, dropout=0.3):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    entradas = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS))
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(entradas)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    salidas = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    modelo = tf.keras.Model(entradas, salidas)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo