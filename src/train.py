from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, IMG_CHANNELS, CLASSES, MODELS_DIR


def compute_balanced_class_weight(labels):

    labels = np.asarray(labels)

    classes = np.unique(labels)

    total = len(labels)
    n_classes = len(classes)

    class_weight = {}

    for class_id in classes:
        count = np.sum(labels == class_id)

        weight = total / (n_classes * count)

        class_weight[int(class_id)] = float(weight)

    return class_weight


def build_model(
    learning_rate=0.001,
    dropout=0.3
):


    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(
            shape=(
                IMG_SIZE[0],
                IMG_SIZE[1],
                IMG_CHANNELS
            )
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(
            dropout
        ),

        tf.keras.layers.Dense(
            len(CLASSES),
            activation="softmax"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


def train_and_log(
    train_ds,
    val_ds,
    run_name,
    learning_rate=0.001,
    dropout=0.3,
    epochs=3,
    fine_tune=False,
    fine_tune_at=None,
    class_weight=None,
    tags=None
):

    print("=" * 60)
    print(f"Entrenamiento: {run_name}")
    print("=" * 60)

    print(f"Learning rate: {learning_rate}")
    print(f"Dropout: {dropout}")
    print(f"Epochs: {epochs}")
    print(f"Fine tuning: {fine_tune}")

    if fine_tune:
        print(
            "Aviso: el modelo CNN actual no utiliza "
            "fine-tuning."
        )


    model = build_model(
        learning_rate=learning_rate,
        dropout=dropout
    )

    model.summary()


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight
    )


    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    safe_name = run_name.replace(
        "/",
        "_"
    )

    model_path = (
        MODELS_DIR /
        f"{safe_name}.keras"
    )

    model.save(model_path)

    print()
    print("Modelo guardado en:")
    print(model_path)


    final_train_accuracy = history.history["accuracy"][-1]
    final_val_accuracy = history.history["val_accuracy"][-1]

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print()
    print("Métricas finales:")
    print(
        f"Train accuracy: "
        f"{final_train_accuracy:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{final_val_accuracy:.4f}"
    )

    print(
        f"Train loss: "
        f"{final_train_loss:.4f}"
    )

    print(
        f"Validation loss: "
        f"{final_val_loss:.4f}"
    )

    return {
        "run_name": run_name,
        "model_path": str(model_path),
        "history": history.history,
        "train_accuracy": float(final_train_accuracy),
        "val_accuracy": float(final_val_accuracy),
        "train_loss": float(final_train_loss),
        "val_loss": float(final_val_loss)
    }
