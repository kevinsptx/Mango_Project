import tensorflow as tf

from src.config import IMG_SIZE, IMG_CHANNELS, BATCH_SIZE


def load_and_preprocess_image(filepath, label):

    image = tf.io.read_file(filepath)

    image = tf.image.decode_image(
        image,
        channels=IMG_CHANNELS,
        expand_animations=False
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    return image, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def dataset_from_manifest(
    manifest,
    split_name,
    batch_size=BATCH_SIZE,
    augment=False
):

    data = manifest[
        manifest["split"] == split_name
    ].copy()

    if data.empty:
        raise ValueError(
            f"No se encontraron imágenes para el split '{split_name}'."
        )

    filepaths = data["filepath"].values
    labels = data["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    if split_name == "train":
        dataset = dataset.shuffle(
            buffer_size=len(data),
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment and split_name == "train":
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset