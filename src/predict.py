import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASSES, IMG_SIZE, MODELS_DIR

MODEL_PATH = MODELS_DIR / "mango_final.keras"

NOMBRES = {
    "UnRipe": "Verde",
    "Ripe": "Maduro",
    "OverRipe": "Sobremaduro",
}

_modelo = None


def obtener_modelo():
    global _modelo
    if _modelo is None:
        _modelo = tf.keras.models.load_model(MODEL_PATH)
    return _modelo


def predecir_imagen(archivo):
    imagen = Image.open(archivo).convert("RGB")
    arreglo = np.array(imagen, dtype="float32") / 255.0
    arreglo = tf.image.resize(arreglo, IMG_SIZE).numpy()
    arreglo = np.expand_dims(arreglo, axis=0)

    probabilidades = obtener_modelo().predict(arreglo, verbose=0)[0]
    indice = int(np.argmax(probabilidades))
    clave = CLASSES[indice]

    return {
        "clave": clave,
        "clase": NOMBRES[clave],
        "confianza": round(float(probabilidades[indice]) * 100, 1),
        "probabilidades": [
            {
                "clave": CLASSES[i],
                "nombre": NOMBRES[CLASSES[i]],
                "valor": round(float(p) * 100, 1),
            }
            for i, p in enumerate(probabilidades)
        ],
    }