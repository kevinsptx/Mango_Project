# Mango: clasificador de madurez

Aplicación web hecha con Django que recibe la foto de un mango y predice su estado de madurez: verde, maduro o sobremaduro. Proyecto académico de Kevin Santiago.

> Este proyecto es académico. Las predicciones pueden contener errores y no deben usarse como control de calidad real.

## Qué hace

La aplicación tiene dos páginas:

- **Inicio:** explica el proyecto y cómo funciona.
- **Predecir:** permite subir una foto y muestra el estado del mango con el porcentaje de confianza de cada clase.

Detrás hay un modelo de visión por computador que usa transfer learning con MobileNetV2 (preentrenada en ImageNet), con una capa final entrenada para las tres clases del problema.

## Dataset

- 975 imágenes de mangos, de aproximadamente 800x800 píxeles.
- Fuente: (completar con el nombre y el enlace del dataset).

| Clase | Nombre en la app | Imágenes |
|---|---|---|
| UnRipe | Verde | 911 |
| Ripe | Maduro | 43 |
| OverRipe | Sobremaduro | 21 |

El dataset está muy desbalanceado: el 93% de las imágenes son de mangos verdes.

## Cómo está armado el proyecto

```text
Mango/
├── core/                  configuración del proyecto Django
├── Mangitos/              aplicación Django
│   ├── templates/         home.html, predecir.html, mango_svg.html y layout/base.html
│   ├── views.py           vistas de inicio y predicción
│   └── url.py             rutas
├── src/
│   ├── config.py          rutas y constantes
│   ├── dataset.py         manifiesto y split del dataset
│   ├── preprocessing.py   carga de imágenes y data augmentation
│   ├── architecture.py    modelo MobileNetV2
│   ├── train.py           utilidades de entrenamiento
│   └── predict.py         predicción usada por la web
├── notebooks/             exploración, preprocesamiento, entrenamiento y evaluación
├── models/                modelos entrenados
├── data/processed/        manifiestos del dataset
├── manage.py
└── requirements.txt
```

## Cómo se construyó el modelo

1. **Exploración (notebook 01):** se arma el manifiesto del dataset, se revisa el balance de clases y se genera el split.
2. **Split por mango:** cada mango tiene dos fotos (`_A` y `_B`). Para evitar que una foto quede en entrenamiento y la otra en prueba, el split se hace por mango y no por imagen. La división es 70% entrenamiento, 15% validación y 15% prueba, manteniendo la proporción de clases. Se verificó que ningún mango queda repartido en más de un grupo.
3. **Preprocesamiento (notebook 02):** las imágenes se redimensionan a 128x128, se normalizan entre 0 y 1 y, solo en entrenamiento, se aplica data augmentation (volteo horizontal, brillo y contraste).
4. **Entrenamiento (notebook 03):** se entrenan dos variantes de MobileNetV2 con la base congelada y se guarda la mejor época de cada una según el F1 macro en validación. Se usan pesos por clase suavizados (raíz cuadrada del peso balanceado) para compensar el desbalance sin desestabilizar el entrenamiento. La mejor variante queda como `models/mango_final.keras`.
5. **Evaluación (notebook 04):** se evalúa el modelo final con el conjunto de prueba.

Variantes entrenadas (15 épocas cada una):

| Variante | Learning rate | Dropout | F1 macro en validación |
|---|---|---|---|
| 1 | 0.001 | 0.3 | 0.6630 |
| 2 | 0.0001 | 0.5 | 0.5964 |

Se eligió la variante 1.

## Resultados en el conjunto de prueba

El conjunto de prueba tiene 152 imágenes: 139 verdes, 8 maduras y 5 sobremaduras.

| Métrica | Valor |
|---|---|
| Accuracy | 0.9671 |
| Precision (macro) | 0.8233 |
| Recall (macro) | 0.8595 |
| F1-score (macro) | 0.8314 |
| ROC-AUC | 0.9832 |

Resultados por clase:

| Clase | Precision | Recall | F1-score | Imágenes |
|---|---|---|---|---|
| Sobremaduro | 0.75 | 0.60 | 0.67 | 5 |
| Maduro | 0.73 | 1.00 | 0.84 | 8 |
| Verde | 0.99 | 0.98 | 0.99 | 139 |

Se reporta el F1 macro y no solo el accuracy, porque con un 93% de mangos verdes un modelo que siempre responde "verde" tendría un accuracy alto sin servir para nada.

El modelo acierta casi todos los verdes y todos los maduros, pero solo 3 de 5 sobremaduros. Con tan pocos ejemplos en el test, las cifras de las clases minoritarias son poco estables.

## Cómo ejecutarlo

Requiere Python 3.11.

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Luego abrir http://127.0.0.1:8000/

La carpeta `mango/` con las imágenes no se sube al repositorio. Para reentrenar, hay que colocarla en la raíz del proyecto (con las subcarpetas `OverRipe`, `Ripe` y `UnRipe`) o indicar otra ruta con la variable de entorno `MANGO_DATA_DIR`.

## Reentrenar el modelo

Ejecutar los notebooks en orden, cada uno con Run All:

1. `01-Exploración.ipynb`
2. `02-Preprocessing.ipynb`
3. `03-Training.ipynb`
4. `04-Evaluación.ipynb`

Al terminar el notebook 03, reiniciar el servidor para que la web cargue el modelo nuevo.

## Limitaciones

- **Desbalance de clases:** hay muy pocas imágenes de mangos maduros (43) y sobremaduros (21), por lo que el modelo reconoce mejor los verdes.
- **Conjunto de prueba pequeño:** solo tiene 8 imágenes de maduros y 5 de sobremaduros, así que las métricas de esas clases son poco estables. Por la misma razón, el F1 de validación (0.66) es más bajo que el de prueba (0.83): la validación solo tiene 2 sobremaduros.
- **Tipo de fotos:** el modelo funciona mejor con imágenes parecidas a las del dataset (buena luz, fondo limpio). Con otras variedades de mango o fotos con fondos complejos puede equivocarse.
- **Madurez y color:** en algunas variedades el color rojo o amarillo no indica madurez, lo que puede confundir al modelo.

## Tecnologías

Python, Django, TensorFlow/Keras, scikit-learn, pandas, Pillow y matplotlib.