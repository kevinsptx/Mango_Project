from pathlib import Path
import pandas as pd
from src.config import CLASSES, SEED, SPLIT_RATIOS


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def build_manifest(data_dir):

    data_dir = Path(data_dir)

    rows = []

    for label_name in CLASSES:

        class_dir = data_dir / label_name

        if not class_dir.exists():
            print(
                f"Advertencia: no se encontró la carpeta "
                f"'{class_dir}'"
            )
            continue

        for image_path in class_dir.rglob("*"):

            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:

                rows.append(
                    {
                        "filepath": str(image_path.resolve()),
                        "filename": image_path.name,
                        "label_name": label_name,
                        "label": CLASSES.index(label_name)
                    }
                )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise ValueError(
            f"No se encontraron imágenes en {data_dir}. "
            f"Revisa la ruta y los nombres de las carpetas."
        )

    manifest = manifest.sort_values(
        by=["label_name", "filename"]
    ).reset_index(drop=True)

    return manifest


def split_manifest(
    manifest,
    ratios=None,
    seed=SEED
):

    if ratios is None:
        ratios = SPLIT_RATIOS

    if not isinstance(manifest, pd.DataFrame):
        raise TypeError(
            "manifest debe ser un pandas.DataFrame"
        )

    required_columns = {
        "filepath",
        "filename",
        "label_name",
        "label"
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el manifiesto: "
            f"{missing_columns}"
        )

    train_parts = []
    val_parts = []
    test_parts = []

    # Dividimos cada clase por separado para conservar
    # el balance de las tres clases.
    for label_name, group in manifest.groupby(
        "label_name",
        sort=False
    ):

        group = group.sample(
            frac=1,
            random_state=seed
        ).reset_index(drop=True)

        n = len(group)

        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        train = group.iloc[:n_train]
        val = group.iloc[n_train:n_train + n_val]
        test = group.iloc[n_train + n_val:]

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat(
        [
            train_df,
            val_df,
            test_df
        ],
        ignore_index=True
    )

    # Mezclamos nuevamente el resultado
    result = result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    return result


def make_balanced_subset(
    manifest,
    per_class,
    seed=SEED
):

    subsets = []

    for label_name in CLASSES:

        class_data = manifest[
            manifest["label_name"] == label_name
        ]

        if len(class_data) < per_class:
            raise ValueError(
                f"La clase '{label_name}' solamente tiene "
                f"{len(class_data)} imágenes, pero se solicitaron "
                f"{per_class}."
            )

        selected = class_data.sample(
            n=per_class,
            random_state=seed
        )

        subsets.append(selected)

    result = pd.concat(
        subsets,
        ignore_index=True
    )

    result = result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    return result

def split_manifest_por_mango(manifest, ratios=None, seed=SEED):
    if ratios is None:
        ratios = SPLIT_RATIOS

    manifest = manifest.copy()
    manifest["mango_id"] = manifest["filename"].str.rsplit("_", n=1).str[0]

    partes = []

    for label_name, grupo in manifest.groupby("label_name", sort=False):
        ids = (
            grupo["mango_id"]
            .drop_duplicates()
            .sample(frac=1, random_state=seed)
            .tolist()
        )

        n = len(ids)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        asignacion = {}
        for i, mango_id in enumerate(ids):
            if i < n_train:
                asignacion[mango_id] = "train"
            elif i < n_train + n_val:
                asignacion[mango_id] = "val"
            else:
                asignacion[mango_id] = "test"

        grupo = grupo.copy()
        grupo["split"] = grupo["mango_id"].map(asignacion)
        partes.append(grupo)

    resultado = pd.concat(partes, ignore_index=True)
    resultado = resultado.drop(columns="mango_id")

    return resultado.sample(frac=1, random_state=seed).reset_index(drop=True)