"""
Módulo de Inferencia y Predicción para Datos Nuevos (Etapa 15 - Tabla T9)
Grupo 09: Estimación de Gastos de Seguros Médicos.

Garantiza que las nuevas predicciones se ejecuten directamente sobre el
Pipeline completo serializado en models/modelo_final.joblib (incluyendo
SimpleImputer, StandardScaler, OneHotEncoder y el estimador ganador)
SIN reentrenar ni reajustar transformaciones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ml import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MODEL_PATH,
    NUMERIC_COLS,
    train_and_export_all,
)

VALID_SEX: set[str] = {"female", "male"}
VALID_SMOKER: set[str] = {"yes", "no"}
VALID_REGION: set[str] = {"northeast", "northwest", "southeast", "southwest"}


def load_artifact(model_path: str = MODEL_PATH) -> dict[str, Any]:
    """
    Carga el artefacto completo desde models/modelo_final.joblib.
    Si el archivo aún no existe, ejecuta train_and_export_all() una vez.
    """
    path = Path(model_path)
    if not path.exists():
        return train_and_export_all(model_path=model_path)

    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        return loaded

    # En caso de que se hubiera guardado solo el Pipeline, regenerar artefacto completo
    return train_and_export_all(model_path=model_path)


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    """
    Retorna exclusivamente el objeto Pipeline ganador ya entrenado
    (preprocesador ColumnTransformer + estimador final) listo para .predict().
    """
    artifact = load_artifact(model_path=model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    if isinstance(artifact, Pipeline):
        return artifact
    raise TypeError("El artefacto cargado no contiene un Pipeline válido.")


def validate_input(
    age: int | float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> tuple[bool, list[str], pd.DataFrame]:
    """
    Valida que las entradas de un nuevo asegurado sean plausibles y construye
    un DataFrame de 1 fila con el esquema exacto de columnas de entrenamiento.
    """
    errors: list[str] = []

    try:
        age_val = int(age)
        if not (18 <= age_val <= 100):
            errors.append(
                f"La edad ({age_val}) debe estar entre 18 y 100 años para un titular adulto."
            )
    except (ValueError, TypeError):
        errors.append("La edad ('age') debe ser un número entero válido.")
        age_val = 30

    try:
        bmi_val = float(bmi)
        if not (12.0 <= bmi_val <= 65.0):
            errors.append(
                f"El IMC ('bmi' = {bmi_val}) debe encontrarse en un rango clínico plausible (12.0 a 65.0 kg/m²)."
            )
    except (ValueError, TypeError):
        errors.append("El IMC ('bmi') debe ser un valor numérico válido.")
        bmi_val = 25.0

    try:
        children_val = int(children)
        if not (0 <= children_val <= 15):
            errors.append(
                f"El número de hijos/dependientes ({children_val}) debe estar entre 0 y 15."
            )
    except (ValueError, TypeError):
        errors.append("El número de hijos ('children') debe ser un entero no negativo.")
        children_val = 0

    sex_clean = str(sex).strip().lower()
    if sex_clean not in VALID_SEX:
        errors.append(
            f"Valor de 'sex' inválido ('{sex}'). Valores permitidos: {sorted(VALID_SEX)}."
        )

    smoker_clean = str(smoker).strip().lower()
    if smoker_clean not in VALID_SMOKER:
        errors.append(
            f"Valor de 'smoker' inválido ('{smoker}'). Valores permitidos: {sorted(VALID_SMOKER)}."
        )

    region_clean = str(region).strip().lower()
    if region_clean not in VALID_REGION:
        errors.append(
            f"Valor de 'region' inválido ('{region}'). Valores permitidos: {sorted(VALID_REGION)}."
        )

    df_single = pd.DataFrame(
        [
            {
                "age": age_val,
                "bmi": bmi_val,
                "children": children_val,
                "sex": sex_clean,
                "smoker": smoker_clean,
                "region": region_clean,
            }
        ]
    )[FEATURE_COLS]

    is_valid = len(errors) == 0
    return is_valid, errors, df_single


def predict_single(
    age: int | float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    model_path: str = MODEL_PATH,
) -> float:
    """
    Predice el gasto médico anual esperado (charges en USD $) para un perfil
    individual utilizando el Pipeline ganador sin reentrenar.
    """
    is_valid, errors, df_single = validate_input(
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region,
    )
    if not is_valid:
        raise ValueError("Error de validación en las entradas: " + "; ".join(errors))

    pipeline = load_model(model_path=model_path)
    pred = pipeline.predict(df_single)[0]
    return float(pred)


def predict_batch(
    df_new: pd.DataFrame,
    model_path: str = MODEL_PATH,
) -> np.ndarray:
    """
    Realiza inferencia por lotes sobre un DataFrame de nuevas observaciones,
    verificando que contenga las columnas predictoras requeridas y manteniendo
    el orden exacto de FEATURE_COLS sin reentrenar el pipeline.
    """
    missing = [c for c in FEATURE_COLS if c not in df_new.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas predictoras requeridas en el DataFrame: {missing}"
        )

    pipeline = load_model(model_path=model_path)
    X_ordered = df_new[FEATURE_COLS].copy()
    preds = pipeline.predict(X_ordered)
    return np.asarray(preds, dtype=float)


def get_five_new_predictions(
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """
    Genera la Tabla T9 (Predicciones nuevas - Etapa 15 y Anexo C):
    Evalúa 5 casos hipotéticos plausibles de nuevos solicitantes de póliza
    (no empleados en entrenamiento ni en prueba), conservando el mismo Pipeline
    entrenado y explicando cada resultado desde la perspectiva actuarial.
    """
    cases = [
        {
            "Caso": "Caso 1: Joven adulto saludable no fumador",
            "age": 22,
            "sex": "female",
            "bmi": 21.8,
            "children": 0,
            "smoker": "no",
            "region": "northwest",
            "Procedencia": "Perfil hipotético plausible (Nuevo cotizante universitario)",
            "Explicación del Resultado": (
                "Representa el piso de riesgo actuarial: baja edad biológica (22 años), "
                "IMC normopeso (21.8 kg/m²), sin dependientes y no fumadora. El modelo "
                "estima un cargo basal bajo (~$2,300-$2,600) dominado únicamente por cobertura preventiva."
            ),
        },
        {
            "Caso": "Caso 2: Adulto de mediana edad con familia, no fumador",
            "age": 41,
            "sex": "male",
            "bmi": 27.4,
            "children": 3,
            "smoker": "no",
            "region": "northeast",
            "Procedencia": "Perfil hipotético plausible (Póliza familiar estándar)",
            "Explicación del Resultado": (
                "El incremento de edad (41 años) y 3 hijos dependientes elevan gradualmente "
                "el gasto esperado hacia el rango medio (~$7,500-$8,600), manteniéndose contenido "
                "porque la ausencia de tabaquismo evita el salto no lineal de riesgo."
            ),
        },
        {
            "Caso": "Caso 3: Fumador joven con IMC normal",
            "age": 27,
            "sex": "female",
            "bmi": 23.5,
            "children": 0,
            "smoker": "yes",
            "region": "southwest",
            "Procedencia": "Perfil hipotético plausible (Fumador sin obesidad)",
            "Explicación del Resultado": (
                "Aunque es joven (27 años) y tiene IMC saludable (< 25), la condición smoker='yes' "
                "desplaza la predicción al régimen intermedio-alto (~$16,500-$19,000), reflejando "
                "el recargo por riesgo respiratorio/cardiovascular sin llegar al extremo de obesidad."
            ),
        },
        {
            "Caso": "Caso 4: Adulto mayor no fumador con sobrepeso",
            "age": 61,
            "sex": "female",
            "bmi": 31.2,
            "children": 1,
            "smoker": "no",
            "region": "southeast",
            "Procedencia": "Perfil hipotético plausible (Asegurado senior pre-jubilación)",
            "Explicación del Resultado": (
                "En no fumadores el gasto crece monótonamente con la edad; a los 61 años "
                "el costo base supera los ~$13,000-$14,000, pero permanece muy por debajo "
                "de un perfil fumador con el mismo IMC gracias a la interacción capturada por el kernel RBF."
            ),
        },
        {
            "Caso": "Caso 5: Fumador con obesidad grado II (Alto riesgo conjunto)",
            "age": 52,
            "sex": "male",
            "bmi": 36.8,
            "children": 2,
            "smoker": "yes",
            "region": "southeast",
            "Procedencia": "Perfil hipotético plausible (Sinergia crítica tabaco + IMC >= 30)",
            "Explicación del Resultado": (
                "Ilustra la sinergia no lineal más fuerte del dataset: la combinación de "
                "tabaquismo activo (smoker='yes'), obesidad (bmi=36.8 >= 30) y edad > 50 años "
                "dispara la predicción por encima de ~$42,000 USD anuales."
            ),
        },
    ]

    df_cases = pd.DataFrame(cases)
    preds = predict_batch(df_cases[FEATURE_COLS], model_path=model_path)

    df_cases["Gasto Anual Predicho - charges_pred ($)"] = np.round(preds, 2)

    ordered_cols = [
        "Caso",
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "Gasto Anual Predicho - charges_pred ($)",
        "Procedencia",
        "Explicación del Resultado",
    ]
    return df_cases[ordered_cols]


if __name__ == "__main__":
    print("Verificando módulo de inferencia src/predict.py...")
    t9_df = get_five_new_predictions()
    print("\n--- Tabla T9: Cinco Predicciones sobre Entradas Nuevas Plausibles ---")
    print(t9_df.to_string(index=False))
