"""
Módulo de carga, validación, calidad de datos y análisis exploratorio (EDA)
para el Grupo 09: Estimación de Gastos de Seguros Médicos (Medical Cost Personal Datasets).

Cumple con las directrices de design_doc.md y ejercicio.md (Etapas 3, 4 y 5, y Tablas T1 y T2),
garantizando que ninguna transformación de preprocesamiento se ajuste sobre el dataset completo
para prevenir fuga de información (data leakage).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constantes globales del proyecto
DEFAULT_DATA_PATH = "data/insurance.csv"
TARGET_COL = "charges"
NUMERIC_COLS = ["age", "bmi", "children"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
REQUIRED_COLS = FEATURE_COLS + [TARGET_COL]

# Configuración estética general para gráficos de Matplotlib / Seaborn
sns.set_theme(style="whitegrid", palette="deep")


def load_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Carga y valida el archivo CSV del dataset Medical Cost Personal Datasets.

    Parameters
    ----------
    filepath : str, optional
        Ruta hacia el archivo CSV (por defecto 'data/insurance.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame validado con las 7 columnas originales sin alterar.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta especificada.
    ValueError
        Si faltan columnas requeridas o el archivo está vacío.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del dataset en '{filepath}'. "
            "Verifique que 'data/insurance.csv' esté presente."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El archivo '{filepath}' está vacío.")

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"El dataset no contiene las columnas obligatorias: {missing_cols}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    return df


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Retorna la ficha técnica y metadatos del dataset para la Etapa 3 del proyecto.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cargado desde insurance.csv.

    Returns
    -------
    Dict[str, Any]
        Diccionario con información de procedencia, dimensiones, duplicados y tipos.
    """
    dup_mask = df.duplicated(keep=False)
    dup_rows = df[dup_mask].to_dict(orient="records") if dup_mask.any() else []
    dup_indices = df[dup_mask].index.tolist() if dup_mask.any() else []

    return {
        "dataset_name": "Medical Cost Personal Datasets (insurance.csv)",
        "author": "Miri Choi (basado en el libro 'Machine Learning with R' de Brett Lantz)",
        "kaggle_url": "https://www.kaggle.com/datasets/mirichoi0218/insurance",
        "license": "Database Contents License (DbCL) v1.0 / Dominio académico público",
        "file_name": "insurance.csv",
        "version": "Versión 1 (Original Kaggle release, 1,338 registros)",
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_nulls_total": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "duplicate_indices": dup_indices,
        "duplicate_rows": dup_rows,
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024.0, 2),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "target_col": TARGET_COL,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }


def get_data_dictionary(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Construye la Tabla T1: Diccionario de variables exigido en el Anexo C y Etapa 3.

    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame para extraer rangos o categorías reales dinámicamente si se provee.

    Returns
    -------
    pd.DataFrame
        Tabla T1 con definición, tipo, rol, unidad/categorías, disponibilidad ex-ante,
        decisión (Incluir/Excluir) y análisis de riesgo de fuga de información.
    """
    records = [
        {
            "Variable": "age",
            "Definición": "Edad del beneficiario titular del seguro médico",
            "Tipo": "Numérica discreta (int64)",
            "Rol": "Predictora (X)",
            "Unidad / Categorías": "Años cumplidos (18 a 64 años)",
            "Disponible antes de predicción": "Sí (al cotizar póliza)",
            "Decisión": "Incluir",
            "Justificación / Riesgo de Fuga": (
                "Sin riesgo de fuga. Dato demográfico básico declarado en la solicitud de afiliación; "
                "captura el incremento basal del gasto médico con el envejecimiento."
            ),
        },
        {
            "Variable": "sex",
            "Definición": "Sexo biológico registrado del contratante",
            "Tipo": "Categórica nominal binaria (str)",
            "Rol": "Predictora (X)",
            "Unidad / Categorías": "Categorías: 'female', 'male'",
            "Disponible antes de predicción": "Sí (al cotizar póliza)",
            "Decisión": "Incluir",
            "Justificación / Riesgo de Fuga": (
                "Sin riesgo de fuga. Variable demográfica estándar conocida previo al inicio de vigencia "
                "de la cobertura."
            ),
        },
        {
            "Variable": "bmi",
            "Definición": "Índice de Masa Corporal (peso en kg dividido por estatura en m²)",
            "Tipo": "Numérica continua (float64)",
            "Rol": "Predictora (X)",
            "Unidad / Categorías": "kg/m² (rango observado: 15.96 a 53.13)",
            "Disponible antes de predicción": "Sí (examen médico de ingreso)",
            "Decisión": "Incluir",
            "Justificación / Riesgo de Fuga": (
                "Sin riesgo de fuga. Se mide en el chequeo de suscripción inicial; indicador clínico "
                "clave de riesgo metabólico y obesidad (BMI >= 30)."
            ),
        },
        {
            "Variable": "children",
            "Definición": "Número de hijos o dependientes cubiertos por el seguro médico",
            "Tipo": "Numérica discreta (int64)",
            "Rol": "Predictora (X)",
            "Unidad / Categorías": "Conteo entero (0 a 5 dependientes)",
            "Disponible antes de predicción": "Sí (declaración familiar)",
            "Decisión": "Incluir",
            "Justificación / Riesgo de Fuga": (
                "Sin riesgo de fuga. El titular registra sus dependientes al contratar la póliza."
            ),
        },
        {
            "Variable": "smoker",
            "Definición": "Condición de consumo habitual de tabaco por parte del asegurado",
            "Tipo": "Categórica nominal binaria (str)",
            "Rol": "Predictora (X)",
            "Unidad / Categorías": "Categorías: 'yes' (fumador), 'no' (no fumador)",
            "Disponible antes de predicción": "Sí (declaración jurada / cotinina)",
            "Decisión": "Incluir",
            "Justificación / Riesgo de Fuga": (
                "Sin riesgo de fuga. Declarado en la suscripción del seguro; factor de mayor asociación "
                "predictiva con gastos médicos severos."
            ),
        },
        {
            "Variable": "region",
            "Definición": "Zona geográfica de residencia del beneficiario en EE. UU.",
            "Tipo": "Categórica nominal (str)",
            "Rol": "Predictora (X)",
            "Unidad / Categorías": "Categorías: 'northeast', 'northwest', 'southeast', 'southwest'",
            "Disponible antes de predicción": "Sí (domicilio del titular)",
            "Decisión": "Incluir",
            "Justificación / Riesgo de Fuga": (
                "Sin riesgo de fuga. Conocida al momento del contrato; controla diferencias de costos "
                "hospitalarios y demográficos regionales."
            ),
        },
        {
            "Variable": "charges",
            "Definición": "Costos médicos individuales facturados por el seguro en el año",
            "Tipo": "Numérica continua (float64)",
            "Rol": "Objetivo (y)",
            "Unidad / Categorías": "Dólares estadounidenses (USD, $1,121.87 a $63,770.43)",
            "Disponible antes de predicción": "No (resultado ex-post al cierre anual)",
            "Decisión": "Variable Objetivo (Excluir de X)",
            "Justificación / Riesgo de Fuga": (
                "Es la variable a predecir (y). Incluirla o usar derivados de ella en X causaría fuga "
                "directa de información (target leakage)."
            ),
        },
    ]
    return pd.DataFrame(records)


def get_quality_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la Tabla T2: Diagnóstico de calidad de datos exigida en el Anexo C y Etapa 5.

    Evalúa nulos, porcentajes, duplicados, valores extremos por criterio IQR (Q3 + 1.5*IQR
    y Q1 - 1.5*IQR) y documenta la decisión metodológica adoptada para cada columna.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cargado desde insurance.csv.

    Returns
    -------
    pd.DataFrame
        Tabla T2 con el diagnóstico de calidad por variable.
    """
    n_total = len(df)
    n_dups = int(df.duplicated().sum())
    dup_text = f"{n_dups} fila exacta (índices 195 y 581; {n_dups / n_total * 100:.2f}%)"

    rows = []
    for col in REQUIRED_COLS:
        n_null = int(df[col].isna().sum())
        pct_null = round((n_null / n_total) * 100.0, 2)
        dtype_str = str(df[col].dtype)

        if col in NUMERIC_COLS + [TARGET_COL]:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            low_fence = q1 - 1.5 * iqr
            high_fence = q3 + 1.5 * iqr
            outliers_mask = (df[col] < low_fence) | (df[col] > high_fence)
            n_outliers = int(outliers_mask.sum())
            pct_outliers = (n_outliers / n_total) * 100.0

            if n_outliers > 0:
                anomaly_str = (
                    f"{n_outliers} extremos ({pct_outliers:.2f}%) > {high_fence:,.2f} "
                    f"(rango [{df[col].min():,.2f}, {df[col].max():,.2f}])"
                )
            else:
                anomaly_str = (
                    f"0 atípicos IQR (rango válido [{df[col].min():,.0f}, {df[col].max():,.0f}])"
                )
        else:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            anomaly_str = f"0 categorías inválidas (valores: {', '.join(map(str, unique_vals))})"

        if col == "charges":
            action = (
                "Conservar íntegros. Los 139 valores altos (> $34,489 USD) corresponden a casos reales "
                "de alta siniestralidad (fumadores con obesidad) y no a errores de captura; se evalúa "
                "con MAE por su robustez ante asimetría derecha."
            )
        elif col == "bmi":
            action = (
                "Conservar los 9 valores extremos (BMI > 47.29 kg/m², obesidad mórbida real); aplicar "
                "SimpleImputer(median) y StandardScaler dentro del Pipeline (ajustado solo en train)."
            )
        elif col in NUMERIC_COLS:
            action = (
                "Conservar todos los registros (incluyendo la fila duplicada por ser perfil demográfico "
                "plausible); aplicar SimpleImputer(median) y StandardScaler dentro del Pipeline."
            )
        else:
            action = (
                "Conservar categorías; aplicar SimpleImputer(most_frequent) y "
                "OneHotEncoder(handle_unknown='ignore') exclusivamente dentro del Pipeline."
            )

        rows.append(
            {
                "Variable": col,
                "Tipo": dtype_str,
                "Nulos (Cant.)": n_null,
                "Nulos (%)": f"{pct_null:.2f}%",
                "Duplicados (Dataset)": dup_text,
                "Valores Anómalos / Extremos (IQR)": anomaly_str,
                "Medida Adoptada": action,
            }
        )

    return pd.DataFrame(rows)


def get_feature_selection_table() -> pd.DataFrame:
    """
    Retorna la tabla de justificación de variables incluidas y excluidas para la Etapa 4.

    Returns
    -------
    pd.DataFrame
        Tabla detallada de selección de predictores X y separación de y sin fuga de datos.
    """
    data = [
        {
            "Columna": "age",
            "Asignación": "Predictora (X)",
            "Estado": "Incluida",
            "Motivo Técnico y de Negocio": (
                "Captura la tendencia creciente casi lineal del costo médico base según la edad del asegurado."
            ),
            "Verificación Anti-Leakage": "Conocida en el instante t=0 (afiliación).",
        },
        {
            "Columna": "sex",
            "Asignación": "Predictora (X)",
            "Estado": "Incluida",
            "Motivo Técnico y de Negocio": (
                "Permite evaluar posibles diferencias demográficas o interacciones de tarifa base."
            ),
            "Verificación Anti-Leakage": "Conocida en el instante t=0 (afiliación).",
        },
        {
            "Columna": "bmi",
            "Asignación": "Predictora (X)",
            "Estado": "Incluida",
            "Motivo Técnico y de Negocio": (
                "Predictor clínico indispensable; a partir de BMI >= 30 dispara el gasto en interacción con tabaquismo."
            ),
            "Verificación Anti-Leakage": "Medida en evaluación médica previa a la póliza.",
        },
        {
            "Columna": "children",
            "Asignación": "Predictora (X)",
            "Estado": "Incluida",
            "Motivo Técnico y de Negocio": (
                "Refleja la carga familiar cubierta por el plan y el incremento moderado en consultas preventivas."
            ),
            "Verificación Anti-Leakage": "Declarada en el contrato inicial.",
        },
        {
            "Columna": "smoker",
            "Asignación": "Predictora (X)",
            "Estado": "Incluida",
            "Motivo Técnico y de Negocio": (
                "Variable de mayor poder discriminante sobre charges; separa los regímenes de gasto moderado y severo."
            ),
            "Verificación Anti-Leakage": "Auditada en la suscripción de la póliza.",
        },
        {
            "Columna": "region",
            "Asignación": "Predictora (X)",
            "Estado": "Incluida",
            "Motivo Técnico y de Negocio": (
                "Controla variaciones geográficas de precios hospitalarios y prevalencia de factores de riesgo."
            ),
            "Verificación Anti-Leakage": "Dirección registrada al contratar.",
        },
        {
            "Columna": "charges",
            "Asignación": "Objetivo (y)",
            "Estado": "Excluida de X (Target)",
            "Motivo Técnico y de Negocio": (
                "Es la magnitud monetaria anual que deseamos estimar; se separa estrictamente en el vector y."
            ),
            "Verificación Anti-Leakage": "Separada antes del split y del preprocesamiento.",
        },
    ]
    return pd.DataFrame(data)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa el DataFrame en la matriz de características predictoras (X) y el vector objetivo (y).

    Verifica que los índices coincidan exactamente y que 'charges' no esté presente en X.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original validado.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Tupla (X, y) con copias independientes para evitar mutaciones accidentales.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    if TARGET_COL in X.columns:
        raise ValueError(f"Fuga detectada: '{TARGET_COL}' no puede estar en X.")
    if not X.index.equals(y.index):
        raise ValueError("Los índices de X e y no coinciden.")

    return X, y


def analyze_charges_asymmetry(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza el análisis especializado de asimetría, curtosis y valores extremos de 'charges'
    y 'bmi', así como el desglose estadístico por hábito tabáquico exigido para el Grupo 09.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset insurance.csv.

    Returns
    -------
    Dict[str, Any]
        Diccionario con métricas de distribución, umbrales IQR, conteo de atípicos y
        comparativa descriptiva entre fumadores y no fumadores.
    """
    charges = df[TARGET_COL]
    q1 = float(charges.quantile(0.25))
    median = float(charges.median())
    q3 = float(charges.quantile(0.75))
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = max(0.0, q1 - 1.5 * iqr)

    outliers_charges = df[charges > upper_fence]
    n_outliers = int(len(outliers_charges))
    pct_outliers = round((n_outliers / len(df)) * 100.0, 2)

    # Análisis de BMI extremos
    bmi = df["bmi"]
    bmi_q1 = float(bmi.quantile(0.25))
    bmi_q3 = float(bmi.quantile(0.75))
    bmi_iqr = bmi_q3 - bmi_q1
    bmi_upper_fence = bmi_q3 + 1.5 * bmi_iqr
    n_bmi_outliers = int((bmi > bmi_upper_fence).sum())

    # Desglose por fumador
    smokers = df[df["smoker"] == "yes"][TARGET_COL]
    non_smokers = df[df["smoker"] == "no"][TARGET_COL]

    # Proporción de fumadores dentro de los outliers de charges
    smokers_in_outliers_pct = round(
        (outliers_charges["smoker"] == "yes").mean() * 100.0, 2
    )

    return {
        "mean": round(float(charges.mean()), 2),
        "median": round(median, 2),
        "std": round(float(charges.std()), 2),
        "min": round(float(charges.min()), 2),
        "max": round(float(charges.max()), 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_fence": round(lower_fence, 2),
        "upper_fence": round(upper_fence, 2),
        "skewness": round(float(charges.skew()), 4),
        "kurtosis": round(float(charges.kurt()), 4),
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "smokers_in_outliers_pct": smokers_in_outliers_pct,
        "bmi_upper_fence": round(bmi_upper_fence, 2),
        "n_bmi_outliers": n_bmi_outliers,
        "smoker_stats": {
            "yes": {
                "count": int(len(smokers)),
                "pct_total": round(len(smokers) / len(df) * 100.0, 2),
                "mean": round(float(smokers.mean()), 2),
                "median": round(float(smokers.median()), 2),
                "std": round(float(smokers.std()), 2),
                "min": round(float(smokers.min()), 2),
                "max": round(float(smokers.max()), 2),
            },
            "no": {
                "count": int(len(non_smokers)),
                "pct_total": round(len(non_smokers) / len(df) * 100.0, 2),
                "mean": round(float(non_smokers.mean()), 2),
                "median": round(float(non_smokers.median()), 2),
                "std": round(float(non_smokers.std()), 2),
                "min": round(float(non_smokers.min()), 2),
                "max": round(float(non_smokers.max()), 2),
            },
        },
    }


def plot_charges_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Gráfico 1 de EDA: Distribución de la variable objetivo `charges` (Histograma + KDE),
    mostrando la asimetría positiva (cola derecha), media, mediana, umbral de atípicos IQR,
    y un segundo panel comparando la densidad por condición de fumador (`smoker`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Returns
    -------
    plt.Figure
        Figura de Matplotlib lista para renderizarse en Streamlit o guardarse.
    """
    stats = analyze_charges_asymmetry(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel izquierdo: Distribución global con estadísticos clave
    ax1 = axes[0]
    sns.histplot(
        df[TARGET_COL],
        bins=35,
        kde=True,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.75,
        ax=ax1,
    )
    ax1.axvline(
        stats["mean"],
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"Media: ${stats['mean']:,.0f}",
    )
    ax1.axvline(
        stats["median"],
        color="#1b9e77",
        linestyle="-",
        linewidth=2,
        label=f"Mediana: ${stats['median']:,.0f}",
    )
    ax1.axvline(
        stats["upper_fence"],
        color="#e7298a",
        linestyle=":",
        linewidth=2.2,
        label=f"Umbral IQR (Q3+1.5·IQR): ${stats['upper_fence']:,.0f}",
    )
    ax1.set_title(
        f"Distribución Global de 'charges' (Skewness = {stats['skewness']:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Gastos Médicos Anuales - charges (USD)", fontsize=10)
    ax1.set_ylabel("Frecuencia de Asegurados", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)

    # Panel derecho: Distribución condicionada por smoker
    ax2 = axes[1]
    palette_smoker = {"no": "#1b9e77", "yes": "#d95f02"}
    sns.histplot(
        data=df,
        x=TARGET_COL,
        hue="smoker",
        bins=30,
        kde=True,
        palette=palette_smoker,
        edgecolor="white",
        alpha=0.6,
        ax=ax2,
    )
    ax2.set_title(
        "Separación Bimodal en Fumadores vs. Régimen Basal en No Fumadores",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Gastos Médicos Anuales - charges (USD)", fontsize=10)
    ax2.set_ylabel("Frecuencia de Asegurados", fontsize=10)
    ax2.legend(
        title="¿Fumador?",
        labels=["Sí (yes)", "No (no)"],
        loc="upper right",
        frameon=True,
    )

    fig.tight_layout()
    return fig


def plot_age_vs_charges(df: pd.DataFrame) -> plt.Figure:
    """
    Gráfico 2 de EDA: Relación entre Edad (`age`) y Gastos Médicos (`charges`)
    diferenciada por condición de tabaquismo (`smoker`) y obesidad (`bmi >= 30`).
    Revela las 3 bandas paralelas características del dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Returns
    -------
    plt.Figure
        Figura de Matplotlib.
    """
    plot_df = df.copy()
    plot_df["Perfil Clínico"] = np.where(
        plot_df["smoker"] == "no",
        "No Fumador (Régimen Basal)",
        np.where(
            plot_df["bmi"] >= 30.0,
            "Fumador con Obesidad (BMI >= 30)",
            "Fumador sin Obesidad (BMI < 30)",
        ),
    )

    palette = {
        "No Fumador (Régimen Basal)": "#1b9e77",
        "Fumador sin Obesidad (BMI < 30)": "#fdae61",
        "Fumador con Obesidad (BMI >= 30)": "#d7191c",
    }

    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.scatterplot(
        data=plot_df,
        x="age",
        y=TARGET_COL,
        hue="Perfil Clínico",
        style="Perfil Clínico",
        palette=palette,
        alpha=0.82,
        s=65,
        ax=ax,
    )

    ax.set_title(
        "Edad vs. Gastos Médicos ('charges'): Evidencia de Tres Regímenes Paralelos",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("Edad del Beneficiario (años)", fontsize=11)
    ax.set_ylabel("Gastos Médicos Anuales - charges (USD)", fontsize=11)
    ax.legend(title="Estrato de Riesgo Observado", loc="upper left", frameon=True)

    fig.tight_layout()
    return fig


def plot_bmi_vs_charges(df: pd.DataFrame) -> plt.Figure:
    """
    Gráfico 3 de EDA: Índice de Masa Corporal (`bmi`) vs Gastos Médicos (`charges`)
    coloreado por `smoker`, destacando el umbral clínico de obesidad de la OMS (BMI = 30 kg/m²)
    donde se produce el salto no lineal en asegurados fumadores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Returns
    -------
    plt.Figure
        Figura de Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(11, 5.8))
    palette_smoker = {"no": "#2b5c8f", "yes": "#d95f02"}

    sns.scatterplot(
        data=df,
        x="bmi",
        y=TARGET_COL,
        hue="smoker",
        style="smoker",
        palette=palette_smoker,
        alpha=0.78,
        s=65,
        ax=ax,
    )

    # Umbral de obesidad OMS (BMI = 30)
    ax.axvline(
        30.0,
        color="#e7298a",
        linestyle="--",
        linewidth=2.2,
        label="Corte Clínico Obesidad OMS (BMI = 30 kg/m²)",
    )

    ax.set_title(
        "Índice de Masa Corporal (BMI) vs. Gastos Médicos por Condición de Fumador",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("Índice de Masa Corporal - BMI (kg/m²)", fontsize=11)
    ax.set_ylabel("Gastos Médicos Anuales - charges (USD)", fontsize=11)
    ax.legend(title="Fumador (smoker) / Referencia", loc="upper left", frameon=True)

    fig.tight_layout()
    return fig


def plot_smoker_boxplot(df: pd.DataFrame) -> plt.Figure:
    """
    Gráfico 4 de EDA: Boxplot y distribución comparativa de `charges` según condición
    de tabaquismo (`smoker`) y sexo (`sex`), evidenciando la dispersión y los valores extremos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Returns
    -------
    plt.Figure
        Figura de Matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel 1: Boxplot + Stripplot por smoker
    ax1 = axes[0]
    sns.boxplot(
        data=df,
        x="smoker",
        y=TARGET_COL,
        hue="smoker",
        palette={"no": "#66c2a5", "yes": "#fc8d62"},
        width=0.45,
        legend=False,
        ax=ax1,
    )
    sns.stripplot(
        data=df,
        x="smoker",
        y=TARGET_COL,
        color="black",
        alpha=0.22,
        size=4,
        jitter=0.18,
        ax=ax1,
    )
    ax1.set_title(
        "Distribución y Atípicos de 'charges' según Tabaquismo",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Condición de Fumador (smoker)", fontsize=10.5)
    ax1.set_ylabel("Gastos Médicos Anuales - charges (USD)", fontsize=10.5)

    # Panel 2: Boxplot agrupado por smoker y sex
    ax2 = axes[1]
    sns.boxplot(
        data=df,
        x="smoker",
        y=TARGET_COL,
        hue="sex",
        palette="Set2",
        width=0.55,
        ax=ax2,
    )
    ax2.set_title(
        "Interacción entre Tabaquismo ('smoker') y Sexo ('sex')",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Condición de Fumador (smoker)", fontsize=10.5)
    ax2.set_ylabel("Gastos Médicos Anuales - charges (USD)", fontsize=10.5)
    ax2.legend(title="Sexo (sex)", loc="upper left", frameon=True)

    fig.tight_layout()
    return fig


def plot_multivariate_and_correlation(df: pd.DataFrame) -> plt.Figure:
    """
    Gráfico 5 de EDA: Panel doble con (1) Mapa de calor de correlaciones de Pearson
    (incluyendo variables numéricas y codificación binaria exploratoria de `smoker` y `sex`)
    y (2) Gasto médico promedio e intervalo de confianza por región (`region`) y número de hijos (`children`).

    Nota metodológica: La codificación binaria usada aquí es exclusivamente descriptiva sobre
    una copia temporal para visualización; el preprocesamiento de modelado se realiza dentro de Pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Returns
    -------
    plt.Figure
        Figura de Matplotlib.
    """
    corr_df = pd.DataFrame(
        {
            "charges": df["charges"],
            "smoker_yes": (df["smoker"] == "yes").astype(int),
            "age": df["age"],
            "bmi": df["bmi"],
            "children": df["children"],
            "sex_male": (df["sex"] == "male").astype(int),
        }
    )
    corr_matrix = corr_df.corr(method="pearson")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    # Panel 1: Matriz de correlación
    ax1 = axes[0]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.8,
        cbar_kws={"shrink": 0.82},
        ax=ax1,
    )
    ax1.set_title(
        "Matriz de Correlación Lineal de Pearson (Exploratoria)",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 2: Gastos por Región y Número de Hijos
    ax2 = axes[1]
    sns.barplot(
        data=df,
        x="region",
        y=TARGET_COL,
        hue="smoker",
        palette={"no": "#2b5c8f", "yes": "#d95f02"},
        errorbar=("ci", 95),
        capsize=0.08,
        ax=ax2,
    )
    ax2.set_title(
        "Gasto Médico Promedio (IC 95%) por Región y Tabaquismo",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Región Geográfica (region)", fontsize=10.5)
    ax2.set_ylabel("Gasto Promedio - charges (USD)", fontsize=10.5)
    ax2.legend(title="¿Fumador?", loc="upper right", frameon=True)

    fig.tight_layout()
    return fig
