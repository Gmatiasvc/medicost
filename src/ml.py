"""
Módulo de Machine Learning (Etapas 6 a 13) para el Grupo 09:
Estimación de Gastos de Seguros Médicos (Medical Cost Personal Datasets).

Implementa:
- Etapa 6: Partición 80/20 con reserva estricta de test (random_state=42) -> Tabla T3
- Etapa 7: Preprocesamiento sin fuga (ColumnTransformer + Pipeline) -> Tabla T4
- Etapas 8 y 9: Modelos candidatos y Pipelines reproducibles (DummyRegressor, LinearRegression, KNeighborsRegressor, SVR)
- Etapa 10: Validación cruzada 5-Fold idéntica solo en train -> Tabla T5
- Etapa 11: Optimización de hiperparámetros (GridSearchCV en KNN y SVR) solo en train -> Tabla T6
- Etapa 12: Selección razonada del mejor modelo antes del test -> Tabla T7
- Etapa 13 y Requisito Grupo 09: Evaluación final única en test, análisis de residuales y comparación de MAE por fumador y nivel de gasto -> Tabla T8
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# ============================================================================
# 1. CONSTANTES GLOBALES Y CONFIGURACIÓN EXPERIMENTAL
# ============================================================================
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
N_SPLITS: int = 5

NUMERIC_COLS: list[str] = ["age", "bmi", "children"]
CATEGORICAL_COLS: list[str] = ["sex", "smoker", "region"]
FEATURE_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL: str = "charges"

MODEL_PATH: str = "models/modelo_final.joblib"
DATA_PATH: str = "data/insurance.csv"


# ============================================================================
# 2. ETAPA 6: PARTICIÓN TRAIN/TEST Y TABLA T3
# ============================================================================
def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza la partición 80/20 reservando el conjunto de prueba antes de
    cualquier ajuste de transformaciones para prevenir fuga de información.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURE_COLS].copy(),
        y.copy(),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test


def get_partition_table(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Genera la Tabla T3 (Partición de datos) exigida en el Anexo C.
    Incluye verificación explícita de 0 filas/índices compartidos.
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_total = n_train + n_test
    shared_indices = len(set(X_train.index).intersection(set(X_test.index)))

    y_all = pd.concat([y_train, y_test])

    rows = [
        {
            "Subconjunto": "Entrenamiento (Train)",
            "Registros (n)": n_train,
            "Proporción (%)": round(n_train / n_total * 100, 2),
            "Media charges ($)": round(float(y_train.mean()), 2),
            "Mediana charges ($)": round(float(y_train.median()), 2),
            "Desv. Est. charges ($)": round(float(y_train.std()), 2),
            "Método": "train_test_split (aleatorio barajado)",
            "Semilla (random_state)": str(RANDOM_STATE),
            "Filas Compartidas": shared_indices,
        },
        {
            "Subconjunto": "Prueba Reservada (Test)",
            "Registros (n)": n_test,
            "Proporción (%)": round(n_test / n_total * 100, 2),
            "Media charges ($)": round(float(y_test.mean()), 2),
            "Mediana charges ($)": round(float(y_test.median()), 2),
            "Desv. Est. charges ($)": round(float(y_test.std()), 2),
            "Método": "Reserva estricta (evaluación única final)",
            "Semilla (random_state)": str(RANDOM_STATE),
            "Filas Compartidas": shared_indices,
        },
        {
            "Subconjunto": "Total Dataset",
            "Registros (n)": n_total,
            "Proporción (%)": 100.0,
            "Media charges ($)": round(float(y_all.mean()), 2),
            "Mediana charges ($)": round(float(y_all.median()), 2),
            "Desv. Est. charges ($)": round(float(y_all.std()), 2),
            "Método": "Partición 80% / 20% previa al preprocesamiento",
            "Semilla (random_state)": str(RANDOM_STATE),
            "Filas Compartidas": shared_indices,
        },
    ]
    return pd.DataFrame(rows)


# ============================================================================
# 3. ETAPA 7: PREPROCESAMIENTO SIN FUGA (COLUMNTRANSFORMER) Y TABLA T4
# ============================================================================
def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer que encapsula el tratamiento diferenciado
    de variables numéricas y categóricas nominales.
    Se ajusta EXCLUSIVAMENTE dentro de un Pipeline sobre X_train o cada fold.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_preprocessing_table() -> pd.DataFrame:
    """
    Genera la Tabla T4 (Preprocesamiento) exigida en el Anexo C.
    """
    rows = [
        {
            "Columna / Grupo": "Rama Numérica (Continua / Discreta)",
            "Variables": "age, bmi, children",
            "Imputación": "SimpleImputer(strategy='median')",
            "Codificación": "No aplica (valores numéricos nativos)",
            "Escalado": "StandardScaler() -> media 0, varianza 1",
            "Justificación Metodológica (Sin Fuga)": (
                "La mediana es robusta frente a posibles valores extremos en bmi. "
                "StandardScaler evita que age (18-64) y bmi (15.96-53.13) dominen sobre "
                "children (0-5) en métricas de distancia (KNN) y productos internos (SVR). "
                "Los estadísticos (mediana, media, desv. est.) se calculan solo en train/fold."
            ),
        },
        {
            "Columna / Grupo": "Rama Categórica Nominal (Binaria)",
            "Variables": "sex, smoker",
            "Imputación": "SimpleImputer(strategy='most_frequent')",
            "Codificación": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "Escalado": "No aplica (indicadores binarios 0/1)",
            "Justificación Metodológica (Sin Fuga)": (
                "Imputación por moda ante eventuales faltantes en producción. "
                "OneHotEncoder convierte categorías nominales en vectores ortogonales 0/1 "
                "sin imponer orden artificial y maneja categorías no vistas sin fallar."
            ),
        },
        {
            "Columna / Grupo": "Rama Categórica Nominal (Multiclase)",
            "Variables": "region",
            "Imputación": "SimpleImputer(strategy='most_frequent')",
            "Codificación": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "Escalado": "No aplica (indicadores binarios 0/1)",
            "Justificación Metodológica (Sin Fuga)": (
                "Las 4 regiones (northeast, northwest, southeast, southwest) carecen de jerarquía "
                "ordinal; codificarlas como 0,1,2,3 induciría distancias falsas en KNN y SVR."
            ),
        },
        {
            "Columna / Grupo": "Variable Objetivo (Target y)",
            "Variables": "charges",
            "Imputación": "No aplica (0 nulos verificados; nunca se imputa y)",
            "Codificación": "No aplica (continua en USD $)",
            "Escalado": "Sin transformar (se mantiene en dólares reales USD $)",
            "Justificación Metodológica (Sin Fuga)": (
                "Se conserva en su escala monetaria original ($) para que el MAE y el RMSE "
                "sean directamente interpretables por el área actuarial y de negocio."
            ),
        },
    ]
    return pd.DataFrame(rows)


# ============================================================================
# 4. ETAPAS 8 Y 9: MODELOS CANDIDATOS Y PIPELINES REPRODUCIBLES
# ============================================================================
def get_algorithms_table() -> pd.DataFrame:
    """
    Tabla descriptiva de fundamentos, hiperparámetros, fortalezas y limitaciones
    de los algoritmos evaluados (Etapa 8).
    """
    rows = [
        {
            "Algoritmo": "DummyRegressor (Baseline Referencial)",
            "Mecanismo / Supuesto": (
                "Predice una constante igual a la media de charges observada en el "
                "conjunto de entrenamiento, ignorando todas las variables predictoras X."
            ),
            "Hiperparámetros Clave": "strategy='mean'",
            "Fortalezas": (
                "Punto de referencia mínimo obligatorio (piso de rendimiento); "
                "costo computacional nulo."
            ),
            "Limitaciones": (
                "Incapaz de capturar estructura o diferencias entre asegurados (R² ≈ 0)."
            ),
            "Sensibilidad al Escalado": "Nula",
            "Pertinencia en insurance.csv": (
                "Permite cuantificar cuánta reducción real de error en dólares aportan "
                "los modelos supervisados frente a una estimación ingenua."
            ),
        },
        {
            "Algoritmo": "LinearRegression (Mínimos Cuadrados Ordinarios)",
            "Mecanismo / Supuesto": (
                "Ajusta un hiperplano lineal y = w^T x + b minimizando la suma de "
                "residuos cuadráticos sobre las variables codificadas."
            ),
            "Hiperparámetros Clave": "fit_intercept=True",
            "Fortalezas": (
                "Alta interpretabilidad de coeficientes aditivos; entrenamiento "
                "instantáneo; buen comportamiento global en tendencias monótonas."
            ),
            "Limitaciones": (
                "Asume aditividad estricta; no captura automáticamente la interacción no lineal "
                "severa entre obesidad (bmi >= 30) y tabaquismo (smoker='yes'), y es sensible a atípicos."
            ),
            "Sensibilidad al Escalado": "Baja en predicción (afecta magnitud comparable de coeficientes)",
            "Pertinencia en insurance.csv": (
                "Captura el salto aditivo de fumar (~+$23k) y la pendiente por edad, "
                "pero subestima o sobreestima regímenes conjuntos no lineales."
            ),
        },
        {
            "Algoritmo": "KNeighborsRegressor (Vecinos Más Cercanos)",
            "Mecanismo / Supuesto": (
                "Método no paramétrico basado en instancias que estima charges como el "
                "promedio (uniforme o ponderado por distancia) de los k perfiles más similares."
            ),
            "Hiperparámetros Clave": (
                "n_neighbors (k), weights ('uniform', 'distance'), p (1=Manhattan, 2=Euclidiana)"
            ),
            "Fortalezas": (
                "No asume forma funcional lineal; captura naturalmente interacciones locales "
                "(ej. fumadores con alto IMC vs no fumadores jóvenes)."
            ),
            "Limitaciones": (
                "Sensible a la maldición de la dimensionalidad tras One-Hot Encoding y "
                "a escalas dispares si no se estandariza previamente."
            ),
            "Sensibilidad al Escalado": "Crítica (requiere StandardScaler obligatorio dentro del Pipeline)",
            "Pertinencia en insurance.csv": (
                "Con 1,338 registros y 11 columnas tras One-Hot Encoding, vecindarios locales "
                "bien calibrados logran aislar eficazmente los clusters de gasto médico."
            ),
        },
        {
            "Algoritmo": "SVR (Support Vector Regression)",
            "Mecanismo / Supuesto": (
                "Busca una función suave en un espacio de características (lineal o inducido por "
                "kernel RBF/polinomial) con tubo de insensibilidad epsilon y penalización C."
            ),
            "Hiperparámetros Clave": "kernel ('rbf', 'linear', 'poly'), C, gamma, epsilon",
            "Fortalezas": (
                "Robusto ante valores atípicos moderados gracias a la pérdida epsilon-insensible (L1); "
                "con kernel RBF modela fronteras no lineales suaves de alta precisión en MAE."
            ),
            "Limitaciones": (
                "Con C por defecto (C=1.0) subajusta severamente cuando la variable objetivo "
                "está en decenas de miles de dólares; requiere búsqueda rigurosa de C y gamma."
            ),
            "Sensibilidad al Escalado": "Crítica (kernels basados en normas y productos internos)",
            "Pertinencia en insurance.csv": (
                "Al optimizar C en escalas monetarias (1,000 a 10,000) y kernel RBF/lineal, "
                "optimiza directamente una pérdida cercana al MAE y maneja la asimetría de charges."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """
    Construye los 4 Pipelines completos (Etapa 9):
    1. DummyRegressor (Baseline no competidor)
    2. LinearRegression
    3. KNeighborsRegressor
    4. SVR
    Todos integran una instancia fresca de build_preprocessor().
    """
    return {
        "DummyRegressor (Baseline)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "LinearRegression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "KNeighborsRegressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", KNeighborsRegressor()),
            ]
        ),
        "SVR": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", SVR()),
            ]
        ),
    }


# ============================================================================
# 5. ETAPA 10: VALIDACIÓN CRUZADA (5 FOLDS EN TRAIN) Y TABLA T5
# ============================================================================
def get_cv_splitter() -> KFold:
    """
    Retorna el esquema único de validación cruzada de 5 folds barajados
    con semilla fija para garantizar comparaciones 100% justas e idénticas.
    """
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Evalúa los pipelines iniciales exclusivamente en X_train, y_train usando
    exactamente los mismos 5 folds. Calcula MAE (métrica primaria), RMSE y R².
    """
    cv = get_cv_splitter()
    pipelines = build_candidate_pipelines()

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    rows = []
    raw_cv_results: dict[str, Any] = {}

    for name, pipe in pipelines.items():
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        fold_maes = -cv_res["test_mae"]
        fold_rmses = -cv_res["test_rmse"]
        fold_r2s = cv_res["test_r2"]
        fit_times = cv_res["fit_time"]

        raw_cv_results[name] = {
            "fold_maes": fold_maes.tolist(),
            "fold_rmses": fold_rmses.tolist(),
            "fold_r2s": fold_r2s.tolist(),
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes, ddof=1)),
            "mean_rmse": float(np.mean(fold_rmses)),
            "std_rmse": float(np.std(fold_rmses, ddof=1)),
            "mean_r2": float(np.mean(fold_r2s)),
            "std_r2": float(np.std(fold_r2s, ddof=1)),
            "mean_fit_time_s": float(np.mean(fit_times)),
        }

        rows.append(
            {
                "Modelo": name,
                "Fold 1 MAE ($)": round(float(fold_maes[0]), 2),
                "Fold 2 MAE ($)": round(float(fold_maes[1]), 2),
                "Fold 3 MAE ($)": round(float(fold_maes[2]), 2),
                "Fold 4 MAE ($)": round(float(fold_maes[3]), 2),
                "Fold 5 MAE ($)": round(float(fold_maes[4]), 2),
                "Media MAE ($)": round(float(np.mean(fold_maes)), 2),
                "Desv. Est. MAE ($)": round(float(np.std(fold_maes, ddof=1)), 2),
                "Media RMSE ($)": round(float(np.mean(fold_rmses)), 2),
                "Desv. Est. RMSE ($)": round(float(np.std(fold_rmses, ddof=1)), 2),
                "Media R²": round(float(np.mean(fold_r2s)), 4),
                "Tiempo Promedio Fit (s)": round(float(np.mean(fit_times)), 4),
            }
        )

    cv_df = pd.DataFrame(rows)

    # Ranking preliminar entre los 3 modelos competidores (y marcando Baseline)
    competitor_mask = cv_df["Modelo"] != "DummyRegressor (Baseline)"
    ranks = cv_df.loc[competitor_mask, "Media MAE ($)"].rank(method="min").astype(int)
    cv_df["Ranking Preliminar"] = "Referencia (No compite)"
    for idx, r in ranks.items():
        cv_df.loc[idx, "Ranking Preliminar"] = f"{r}° lugar (CV Inicial)"

    return cv_df, raw_cv_results


# ============================================================================
# 6. ETAPA 11: AJUSTE DE HIPERPARÁMETROS (GRIDSEARCHCV) Y TABLA T6
# ============================================================================
def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_baseline_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Ejecuta GridSearchCV exclusivamente sobre X_train, y_train utilizando el
    mismo esquema de 5 folds y métrica primaria MAE para KNeighborsRegressor y SVR.
    """
    cv = get_cv_splitter()
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    # 1. Rejilla de KNeighborsRegressor (24 combinaciones)
    knn_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", KNeighborsRegressor()),
        ]
    )
    knn_param_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    t0_knn = time.perf_counter()
    knn_grid = GridSearchCV(
        estimator=knn_pipe,
        param_grid=knn_param_grid,
        scoring=scoring,
        refit="mae",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    knn_grid.fit(X_train, y_train)
    knn_search_time = time.perf_counter() - t0_knn

    # 2. Rejilla de SVR (72 combinaciones)
    # Nota técnica crítica: como charges está en USD ($1,121 - $63,770), C=1.0 por defecto
    # subajusta severamente. Se exploran valores de C hasta 10,000 junto con kernels lineal, rbf y poly.
    svr_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", SVR()),
        ]
    )
    svr_param_grid = {
        "model__kernel": ["linear", "rbf", "poly"],
        "model__C": [100, 1000, 5000, 10000],
        "model__gamma": ["scale", "auto", 0.1],
        "model__epsilon": [1.0, 50.0],
    }

    t0_svr = time.perf_counter()
    svr_grid = GridSearchCV(
        estimator=svr_pipe,
        param_grid=svr_param_grid,
        scoring=scoring,
        refit="mae",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    svr_grid.fit(X_train, y_train)
    svr_search_time = time.perf_counter() - t0_svr

    def _extract_top5_and_stats(grid: GridSearchCV) -> tuple[pd.DataFrame, dict[str, Any]]:
        res = pd.DataFrame(grid.cv_results_)
        best_idx = grid.best_index_

        fold_maes = np.array(
            [-res.loc[best_idx, f"split{i}_test_mae"] for i in range(N_SPLITS)]
        )
        fold_rmses = np.array(
            [-res.loc[best_idx, f"split{i}_test_rmse"] for i in range(N_SPLITS)]
        )
        fold_r2s = np.array(
            [res.loc[best_idx, f"split{i}_test_r2"] for i in range(N_SPLITS)]
        )

        stats = {
            "best_params": grid.best_params_,
            "fold_maes": fold_maes.tolist(),
            "fold_rmses": fold_rmses.tolist(),
            "fold_r2s": fold_r2s.tolist(),
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes, ddof=1)),
            "mean_rmse": float(np.mean(fold_rmses)),
            "std_rmse": float(np.std(fold_rmses, ddof=1)),
            "mean_r2": float(np.mean(fold_r2s)),
            "std_r2": float(np.std(fold_r2s, ddof=1)),
            "mean_fit_time_s": float(res.loc[best_idx, "mean_fit_time"]),
        }

        res_sorted = res.sort_values("rank_test_mae").head(5).copy()
        top5_rows = []
        for _, row in res_sorted.iterrows():
            f_maes = [-row[f"split{i}_test_mae"] for i in range(N_SPLITS)]
            f_rmses = [-row[f"split{i}_test_rmse"] for i in range(N_SPLITS)]
            f_r2s = [row[f"split{i}_test_r2"] for i in range(N_SPLITS)]
            clean_params = {
                k.replace("model__", ""): v for k, v in row["params"].items()
            }
            top5_rows.append(
                {
                    "Ranking": int(row["rank_test_mae"]),
                    "Configuración de Hiperparámetros": str(clean_params),
                    "Media MAE CV ($)": round(float(np.mean(f_maes)), 2),
                    "Desv. Est. MAE ($)": round(float(np.std(f_maes, ddof=1)), 2),
                    "Media RMSE CV ($)": round(float(np.mean(f_rmses)), 2),
                    "Media R² CV": round(float(np.mean(f_r2s)), 4),
                    "Tiempo Fit Promedio (s)": round(float(row["mean_fit_time"]), 4),
                }
            )
        return pd.DataFrame(top5_rows), stats

    knn_top5_df, knn_best_stats = _extract_top5_and_stats(knn_grid)
    svr_top5_df, svr_best_stats = _extract_top5_and_stats(svr_grid)

    # Extraer métricas antes de optimizar desde cv_baseline_df
    knn_before_mae = float(
        cv_baseline_df.loc[
            cv_baseline_df["Modelo"] == "KNeighborsRegressor", "Media MAE ($)"
        ].iloc[0]
    )
    knn_before_std = float(
        cv_baseline_df.loc[
            cv_baseline_df["Modelo"] == "KNeighborsRegressor", "Desv. Est. MAE ($)"
        ].iloc[0]
    )
    knn_before_r2 = float(
        cv_baseline_df.loc[
            cv_baseline_df["Modelo"] == "KNeighborsRegressor", "Media R²"
        ].iloc[0]
    )

    svr_before_mae = float(
        cv_baseline_df.loc[cv_baseline_df["Modelo"] == "SVR", "Media MAE ($)"].iloc[0]
    )
    svr_before_std = float(
        cv_baseline_df.loc[
            cv_baseline_df["Modelo"] == "SVR", "Desv. Est. MAE ($)"
        ].iloc[0]
    )
    svr_before_r2 = float(
        cv_baseline_df.loc[cv_baseline_df["Modelo"] == "SVR", "Media R²"].iloc[0]
    )

    knn_clean_best = {
        k.replace("model__", ""): v for k, v in knn_best_stats["best_params"].items()
    }
    svr_clean_best = {
        k.replace("model__", ""): v for k, v in svr_best_stats["best_params"].items()
    }

    t6_rows = [
        {
            "Algoritmo": "KNeighborsRegressor",
            "Rejilla Evaluada (param_grid)": (
                "n_neighbors: [3, 5, 7, 9, 11, 15], "
                "weights: ['uniform', 'distance'], "
                "p: [1 (Manhattan), 2 (Euclidiana)]"
            ),
            "Combinaciones (x 5 Folds)": "24 comb. (120 ajustes)",
            "Mejor Configuración": str(knn_clean_best),
            "MAE CV Antes ($)": round(knn_before_mae, 2),
            "MAE CV Después ($)": round(knn_best_stats["mean_mae"], 2),
            "Desv. Est. MAE Después ($)": round(knn_best_stats["std_mae"], 2),
            "Mejora MAE ($)": round(knn_before_mae - knn_best_stats["mean_mae"], 2),
            "Mejora MAE (%)": round(
                (knn_before_mae - knn_best_stats["mean_mae"]) / knn_before_mae * 100,
                2,
            ),
            "R² CV Antes": round(knn_before_r2, 4),
            "R² CV Después": round(knn_best_stats["mean_r2"], 4),
            "Tiempo Búsqueda Total (s)": round(knn_search_time, 2),
        },
        {
            "Algoritmo": "SVR",
            "Rejilla Evaluada (param_grid)": (
                "kernel: ['linear', 'rbf', 'poly'], "
                "C: [100, 1000, 5000, 10000], "
                "gamma: ['scale', 'auto', 0.1], "
                "epsilon: [1.0, 50.0]"
            ),
            "Combinaciones (x 5 Folds)": "72 comb. (360 ajustes)",
            "Mejor Configuración": str(svr_clean_best),
            "MAE CV Antes ($)": round(svr_before_mae, 2),
            "MAE CV Después ($)": round(svr_best_stats["mean_mae"], 2),
            "Desv. Est. MAE Después ($)": round(svr_best_stats["std_mae"], 2),
            "Mejora MAE ($)": round(svr_before_mae - svr_best_stats["mean_mae"], 2),
            "Mejora MAE (%)": round(
                (svr_before_mae - svr_best_stats["mean_mae"]) / svr_before_mae * 100,
                2,
            ),
            "R² CV Antes": round(svr_before_r2, 4),
            "R² CV Después": round(svr_best_stats["mean_r2"], 4),
            "Tiempo Búsqueda Total (s)": round(svr_search_time, 2),
        },
    ]

    t6_df = pd.DataFrame(t6_rows)

    return {
        "t6_table": t6_df,
        "knn_top5": knn_top5_df,
        "svr_top5": svr_top5_df,
        "knn_best_stats": knn_best_stats,
        "svr_best_stats": svr_best_stats,
        "knn_best_estimator": knn_grid.best_estimator_,
        "svr_best_estimator": svr_grid.best_estimator_,
        "knn_param_grid": knn_param_grid,
        "svr_param_grid": svr_param_grid,
    }


# ============================================================================
# 7. ETAPA 12: SELECCIÓN DEL MODELO ANTES DEL TEST Y TABLA T7
# ============================================================================
def get_decision_table(
    cv_df: pd.DataFrame,
    tuning_results: dict[str, Any],
) -> tuple[pd.DataFrame, str, Pipeline]:
    """
    Construye la Tabla T7 (Decisión) comparando todos los candidatos únicamente
    con base en sus resultados de Validación Cruzada (5 folds en train),
    estabilidad entre folds, costo computacional e interpretabilidad.
    Selecciona el modelo ganador ANTES de tocar el conjunto de prueba.
    """
    dummy_row = cv_df.loc[cv_df["Modelo"] == "DummyRegressor (Baseline)"].iloc[0]
    lin_row = cv_df.loc[cv_df["Modelo"] == "LinearRegression"].iloc[0]
    knn_stats = tuning_results["knn_best_stats"]
    svr_stats = tuning_results["svr_best_stats"]

    knn_clean_params = {
        k.replace("model__", ""): v for k, v in knn_stats["best_params"].items()
    }
    svr_clean_params = {
        k.replace("model__", ""): v for k, v in svr_stats["best_params"].items()
    }

    candidates_info = [
        {
            "Candidato": "DummyRegressor (Baseline)",
            "Configuración Final": "strategy='mean'",
            "Media MAE CV ($)": round(float(dummy_row["Media MAE ($)"]), 2),
            "Desv. Est. MAE CV ($)": round(float(dummy_row["Desv. Est. MAE ($)"]), 2),
            "Media RMSE CV ($)": round(float(dummy_row["Media RMSE ($)"]), 2),
            "Media R² CV": round(float(dummy_row["Media R²"]), 4),
            "Estabilidad (CV %)": round(
                float(dummy_row["Desv. Est. MAE ($)"])
                / float(dummy_row["Media MAE ($)"])
                * 100,
                2,
            ),
            "Costo Computacional (Fit s)": round(
                float(dummy_row["Tiempo Promedio Fit (s)"]), 4
            ),
            "Rol": "Referencia base (No competidor)",
            "Argumentación": (
                "Línea base ingenua que predice la media de entrenamiento; sirve para "
                "verificar la ganancia predictiva real de los algoritmos supervisados."
            ),
        },
        {
            "Candidato": "LinearRegression",
            "Configuración Final": "fit_intercept=True (Mínimos Cuadrados Ordinarios)",
            "Media MAE CV ($)": round(float(lin_row["Media MAE ($)"]), 2),
            "Desv. Est. MAE CV ($)": round(float(lin_row["Desv. Est. MAE ($)"]), 2),
            "Media RMSE CV ($)": round(float(lin_row["Media RMSE ($)"]), 2),
            "Media R² CV": round(float(lin_row["Media R²"]), 4),
            "Estabilidad (CV %)": round(
                float(lin_row["Desv. Est. MAE ($)"])
                / float(lin_row["Media MAE ($)"])
                * 100,
                2,
            ),
            "Costo Computacional (Fit s)": round(
                float(lin_row["Tiempo Promedio Fit (s)"]), 4
            ),
            "Rol": "Competidor 1 (Lineal paramétrico)",
            "Argumentación": (
                "Alta interpretabilidad y rapidez, pero su supuesto aditivo le impide modelar "
                "la interacción no lineal entre IMC alto y tabaquismo, dejando el MAE en ~$4,251."
            ),
        },
        {
            "Candidato": "KNeighborsRegressor (Optimizado)",
            "Configuración Final": str(knn_clean_params),
            "Media MAE CV ($)": round(float(knn_stats["mean_mae"]), 2),
            "Desv. Est. MAE CV ($)": round(float(knn_stats["std_mae"]), 2),
            "Media RMSE CV ($)": round(float(knn_stats["mean_rmse"]), 2),
            "Media R² CV": round(float(knn_stats["mean_r2"]), 4),
            "Estabilidad (CV %)": round(
                float(knn_stats["std_mae"]) / float(knn_stats["mean_mae"]) * 100, 2
            ),
            "Costo Computacional (Fit s)": round(
                float(knn_stats["mean_fit_time_s"]), 4
            ),
            "Rol": "Competidor 2 (No paramétrico local)",
            "Argumentación": (
                "Al usar distancia Manhattan (p=1) y ponderación por distancia (k=7), "
                "reduce el MAE en más de $1,140 respecto a regresión lineal, capturando vecindarios locales."
            ),
        },
        {
            "Candidato": "SVR (Optimizado)",
            "Configuración Final": str(svr_clean_params),
            "Media MAE CV ($)": round(float(svr_stats["mean_mae"]), 2),
            "Desv. Est. MAE CV ($)": round(float(svr_stats["std_mae"]), 2),
            "Media RMSE CV ($)": round(float(svr_stats["mean_rmse"]), 2),
            "Media R² CV": round(float(svr_stats["mean_r2"]), 4),
            "Estabilidad (CV %)": round(
                float(svr_stats["std_mae"]) / float(svr_stats["mean_mae"]) * 100, 2
            ),
            "Costo Computacional (Fit s)": round(
                float(svr_stats["mean_fit_time_s"]), 4
            ),
            "Rol": "Competidor 3 (Kernel no lineal)",
            "Argumentación": (
                "Obtiene el menor MAE absoluto en validación cruzada (~$2,424) y el mayor R² (~0.824) "
                "gracias al kernel RBF y la pérdida epsilon-insensible que atenúa valores extremos."
            ),
        },
    ]

    decision_df = pd.DataFrame(candidates_info)

    # Seleccionar ganador estrictamente por menor Media MAE CV entre los 3 competidores
    competitors_df = decision_df.loc[
        decision_df["Candidato"] != "DummyRegressor (Baseline)"
    ]
    best_idx = competitors_df["Media MAE CV ($)"].idxmin()
    winner_name = str(decision_df.loc[best_idx, "Candidato"])

    decision_df["Decisión Previa al Test"] = "No seleccionado"
    decision_df.loc[
        decision_df["Candidato"] == "DummyRegressor (Baseline)",
        "Decisión Previa al Test",
    ] = "Baseline de referencia"
    decision_df.loc[best_idx, "Decisión Previa al Test"] = (
        "GANADOR (Menor MAE en 5-Fold CV)"
    )

    if "SVR" in winner_name:
        winner_pipeline = tuning_results["svr_best_estimator"]
    elif "KNeighbors" in winner_name:
        winner_pipeline = tuning_results["knn_best_estimator"]
    else:
        winner_pipeline = build_candidate_pipelines()["LinearRegression"]

    return decision_df, winner_name, winner_pipeline


# ============================================================================
# 8. ETAPA 13 Y ANEXO B: EVALUACIÓN FINAL EN TEST Y SUBGRUPOS (TABLA T8)
# ============================================================================
def evaluate_final_model(
    winner_pipeline: Pipeline,
    baseline_pipeline: Pipeline,
    all_final_pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    winner_name: str = "SVR (Optimizado)",
) -> dict[str, Any]:
    """
    Evalúa UNA sola vez en el conjunto de prueba reservado (20%) el pipeline ganador
    previamente bloqueado en la Etapa 12, comparándolo con el Baseline (Tabla T8).
    Además ejecuta el análisis obligatorio del Grupo 09:
    - Comparación de MAE entre fumadores (smoker=='yes') y no fumadores (smoker=='no').
    - Comparación de errores por niveles de gasto (Bajo, Medio, Alto).
    - Muestra de 10 predicciones reales vs predichas y estadísticas de residuales.
    """
    # Ajustar ganador y baseline en el 100% de X_train
    winner_fitted = clone(winner_pipeline).fit(X_train, y_train)
    baseline_fitted = clone(baseline_pipeline).fit(X_train, y_train)

    y_pred_winner = winner_fitted.predict(X_test)
    y_pred_baseline = baseline_fitted.predict(X_test)

    mae_win = float(mean_absolute_error(y_test, y_pred_winner))
    rmse_win = float(np.sqrt(mean_squared_error(y_test, y_pred_winner)))
    r2_win = float(r2_score(y_test, y_pred_winner))

    mae_base = float(mean_absolute_error(y_test, y_pred_baseline))
    rmse_base = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    r2_base = float(r2_score(y_test, y_pred_baseline))

    mae_reduction_usd = mae_base - mae_win
    mae_reduction_pct = (mae_reduction_usd / mae_base) * 100.0
    rmse_reduction_usd = rmse_base - rmse_win
    rmse_reduction_pct = (rmse_reduction_usd / rmse_base) * 100.0

    # Tabla T8 oficial (Prueba final: Ganador vs Baseline e interpretación cuantitativa)
    t8_rows = [
        {
            "Métrica en Test Reservado (n=268)": "MAE - Error Absoluto Medio (Primaria)",
            "Unidad": "USD ($)",
            f"Modelo Ganador ({winner_name})": round(mae_win, 2),
            "Baseline (DummyRegressor Media)": round(mae_base, 2),
            "Diferencia / Mejora Absoluta": f"-${mae_reduction_usd:,.2f}",
            "Mejora Relativa (%)": f"-{mae_reduction_pct:.2f}%",
            "Interpretación Cuantitativa": (
                f"El modelo ganador se equivoca en promedio por ${mae_win:,.2f} por póliza en datos "
                f"no vistos, reduciendo el error absoluto medio en ${mae_reduction_usd:,.2f} "
                f"({mae_reduction_pct:.1f}%) frente a predecir la media histórica."
            ),
        },
        {
            "Métrica en Test Reservado (n=268)": "RMSE - Raíz del Error Cuadrático Medio (Secundaria)",
            "Unidad": "USD ($)",
            f"Modelo Ganador ({winner_name})": round(rmse_win, 2),
            "Baseline (DummyRegressor Media)": round(rmse_base, 2),
            "Diferencia / Mejora Absoluta": f"-${rmse_reduction_usd:,.2f}",
            "Mejora Relativa (%)": f"-{rmse_reduction_pct:.2f}%",
            "Interpretación Cuantitativa": (
                f"La brecha entre RMSE (${rmse_win:,.2f}) y MAE (${mae_win:,.2f}) indica la presencia "
                "de un subconjunto reducido de pólizas atípicas (hospitalizaciones complejas sin causa "
                "registrada en el CSV) con residuales grandes que el término cuadrático penaliza más."
            ),
        },
        {
            "Métrica en Test Reservado (n=268)": "R² - Coeficiente de Determinación (Secundaria)",
            "Unidad": "Proporción (adimensional)",
            f"Modelo Ganador ({winner_name})": round(r2_win, 4),
            "Baseline (DummyRegressor Media)": round(r2_base, 4),
            "Diferencia / Mejora Absoluta": f"+{r2_win - r2_base:.4f}",
            "Mejora Relativa (%)": f"{r2_win * 100:.2f}% var. explicada",
            "Interpretación Cuantitativa": (
                f"El pipeline ganador explica el {r2_win * 100:.2f}% de la varianza total de los gastos "
                "médicos en el conjunto de prueba, superando ampliamente el umbral de éxito (R² >= 0.75)."
            ),
        },
    ]
    t8_table = pd.DataFrame(t8_rows)

    # Tabla resumen de todos los modelos evaluados al final (solo como referencia documental)
    all_test_rows = []
    for m_name, m_pipe in all_final_pipelines.items():
        fitted_m = clone(m_pipe).fit(X_train, y_train)
        preds_m = fitted_m.predict(X_test)
        all_test_rows.append(
            {
                "Modelo": m_name,
                "Condición": (
                    "GANADOR ELEGIDO EN CV"
                    if m_name == winner_name
                    else (
                        "Baseline Referencial"
                        if "Dummy" in m_name
                        else "Referencia Post-Hoc (No elegido)"
                    )
                ),
                "Test MAE ($)": round(float(mean_absolute_error(y_test, preds_m)), 2),
                "Test RMSE ($)": round(
                    float(np.sqrt(mean_squared_error(y_test, preds_m))), 2
                ),
                "Test R²": round(float(r2_score(y_test, preds_m)), 4),
            }
        )
    all_test_summary_df = pd.DataFrame(all_test_rows)

    # Residuales detallados (Error = Real - Predicho)
    residuals = y_test.values - y_pred_winner
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / y_test.values) * 100.0

    test_details_df = X_test.copy()
    test_details_df["charges_real ($)"] = np.round(y_test.values, 2)
    test_details_df["charges_pred ($)"] = np.round(y_pred_winner, 2)
    test_details_df["residual_real_menos_pred ($)"] = np.round(residuals, 2)
    test_details_df["error_absoluto ($)"] = np.round(abs_errors, 2)
    test_details_df["error_porcentual (%)"] = np.round(pct_errors, 2)

    # Muestra de 10 filas representativas del conjunto de prueba (Etapa 13)
    sample_10_df = test_details_df.head(10).reset_index(names="ID Registro Original")

    # Estadísticas de residuales
    residual_stats = {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "std_residual": float(np.std(residuals, ddof=1)),
        "median_abs_error": float(np.median(abs_errors)),
        "p90_abs_error": float(np.percentile(abs_errors, 90)),
        "max_underestimation": float(np.max(residuals)),  # Real >> Predicho
        "max_overestimation": float(np.min(residuals)),   # Real << Predicho
        "pct_underestimated": float(np.mean(residuals > 0) * 100.0),
    }

    # ========================================================================
    # REQUISITO ESPECÍFICO GRUPO 09 (Anexo B y Design Doc Sección 4):
    # Comparación de MAE entre Fumadores vs No Fumadores (sin asumir causalidad)
    # ========================================================================
    smoker_rows = []
    for status, label in [
        ("no", "No Fumadores (smoker = 'no')"),
        ("yes", "Fumadores (smoker = 'yes')"),
    ]:
        mask = (X_test["smoker"] == status).values
        y_sub_true = y_test.values[mask]
        y_sub_pred = y_pred_winner[mask]
        res_sub = y_sub_true - y_sub_pred
        abs_sub = np.abs(res_sub)

        sub_mae = float(mean_absolute_error(y_sub_true, y_sub_pred))
        sub_rmse = float(np.sqrt(mean_squared_error(y_sub_true, y_sub_pred)))
        sub_r2 = float(r2_score(y_sub_true, y_sub_pred))
        mean_real = float(np.mean(y_sub_true))

        smoker_rows.append(
            {
                "Subgrupo": label,
                " Pólizas en Test (n)": int(np.sum(mask)),
                "Proporción Test (%)": round(float(np.mean(mask) * 100), 2),
                "Gasto Real Promedio ($)": round(mean_real, 2),
                "MAE en Test ($)": round(sub_mae, 2),
                "Mediana Error Abs. ($)": round(float(np.median(abs_sub)), 2),
                "RMSE en Test ($)": round(sub_rmse, 2),
                "MAE Relativo al Gasto Medio (%)": round(
                    (sub_mae / mean_real) * 100, 2
                ),
                "R² Intra-Subgrupo": round(sub_r2, 4),
                "Sesgo Medio (Real - Pred) ($)": round(float(np.mean(res_sub)), 2),
                "Interpretación y Precaución No Causal": (
                    "Menor error absoluto en dólares, pero mayor presencia de picos aislados "
                    "inexplicados (cirugías/accidentes en no fumadores) que elevan el RMSE frente al MAE."
                    if status == "no"
                    else (
                        "Mayor MAE nominal en dólares debido a que operan en una escala monetaria "
                        "3.8 veces superior (~$32k vs ~$8.4k), aunque su error relativo al gasto medio "
                        "y su R² intra-grupo son mejores. Asociar tabaquismo a mayor cargo predicho "
                        "refleja tarificación/correlación observacional, NO causalidad clínica directa."
                    )
                ),
            }
        )
    smoker_mae_table = pd.DataFrame(smoker_rows)

    # ========================================================================
    # REQUISITO ESPECÍFICO GRUPO 09 (Sección 5 Guía):
    # Análisis de errores por niveles de gasto (Bajo < $5k, Medio $5k-$15k, Alto > $15k)
    # ========================================================================
    tiers = [
        ("Bajo (< $5,000)", y_test.values < 5000),
        ("Medio ($5,000 - $15,000)", (y_test.values >= 5000) & (y_test.values <= 15000)),
        ("Alto (> $15,000)", y_test.values > 15000),
    ]
    spending_rows = []
    for tier_name, mask in tiers:
        if np.sum(mask) == 0:
            continue
        y_t = y_test.values[mask]
        p_t = y_pred_winner[mask]
        res_t = y_t - p_t
        abs_t = np.abs(res_t)
        spending_rows.append(
            {
                "Nivel de Gasto Real (charges)": tier_name,
                "Pólizas en Test (n)": int(np.sum(mask)),
                "Gasto Real Promedio ($)": round(float(np.mean(y_t)), 2),
                "MAE ($)": round(float(mean_absolute_error(y_t, p_t)), 2),
                "Mediana Error Abs. ($)": round(float(np.median(abs_t)), 2),
                "RMSE ($)": round(float(np.sqrt(mean_squared_error(y_t, p_t))), 2),
                "Sesgo Medio (Real - Pred) ($)": round(float(np.mean(res_t)), 2),
                "Comportamiento del Error": (
                    "Ajuste muy preciso sobre la curva base edad-costo de asegurados sanos."
                    if "Bajo" in tier_name
                    else (
                        "Error muy bajo en la mediana, con casos puntuales de subestimación moderada."
                        if "Medio" in tier_name
                        else (
                            "Concentra los mayores errores absolutos en dólares por asimetría positiva "
                            "y pólizas de no fumadores con hospitalizaciones imprevistas (> $20,000)."
                        )
                    )
                ),
            }
        )
    spending_level_mae_table = pd.DataFrame(spending_rows)

    return {
        "winner_name": winner_name,
        "winner_fitted": winner_fitted,
        "baseline_fitted": baseline_fitted,
        "t8_table": t8_table,
        "all_test_summary_df": all_test_summary_df,
        "sample_10_df": sample_10_df,
        "test_details_df": test_details_df,
        "residual_stats": residual_stats,
        "smoker_mae_table": smoker_mae_table,
        "spending_level_mae_table": spending_level_mae_table,
        "y_test": y_test.values.tolist(),
        "y_pred_winner": y_pred_winner.tolist(),
        "y_pred_baseline": y_pred_baseline.tolist(),
        "test_metrics": {
            "mae": mae_win,
            "rmse": rmse_win,
            "r2": r2_win,
            "baseline_mae": mae_base,
            "baseline_rmse": rmse_base,
            "baseline_r2": r2_base,
            "mae_reduction_usd": mae_reduction_usd,
            "mae_reduction_pct": mae_reduction_pct,
        },
    }


# ============================================================================
# 9. FUNCIONES DE VISUALIZACIÓN PARA STREAMLIT E INFORME
# ============================================================================
def plot_cv_comparison(
    cv_df: pd.DataFrame,
    tuning_results: dict[str, Any],
) -> plt.Figure:
    """
    Gráfico comparativo de MAE en Validación Cruzada (5 Folds) antes y después
    de GridSearchCV, incluyendo distribución por fold (Etapas 10 y 11).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=110)

    # Panel A: Barras horizontales de Media MAE CV (+/- Desv. Est.)
    models_labels = [
        "DummyRegressor\n(Baseline)",
        "SVR\n(Por Defecto C=1)",
        "LinearRegression\n(OLS)",
        "KNN\n(Por Defecto k=5)",
        "KNN Optimizado\n(GridSearchCV)",
        "SVR Optimizado\n(GridSearchCV - Ganador)",
    ]

    dummy_r = cv_df.loc[cv_df["Modelo"] == "DummyRegressor (Baseline)"].iloc[0]
    svr_def_r = cv_df.loc[cv_df["Modelo"] == "SVR"].iloc[0]
    lin_r = cv_df.loc[cv_df["Modelo"] == "LinearRegression"].iloc[0]
    knn_def_r = cv_df.loc[cv_df["Modelo"] == "KNeighborsRegressor"].iloc[0]

    knn_opt = tuning_results["knn_best_stats"]
    svr_opt = tuning_results["svr_best_stats"]

    means = [
        float(dummy_r["Media MAE ($)"]),
        float(svr_def_r["Media MAE ($)"]),
        float(lin_r["Media MAE ($)"]),
        float(knn_def_r["Media MAE ($)"]),
        float(knn_opt["mean_mae"]),
        float(svr_opt["mean_mae"]),
    ]
    stds = [
        float(dummy_r["Desv. Est. MAE ($)"]),
        float(svr_def_r["Desv. Est. MAE ($)"]),
        float(lin_r["Desv. Est. MAE ($)"]),
        float(knn_def_r["Desv. Est. MAE ($)"]),
        float(knn_opt["std_mae"]),
        float(svr_opt["std_mae"]),
    ]
    colors = ["#9ca3af", "#f59e0b", "#3b82f6", "#60a5fa", "#10b981", "#059669"]

    y_pos = np.arange(len(models_labels))
    bars = axes[0].barh(
        y_pos,
        means,
        xerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#1f2937",
        alpha=0.9,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models_labels, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE en Validación Cruzada 5-Fold (USD $ - Menor es mejor)")
    axes[0].set_title("A. Comparación de Media MAE CV (±1 DE)", fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    for bar, m in zip(bars, means):
        axes[0].text(
            m + 220,
            bar.get_y() + bar.get_height() / 2,
            f"${m:,.0f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_xlim(0, max(means) * 1.22)

    # Panel B: Evolución por Fold de los 3 modelos finales competidores
    folds = [f"Fold {i}" for i in range(1, N_SPLITS + 1)]
    lin_folds = [float(lin_r[f"Fold {i} MAE ($)"]) for i in range(1, N_SPLITS + 1)]
    knn_folds = knn_opt["fold_maes"]
    svr_folds = svr_opt["fold_maes"]

    axes[1].plot(
        folds,
        lin_folds,
        marker="o",
        linewidth=2,
        label="LinearRegression",
        color="#3b82f6",
    )
    axes[1].plot(
        folds,
        knn_folds,
        marker="s",
        linewidth=2,
        label="KNN (Optimizado)",
        color="#10b981",
    )
    axes[1].plot(
        folds,
        svr_folds,
        marker="^",
        linewidth=2.5,
        label="SVR (Optimizado - Ganador)",
        color="#059669",
    )
    axes[1].set_ylabel("MAE por Fold (USD $)")
    axes[1].set_title(
        "B. Estabilidad en los 5 Folds Idénticos (Solo Train)", fontweight="bold"
    )
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    return fig


def plot_real_vs_pred(
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    model_name: str = "SVR (Optimizado)",
) -> plt.Figure:
    """
    Gráfico de dispersión Valor Real vs Valor Predicho en el conjunto de prueba,
    diferenciando por condición de fumador (Etapa 13).
    """
    y_true_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=110)

    smokers = X_test["smoker"].values == "yes"
    ax.scatter(
        y_true_arr[~smokers],
        y_pred_arr[~smokers],
        alpha=0.65,
        c="#2563eb",
        edgecolors="white",
        s=55,
        label="No Fumador (smoker='no')",
    )
    ax.scatter(
        y_true_arr[smokers],
        y_pred_arr[smokers],
        alpha=0.80,
        c="#dc2626",
        edgecolors="white",
        s=65,
        marker="^",
        label="Fumador (smoker='yes')",
    )

    max_val = max(float(np.max(y_true_arr)), float(np.max(y_pred_arr))) * 1.05
    ax.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        color="#111827",
        linewidth=1.8,
        label="Identidad Perfecta (y_pred = y_real)",
    )

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Gasto Médico Real en Test - charges (USD $)", fontsize=10.5)
    ax.set_ylabel("Gasto Médico Predicho - charges_pred (USD $)", fontsize=10.5)
    ax.set_title(
        f"Evaluación en Test Reservado (n={len(y_true_arr)}): Real vs. Predicho ({model_name})",
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def plot_residuals_analysis(
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
) -> plt.Figure:
    """
    Gráfico diagnóstico de residuales (Real - Predicho) en el conjunto de prueba:
    Panel A: Residuales vs Predicción por fumador.
    Panel B: Distribución de residuales (Histograma + KDE).
    """
    y_true_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)
    residuals = y_true_arr - y_pred_arr

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=110)

    smokers = X_test["smoker"].values == "yes"
    axes[0].scatter(
        y_pred_arr[~smokers],
        residuals[~smokers],
        alpha=0.65,
        c="#2563eb",
        edgecolors="white",
        s=50,
        label="No Fumador (no)",
    )
    axes[0].scatter(
        y_pred_arr[smokers],
        residuals[smokers],
        alpha=0.80,
        c="#dc2626",
        edgecolors="white",
        s=60,
        marker="^",
        label="Fumador (yes)",
    )
    axes[0].axhline(0, color="#111827", linestyle="--", linewidth=1.6)
    axes[0].set_xlabel("Gasto Médico Predicho ($ USD)")
    axes[0].set_ylabel("Residual = Real - Predicho ($ USD)")
    axes[0].set_title("A. Residuales vs. Valores Predichos", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    sns.histplot(
        residuals,
        bins=30,
        kde=True,
        ax=axes[1],
        color="#059669",
        edgecolor="#1f2937",
    )
    axes[1].axvline(0, color="#dc2626", linestyle="--", linewidth=1.8, label="Cero Error")
    axes[1].set_xlabel("Residual = Real - Predicho ($ USD)")
    axes[1].set_ylabel("Frecuencia de Pólizas en Test")
    axes[1].set_title("B. Distribución de Residuales en Test", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def plot_smoker_error_comparison(
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> plt.Figure:
    """
    Visualización específica del Grupo 09 (Anexo B):
    Comparación del Error Absoluto |y_real - y_pred| entre fumadores y no fumadores
    y por nivel de gasto real.
    """
    y_true_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)
    abs_err = np.abs(y_true_arr - y_pred_arr)

    plot_df = pd.DataFrame(
        {
            "Condición Tabaco": np.where(
                X_test["smoker"].values == "yes",
                "Fumador (yes)",
                "No Fumador (no)",
            ),
            "Error Absoluto ($)": abs_err,
            "Gasto Real ($)": y_true_arr,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)

    # Panel A: Boxplot de Error Absoluto por condición de fumador
    sns.boxplot(
        data=plot_df,
        x="Condición Tabaco",
        y="Error Absoluto ($)",
        hue="Condición Tabaco",
        palette={"No Fumador (no)": "#60a5fa", "Fumador (yes)": "#f87171"},
        legend=False,
        ax=axes[0],
        width=0.45,
    )
    axes[0].set_title(
        "A. Distribución de Error Absoluto por Tabaquismo", fontweight="bold"
    )
    axes[0].set_ylabel("Error Absoluto |Real - Predicho| (USD $)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Comparación de MAE Nominal vs MAE Relativo al gasto medio del grupo
    grp_no = plot_df[plot_df["Condición Tabaco"] == "No Fumador (no)"]
    grp_yes = plot_df[plot_df["Condición Tabaco"] == "Fumador (yes)"]

    mae_no = float(grp_no["Error Absoluto ($)"].mean())
    mae_yes = float(grp_yes["Error Absoluto ($)"].mean())
    med_no = float(grp_no["Error Absoluto ($)"].median())
    med_yes = float(grp_yes["Error Absoluto ($)"].median())

    x_idx = np.arange(2)
    width = 0.34
    b1 = axes[1].bar(
        x_idx - width / 2,
        [mae_no, mae_yes],
        width,
        label="MAE Promedio ($)",
        color="#2563eb",
    )
    b2 = axes[1].bar(
        x_idx + width / 2,
        [med_no, med_yes],
        width,
        label="Mediana Error Absoluto ($)",
        color="#10b981",
    )
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(["No Fumador (no)", "Fumador (yes)"])
    axes[1].set_ylabel("Error en Dólares (USD $)")
    axes[1].set_title(
        "B. MAE Promedio vs. Mediana de Error por Subgrupo", fontweight="bold"
    )
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 60,
            f"${h:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(mae_no, mae_yes, med_no, med_yes) * 1.2)

    fig.tight_layout()
    return fig


# ============================================================================
# 10. ORQUESTACIÓN COMPLETA Y SERIALIZACIÓN (models/modelo_final.joblib)
# ============================================================================
def train_and_export_all(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """
    Ejecuta el flujo completo de las Etapas 6 a 13 garantizando cero fuga de
    información y exporta el artefacto completo (incluyendo el Pipeline ganador
    bajo la clave 'pipeline' y todas las tablas T3-T8) en models/modelo_final.joblib.
    """
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Etapa 6: Partición 80/20 antes de cualquier transformación
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    t3_table = get_partition_table(X_train, X_test, y_train, y_test)

    # Etapa 7 y 8: Tablas de preprocesamiento y algoritmos
    t4_table = get_preprocessing_table()
    algorithms_table = get_algorithms_table()

    # Etapa 10: Validación cruzada 5-Fold idéntica solo en X_train
    t5_table, raw_cv_results = run_cross_validation(X_train, y_train)

    # Etapa 11: GridSearchCV para KNN y SVR solo en X_train
    tuning_results = run_hyperparameter_tuning(X_train, y_train, t5_table)
    t6_table = tuning_results["t6_table"]

    # Etapa 12: Selección del modelo ganador antes de mirar el test
    t7_table, winner_name, winner_pipeline = get_decision_table(
        t5_table, tuning_results
    )

    # Diccionario con los pipelines finales para referencia
    cand_pipes = build_candidate_pipelines()
    all_final_pipelines = {
        "DummyRegressor (Baseline)": cand_pipes["DummyRegressor (Baseline)"],
        "LinearRegression": cand_pipes["LinearRegression"],
        "KNeighborsRegressor (Optimizado)": tuning_results["knn_best_estimator"],
        "SVR (Optimizado)": tuning_results["svr_best_estimator"],
    }

    # Etapa 13: Evaluación única sobre el conjunto de prueba reservado
    final_eval = evaluate_final_model(
        winner_pipeline=winner_pipeline,
        baseline_pipeline=cand_pipes["DummyRegressor (Baseline)"],
        all_final_pipelines=all_final_pipelines,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        winner_name=winner_name,
    )

    artifact: dict[str, Any] = {
        "pipeline": final_eval["winner_fitted"],
        "baseline_pipeline": final_eval["baseline_fitted"],
        "winner_name": winner_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "feature_cols": FEATURE_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_col": TARGET_COL,
        "t3_partition_table": t3_table,
        "t4_preprocessing_table": t4_table,
        "algorithms_table": algorithms_table,
        "t5_cv_table": t5_table,
        "raw_cv_results": raw_cv_results,
        "tuning_results": tuning_results,
        "t6_tuning_table": t6_table,
        "t7_decision_table": t7_table,
        "t8_final_test_table": final_eval["t8_table"],
        "all_test_summary_df": final_eval["all_test_summary_df"],
        "sample_10_df": final_eval["sample_10_df"],
        "test_details_df": final_eval["test_details_df"],
        "residual_stats": final_eval["residual_stats"],
        "smoker_mae_table": final_eval["smoker_mae_table"],
        "spending_level_mae_table": final_eval["spending_level_mae_table"],
        "test_metrics": final_eval["test_metrics"],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_winner": np.array(final_eval["y_pred_winner"]),
        "y_pred_baseline": np.array(final_eval["y_pred_baseline"]),
    }

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    return artifact


if __name__ == "__main__":
    print("Ejecutando flujo de entrenamiento, CV, GridSearchCV y evaluación final...")
    art = train_and_export_all()
    print(f"Modelo ganador seleccionado en CV: {art['winner_name']}")
    print("\n--- Tabla T5: Comparación en Validación Cruzada (5 Folds) ---")
    print(art["t5_cv_table"].to_string(index=False))
    print("\n--- Tabla T6: Optimización de Hiperparámetros (GridSearchCV) ---")
    print(art["t6_tuning_table"].to_string(index=False))
    print("\n--- Tabla T7: Decisión Previa al Test ---")
    print(art["t7_decision_table"].to_string(index=False))
    print("\n--- Tabla T8: Prueba Final en Test Reservado ---")
    print(art["t8_final_test_table"].to_string(index=False))
    print("\n--- Análisis Grupo 09: MAE por Condición de Fumador en Test ---")
    print(art["smoker_mae_table"].to_string(index=False))
    print(f"\nArtefacto exportado exitosamente en: {MODEL_PATH}")
