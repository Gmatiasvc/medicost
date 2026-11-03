"""
Aplicación interactiva en Streamlit para el Grupo 09:
Estimación de Gastos de Seguros Médicos (Medical Cost Personal Datasets).

Implementa la arquitectura de 6 pantallas exigida en el Anexo H de ejercicio.md
y design_doc.md, cubriendo las 16 etapas obligatorias (Anexo A / Anexo E),
las 9 tablas estándar (T1 a T9, Anexo C) y los requisitos analíticos específicos
del Grupo 09 (asimetría y atípicos de `charges`, comparación de MAE entre
fumadores y no fumadores por niveles de gasto, y advertencia explícita de no causalidad).
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from src.data import (
    CATEGORICAL_COLS,
    DEFAULT_DATA_PATH,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    analyze_charges_asymmetry,
    get_data_dictionary,
    get_dataset_metadata,
    get_feature_selection_table,
    get_quality_diagnostics,
    load_data,
    plot_age_vs_charges,
    plot_bmi_vs_charges,
    plot_charges_distribution,
    plot_multivariate_and_correlation,
    plot_smoker_boxplot,
    split_features_target,
)
from src.ml import (
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    plot_cv_comparison,
    plot_real_vs_pred,
    plot_residuals_analysis,
    plot_smoker_error_comparison,
    train_and_export_all,
)
from src.predict import (
    get_five_new_predictions,
    load_artifact,
    predict_single,
    validate_input,
)

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS VISUALES
# ============================================================================
st.set_page_config(
    page_title="Grupo 09 | Estimación de Gastos de Seguros Médicos",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0d9488 100%);
        color: #ffffff;
        padding: 1.5rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    .main-banner h1 {
        color: #ffffff !important;
        font-size: 1.75rem !important;
        margin-bottom: 0.35rem !important;
        font-weight: 700 !important;
    }
    .main-banner p {
        color: #e2e8f0 !important;
        font-size: 0.98rem !important;
        margin: 0 !important;
    }
    .stage-badge {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .table-badge {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .group02-box {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .causal-warning {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# FUNCIONES CACHEADAS PARA RENDIMIENTO ÓPTIMO
# ============================================================================
@st.cache_data(show_spinner=False)
def cached_load_default_data() -> pd.DataFrame:
    """Carga y cachea el dataset oficial data/insurance.csv."""
    return load_data(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def cached_eda_summaries(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula y cachea las tablas y estadísticos descriptivos del EDA."""
    return {
        "metadata": get_dataset_metadata(df),
        "t1_dict": get_data_dictionary(df),
        "t2_quality": get_quality_diagnostics(df),
        "feature_selection": get_feature_selection_table(),
        "asymmetry": analyze_charges_asymmetry(df),
    }


@st.cache_resource(show_spinner="Cargando Pipeline ganador y tablas experimentales (T3-T8)...")
def cached_load_artifact() -> dict[str, Any]:
    """Carga el artefacto serializado models/modelo_final.joblib."""
    return load_artifact(MODEL_PATH)


@st.cache_data(show_spinner=False)
def cached_five_new_predictions() -> pd.DataFrame:
    """Genera y cachea la Tabla T9 de 5 casos nuevos plausibles."""
    return get_five_new_predictions(MODEL_PATH)


def render_header(df: pd.DataFrame, artifact: dict[str, Any]) -> None:
    """Renderiza el encabezado institucional del Grupo 09."""
    st.markdown(
        """
        <div class="main-banner">
            <h1>🏥 Grupo 09 — Estimación de Gastos de Seguros Médicos</h1>
            <p>
                Aprendizaje Supervisado (Regresión) sobre <b>Medical Cost Personal Datasets (insurance.csv)</b> ·
                Pipeline reproducible sin Fuga de Datos (Data Leakage) · Métrica Primaria: <b>MAE (USD $)</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# PANTALLA 1: INICIO Y PROBLEMA (ETAPAS 1 Y 2)
# ============================================================================
def render_screen_1_inicio(df: pd.DataFrame, artifact: dict[str, Any]) -> None:
    """Pantalla 1 (Anexo H): Contexto, pregunta predictiva, objetivos, criterios y uso responsable."""
    st.markdown(
        '<span class="stage-badge">Etapa 1: Definición del problema</span>'
        '<span class="stage-badge">Etapa 2: Objetivos, criterios de éxito y uso responsable</span>',
        unsafe_allow_html=True,
    )
    st.header("1. Inicio y Definición del Problema Predictivo")

    test_metrics = artifact["test_metrics"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pólizas Analizadas", f"{len(df):,}", "7 columnas (6 X + 1 y)")
    k2.metric("Partición Train / Test", "80% / 20%", f"1,070 train / 268 test (seed={RANDOM_STATE})")
    k3.metric("Modelo Ganador (CV)", artifact["winner_name"], "RBF Kernel · C=10,000")
    k4.metric(
        "MAE en Test Reservado",
        f"${test_metrics['mae']:,.2f}",
        f"-{test_metrics['mae_reduction_pct']:.1f}% vs. Baseline (${test_metrics['baseline_mae']:,.0f})",
        delta_color="inverse",
    )
    k5.metric(
        "R² en Test Reservado",
        f"{test_metrics['r2']:.4f}",
        f"{test_metrics['r2'] * 100:.2f}% varianza explicada",
    )

    st.divider()

    # --- ETAPA 1 ---
    st.subheader("📌 Etapa 1: Definición del Problema y Formulación Predictiva")
    col_a, col_b = st.columns([1.25, 1.0])

    with col_a:
        st.markdown(
            """
            #### 1.1 Contexto del Problema Real y Relevancia Actuarial
            En la gestión de seguros de salud individuales y colectivos, la **tarificación técnica de pólizas (*underwriting*)**
            y la **constitución de reservas técnicas de siniestralidad** requieren estimar con antelación el desembolso médico
            anual que generará cada beneficiario.

            La distribución del gasto médico individual (`charges`) presenta dos desafíos estadísticos fundamentales:
            1. **Fuerte asimetría positiva (cola derecha):** la mayoría de asegurados demanda atención preventiva o ambulatoria
               de bajo costo (mediana de **$9,382.03 USD**), mientras que una fracción experimenta eventos hospitalarios o
               tratamientos crónicos que superan los **$34,489 USD** y alcanzan hasta **$63,770.43 USD**.
            2. **Interacciones no lineales severas:** el efecto conjunto de factores de riesgo metabólicos (obesidad, `bmi >= 30`)
               y conductuales (tabaquismo, `smoker = 'yes'`) produce saltos discretos en el gasto que una media histórica o una
               fórmula lineal aditiva no logran capturar adecuadamente.

            #### 1.2 Pregunta Predictiva Precisa
            > **¿Cuál será el gasto médico anual facturado (`charges`, en dólares estadounidenses USD) para un beneficiario titular
            > de seguro médico, estimado en el momento previo a la vigencia de la póliza ($t=0$) a partir de su edad (`age`),
            > sexo (`sex`), índice de masa corporal (`bmi`), número de hijos dependientes (`children`), condición de fumador (`smoker`)
            > y región geográfica de residencia (`region`)?**
            """
        )

    with col_b:
        st.info(
            """
            **Ficha Operativa de la Decisión (Etapa 1)**
            - **Tipo de aprendizaje supervisado:** Regresión (variable respuesta numérica continua en USD $).
            - **Usuario decisor:** Analista actuarial, suscriptor técnico de pólizas de salud (*medical underwriter*) y comité de gestión de reservas.
            - **Momento exacto de uso ($t=0$):** Durante el proceso de cotización, suscripción inicial o renovación anual de la póliza, **antes** de que ocurran los siniestros médicos del período.
            - **Decisión que apoya:** Estimación de prima pura de riesgo individualizada, cálculo de suficiencia de reserva técnica y segmentación preventiva de cartera.
            - **Garantía de disponibilidad ex-ante:** Las 6 variables predictoras (`X`) se recaban en el formulario de afiliación y examen médico de ingreso; ninguna depende de eventos futuros.
            """
        )

        st.markdown(r"##### Esquema Conceptual: Entradas ($\mathbf{X}$) $\longrightarrow$ Salida ($y$)")
        schema_df = pd.DataFrame(
            [
                {
                    "Componente": "Entradas Predictoras (X)",
                    "Variables": "age, sex, bmi, children, smoker, region (6 columnas)",
                    "Instante": "t = 0 (Suscripción / Afiliación)",
                },
                {
                    "Componente": "Pipeline Supervisado",
                    "Variables": "ColumnTransformer (Imputación + Escalado + OneHot) + Estimador",
                    "Instante": "Inferencia sin reajuste (.predict)",
                },
                {
                    "Componente": "Salida Predicha (ŷ)",
                    "Variables": "charges_pred (Gasto médico anual esperado en USD $)",
                    "Instante": "Proyección anual [t, t+1]",
                },
            ]
        )
        st.dataframe(schema_df, width="stretch", hide_index=True)

    st.divider()

    # --- ETAPA 2 ---
    st.subheader("🎯 Etapa 2: Objetivos, Criterios de Éxito y Uso Responsable")

    st.markdown(
        r"""
        #### 2.1 Objetivo General Medible
        Desarrollar, validar y desplegar en **Streamlit** un flujo reproducible de aprendizaje supervisado para regresión
        que estime el gasto médico anual individual (`charges`) sobre el conjunto *Medical Cost Personal Datasets* (`n = 1,338`),
        comparando bajo validación cruzada de 5 folds idénticos tres familias algorítmicas obligatorias (**`LinearRegression`**,
        **`KNeighborsRegressor`** y **`SVR`**) frente a una línea base (**`DummyRegressor`**), previniendo toda fuga de información
        mediante el uso estricto de `Pipeline` y `ColumnTransformer`, y alcanzando en el conjunto de prueba reservado (20%)
        un **Error Absoluto Medio (`MAE`) inferior a $3,500 USD** y un **$R^2 \ge 0.75$**.
        """
    )

    st.markdown(r"#### 2.2 Objetivos Específicos y Matriz de Verificación (Objetivo $\rightarrow$ Actividad $\rightarrow$ Evidencia)")
    obj_df = pd.DataFrame(
        [
            {
                "N°": "OE1",
                "Objetivo Específico": (
                    "Diagnosticar la estructura, calidad, asimetría y valores atípicos de la variable monetaria 'charges' "
                    "y sus relaciones con las variables predictoras."
                ),
                "Actividad Técnica Realizada": (
                    "Auditoría de tipos, nulos, duplicados, cálculo de sesgo (Skewness=1.52), curtosis (1.61), "
                    "cercos IQR y generación de 5 visualizaciones multivariadas."
                ),
                "Evidencia Verificable en la App": (
                    "Pantalla 2 (Datos y EDA): Tablas T1 y T2, KPIs de asimetría de charges y 5 gráficos interpretados."
                ),
            },
            {
                "N°": "OE2",
                "Objetivo Específico": (
                    "Construir una arquitectura de partición 80/20 y preprocesamiento diferenciado libre de fuga de "
                    "información (Data Leakage)."
                ),
                "Actividad Técnica Realizada": (
                    "Separación X/y previa, train_test_split(test_size=0.20, random_state=42) y encapsulamiento de "
                    "SimpleImputer, StandardScaler y OneHotEncoder dentro de ColumnTransformer."
                ),
                "Evidencia Verificable en la App": (
                    "Pantalla 3 (Variables y preparación): Tablas T3 y T4, verificación de 0 filas compartidas y prueba en vivo."
                ),
            },
            {
                "N°": "OE3",
                "Objetivo Específico": (
                    "Entrenar, comparar en validación cruzada (5-Fold) y optimizar hiperparámetros de KNN y SVR "
                    "exclusivamente sobre entrenamiento antes de abrir el conjunto de prueba."
                ),
                "Actividad Técnica Realizada": (
                    "Evaluación 5-Fold KFold(shuffle=True, random_state=42) de los 4 pipelines y ajuste con GridSearchCV "
                    "optimizando MAE (24 combinaciones en KNN y 72 en SVR)."
                ),
                "Evidencia Verificable en la App": (
                    "Pantalla 4 (Modelos y comparación): Tablas T5, T6 y T7, ranking Top 5 y justificación del modelo ganador."
                ),
            },
            {
                "N°": "OE4",
                "Objetivo Específico": (
                    "Evaluar por única vez el modelo ganador en el test reservado, analizando el MAE entre fumadores y "
                    "no fumadores y por niveles de gasto sin asumir causalidad directa, e implementar inferencia en producción."
                ),
                "Actividad Técnica Realizada": (
                    "Cálculo de Tabla T8 en test (n=268), desglose de residuales por tabaquismo y nivel de gasto, "
                    "y serialización en models/modelo_final.joblib para predicciones nuevas."
                ),
                "Evidencia Verificable en la App": (
                    "Pantallas 5 (Evaluación) y 6 (Predicciones): Tabla T8, análisis Grupo 09, formulario y Tabla T9."
                ),
            },
        ]
    )
    st.dataframe(obj_df, width="stretch", hide_index=True)

    c_crit1, c_crit2 = st.columns(2)
    with c_crit1:
        st.markdown("#### 2.3 Criterio Cuantitativo de Éxito y Justificación de Métricas")
        st.markdown(
            f"""
            - **Métrica Primaria de Decisión — `MAE` (Error Absoluto Medio, menor es mejor):**
              Expresa en dólares reales ($ USD) la desviación promedio por póliza:
              $$\\text{{MAE}} = \\frac{{1}}{{n}}\\sum_{{i=1}}^{{n}} |y_i - \\hat{{y}}_i|$$
              En presencia de la pronunciada asimetría derecha de `charges` (139 valores extremos $> \\$34,489$),
              el `MAE` no eleva al cuadrado los errores de hospitalizaciones fortuitas, reflejando con fidelidad
              la precisión operativa sobre la cartera típica.
            - **Métricas Secundarias — `RMSE` y `$R^2$`:**
              El `RMSE` penaliza cuadráticamente errores grandes (útil para vigilar el riesgo de cola actuarial),
              mientras que el coeficiente de determinación `$R^2$` cuantifica la proporción de varianza explicada.
            - **Umbral de Éxito Acordado Ex-Ante vs. Resultado Logrado:**
              1. Superar ampliamente al baseline ingenuo (`DummyRegressor` MAE = **${test_metrics['baseline_mae']:,.2f}**).
              2. Alcanzar **MAE $< \\$3,500 USD** y **$R^2 \\ge 0.75$** en test reservado.
              3. **Resultado alcanzado por `{artifact['winner_name']}`:** **MAE = ${test_metrics['mae']:,.2f} USD**
                 (reducción del **{test_metrics['mae_reduction_pct']:.2f}%** del error base) y **$R^2 = {test_metrics['r2']:.4f}$**.
            """
        )

    with c_crit2:
        st.markdown("#### 2.4 Costo Contextual de los Errores en el Dominio Asegurador")
        st.markdown(
            r"""
            En tarificación y reservas de salud, los errores positivos y negativos tienen consecuencias asimétricas:
            - **Subestimación ($\hat{y}_i \ll y_i$, residual positivo grande):**
              Si el modelo predice un gasto de $\$7,000$ para un asegurado que termina facturando $\$35,000$,
              la aseguradora sufre **insuficiencia de prima y déficit de reserva técnica (riesgo de solvencia)**.
              En nuestro conjunto de prueba, esto ocurre principalmente en un pequeño grupo de *no fumadores* que
              sufren emergencias quirúrgicas o accidentes graves no anticipables con las 6 variables de entrada.
            - **Sobreestimación ($\hat{y}_i \gg y_i$, residual negativo):**
              Si el modelo asigna un costo esperado excesivo a un cotizante saludable, la prima resultante será
              poco competitiva en el mercado, provocando **selección adversa y pérdida comercial de clientes de bajo riesgo**.
            """
        )

    st.markdown(
        """
        <div class="causal-warning">
            <b>⚠️ Advertencia Metodológica Obligatoria (Grupo 09): Uso Responsable, Límites Éticos y No Causalidad</b><br>
            1. <b>Asociación predictiva vs. Causalidad directa:</b> Este modelo aprende patrones estadísticos observacionales en datos históricos
            de facturación. Aunque <code>smoker = 'yes'</code> y <code>bmi >= 30</code> presentan la mayor asociación con incrementos en
            <code>charges</code>, <b>no debe interpretarse como una estimación de efecto causal directo</b> (no existe asignación aleatoria
            ni control por variables confusoras omitidas como historial genético, ocupación, nivel socioeconómico o enfermedades preexistentes).<br>
            2. <b>Apoyo a la decisión, no reemplazo ni discriminación automática:</b> En marcos regulatorios reales de salud (p. ej., ACA en EE. UU.
            o normativas de supervisión de seguros), ciertas variables demográficas como el sexo biológico (<code>sex</code>) o condiciones médicas
            tienen restricciones legales estrictas para fijar primas individuales. El sistema constituye una <b>herramienta académica y de soporte
            actuarial para estimación agregada de reservas</b> y nunca debe utilizarse para denegar cobertura médica de forma automatizada.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# PANTALLA 2: DATOS Y EDA (ETAPAS 3 Y 5 + REQUISITO GRUPO 09)
# ============================================================================
def render_screen_2_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Pantalla 2 (Anexo H): Fuente del dataset, T1, calidad T2, asimetría de charges y 5 gráficos EDA."""
    st.markdown(
        '<span class="stage-badge">Etapa 3: Fuente del dataset y diccionario</span>'
        '<span class="stage-badge">Etapa 5: Análisis exploratorio de datos (EDA)</span>'
        '<span class="table-badge">Tabla T1: Diccionario de variables</span>'
        '<span class="table-badge">Tabla T2: Diagnóstico de calidad</span>',
        unsafe_allow_html=True,
    )
    st.header("2. Datos, Diccionario de Variables y Análisis Exploratorio (EDA)")

    # Opción de carga reproducible o subida de archivo compatible
    with st.expander("📂 Configuración de Fuente de Datos (Carga Reproducible por Defecto o Archivo CSV Compatible)", expanded=False):
        st.markdown(
            "Por defecto, la aplicación carga de manera automática y reproducible el archivo local **`data/insurance.csv`**. "
            "Opcionalmente puede cargar un archivo CSV con el mismo esquema de 7 columnas para auditarlo en vivo."
        )
        uploaded_file = st.file_uploader("Subir archivo CSV compatible (opcional)", type=["csv"])
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in custom_df.columns]
                if missing:
                    st.error(f"El archivo subido no contiene las columnas requeridas: {missing}. Usando dataset oficial.")
                else:
                    df = custom_df
                    st.success(f"Archivo personalizado cargado exitosamente ({len(df):,} filas x {df.shape[1]} columnas).")
            except Exception as exc:
                st.error(f"Error al leer el archivo subido: {exc}. Se mantiene `data/insurance.csv`.")

    eda_data = cached_eda_summaries(df)
    meta = eda_data["metadata"]
    t1_df = eda_data["t1_dict"]
    t2_df = eda_data["t2_quality"]
    asym = eda_data["asymmetry"]

    # --- ETAPA 3 ---
    st.subheader("📋 Etapa 3: Ficha Técnica de la Fuente de Datos y Tabla T1")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dimensiones (Filas x Columnas)", f"{meta['n_rows']:,} × {meta['n_cols']}", "Unidad: 1 póliza / titular anual")
    m2.metric("Valores Nulos Totales", f"{meta['n_nulls_total']}", "0.00% en las 7 columnas")
    m3.metric("Filas Duplicadas Exactas", f"{meta['n_duplicates']}", f"Índices: {meta['duplicate_indices']}")
    m4.metric("Memoria en RAM", f"{meta['memory_kb']} KB", "3 numéricas, 3 categóricas, 1 target")

    st.markdown(
        f"""
        - **Nombre oficial del conjunto de datos:** *{meta['dataset_name']}*
        - **Autor / Compilador:** {meta['author']}
        - **Enlace directo verificable (Kaggle):** [{meta['kaggle_url']}]({meta['kaggle_url']})
        - **Licencia y versión:** {meta['license']} — {meta['version']}
        - **Ruta local reproducible:** `{DEFAULT_DATA_PATH}`
        """
    )

    st.markdown("#### Tabla T1 — Diccionario de Variables y Análisis de Disponibilidad Ex-Ante (Anexo C)")
    st.dataframe(t1_df, width="stretch", hide_index=True)

    st.divider()

    # --- ETAPA 5: INSPECCIÓN Y CALIDAD (TABLA T2) ---
    st.subheader("🔍 Etapa 5: Inspección Estructural, Estadísticos Descriptivos y Diagnóstico de Calidad (Tabla T2)")

    tab_prev, tab_stats, tab_t2 = st.tabs(
        [
            "📄 Vista Previa (head) y Tipos de Datos",
            "📊 Estadísticos Descriptivos (Numéricas y Categóricas)",
            "🛡️ Tabla T2: Diagnóstico de Calidad y Decisiones",
        ]
    )

    with tab_prev:
        c_p1, c_p2 = st.columns([1.6, 1.0])
        with c_p1:
            st.markdown("##### Primeras 10 filas del dataset (`df.head(10)`)")
            st.dataframe(df.head(10), width="stretch")
        with c_p2:
            st.markdown("##### Estructura de Tipos y Conteo de No-Nulos")
            dtypes_df = pd.DataFrame(
                [
                    {
                        "Columna": col,
                        "Tipo en Pandas": str(df[col].dtype),
                        "No Nulos": int(df[col].notna().sum()),
                        "Valores Únicos": int(df[col].nunique()),
                    }
                    for col in df.columns
                ]
            )
            st.dataframe(dtypes_df, width="stretch", hide_index=True)
            if meta["n_duplicates"] > 0:
                st.caption(
                    f"Registro duplicado detectado en los índices `{meta['duplicate_indices']}`: "
                    f"`{meta['duplicate_rows'][0]}` (perfil demográfico joven plausible: 19 años, male, bmi=30.59, smoker='no')."
                )

    with tab_stats:
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown("##### Variables Numéricas (`age`, `bmi`, `children`, `charges`)")
            num_desc = df[NUMERIC_COLS + [TARGET_COL]].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]).T
            num_desc["iqr"] = num_desc["75%"] - num_desc["25%"]
            num_desc["skewness"] = df[NUMERIC_COLS + [TARGET_COL]].skew()
            st.dataframe(num_desc.round(2), width="stretch")
        with c_s2:
            st.markdown("##### Frecuencias de Variables Categóricas (`sex`, `smoker`, `region`)")
            cat_rows = []
            for c in CATEGORICAL_COLS:
                vc = df[c].value_counts()
                vcp = df[c].value_counts(normalize=True) * 100
                for val in vc.index:
                    cat_rows.append(
                        {
                            "Variable": c,
                            "Categoría": val,
                            "Frecuencia Absoluta (n)": int(vc[val]),
                            "Proporción (%)": round(float(vcp[val]), 2),
                        }
                    )
            st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)

    with tab_t2:
        st.markdown("##### Tabla T2 — Diagnóstico de Calidad de Datos (Nulos, Duplicados, Atípicos IQR y Medida Adoptada)")
        st.dataframe(t2_df, width="stretch", hide_index=True)

    st.divider()

    # --- REQUISITO ESPECÍFICO GRUPO 09: ASIMETRÍA Y VALORES ATÍPICOS DE CHARGES ---
    st.subheader("📈 Requisito Específico Grupo 09: Análisis de Asimetría y Valores Atípicos de `charges`")
    st.markdown(
        """
        <div class="group02-box">
            <b>Foco Analítico del Grupo 09 (Anexo B y Design Doc Sección 4):</b><br>
            La variable monetaria objetivo <code>charges</code> presenta una marcada <b>asimetría positiva (cola derecha)</b>.
            La media global (<b>$13,270.42</b>) supera en casi <b>$3,888 USD</b> a la mediana (<b>$9,382.03</b>) debido a una subpoblación
            de asegurados de alto costo. Bajo el criterio intercuartílico clásico ($Q_3 + 1.5 \\times \\text{IQR}$), existen
            <b>139 registros extremos (&gt; $34,489.35 USD, 10.39% del dataset)</b>. Sin embargo, el análisis cruzado demuestra que el
            <b>97.84% de estos valores extremos (136 de 139 casos) corresponden a asegurados fumadores (<code>smoker = 'yes'</code>)</b>,
            principalmente con obesidad (<code>bmi &gt;= 30</code>). Por tanto, <b>NO son errores de captura sino siniestralidad real</b>
            y deben conservarse íntegros en el modelado, eligiendo el <b>MAE</b> como métrica primaria por su robustez ante colas pesadas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Media de charges", f"${asym['mean']:,.2f}", f"DE: ${asym['std']:,.2f}")
    a2.metric("Mediana (Q2)", f"${asym['median']:,.2f}", f"IQR: ${asym['iqr']:,.2f}")
    a3.metric("Coef. Asimetría (Skew)", f"{asym['skewness']:.4f}", "Cola derecha severa (> 1)")
    a4.metric("Curtosis (Exceso)", f"{asym['kurtosis']:.4f}", "Leptocúrtica / colas pesadas")
    a5.metric("Umbral Atípicos IQR", f"${asym['upper_fence']:,.2f}", f"{asym['n_outliers']} casos ({asym['pct_outliers']}%)")
    a6.metric("Fumadores en Atípicos", f"{asym['smokers_in_outliers_pct']}%", "136 de los 139 casos > umbral")

    smoker_comp_df = pd.DataFrame(
        [
            {
                "Estrato de Asegurados": "No Fumadores (smoker = 'no')",
                "Cantidad (n)": asym["smoker_stats"]["no"]["count"],
                "% del Dataset": f"{asym['smoker_stats']['no']['pct_total']}%",
                "Media charges ($)": f"${asym['smoker_stats']['no']['mean']:,.2f}",
                "Mediana charges ($)": f"${asym['smoker_stats']['no']['median']:,.2f}",
                "Desv. Estándar ($)": f"${asym['smoker_stats']['no']['std']:,.2f}",
                "Mínimo ($)": f"${asym['smoker_stats']['no']['min']:,.2f}",
                "Máximo ($)": f"${asym['smoker_stats']['no']['max']:,.2f}",
            },
            {
                "Estrato de Asegurados": "Fumadores (smoker = 'yes')",
                "Cantidad (n)": asym["smoker_stats"]["yes"]["count"],
                "% del Dataset": f"{asym['smoker_stats']['yes']['pct_total']}%",
                "Media charges ($)": f"${asym['smoker_stats']['yes']['mean']:,.2f}",
                "Mediana charges ($)": f"${asym['smoker_stats']['yes']['median']:,.2f}",
                "Desv. Estándar ($)": f"${asym['smoker_stats']['yes']['std']:,.2f}",
                "Mínimo ($)": f"${asym['smoker_stats']['yes']['min']:,.2f}",
                "Máximo ($)": f"${asym['smoker_stats']['yes']['max']:,.2f}",
            },
        ]
    )
    st.dataframe(smoker_comp_df, width="stretch", hide_index=True)

    st.divider()

    # --- LOS 5 GRÁFICOS OBLIGATORIOS DE EDA ---
    st.subheader("📊 Cinco Visualizaciones Exploratorias Interpretadas (Etapa 5)")

    g_tabs = st.tabs(
        [
            "1️⃣ Distribución y Asimetría de charges",
            "2️⃣ Edad vs. charges (3 Regímenes)",
            "3️⃣ IMC (BMI) vs. charges y Corte OMS",
            "4️⃣ Boxplots por Fumador y Sexo",
            "5️⃣ Correlaciones y Región / Hijos",
        ]
    )

    with g_tabs[0]:
        fig1 = plot_charges_distribution(df)
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)
        st.info(
            "**Interpretación del Gráfico 1:** El panel izquierdo confirma el sesgo positivo (`Skewness = 1.52`): "
            "la concentración principal se ubica entre $1,121 y $14,000 USD, con la mediana ($9,382) muy por debajo de la media ($13,270). "
            "El panel derecho revela el origen estructural de la cola derecha: mientras los no fumadores siguen una distribución unimodal "
            "compacta debajo de $20,000, los fumadores exhiben una distribución **bimodal** con dos lóbulos centrados alrededor de ~$21,000 "
            "y ~$41,000 USD."
        )

    with g_tabs[1]:
        fig2 = plot_age_vs_charges(df)
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)
        st.info(
            "**Interpretación del Gráfico 2:** Al graficar `age` contra `charges` distinguiendo el perfil clínico, emergen con nitidez "
            "**tres bandas paralelas ascendentes**: (1) en la base (verde), los no fumadores cuyo gasto crece suavemente con la edad "
            "(~$250-$270 por año cumplido); (2) en la franja intermedia (naranja, $15,000 a $28,000), los fumadores sin obesidad (`bmi < 30`); "
            "y (3) en la banda superior (rojo, $32,000 a $63,770), los fumadores con obesidad (`bmi >= 30`)."
        )

    with g_tabs[2]:
        fig3 = plot_bmi_vs_charges(df)
        st.pyplot(fig3, width="stretch")
        plt.close(fig3)
        st.info(
            "**Interpretación del Gráfico 3:** En asegurados no fumadores (azul), un aumento del IMC de 20 a 45 kg/m² apenas incrementa "
            "el gasto médico anual. En cambio, en asegurados fumadores (naranja), al cruzar el umbral clínico de obesidad de la OMS "
            "(`BMI = 30 kg/m²`, línea vertical punteada), el gasto salta abruptamente de ~$21,000 a más de ~$40,000 USD. Esta **interacción "
            "no lineal y no aditiva (`smoker × bmi`)** explica por qué `SVR (kernel RBF)` y `KNN` superan ampliamente a `LinearRegression`."
        )

    with g_tabs[3]:
        fig4 = plot_smoker_boxplot(df)
        st.pyplot(fig4, width="stretch")
        plt.close(fig4)
        st.info(
            "**Interpretación del Gráfico 4:** La mediana de los fumadores ($34,456 USD) cuadruplica con creces la mediana de los no fumadores "
            "($7,345 USD), sin solapamiento entre sus cajas intercuartílicas. En el grupo `smoker='no'` se observan puntos aislados por encima "
            "del bigote superior ($18,000-$36,000 USD) que corresponden a episodios hospitalarios fortuitos en no fumadores. El segundo panel "
            "muestra que el sexo (`sex`) tiene un efecto marginal casi nulo en no fumadores y solo una ligera mayor dispersión en hombres fumadores."
        )

    with g_tabs[4]:
        fig5 = plot_multivariate_and_correlation(df)
        st.pyplot(fig5, width="stretch")
        plt.close(fig5)
        st.info(
            r"**Interpretación del Gráfico 5:** El mapa de calor (calculado sobre una copia temporal estrictamente descriptiva) muestra que "
            r"`smoker_yes` posee la mayor correlación lineal con `charges` ($r = 0.79$), seguida de `age` ($r = 0.30$) y `bmi` ($r = 0.20$), "
            r"sin multicolinealidad entre predictores ($|r| \le 0.11$). El gráfico de barras derecho indica que la región `southeast` registra "
            r"el mayor gasto promedio debido a una mayor prevalencia relativa de IMC elevado y tabaquismo."
        )

    st.markdown("#### 💡 Resumen de los 5 Hallazgos Clave del EDA y sus Implicaciones de Modelado")
    findings_df = pd.DataFrame(
        [
            {
                "Hallazgo Exploratorio": "1. Asimetría derecha severa en charges (Skew = 1.52) con 139 valores > $34,489",
                "Evidencia Cuantitativa": "Media ($13,270) >> Mediana ($9,382); 97.84% de extremos son fumadores reales",
                "Decisión de Ingeniería / Modelado": "Conservar todos los registros y adoptar MAE como métrica primaria en CV y GridSearchCV.",
            },
            {
                "Hallazgo Exploratorio": "2. Tres regímenes paralelos según edad, tabaquismo y obesidad",
                "Evidencia Cuantitativa": "No fumadores (~$8.4k), Fumadores BMI<30 (~$21.3k), Fumadores BMI>=30 (~$41.5k)",
                "Decisión de Ingeniería / Modelado": "Evaluar modelos capaces de capturar fronteras no lineales y vecindarios locales (SVR RBF y KNN).",
            },
            {
                "Hallazgo Exploratorio": "3. Interacción crítica de umbral en BMI = 30 kg/m² exclusiva de fumadores",
                "Evidencia Cuantitativa": "Salto de +$20,000 USD al cruzar BMI >= 30 cuando smoker == 'yes'",
                "Decisión de Ingeniería / Modelado": "Evitar modelos puramente aditivos como único enfoque; optimizar C y gamma en SVR.",
            },
            {
                "Hallazgo Exploratorio": "4. Escalas numéricas heterogéneas entre predictores",
                "Evidencia Cuantitativa": "age [18..64], bmi [15.96..53.13], children [0..5] vs dummies [0, 1]",
                "Decisión de Ingeniería / Modelado": "Aplicar StandardScaler obligatorio dentro de ColumnTransformer ajustado solo en train.",
            },
            {
                "Hallazgo Exploratorio": "5. Ruido asimétrico en no fumadores con hospitalizaciones imprevistas",
                "Evidencia Cuantitativa": "Pólizas smoker='no' con cargos de $20,000 a $36,000 sin causa observable en X",
                "Decisión de Ingeniería / Modelado": "Usar pérdida epsilon-insensible (SVR) y analizar residuales por subgrupo en Etapa 13.",
            },
        ]
    )
    st.dataframe(findings_df, width="stretch", hide_index=True)

    return df


# ============================================================================
# PANTALLA 3: VARIABLES Y PREPARACIÓN (ETAPAS 4, 6 Y 7)
# ============================================================================
def render_screen_3_preparacion(df: pd.DataFrame, artifact: dict[str, Any]) -> None:
    """Pantalla 3 (Anexo H): Selección X/y, partición 80/20 T3 y ColumnTransformer sin fuga T4."""
    st.markdown(
        '<span class="stage-badge">Etapa 4: Variable objetivo y predictoras</span>'
        '<span class="stage-badge">Etapa 6: Partición 80/20</span>'
        '<span class="stage-badge">Etapa 7: Preprocesamiento sin fuga</span>'
        '<span class="table-badge">Tabla T3: Partición de datos</span>'
        '<span class="table-badge">Tabla T4: Preprocesamiento</span>',
        unsafe_allow_html=True,
    )
    st.header("3. Selección de Variables, Partición 80/20 y Preprocesamiento sin Fuga")

    X, y = split_features_target(df)

    # --- ETAPA 4 ---
    st.subheader(r"🎯 Etapa 4: Definición de Variable Objetivo ($y$) y Matriz Predictora ($\mathbf{X}$)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dimensión de X (Predictoras)", f"{X.shape[0]:,} × {X.shape[1]}", f"Columnas: {', '.join(FEATURE_COLS)}")
    c2.metric("Dimensión de y (Objetivo)", f"{y.shape[0]:,} × 1", f"Columna: '{TARGET_COL}' (float64)")
    c3.metric("Verificación de Índices y Anti-Fuga", "100% Coincidentes", f"'{TARGET_COL}' excluida de X")

    st.code(
        """# Etapa 4: Separación explícita y verificada de X e y (src/data.py)
FEATURE_COLS = ["age", "bmi", "children", "sex", "smoker", "region"]
TARGET_COL = "charges"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()
assert TARGET_COL not in X.columns and X.index.equals(y.index)""",
        language="python",
    )

    st.markdown("##### Tabla de Justificación de Variables Incluidas y Excluidas (Prevención de Target Leakage)")
    st.dataframe(get_feature_selection_table(), width="stretch", hide_index=True)

    st.divider()

    # --- ETAPA 6 ---
    st.subheader("✂️ Etapa 6: Protocolo de Partición Entrenamiento / Prueba (80% / 20%) — Tabla T3")
    st.markdown(
        """
        Antes de calcular cualquier estadístico de imputación (mediana, moda), escalado (media, desviación estándar)
        o codificación categórica, se divide el conjunto de datos en **80% de entrenamiento (`n = 1,070`)** y
        **20% de prueba reservada (`n = 268`)** empleando `train_test_split` con semilla fija `random_state = 42`.
        - **¿Por qué no se aplica `stratify` directo?** En problemas de regresión con variable respuesta continua (`charges`),
          `train_test_split` no admite estratificación directa sobre `y` continuo; la semilla fija `random_state=42` con
          barajado aleatorio preserva fielmente la media, mediana y dispersión entre ambos subconjuntos, como demuestra la **Tabla T3**.
        """
    )

    st.code(
        """# Etapa 6: Reserva estricta del conjunto de prueba antes del preprocesamiento (src/ml.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)
assert len(set(X_train.index).intersection(set(X_test.index))) == 0""",
        language="python",
    )

    st.markdown("#### Tabla T3 — Partición de Datos y Verificación de 0 Filas Compartidas (Anexo C)")
    st.dataframe(artifact["t3_partition_table"], width="stretch", hide_index=True)

    st.divider()

    # --- ETAPA 7 ---
    st.subheader("⚙️ Etapa 7: Preprocesamiento Diferenciado con `ColumnTransformer` y `Pipeline` — Tabla T4")
    st.markdown(
        """
        Para cumplir con la **Regla de Cero Fuga de Información (Sección 2 de la guía)**, ningún transformador se ajusta
        sobre el dataset completo `df`. Se diseña un objeto `ColumnTransformer` que divide el flujo en dos ramas paralelas:
        1. **Rama Numérica (`age`, `bmi`, `children`):** `SimpleImputer(strategy='median')` $\\rightarrow$ `StandardScaler()`.
        2. **Rama Categórica (`sex`, `smoker`, `region`):** `SimpleImputer(strategy='most_frequent')` $\\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
        """
    )

    st.code(
        """# Etapa 7: Construcción del ColumnTransformer libre de fuga de datos (src/ml.py)
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, ["age", "bmi", "children"]),
        ("cat", categorical_pipeline, ["sex", "smoker", "region"]),
    ],
    remainder="drop",
)""",
        language="python",
    )

    st.markdown("#### Tabla T4 — Diseño del Preprocesamiento por Grupo de Variables (Anexo C)")
    st.dataframe(artifact["t4_preprocessing_table"], width="stretch", hide_index=True)

    # Demostración en vivo de fit_transform(X_train) vs transform(X_test)
    st.markdown("#### 🔬 Verificación Técnica en Vivo: `fit_transform(X_train)` vs. `transform(X_test)`")
    X_train = artifact["X_train"]
    X_test = artifact["X_test"]
    demo_prep = build_preprocessor()
    X_train_trans = demo_prep.fit_transform(X_train)
    X_test_trans = demo_prep.transform(X_test)
    out_feature_names = [name.replace("num__", "").replace("cat__", "") for name in demo_prep.get_feature_names_out()]

    d1, d2, d3 = st.columns(3)
    d1.metric("Matriz X_train Transformada", f"{X_train_trans.shape[0]} × {X_train_trans.shape[1]}", "Ajustada con fit_transform(X_train)")
    d2.metric("Matriz X_test Transformada", f"{X_test_trans.shape[0]} × {X_test_trans.shape[1]}", "Solo transform(X_test) sin reajustar")
    d3.metric("Columnas Resultantes", f"{len(out_feature_names)} columnas", "3 escaladas + 8 indicadores One-Hot")

    trans_preview_df = pd.DataFrame(X_train_trans[:5], columns=out_feature_names).round(4)
    st.markdown("##### Primeras 5 filas de `X_train` tras aplicar `ColumnTransformer` (3 numéricas estandarizadas + 8 binarias One-Hot)")
    st.dataframe(trans_preview_df, width="stretch", hide_index=True)


# ============================================================================
# PANTALLA 4: MODELOS Y COMPARACIÓN (ETAPAS 8, 9, 10, 11 Y 12)
# ============================================================================
def render_screen_4_modelos(artifact: dict[str, Any]) -> None:
    """Pantalla 4 (Anexo H): Algoritmos, Pipelines, CV 5-Fold T5, GridSearchCV T6 y Decisión previa al test T7."""
    st.markdown(
        '<span class="stage-badge">Etapa 8: Fundamentos de algoritmos</span>'
        '<span class="stage-badge">Etapa 9: Entrenamiento reproducible</span>'
        '<span class="stage-badge">Etapa 10: Validación cruzada 5-Fold</span>'
        '<span class="stage-badge">Etapa 11: Optimización GridSearchCV</span>'
        '<span class="stage-badge">Etapa 12: Selección previa al test</span>'
        '<span class="table-badge">Tabla T5</span>'
        '<span class="table-badge">Tabla T6</span>'
        '<span class="table-badge">Tabla T7</span>',
        unsafe_allow_html=True,
    )
    st.header("4. Modelos Supervisados, Validación Cruzada (5-Fold), Optimización y Selección")

    # --- ETAPA 8 ---
    st.subheader("🧠 Etapa 8: Fundamentos Técnicos de los Tres Algoritmos Competidores y el Baseline")
    st.markdown(
        """
        Se evalúan los **tres estimadores de regresión obligatorios de Scikit-Learn** junto con un modelo base de referencia:
        1. **`DummyRegressor(strategy='mean')` (Baseline no competidor):** Predice siempre el gasto promedio de entrenamiento. Establece el techo de error admisible.
        2. **`LinearRegression` (Mínimos Cuadrados Ordinarios):** Ajusta una combinación lineal aditiva $\\hat{y} = \\mathbf{w}^\\top \\mathbf{x} + b$. Altamente interpretable, pero limitado frente a sinergias multiplicativas como `smoker × bmi`.
        3. **`KNeighborsRegressor` (Vecinos Más Cercanos):** Método no paramétrico basado en distancias en el espacio preprocesado de 11 dimensiones. Captura vecindarios clínicos locales (ej. fumadores obesos de edad similar).
        4. **`SVR` (Support Vector Regression):** Minimiza una función de pérdida $\\varepsilon$-insensible (norma $L_1$ fuera del tubo $\\varepsilon$) proyectando los datos mediante un kernel (lineal, polinomial o gaussiano RBF).
        """
    )
    st.dataframe(artifact["algorithms_table"], width="stretch", hide_index=True)

    st.divider()

    # --- ETAPA 9 ---
    st.subheader("🔁 Etapa 9: Implementación Reproducible de Pipelines y Entorno de Ejecución")
    c_env1, c_env2 = st.columns([1.4, 1.0])
    with c_env1:
        st.code(
            """# Etapa 9: Cada estimador se encapsula en su propio Pipeline (src/ml.py)
pipelines = {
    "DummyRegressor (Baseline)": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", DummyRegressor(strategy="mean")),
    ]),
    "LinearRegression": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LinearRegression()),
    ]),
    "KNeighborsRegressor": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", KNeighborsRegressor()),
    ]),
    "SVR": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", SVR()),
    ]),
}""",
            language="python",
        )
    with c_env2:
        env_df = pd.DataFrame(
            [
                {"Parámetro de Reproducibilidad": "Semilla Aleatoria Global (random_state)", "Valor": str(RANDOM_STATE)},
                {"Parámetro de Reproducibilidad": "Esquema de Validación Cruzada", "Valor": f"KFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})"},
                {"Parámetro de Reproducibilidad": "Versión de Python", "Valor": sys.version.split()[0]},
                {"Parámetro de Reproducibilidad": "Versión de Scikit-Learn", "Valor": sklearn.__version__},
                {"Parámetro de Reproducibilidad": "Versión de Pandas / NumPy", "Valor": f"{pd.__version__} / {np.__version__}"},
                {"Parámetro de Reproducibilidad": "Sistema Operativo", "Valor": f"{platform.system()} ({platform.release()})"},
            ]
        )
        st.dataframe(env_df, width="stretch", hide_index=True)

    st.divider()

    # --- ETAPA 10: VALIDACIÓN CRUZADA (TABLA T5) ---
    st.subheader("📊 Etapa 10: Comparación Justa con Validación Cruzada de 5 Folds en Entrenamiento — Tabla T5")
    st.markdown(
        """
        Los cuatro pipelines se evalúan **exclusivamente sobre `X_train, y_train` (`n = 1,070`)** utilizando exactamente las mismas
        5 particiones (`KFold(n_splits=5, shuffle=True, random_state=42)`). En cada iteración, el `ColumnTransformer` se ajusta
        solo con los 4 folds de entrenamiento (856 registros) y transforma el fold de validación (214 registros).
        """
    )
    st.markdown("#### Tabla T5 — Resultados de Validación Cruzada 5-Fold Inicial (Métrica Primaria: MAE en USD $)")
    st.dataframe(artifact["t5_cv_table"], width="stretch", hide_index=True)
    st.info(
        "**Análisis de la Validación Cruzada Inicial (Etapa 10):** Con hiperparámetros por defecto, **`KNeighborsRegressor` (`k=5`, distancia euclidiana)** "
        "alcanza el mejor desempeño inicial (**Media MAE = $3,314.12 ± $183.75 USD**, $R^2 = 0.7745$), superando a **`LinearRegression`** "
        "(**Media MAE = $4,251.40 ± $188.09 USD**, $R^2 = 0.7338$). Por su parte, **`SVR` por defecto (`C=1.0`)** sufre subajuste severo "
        "(**Media MAE = $8,266.66 USD**) porque una penalización $C=1$ sólo permite desplazar las predicciones unos pocos dólares alrededor "
        "de la mediana cuando la variable objetivo se mide en decenas de miles de dólares."
    )

    st.divider()

    # --- ETAPA 11: OPTIMIZACIÓN GRIDSEARCHCV (TABLA T6) ---
    st.subheader("🎛️ Etapa 11: Ajuste de Hiperparámetros con `GridSearchCV` (Solo en Entrenamiento) — Tabla T6")
    st.markdown(
        """
        Se ejecuta `GridSearchCV` (con `refit='mae'`, optimizando `neg_mean_absolute_error` sobre los mismos 5 folds de `X_train`)
        para los dos algoritmos con hiperparámetros críticos:
        - **Rejilla de `KNeighborsRegressor` (24 combinaciones $\\times$ 5 folds = 120 ajustes):**
          `n_neighbors`: `[3, 5, 7, 9, 11, 15]`, `weights`: `['uniform', 'distance']`, `p`: `[1 (Manhattan), 2 (Euclidiana)]`.
        - **Rejilla de `SVR` (72 combinaciones $\\times$ 5 folds = 360 ajustes):**
          `kernel`: `['linear', 'rbf', 'poly']`, `C`: `[100, 1000, 5000, 10000]`, `gamma`: `['scale', 'auto', 0.1]`, `epsilon`: `[1.0, 50.0]`.
        """
    )

    st.markdown("#### Tabla T6 — Comparación Antes vs. Después de `GridSearchCV` en Validación Cruzada (Anexo C)")
    st.dataframe(artifact["t6_tuning_table"], width="stretch", hide_index=True)

    col_top_knn, col_top_svr = st.columns(2)
    with col_top_knn:
        st.markdown("##### Top 5 Configuraciones de `KNeighborsRegressor` en 5-Fold CV")
        st.dataframe(artifact["tuning_results"]["knn_top5"], width="stretch", hide_index=True)
    with col_top_svr:
        st.markdown("##### Top 5 Configuraciones de `SVR` en 5-Fold CV")
        st.dataframe(artifact["tuning_results"]["svr_top5"], width="stretch", hide_index=True)

    fig_cv = plot_cv_comparison(artifact["t5_cv_table"], artifact["tuning_results"])
    st.pyplot(fig_cv, width="stretch")
    plt.close(fig_cv)

    st.divider()

    # --- ETAPA 12: SELECCIÓN DEL MODELO GANADOR ANTES DEL TEST (TABLA T7) ---
    st.subheader("🏆 Etapa 12: Selección Razonada del Modelo Ganador Antes del Conjunto de Prueba — Tabla T7")
    st.markdown("#### Tabla T7 — Matriz de Decisión Previa al Test (Anexo C)")
    st.dataframe(artifact["t7_decision_table"], width="stretch", hide_index=True)

    st.success(
        f"""
        **Justificación Razonada de Selección Previa al Test (`{artifact['winner_name']}`):**

        Antes de desbloquear el conjunto de prueba reservado (`X_test`), la evidencia experimental obtenida exclusivamente mediante
        validación cruzada de 5 folds sobre `X_train` demuestra que **`SVR (Optimizado)`** con configuración
        `{{'C': 10000, 'epsilon': 50.0, 'gamma': 0.1, 'kernel': 'rbf'}}` es el modelo ganador indiscutible por cuatro razones técnicas:

        1. **Superioridad en la métrica primaria acordada (`MAE`):** Alcanza una **Media MAE en CV de $2,424.16 USD**, reduciendo el error
           absoluto medio en **$686.57 USD por póliza (-22.07%)** frente a `KNeighborsRegressor (Optimizado)` ($3,110.73 USD) y en
           **$1,827.24 USD (-42.98%)** frente a `LinearRegression` ($4,251.40 USD).
        2. **Mayor poder explicativo global y menor error cuadrático:** Obtiene el mayor coeficiente de determinación en validación cruzada
           (**$R^2 = 0.8240$**) y el menor `RMSE` (**$4,946.46 USD** frente a $5,196.46 de KNN y $6,135.14 de regresión lineal).
        3. **Estabilidad inter-fold:** Presenta una desviación estándar entre folds de **$166.96 USD** (coeficiente de variación de apenas
           **6.89%**), dominando consistentemente a `LinearRegression` y `KNN` en cada uno de los 5 folds individuales (Panel B del gráfico).
        4. **Adecuación matemática a la estructura de `insurance.csv`:** El kernel gaussiano RBF modela con suavidad la interacción no lineal
           entre `smoker='yes'` y `bmi >= 30`, mientras que la función de pérdida $\\varepsilon$-insensible (norma $L_1$) impide que los
           casos aislados de hospitalizaciones fortuitas en no fumadores distorsionen el hiperplano de regresión, y el costo de ajuste
           (~0.06 segundos) es instantáneo para producción.
        """
    )

    with st.expander("⚙️ Re-ejecución Opcional en Vivo del Flujo de Entrenamiento y Exportación (`src/ml.py`)", expanded=False):
        st.caption(
            "El artefacto `models/modelo_final.joblib` ya se encuentra pre-calculado y verificado con `random_state=42`. "
            "Si desea re-ejecutar en vivo la validación cruzada, las rejillas de `GridSearchCV` y regenerar el archivo `.joblib`, "
            "pulse el siguiente botón."
        )
        if st.button("🔄 Re-entrenar Pipelines y Exportar `models/modelo_final.joblib`"):
            with st.spinner("Ejecutando 5-Fold CV y GridSearchCV en X_train..."):
                train_and_export_all(DEFAULT_DATA_PATH, MODEL_PATH)
                st.cache_resource.clear()
                st.cache_data.clear()
            st.success("Entrenamiento reproducible finalizado y artefacto actualizado en `models/modelo_final.joblib`.")
            st.rerun()


# ============================================================================
# PANTALLA 5: EVALUACIÓN, ANÁLISIS CRÍTICO Y CONCLUSIONES (ETAPAS 13, 14 Y 16)
# ============================================================================
def render_screen_5_evaluacion(artifact: dict[str, Any]) -> None:
    """Pantalla 5 (Anexo H): Evaluación final única en test T8, análisis Grupo 09 fumadores vs no fumadores, Etapa 14 y Etapa 16."""
    st.markdown(
        '<span class="stage-badge">Etapa 13: Evaluación final en test</span>'
        '<span class="stage-badge">Etapa 14: Análisis crítico y limitaciones</span>'
        '<span class="stage-badge">Etapa 16: Conclusiones y recomendaciones</span>'
        '<span class="table-badge">Tabla T8: Prueba final</span>',
        unsafe_allow_html=True,
    )
    st.header("5. Evaluación Final en Test Reservado, Análisis Grupo 09 y Conclusiones")

    tm = artifact["test_metrics"]
    res_stats = artifact["residual_stats"]

    # --- ETAPA 13 ---
    st.subheader("🏁 Etapa 13: Evaluación Única en el Conjunto de Prueba Reservado (`n = 268`, 20%) — Tabla T8")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric(
        "MAE en Test (Primaria)",
        f"${tm['mae']:,.2f}",
        f"-${tm['mae_reduction_usd']:,.2f} (-{tm['mae_reduction_pct']:.1f}%) vs Baseline",
        delta_color="inverse",
    )
    e2.metric("RMSE en Test (Secundaria)", f"${tm['rmse']:,.2f}", f"Baseline RMSE: ${tm['baseline_rmse']:,.2f}")
    e3.metric("R² en Test (Varianza Explicada)", f"{tm['r2']:.4f}", f"Supera meta ex-ante (R² >= 0.75)")
    e4.metric("Mediana del Error Absoluto", f"${res_stats['median_abs_error']:,.2f}", "50% de pólizas con error < $342")

    st.markdown("#### Tabla T8 — Prueba Final del Modelo Ganador vs. Baseline en Test Reservado (Anexo C)")
    st.dataframe(artifact["t8_final_test_table"], width="stretch", hide_index=True)

    with st.expander("📋 Verificación Post-Hoc Referencial de Todos los Modelos en Test y Muestra de 10 Predicciones Reales vs. Predichas", expanded=True):
        c_t1, c_t2 = st.columns([1.0, 1.4])
        with c_t1:
            st.markdown("##### Comparativa Referencial en Test (sin alterar la decisión de Etapa 12)")
            st.dataframe(artifact["all_test_summary_df"], width="stretch", hide_index=True)
        with c_t2:
            st.markdown("##### Muestra de 10 Filas del Conjunto de Prueba (`Real vs. Predicho vs. Residual`)")
            st.dataframe(artifact["sample_10_df"], width="stretch", hide_index=True)

    col_g1, col_g2 = st.columns([1.0, 1.25])
    with col_g1:
        fig_rvp = plot_real_vs_pred(
            artifact["y_test"],
            artifact["y_pred_winner"],
            artifact["X_test"],
            artifact["winner_name"],
        )
        st.pyplot(fig_rvp, width="stretch")
        plt.close(fig_rvp)
    with col_g2:
        fig_res = plot_residuals_analysis(
            artifact["y_test"],
            artifact["y_pred_winner"],
            artifact["X_test"],
        )
        st.pyplot(fig_res, width="stretch")
        plt.close(fig_res)

    st.info(
        "**Interpretación de Real vs. Predicho y Residuales (Etapa 13):** El grueso de las 268 pólizas de prueba se alinea de manera "
        "sobresaliente sobre la diagonal de identidad perfecta ($\\hat{y} = y$), logrando una **mediana de error absoluto de apenas "
        f"${res_stats['median_abs_error']:,.2f} USD** (la mitad de las predicciones en test difiere en menos de $342 USD del valor real). "
        "El gráfico de residuales muestra un pico agudo centrado en $0 USD y un pequeño conjunto de residuales positivos altos "
        "($y_{\\text{real}} - \\hat{y} > \\$10,000$) que corresponden a asegurados cuyos costos hospitalarios reales superaron ampliamente "
        "lo predecible por su perfil demográfico básico."
    )

    st.divider()

    # --- REQUISITO ANALÍTICO ESPECÍFICO DEL GRUPO 09 ---
    st.subheader("🔬 Requisito Analítico Grupo 09: Comparación del MAE entre Fumadores y No Fumadores y por Nivel de Gasto")
    st.markdown(
        """
        <div class="group02-box">
            <b>Hallazgo Diferencial en Subgrupos (Grupo 09 — Anexo B y Sección 5):</b><br>
            Al desagregar el desempeño del modelo ganador (<code>SVR Optimizado</code>) en el conjunto de prueba según la condición
            de tabaquismo y el nivel de gasto real, se descubren dos comportamientos complementarios:<br>
            • <b>No Fumadores (<code>smoker = 'no'</code>, n = 214 en test):</b> Presentan un <b>MAE nominal menor ($1,363.24 USD)</b> y una
              <b>mediana de error extraordinariamente baja ($257.33 USD)</b>. Sin embargo, su RMSE ($4,768.25) es 3.5 veces mayor que su MAE
              debido a ~12 pólizas de no fumadores que tuvieron hospitalizaciones agudas inesperadas (&gt; $20,000 USD).<br>
            • <b>Fumadores (<code>smoker = 'yes'</code>, n = 54 en test):</b> Presentan un <b>MAE nominal mayor ($3,547.97 USD)</b> simplemente
              porque su gasto médico promedio real es <b>3.82 veces más alto ($32,050.23 vs. $8,384.59 USD)</b>. En términos relativos al gasto
              medio de su grupo, el modelo es <b>más preciso en fumadores (MAE relativo = 11.07% e $R^2$ intra-grupo = 0.8075)</b> que en no
              fumadores (MAE relativo = 16.26%).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 1. Comparación de Errores en Test: Fumadores (`smoker='yes'`) vs. No Fumadores (`smoker='no'`)")
    st.dataframe(artifact["smoker_mae_table"], width="stretch", hide_index=True)

    st.markdown("##### 2. Comparación de Errores en Test por Niveles de Gasto Real (`Bajo < $5k`, `Medio $5k-$15k`, `Alto > $15k`)")
    st.dataframe(artifact["spending_level_mae_table"], width="stretch", hide_index=True)

    fig_smk = plot_smoker_error_comparison(
        artifact["X_test"],
        artifact["y_test"],
        artifact["y_pred_winner"],
    )
    st.pyplot(fig_smk, width="stretch")
    plt.close(fig_smk)

    st.markdown(
        """
        <div class="causal-warning">
            <b>⚠️ Aclaración Metodológica Explícita sobre No Causalidad (Requisito Grupo 09):</b><br>
            La diferencia observada en los gastos predichos y en el MAE entre fumadores y no fumadores refleja una
            <b>asociación estadística y actuarial en el conjunto de datos observacional</b>, pero <b>NO demuestra causalidad directa</b>
            por tres razones metodológicas:<br>
            1. <b>Diseño observacional transversal (no experimental):</b> No existe asignación aleatoria del hábito tabáquico ni seguimiento longitudinal.<br>
            2. <b>Variables de confusión omitidas (Omitted Variable Bias):</b> El dataset carece de información clínica crítica (antecedentes
               oncológicos/cardiovasculares familiares, consumo de alcohol, riesgos laborales, sedentarismo o tipo de plan contratado) que pueden
               correlacionarse simultáneamente con fumar y con mayores cargos médicos.<br>
            3. <b>Naturaleza administrativa de <code>charges</code>:</b> La variable objetivo mide montos facturados según reglas tarifarias de la
               aseguradora, no directamente el estado biológico de salud del paciente.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- ETAPA 14: ANÁLISIS CRÍTICO, LIMITACIONES Y PREGUNTAS DE SUSTENTACIÓN ---
    st.subheader("🧐 Etapa 14: Análisis Crítico de Resultados, Limitaciones y Sustentación Metodológica")

    crit_col1, crit_col2 = st.columns(2)
    with crit_col1:
        st.markdown(
            r"""
            #### ¿Para qué sirve bien el modelo y cuándo NO debe usarse?
            - **Usos donde el modelo sobresale:**
              1. Estimación de **reservas técnicas agregadas** y presupuesto anual de siniestralidad para carteras de asegurados adultos (18 a 64 años).
              2. Cotización referencial de **prima pura de riesgo** en el régimen basal de asegurados saludables y en los regímenes de tabaquismo/obesidad.
              3. Identificación temprana de perfiles de alta sinergia metabólica (`smoker='yes'` + `bmi >= 30`) para programas preventivos de salud.
            - **Situaciones donde el modelo NO sirve o falla:**
              1. **Eventos agudos fortuitos en no fumadores:** El modelo subestima pólizas de no fumadores jóvenes que sufren accidentes de tránsito,
                 apendicectomías complicadas o diagnósticos súbitos graves (residuales de $+\$15,000$ a $+\$24,000$), pues esos eventos son
                 estocásticos respecto a las 6 variables demográficas de entrada.
              2. **Extrapolación fuera del dominio de entrenamiento:** No es válido para menores de 18 años, adultos mayores en régimen *Medicare*
                 ($> 65$ años) ni geografías o sistemas de salud fuera de EE. UU. con estructuras de precios hospitalarios distintas.
            """
        )
    with crit_col2:
        st.markdown(
            """
            #### Limitaciones Internas y Externas del Estudio
            - **Limitaciones internas (del dataset `insurance.csv`):**
              - Ausencia de variables clínicas esenciales: historial de enfermedades crónicas (diabetes, hipertensión), hospitalizaciones del año
                anterior, presión arterial, perfil lipídico y tipo de cobertura/deducible del plan.
              - Tamaño muestral moderado (`n = 1,338`), con solo 274 fumadores totales (214 en train y 54 en test), lo que incrementa la varianza
                absoluta en el estrato de gasto alto.
            - **Limitaciones externas (validez ecológica y temporal):**
              - **Inflación médica (*Medical Cost Drift*):** Los montos en dólares corresponden a una cohorte histórica fija; su despliegue actual
                requiere indexación actuarial por inflación hospitalaria.
              - **Sesgo de declaración en `smoker`:** En contextos reales de suscripción, un solicitante puede ocultar el consumo ocasional de tabaco
                para evitar recargos si no se realiza prueba bioquímica de cotinina.
            """
        )

    with st.expander("🎓 Respuestas Completas a las 8 Preguntas de Sustentación Oral (Sección 6 de `ejercicio.md`)", expanded=False):
        st.markdown(
            """
            1. **¿Por qué se aplicó `StandardScaler` dentro de un `Pipeline` y no sobre todo el dataset antes del `train_test_split`?**
               Porque calcular la media y desviación estándar sobre las 1,338 filas antes de partir contamina el entrenamiento con información
               estadística del conjunto de prueba (*data leakage*), produciendo métricas artificialmente optimistas en algoritmos basados en
               distancias (`KNN`) y productos internos (`SVR`).
            2. **¿Por qué se eligió `MAE` como métrica primaria en lugar de `MSE` o `RMSE`?**
               Porque `charges` presenta una fuerte asimetría positiva (`Skewness = 1.52`) con 139 valores extremos reales ($> \\$34,489$).
               El `MSE`/`RMSE` eleva al cuadrado los residuales de hospitalizaciones fortuitas e induce al modelo a sobrepredecir a la mayoría
               de asegurados sanos; el `MAE` mide el error promedio directo en dólares por póliza con robustez ante valores extremos.
            3. **¿Por qué `SVR` tuvo un desempeño deficiente antes de `GridSearchCV` y pasó a ser el ganador después?**
               Porque el hiperparámetro por defecto `C = 1.0` impone una regularización extrema que limita las predicciones a un rango estrecho
               alrededor de la mediana (~$9,382 USD), subajustando una variable que llega a $63,770 USD. Al ampliar `C = 10,000` con `kernel = 'rbf'`
               y `gamma = 0.1`, el modelo adquiere la capacidad de ajustarse a los tres regímenes de gasto reduciendo su MAE CV de $8,266.66 a $2,424.16.
            4. **¿Por qué `SVR (Optimizado)` y `KNN (Optimizado)` superan por más de $1,800 USD de MAE a `LinearRegression`?**
               Porque `LinearRegression` asume aditividad estricta ($\\Delta \\text{charges}$ constante al fumar, sin importar el IMC). En los datos
               reales, tener `bmi >= 30` casi no aumenta el gasto en no fumadores, pero dispara el costo en +$20,000 USD en fumadores. El kernel RBF
               y los vecindarios locales capturan esa interacción no lineal sin necesidad de especificar términos polinomiales manuales.
            5. **¿Por qué no se eliminaron los 139 valores atípicos de `charges` detectados por el criterio IQR?**
               Porque el 97.84% de ellos (136 de 139) corresponden a asegurados fumadores con obesidad. Eliminarlos mutilaría el fenómeno real de
               alta siniestralidad que la aseguradora necesita prever y sesgaría gravemente las reservas técnicas hacia abajo.
            6. **¿Por qué el MAE en dólares es mayor en fumadores ($3,547.97) que en no fumadores ($1,363.24), pero el modelo es más explicativo en fumadores?**
               Porque los fumadores operan en una escala monetaria casi 4 veces mayor (gasto medio de $32,050 vs. $8,384 USD). Al normalizar por el
               gasto medio de cada subgrupo, el error relativo en fumadores es menor (**11.07%** vs. **16.26%** en no fumadores) y el $R^2$ intra-grupo
               alcanza **0.8075** frente a **0.4150** en no fumadores (donde pesan las cirugías imprevistas).
            7. **¿Por qué no podemos afirmar que fumar "causa" exactamente el incremento en dólares estimado por el modelo?**
               Porque el modelo estima una esperanza condicional $\\mathbb{E}[Y \\mid X]$ sobre datos observacionales transversales, no un efecto causal
               intervencionista $\\mathbb{E}[Y \\mid \\text{do}(\\text{smoker}=\\text{yes})]$, existiendo confusores no medidos y reglas tarifarias administrativas.
            8. **¿Cómo se garantiza que en la Pantalla 6 (Predicciones) no ocurra fuga ni inconsistencia de columnas al evaluar un nuevo cliente?**
               Serializando el objeto `Pipeline` completo (`ColumnTransformer` + `SVR`) en `models/modelo_final.joblib` e invocando únicamente
               `pipeline.predict(df_nuevo)`, de modo que el nuevo registro se impute, escale y codifique con los parámetros aprendidos en `X_train`.
            """
        )

    st.divider()

    # --- ETAPA 16: CONCLUSIONES Y RECOMENDACIONES ---
    st.subheader("✅ Etapa 16: Conclusiones Respaldadas por Evidencia, Mejoras Futuras y Decisión de Despliegue")

    concl_df = pd.DataFrame(
        [
            {
                "N°": "C1",
                "Conclusión Técnica y de Negocio": (
                    "El modelo ganador SVR (Optimizado con kernel RBF, C=10000, gamma=0.1, epsilon=50) supera holgadamente "
                    "todos los criterios cuantitativos de éxito establecidos en la Etapa 2."
                ),
                "Evidencia Cuantitativa Verificable": (
                    f"En test reservado (n=268): MAE = ${tm['mae']:,.2f} (< meta $3,500), R² = {tm['r2']:.4f} (>= meta 0.75), "
                    f"reduciendo el MAE del baseline (${tm['baseline_mae']:,.2f}) en un {tm['mae_reduction_pct']:.2f}%."
                ),
            },
            {
                "N°": "C2",
                "Conclusión Técnica y de Negocio": (
                    "La captura de interacciones no lineales (tabaquismo × obesidad BMI >= 30) es indispensable para "
                    "estimar gastos médicos con alta precisión."
                ),
                "Evidencia Cuantitativa Verificable": (
                    "En 5-Fold CV sobre train, SVR Optimizado ($2,424.16) y KNN Optimizado ($3,110.73) reducen el MAE de "
                    "LinearRegression ($4,251.40) en 42.98% y 26.83% respectivamente."
                ),
            },
            {
                "N°": "C3",
                "Conclusión Técnica y de Negocio": (
                    "La asimetría de charges (Skewness = 1.52) obedece a tres regímenes clínicos reales y no a errores "
                    "de captura; la pérdida epsilon-insensible logra una precisión mediana excepcional en la cartera típica."
                ),
                "Evidencia Cuantitativa Verificable": (
                    f"El 97.84% de los 139 valores > $34,489 son fumadores; en test, la mediana del error absoluto de SVR "
                    f"es de solo ${res_stats['median_abs_error']:,.2f} USD por póliza."
                ),
            },
            {
                "N°": "C4",
                "Conclusión Técnica y de Negocio": (
                    "El error absoluto nominal es mayor en fumadores por efecto de escala monetaria, pero el error "
                    "relativo es menor y la principal fuente de error extremo reside en hospitalizaciones fortuitas de no fumadores."
                ),
                "Evidencia Cuantitativa Verificable": (
                    "En test: MAE relativo de 11.07% en fumadores (R² intra-grupo = 0.8075) frente a 16.26% en no fumadores "
                    "(donde el RMSE sube a $4,768.25 por siniestros agudos no observables en X)."
                ),
            },
            {
                "N°": "C5",
                "Conclusión Técnica y de Negocio": (
                    "La arquitectura de Pipeline serializada garantiza cero fuga de información durante la validación y "
                    "trazabilidad total en inferencia sobre nuevos asegurados."
                ),
                "Evidencia Cuantitativa Verificable": (
                    "0 filas compartidas entre train (1,070) y test (268); consistencia entre MAE CV ($2,424.16) y MAE Test ($1,803.44)."
                ),
            },
        ]
    )
    st.dataframe(concl_df, width="stretch", hide_index=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            r"""
            #### 🚀 Tres Mejoras Futuras Priorizadas
            1. **Prioridad 1 (Datos Clínicos e Historial Previo):** Incorporar diagnósticos preexistentes (código CIE-10 de hipertensión, diabetes),
               siniestralidad del año $t-1$ y nivel de deducible/copago de la póliza para explicar los picos de $+\$20,000$ en no fumadores.
            2. **Prioridad 2 (Intervalos de Predicción Conformal / Regresión Cuantílica):** Complementar la estimación puntual de `SVR` con intervalos
               de predicción al 90% (*Conformal Prediction*) para cuantificar el margen de solvencia y recargo de seguridad actuarial por póliza.
            3. **Prioridad 3 (Exploración de Ensambles de Gradiente e Interacciones Explícitas):** Evaluar `HistGradientBoostingRegressor` y términos
               de interacción explícitos $\mathbb{I}(\text{BMI} \ge 30) \times \mathbb{I}(\text{smoker}=\text{'yes'})$ con explicabilidad SHAP.
            """
        )
    with r2:
        st.markdown(
            r"""
            #### 🏛️ Decisión Argumentada: ¿Listo para Uso Experimental o Producción?
            - **Dictamen:** **Apto para Despliegue en Entorno Experimental / Piloto Actuarial de Soporte a la Decisión (*Shadow Mode*)**,
              pero **no para tarificación individual 100% automatizada sin supervisión humana**.
            - **Argumentación técnica y regulatoria:** Aunque el rendimiento estadístico en datos fuera de muestra es sobresaliente
              ($\text{MAE} = \$1,803.44$, $R^2 = 0.8496$), un despliegue productivo directo en una aseguradora real exige previamente:
              (a) recalibrar los montos monetarios según la inflación médica del año en curso,
              (b) auditar el cumplimiento normativo sobre uso de variables como `sex` según la jurisdicción local, y
              (c) acoplar reglas actuariales de reaseguro (*stop-loss*) para cubrir los siniestros fortuitos de cola derecha.
            """
        )


# ============================================================================
# PANTALLA 6: PREDICCIONES EN VIVO Y TABLA T9 (ETAPA 15)
# ============================================================================
def render_screen_6_predicciones(df: pd.DataFrame, artifact: dict[str, Any]) -> None:
    """Pantalla 6 (Anexo H): Formulario interactivo validado con src/predict.py y Tabla T9 de 5 casos nuevos."""
    st.markdown(
        '<span class="stage-badge">Etapa 15: Predicción con datos nuevos</span>'
        '<span class="table-badge">Tabla T9: Cinco predicciones nuevas plausibles</span>',
        unsafe_allow_html=True,
    )
    st.header("6. Simulador Interactivo de Pólizas y Demostración con Datos Nuevos (Etapa 15)")

    st.markdown(
        f"""
        Esta pantalla ejecuta inferencia en tiempo real invocando **`src/predict.py`** sobre el objeto `Pipeline` ganador
        serializado en **`{MODEL_PATH}`** (**`{artifact['winner_name']}`**).
        - **Garantía de no reentrenamiento:** Al presionar **Predecir Gasto Médico**, el sistema valida los tipos y rangos de entrada,
          construye un `DataFrame` con las 6 columnas originales (`age`, `bmi`, `children`, `sex`, `smoker`, `region`) y ejecuta
          exclusivamente `pipeline.predict(df_nuevo)`, aplicando los parámetros de imputación, escalado (`StandardScaler`) y
          codificación (`OneHotEncoder`) aprendidos en `X_train`.
        """
    )

    col_form, col_result = st.columns([1.05, 1.25])

    with col_form:
        st.subheader("📝 Formulario de Cotización de Nuevo Asegurado")
        with st.form("prediction_form"):
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                age_in = st.number_input(
                    "Edad del titular (age, años)",
                    min_value=18,
                    max_value=100,
                    value=35,
                    step=1,
                    help="Rango observado en el dataset: 18 a 64 años (admite hasta 100 años).",
                )
                bmi_in = st.number_input(
                    "Índice de Masa Corporal (bmi, kg/m²)",
                    min_value=12.0,
                    max_value=65.0,
                    value=28.5,
                    step=0.1,
                    format="%.1f",
                    help="Corte clínico de obesidad OMS: BMI >= 30.0 kg/m².",
                )
                children_in = st.number_input(
                    "Hijos / dependientes cubiertos (children)",
                    min_value=0,
                    max_value=15,
                    value=1,
                    step=1,
                    help="Número de dependientes inscritos en la póliza (0 a 5 en dataset original).",
                )
            with f_c2:
                sex_label = st.selectbox(
                    "Sexo biológico registrado (sex)",
                    options=["female (Femenino)", "male (Masculino)"],
                    index=0,
                )
                smoker_label = st.selectbox(
                    "¿Fumador activo? (smoker)",
                    options=["no (No Fumador)", "yes (Fumador)"],
                    index=0,
                )
                region_label = st.selectbox(
                    "Región de residencia (region)",
                    options=[
                        "northwest (Noroeste)",
                        "northeast (Noreste)",
                        "southeast (Sureste)",
                        "southwest (Suroeste)",
                    ],
                    index=0,
                )

            submitted = st.form_submit_button("🧮 Predecir Gasto Médico Anual (charges)", width="stretch")

    sex_val = sex_label.split()[0]
    smoker_val = smoker_label.split()[0]
    region_val = region_label.split()[0]

    with col_result:
        st.subheader("📊 Resultado de la Estimación Actuarial")
        is_valid, val_errors, df_single = validate_input(
            age=age_in,
            sex=sex_val,
            bmi=bmi_in,
            children=children_in,
            smoker=smoker_val,
            region=region_val,
        )

        if not is_valid:
            for err in val_errors:
                st.error(err)
        else:
            pred_usd = predict_single(
                age=age_in,
                sex=sex_val,
                bmi=bmi_in,
                children=children_in,
                smoker=smoker_val,
                region=region_val,
                model_path=MODEL_PATH,
            )
            mean_global = float(df[TARGET_COL].mean())
            median_global = float(df[TARGET_COL].median())
            diff_vs_median = pred_usd - median_global

            # Clasificación clínica OMS de IMC y estrato de riesgo
            if bmi_in < 18.5:
                bmi_cat = "Bajo peso (< 18.5)"
            elif bmi_in < 25.0:
                bmi_cat = "Normopeso (18.5 - 24.9)"
            elif bmi_in < 30.0:
                bmi_cat = "Sobrepeso (25.0 - 29.9)"
            else:
                bmi_cat = "Obesidad OMS (>= 30.0)"

            if smoker_val == "no":
                risk_regime = "Régimen 1 — No Fumador (Riesgo Basal Moderado)"
            elif bmi_in < 30.0:
                risk_regime = "Régimen 2 — Fumador sin Obesidad (Riesgo Intermedio-Alto)"
            else:
                risk_regime = "Régimen 3 — Sinergia Crítica: Fumador con Obesidad (Riesgo Muy Alto)"

            r_m1, r_m2, r_m3 = st.columns(3)
            r_m1.metric(
                "Gasto Anual Predicho (ŷ)",
                f"${pred_usd:,.2f} USD",
                f"{'+' if diff_vs_median >= 0 else ''}${diff_vs_median:,.2f} vs. Mediana ($9,382)",
                delta_color="inverse",
            )
            r_m2.metric("Equivalente Mensual Puro", f"${pred_usd / 12.0:,.2f} / mes", "Prima pura sin recargo adm.")
            r_m3.metric("Categoría IMC (OMS)", bmi_cat, f"BMI = {bmi_in:.1f} kg/m²")

            st.info(
                f"**Diagnóstico del Perfil Cotizado:**\n"
                f"- **Estrato Actuarial Identificado:** `{risk_regime}`\n"
                f"- **Banda de Referencia Operativa (± MAE de Test):** Entre **${max(0.0, pred_usd - artifact['test_metrics']['mae']):,.2f}** "
                f"y **${pred_usd + artifact['test_metrics']['mae']:,.2f} USD** anuales.\n"
                f"- **Comparación con la Cartera Histórica:** El valor estimado equivale al **{pred_usd / mean_global * 100:.1f}%** del gasto "
                f"medio histórico (${mean_global:,.2f} USD)."
            )

            st.markdown("##### Vector de Entrada Validado Enviado a `pipeline.predict()`")
            st.dataframe(df_single, width="stretch", hide_index=True)

            if age_in > 64:
                st.warning(
                    "⚠️ **Aviso de extrapolación de edad:** La edad ingresada supera los 64 años (máximo observado en `insurance.csv`). "
                    "Con un kernel RBF gaussiano, alejarse demasiado del soporte de entrenamiento puede atenuar la predicción hacia la media global."
                )

    st.divider()

    # --- TABLA T9: 5 CASOS NUEVOS PLAUSIBLES ---
    st.subheader("📋 Tabla T9 — Demostración Obligatoria con 5 Casos Nuevos Plausibles (Anexo C)")
    st.markdown(
        """
        A continuación se presentan **5 perfiles hipotéticos plausibles de nuevos solicitantes de póliza** (no pertenecientes al
        entrenamiento ni al conjunto de prueba) evaluados por lotes mediante `get_five_new_predictions()` en `src/predict.py`,
        cubriendo todo el espectro actuarial del dominio:
        """
    )

    t9_df = cached_five_new_predictions()
    st.dataframe(t9_df, width="stretch", hide_index=True)

    # Gráfico visual de los 5 casos nuevos de la Tabla T9
    fig_t9, ax_t9 = plt.subplots(figsize=(11.5, 4.3), dpi=110)
    case_labels = [c.split(":")[0] + "\n(" + c.split(":")[1].strip()[:26] + "...)" for c in t9_df["Caso"]]
    preds_t9 = t9_df["Gasto Anual Predicho - charges_pred ($)"].values
    bar_colors = ["#10b981", "#3b82f6", "#f59e0b", "#6366f1", "#ef4444"]

    bars = ax_t9.barh(case_labels, preds_t9, color=bar_colors, edgecolor="#1f2937", height=0.58)
    ax_t9.invert_yaxis()
    ax_t9.axvline(float(df[TARGET_COL].median()), color="#0f172a", linestyle="--", linewidth=1.6, label="Mediana Histórica ($9,382)")
    ax_t9.axvline(float(df[TARGET_COL].mean()), color="#d97706", linestyle=":", linewidth=1.8, label="Media Histórica ($13,270)")
    ax_t9.set_xlabel("Gasto Médico Anual Predicho por SVR Optimizado (USD $)")
    ax_t9.set_title("Comparación Cuantitativa de los 5 Casos Nuevos Plausibles (Tabla T9)", fontweight="bold")
    ax_t9.legend(loc="lower right", frameon=True)
    ax_t9.grid(axis="x", linestyle="--", alpha=0.4)

    for b, val in zip(bars, preds_t9):
        ax_t9.text(
            val + 550,
            b.get_y() + b.get_height() / 2,
            f"${val:,.2f}",
            va="center",
            fontweight="bold",
            fontsize=9.5,
        )
    ax_t9.set_xlim(0, max(preds_t9) * 1.18)
    fig_t9.tight_layout()
    st.pyplot(fig_t9, width="stretch")
    plt.close(fig_t9)


# ============================================================================
# NAVEGACIÓN PRINCIPAL EN BARRA LATERAL (ANEXO H)
# ============================================================================
def main() -> None:
    df = cached_load_default_data()
    artifact = cached_load_artifact()

    with st.sidebar:
        st.markdown("## 🏥 Grupo 09 — MediCost ML")
        st.caption("Estimación de Gastos de Seguros Médicos (`insurance.csv`)")
        st.divider()

        screens = [
            "1. Inicio y problema",
            "2. Datos y EDA",
            "3. Variables y preparación",
            "4. Modelos y comparación",
            "5. Evaluación",
            "6. Predicciones",
        ]
        selected_screen = st.radio(
            "🧭 Navegación de Módulos (Anexo H):",
            options=screens,
            index=0,
        )

        st.divider()
        st.markdown("### 📑 Mapa de Etapas y Tablas")
        st.markdown(
            """
            - **1. Inicio y problema:** Etapas 1 y 2
            - **2. Datos y EDA:** Etapas 3 y 5 · **T1, T2**
            - **3. Variables y preparación:** Etapas 4, 6 y 7 · **T3, T4**
            - **4. Modelos y comparación:** Etapas 8 a 12 · **T5, T6, T7**
            - **5. Evaluación:** Etapas 13, 14 y 16 · **T8** + Grupo 09
            - **6. Predicciones:** Etapa 15 · **T9** + Simulador
            """
        )
        st.divider()
        st.success(
            f"**Modelo Activo:** `{artifact['winner_name']}`\n\n"
            f"- **MAE Test:** `${artifact['test_metrics']['mae']:,.2f}`\n"
            f"- **RMSE Test:** `${artifact['test_metrics']['rmse']:,.2f}`\n"
            f"- **R² Test:** `{artifact['test_metrics']['r2']:.4f}`"
        )

    render_header(df, artifact)

    if selected_screen == "1. Inicio y problema":
        render_screen_1_inicio(df, artifact)
    elif selected_screen == "2. Datos y EDA":
        render_screen_2_eda(df)
    elif selected_screen == "3. Variables y preparación":
        render_screen_3_preparacion(df, artifact)
    elif selected_screen == "4. Modelos y comparación":
        render_screen_4_modelos(artifact)
    elif selected_screen == "5. Evaluación":
        render_screen_5_evaluacion(artifact)
    elif selected_screen == "6. Predicciones":
        render_screen_6_predicciones(df, artifact)


if __name__ == "__main__":
    main()
