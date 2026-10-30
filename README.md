# 🏥 Grupo 02 — Estimación de Gastos de Seguros Médicos (`insurance.csv`)

Aplicación interactiva en **Streamlit** y proyecto integral de **Aprendizaje Supervisado (Regresión)** desarrollado para el **Grupo 02** siguiendo las especificaciones de `ejercicio.md` (Anexos A, B, C, E y H) y `design_doc.md`.

El sistema estima el gasto médico anual individual facturado (`charges`, en dólares estadounidenses USD) de un beneficiario titular a partir de sus características demográficas y clínicas básicas (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) conocidas en el instante previo a la suscripción de la póliza ($t = 0$).

---

## 📊 1. Fuente del Conjunto de Datos

- **Nombre oficial:** *Medical Cost Personal Datasets (`insurance.csv`)*
- **Autor / Compilador:** Miri Choi (basado en el caso de estudio actuarial de Brett Lantz, *Machine Learning with R*)
- **Enlace directo en Kaggle:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Ruta local incluida en el repositorio:** `data/insurance.csv`
- **Dimensiones:** `1,338 filas × 7 columnas` (`0` valores nulos, `1` fila duplicada exacta en los índices `195` y `581`).
- **Variables:**
  - **Predictoras ($\mathbf{X}$, 6 columnas):**
    - Numéricas (`3`): `age` (18 a 64 años), `bmi` (15.96 a 53.13 $\text{kg/m}^2$), `children` (0 a 5 dependientes).
    - Categóricas nominales (`3`): `sex` (`female`, `male`), `smoker` (`yes`, `no`), `region` (`northeast`, `northwest`, `southeast`, `southwest`).
  - **Objetivo ($y$, 1 columna):**
    - `charges`: Costo médico individual anual facturado en dólares estadounidenses ($\text{rango}: \$1,121.87 \text{ a } \$63,770.43\text{ USD}$).

---

## 🗂️ 2. Estructura del Proyecto

```text
medicost/
├── app.py                      # Aplicación principal en Streamlit (6 pantallas del Anexo H, Etapas 1-16 y Tablas T1-T9)
├── requirements.txt            # Dependencias reproducibles del entorno Python
├── README.md                   # Documentación técnica y guía de ejecución
├── design_doc.md               # Documento de diseño arquitectónico y metodológico
├── agents.md                   # Definición de roles y reglas de desarrollo
├── data/
│   └── insurance.csv           # Dataset original Medical Cost Personal Datasets (1,338 registros)
├── models/
│   └── modelo_final.joblib     # Artefacto serializado con el Pipeline ganador (SVR Optimizado) y tablas T3-T8
└── src/
    ├── __init__.py
    ├── data.py                 # Carga validada, diccionario T1, diagnóstico de calidad T2, asimetría y 5 gráficos EDA
    ├── ml.py                   # Partición 80/20 T3, ColumnTransformer T4, 5-Fold CV T5, GridSearchCV T6, Decisión T7 y Test T8
    └── predict.py              # Validación de entradas nuevas, inferencia sin reentrenar y 5 casos plausibles (Tabla T9)
```

---

## ⚙️ 3. Requisitos e Instalación (Entorno Virtual `.venv`)

1. **Crear y activar el entorno virtual (si aún no existe):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instalar las dependencias desde `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 4. Ejecución de la Aplicación en Streamlit

Para iniciar la aplicación interactiva en el navegador:

```bash
.venv/bin/streamlit run app.py
```

Si desea regenerar desde la terminal el artefacto `models/modelo_final.joblib` o verificar el módulo de inferencia:

```bash
.venv/bin/python -m src.ml
.venv/bin/python -m src.predict
```

---

## 🛡️ 5. Metodología Anti-Fuga de Información (Zero Data Leakage)

El flujo experimental cumple estrictamente con todas las reglas metodológicas de la guía:

1. **Separación previa de $\mathbf{X}$ e $y$ (Etapa 4):** `charges` se separa antes de cualquier transformación.
2. **Partición 80% Entrenamiento / 20% Prueba Reservada (Etapa 6 — Tabla T3):**
   - `train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)` $\rightarrow$ `1,070` registros en entrenamiento y `268` en prueba reservada (`0` filas compartidas).
3. **Preprocesamiento encapsulado en `ColumnTransformer` + `Pipeline` (Etapa 7 — Tabla T4):**
   - **Rama Numérica (`age`, `bmi`, `children`):** `SimpleImputer(strategy='median')` + `StandardScaler()`.
   - **Rama Categórica (`sex`, `smoker`, `region`):** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - Ningún transformador se ajusta sobre el dataset completo; todos se ajustan exclusivamente dentro de `Pipeline` sobre `X_train` o cada fold de validación.
4. **Validación Cruzada Idéntica de 5 Folds en Train (Etapa 10 — Tabla T5):**
   - `KFold(n_splits=5, shuffle=True, random_state=42)` evaluando `DummyRegressor`, `LinearRegression`, `KNeighborsRegressor` y `SVR` con **MAE** como métrica primaria.
5. **Ajuste de Hiperparámetros con `GridSearchCV` Solo en Train (Etapa 11 — Tabla T6):**
   - Optimización de `KNeighborsRegressor` (24 combinaciones) y `SVR` (72 combinaciones) usando `refit='mae'`.
6. **Selección Previa al Test y Evaluación Final Única (Etapas 12 y 13 — Tablas T7 y T8):**
   - El modelo ganador se bloquea antes de abrir `X_test` y se evalúa una única vez en el conjunto de prueba reservado.

---

## 📈 6. Resumen de Resultados Experimentales y Hallazgos del Grupo 02

### Comparación de Modelos en Validación Cruzada (5 Folds en `X_train`, $n=1,070$) y Prueba Reservada ($n=268$)

| Modelo / Pipeline | Configuración | Media MAE CV ($) | Desv. Est. MAE CV ($) | Media $R^2$ CV | MAE Test Reservado ($) | RMSE Test ($) | $R^2$ Test |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyRegressor (Baseline)** | `strategy='mean'` | $9,000.66 | $289.18 | -0.0029 | $9,593.34 | $12,465.61 | -0.0009 |
| **LinearRegression** | `fit_intercept=True` (OLS) | $4,251.40 | $188.09 | 0.7338 | $4,181.19 | $5,796.28 | 0.7836 |
| **KNeighborsRegressor (Optimizado)** | `n_neighbors=7, p=1, weights='distance'` | $3,110.73 | $227.60 | 0.7897 | $2,916.21 | $5,221.37 | 0.8244 |
| **🏆 SVR (Optimizado — Ganador)** | `kernel='rbf', C=10000, gamma=0.1, epsilon=50.0` | **$2,424.16** | **$166.96** | **0.8240** | **$1,803.44** | **$4,832.09** | **0.8496** |

### Hallazgos Específicos del Grupo 02

1. **Asimetría y Valores Atípicos de `charges`:**
   - `charges` presenta un coeficiente de asimetría positivo de **`1.5159`** y curtosis de **`1.6063`**, con media (**`$13,270.42`**) muy superior a la mediana (**`$9,382.03`**).
   - Existen **`139` registros (`10.39%`)** por encima del cerco intercuartílico superior ($Q_3 + 1.5 \times \text{IQR} = \$34,489.35$), de los cuales el **`97.84%` (`136` de `139`) corresponden a asegurados fumadores (`smoker='yes'`)**. No son errores de medición sino siniestralidad severa real.
2. **Comparación del MAE entre Fumadores y No Fumadores (Advertencia de No Causalidad):**
   - En el conjunto de prueba (`n=268`), los **no fumadores (`n=214`)** registran un **MAE nominal de `$1,363.24 USD`** (mediana de error de solo **`$257.33 USD`**), mientras que los **fumadores (`n=54`)** registran un **MAE nominal de `$3,547.97 USD`**.
   - Esta diferencia se explica por la **escala monetaria 3.82 veces mayor** en fumadores (gasto medio real de `$32,050.23` vs. `$8,384.59`). En términos relativos al gasto promedio de su propio grupo, el modelo es **más preciso en fumadores (`MAE Relativo = 11.07%`, $R^2\text{ intra-grupo} = 0.8075$)** que en no fumadores (`MAE Relativo = 16.26%`).
   - **No causalidad:** El modelo captura asociaciones predictivas y tarifarias en datos observacionales transversales; no demuestra un efecto causal clínico directo debido a variables confusoras no observadas en el dataset.

---

## 🧭 7. Módulos de Navegación en `app.py` (Anexo H)

1. **`1. Inicio y problema`:** Etapas 1 y 2 (Contexto actuarial, pregunta predictiva, objetivos general/específicos, criterios de éxito y uso responsable).
2. **`2. Datos y EDA`:** Etapas 3 y 5 (**Tabla T1**, **Tabla T2**, análisis de asimetría/atípicos de `charges` del Grupo 02 y 5 gráficos interpretados).
3. **`3. Variables y preparación`:** Etapas 4, 6 y 7 (Separación `X`/`y`, **Tabla T3** de partición 80/20 y **Tabla T4** del `ColumnTransformer` con demostración en vivo).
4. **`4. Modelos y comparación`:** Etapas 8, 9, 10, 11 y 12 (Fundamentos, Pipelines reproducibles, **Tabla T5** de 5-Fold CV, **Tabla T6** de `GridSearchCV` y **Tabla T7** de decisión previa al test).
5. **`5. Evaluación`:** Etapas 13, 14 y 16 (**Tabla T8** en test reservado, análisis Grupo 02 de MAE en fumadores vs. no fumadores y por niveles de gasto, gráficos de residuales, 8 preguntas de sustentación y conclusiones).
6. **`6. Predicciones`:** Etapa 15 (Simulador interactivo con validación en `src/predict.py` y **Tabla T9** con 5 casos nuevos plausibles explicados).
