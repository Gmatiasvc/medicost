# Antigravity CLI - Agent Task Delegation

Este documento define la orquestación de agentes en Antigravity para el pipeline de Machine Learning y la app Streamlit.
**Seguridad:** Limita el servidor MCP `filesystem` estrictamente a este directorio de trabajo; no otorgues acceso global.

## Agent 1: Data Engineer (EDA & Data Prep)
- **Servidores MCP:** `filesystem`, `python_interpreter`.
- **Directivas:**
  1. Crear `src/data.py`.
  2. Implementar funciones de carga del CSV.
  3. Ejecutar EDA: detectar nulos, duplicados, distribución de `charges`.
  4. Generar visualizaciones: histograma de `charges`, edad vs cargos, BMI vs cargos, y boxplot por fumador.

## Agent 2: ML Engineer (Pipelines & CV)
- **Servidores MCP:** `filesystem`, `python_interpreter`.
- **Directivas:**
  1. Crear `src/ml.py`.
  2. Implementar `train_test_split` garantizando la reserva del 20% de test sin fugas.
  3. Estructurar el `ColumnTransformer` y los 3 `Pipelines` (LinearRegression, KNeighborsRegressor, SVR).
  4. Ejecutar Cross-Validation (5 folds) y `GridSearchCV` para optimizar hiperparámetros de KNN y SVR.
  5. Serializar el mejor modelo a `models/modelo_final.joblib`.

## Agent 3: Frontend / Streamlit Developer
- **Servidores MCP:** `filesystem`, `desktop-commander`.
- **Directivas:**
  1. Escribir `app.py` implementando la navegación en 6 secciones.
  2. Integrar las tablas requeridas por la rúbrica (T1-T9).
  3. En la sección 6, crear el formulario de inferencia interactiva.
  4. Ejecutar el subproceso `streamlit run app.py` usando `desktop-commander` para verificar el despliegue local en el entorno de desarrollo.
