# Design Document: Estimación de Gastos de Seguros Médicos (Grupo 09)

## 1. Contexto y Objetivos
Desarrollo de un sistema predictivo supervisado (Regresión) para estimar `charges` en el dataset *Medical Cost Personal Datasets* [cite: 1].
- **Métrica principal de decisión:** MAE (Menor es mejor) [cite: 1].
- **Métricas secundarias:** RMSE, R² [cite: 1].
- **Modelos obligatorios:** LinearRegression, KNeighborsRegressor, SVR. (Opcional: DummyRegressor como baseline) [cite: 1].

## 2. Arquitectura del Proyecto (Streamlit)
Estructura de archivos y módulos exigida [cite: 1]:
- `app.py`: Interfaz principal con las 6 secciones requeridas (Inicio, EDA, Variables, Modelos, Evaluación, Predicciones) [cite: 1].
- `src/data.py`: Carga y validación inicial del CSV [cite: 1].
- `src/ml.py`: Funciones de preprocesamiento, CV, GridSearchCV y evaluación [cite: 1].
- `src/predict.py`: Carga del modelo y lógica de inferencia para datos nuevos [cite: 1].
- `models/modelo_final.joblib`: Pipeline ganador exportado [cite: 1].
- `data/`: Contenedor para `insurance.csv` [cite: 1].

## 3. Restricciones Metodológicas Críticas (Prevención de Data Leakage)
- **Partición:** Separación 80/20 con `random_state=42`. El conjunto de test se reserva y se evalúa UNA sola vez al final [cite: 1].
- **Pipelines:** `ColumnTransformer` (SimpleImputer, StandardScaler, OneHotEncoder(handle_unknown="ignore")) debe aplicarse **EXCLUSIVAMENTE** dentro de un `Pipeline` sobre los datos de entrenamiento/folds [cite: 1].
- **Validación Cruzada:** 5 folds idénticos para comparar los 3 estimadores [cite: 1].
- **Optimización:** `GridSearchCV` exclusivo para KNN y SVR sobre train [cite: 1].

## 4. Requisitos Analíticos del Grupo 09
- Analizar la asimetría y los valores atípicos de la variable monetaria `charges` [cite: 1].
- Comparar el MAE entre fumadores y no fumadores, estableciendo claramente que no se asume causalidad directa [cite: 1].
