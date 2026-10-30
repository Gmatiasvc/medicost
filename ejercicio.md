# SISTEMAS INTELIGENTES Y MACHINE LEARNING | GUÍA DE TRABAJO APLICATIVO

**GRUPO 02. Estimación de gastos de seguros médicos**  
**Modalidad:** Trabajo grupal de aprendizaje supervisado – Regresión.  
**Dataset asignado:** Medical Cost Personal Datasets  
**Fuente Kaggle:** `https://www.kaggle.com/datasets/mirichoi0218/insurance`  
**Variable objetivo propuesta:** `charges`  
**Predictores candidatos:** edad, índice de masa corporal, número de hijos, consumo de tabaco y variables categóricas disponibles. Su selección final debe justificarse según el CSV realmente descargado.

---

## 1. Situación problemática

¿Qué precisión alcanza un modelo al estimar el costo individual del seguro? La organización requiere una evaluación reproducible para decidir si estas predicciones serían útiles. No basta con mostrar código: deben interpretar resultados y justificar decisiones.

---

## 2. Propósito y objetivos exigidos

**Objetivo general:** Desarrollar y evaluar un sistema predictivo que responda al problema del grupo mediante tres modelos supervisados, sin fuga de información (*data leakage*) y con evaluación final reservada.

* Identificar variables predictoras y riesgos de calidad del dataset.
* Diseñar un preprocesamiento reproducible mediante `Pipeline` y `ColumnTransformer`.
* Comparar tres algoritmos mediante validación cruzada y optimización.
* Evaluar el modelo seleccionado y argumentar su utilidad y limitaciones.

---

## 3. Algoritmos y métricas obligatorias

* **Modelos candidatos:** `LinearRegression`, `KNeighborsRegressor` y `SVR`.
* **Métrica primaria de selección:** MAE (menor es mejor).
* **Métricas secundarias:** RMSE y $R^2$.
* **Modelo base:** Incluir `DummyRegressor` como referencia no competidora opcional.

---

## 4. Desarrollo obligatorio: 16 etapas

1. **Planteamiento del problema:** Describan el contexto del caso y elaboren una pregunta predictiva precisa. Indiquen quién utilizaría la predicción, en qué momento y para qué decisión.
2. **Objetivos:** Redacten un objetivo general medible y cuatro objetivos específicos vinculados a análisis de datos, entrenamiento, validación y evaluación.
3. **Dataset y diccionario de datos:** Descarguen de Kaggle el archivo con etiquetas, documenten autor, URL, licencia, nombre y versión del CSV, cantidad real de filas/columnas, tipos de datos y significado de las variables.
4. **Separación de variables $X$ e $y$:** Construyan $y = \text{columna objetivo}$ y $X = \text{columnas predictoras}$. Muestren el código y una tabla con variables incluidas/excluidas y razones; revisen posible fuga de información.
5. **Análisis exploratorio (EDA):** Presenten `head()`, `info()`, `describe()`, nulos, duplicados, distribución del objetivo, variables relevantes y mínimo cuatro gráficos interpretados; distingan análisis descriptivo de decisiones ajustadas al entrenamiento.  
   *Evidencia mínima:* tabla de calidad de datos, histograma del objetivo, gráfico de relación entre dos variables y una visualización adicional útil al caso.
6. **Entrenamiento/prueba:** Reservar un 20% para prueba final cuando el diseño lo permita; declarar `random_state=42`. En clasificación usar `stratify` cuando proceda; en datos temporales respetar orden cronológico.
7. **Preprocesamiento sin fuga:** Construyan `ColumnTransformer`: `SimpleImputer` para numéricas/categóricas, `OneHotEncoder(handle_unknown="ignore")` para nominales y `StandardScaler` cuando corresponda. Ajusten transformaciones SOLO dentro de `Pipeline` sobre entrenamiento y folds.
8. **Modelos candidatos:** Expliquen los fundamentos de cada algoritmo, parámetros principales, fortalezas, debilidades y pertinencia con respecto al dataset.
9. **Entrenamiento reproducible:** Construyan tres pipelines completos con estimadores distintos. Muestren importaciones, ajuste, semillas, librerías y configuración utilizada.
10. **Validación cruzada:** Comparen los tres algoritmos usando los mismos cinco folds de entrenamiento; muestren cada resultado, media, desviación estándar y comentario de estabilidad. No miren test.
11. **Ajuste de hiperparámetros:** Ejecuten `GridSearchCV` para KNN y SVM con rejillas justificadas y el mismo esquema de CV. Registren mejores parámetros y mejor score; no optimicen con test.
12. **Selección antes del test:** Presenten una tabla comparativa con media/desviación de CV, parámetros y costo computacional aproximado. Justifiquen la elección previa a abrir test.
13. **Evaluación final:** Evalúen UNA vez sobre test reservado al pipeline seleccionado y calculen todas las métricas exigidas; interpreten magnitud y dirección de los errores. Incluyan MAE, RMSE y $R^2$ en test; muestren residuales y gráfica real vs. predicho.
14. **Análisis crítico:** Expliquen para qué serviría y cuándo NO serviría el modelo; identifiquen errores frecuentes, limitaciones, riesgos de sesgo y posibles fuentes de sobreajuste.
15. **Nuevas predicciones:** Realicen cinco predicciones sobre entradas nuevas plausibles o registros externos no usados en entrenamiento. Aclaren procedencia y no las presenten como prueba de exactitud si no hay valores reales.
16. **Conclusiones y mejoras:** Redacten cinco conclusiones basadas en evidencias, tres mejoras realizables y la decisión de despliegue/no despliegue para el contexto.

---

## 5. Requisitos particulares del grupo

* **Enfoque analítico:** Analizar errores por niveles de gasto e influencia de condiciones de entrada.
* **Precaución metodológica:** Analizar asimetría y valores atípicos; evitar atribuir causalidad a las asociaciones predictivas.

---

## 6. Preguntas para el informe y sustentación

1. ¿Qué precisión alcanza un modelo al estimar el costo individual del seguro?
2. ¿Qué predictores permanecen disponibles en el momento real de utilizar el modelo?
3. ¿Qué cambios concretos produce imputar, codificar y escalar dentro de `Pipeline`?
4. ¿Cómo varía el rendimiento entre folds y qué revela esa variación?
5. ¿Qué parámetros optimizaron para KNN y SVM, y por qué?
6. ¿Qué error del modelo resultaría más costoso en este contexto?
7. ¿Por qué el modelo elegido debería generalizar a nuevos casos?
8. ¿Qué limitación impide afirmar que el modelo está listo para producción?

---

## 7. Entregables obligatorios

* **Informe académico de 12-20 páginas** (sin anexos), con introducción, objetivos, metodología, tablas, discusión, conclusiones y referencias APA 7.
* **Aplicación web Streamlit operativa (`app.py`)**, con módulos de carga, análisis, entrenamiento, comparación, evaluación y predicción. Debe ejecutarse localmente mediante `streamlit run app.py`.
* **CSV original** permitido por la licencia o instrucciones exactas de descarga; archivo `requirements.txt` o listado de versiones.
* **Presentación PowerPoint** de 10-15 diapositivas y defensa oral participativa.
* **Tabla comparativa** de 3 algoritmos, rejillas de búsqueda, resultados por fold y métricas del modelo elegido.
* **Cinco predicciones nuevas** claramente explicadas; declaración de contribución de cada integrante.

---

## 8. Rúbrica de calificación (20 puntos)

| Criterio | Puntaje |
| :--- | :---: |
| Planteamiento y objetivos | 2 |
| EDA y comprensión del dataset | 3 |
| Preprocesamiento sin fuga | 3 |
| Implementación de tres modelos | 4 |
| Validación cruzada y GridSearch | 3 |
| Selección y evaluación interpretada | 3 |
| Reproducibilidad, informe y sustentación | 2 |
| **TOTAL** | **20** |

---

## 9. Condiciones de entrega y control de calidad

No reutilicen el test durante el ajuste; no entrenen una transformación con todo el CSV; mantengan el mismo esquema de CV para una comparación justa. El trabajo debe citar Kaggle y la documentación técnica utilizada. Los resultados dependen de la versión descargada y no deben inventarse. Fecha y medio de entrega: los indicará el docente.

---

## ANEXO A. INSTRUCCIONES OPERATIVAS Y EVIDENCIAS POR ETAPA

Este anexo amplía la guía anterior. Cada etapa debe mostrarse en la aplicación Streamlit con explicaciones visibles en pantalla y en el informe mediante evidencias seleccionadas. No se aceptarán capturas de código sin salidas ni tablas sin interpretación.

1. **Problema y finalidad**
   * *Actividades:* Redactar 2-3 párrafos de contexto, situación a resolver, quién tomaría decisiones, instante de predicción y beneficios esperados.
   * *Qué presentar:* Contexto documentado; pregunta de predicción; justificación del enfoque supervisado; esquema de entradas y salida.
2. **Objetivos y criterios de éxito**
   * *Actividades:* Formular un objetivo general, cuatro específicos y un criterio de aceptación medible y razonado, acordado antes de experimentar.
   * *Qué presentar:* Objetivos verificables; tabla objetivo-actividad-evidencia; criterio de éxito y costo de error contextual.
3. **Procedencia y diccionario de variables**
   * *Actividades:* Descargar conjunto etiquetado, anotar URL y nombre exacto del archivo; informar filas, columnas, licencia/condiciones, fecha de consulta y significado de cada campo.
   * *Qué presentar:* Ficha del dataset; diccionario con columna, definición, unidad, tipo, rol, momento de disponibilidad y riesgo de fuga.
4. **Definir $X$ e $y$**
   * *Actividades:* Escribir `y = df[columna_objetivo]` y `X = df.drop(columns=[columna_objetivo, ...])`; justificar qué identificadores, datos posteriores y variables redundantes se excluyen.
   * *Qué presentar:* Celda Python con $X$ e $y$; capturas de dimensiones; tabla de exclusiones y razones; comprobación de coincidencia de índices.
5. **Análisis exploratorio**
   * *Actividades:* Revisar `dtypes`, `head`, `describe`, faltantes, duplicados, categorías, valores improbables y anomalías; al menos 5 gráficos pertinentes y descripción debajo de cada uno.
   * *Qué presentar:* Tabla de nulos (%), duplicados, distribución de $y$, histogramas/boxplots, asociaciones y resumen de cinco hallazgos con implicaciones.
6. **División y protocolo experimental**
   * *Actividades:* Reservar test antes de ajustar transformaciones; 80/20 orientativo. Clasificación: estratificar si es factible; tiempo: ordenar cronológicamente. Registrar `random_state` si aplica.
   * *Qué presentar:* Código `train_test_split` o corte temporal; números y porcentajes; comprobación de ausencia de filas compartidas; justificación del esquema.
7. **Tratamiento y transformación**
   * *Actividades:* Definir columnas numéricas/categóricas; crear `SimpleImputer`, `StandardScaler` y `OneHotEncoder(handle_unknown="ignore")` cuando correspondan, integrados en `ColumnTransformer` y `Pipeline`.
   * *Qué presentar:* Diagrama simple de ramas numéricas/categóricas; código completo; explicación de nulos, encoding y escala; prueba de transformaciones en datos no vistos.
8. **Justificación de algoritmos**
   * *Actividades:* Explicar la idea de los tres algoritmos con sus supuestos, costes y sensibilidad al escalado; especificar sus hiperparámetros relevantes.
   * *Qué presentar:* Tabla algoritmo-mecanismo-fortaleza-limitación, hiperparámetros y justificación de aplicación al caso.
9. **Implementación**
   * *Actividades:* Instanciar tres pipelines equivalentes con distinto estimador; conservar misma variable y particiones; crear baseline simple como referencia si es viable.
   * *Qué presentar:* Código Python reproducible; tabla de configuración inicial; salida de entrenamiento; confirmación de uso del mismo train.
10. **Validación cruzada**
    * *Actividades:* Evaluar los tres candidatos exclusivamente en entrenamiento con cinco folds; usar particiones idénticas y la métrica primaria acordada.
    * *Qué presentar:* Tabla de 5 scores por modelo, media, desviación, ranking preliminar; gráfica comparativa e interpretación de variabilidad.
11. **Búsqueda de hiperparámetros**
    * *Actividades:* Definir `param_grid` viable para KNN y SVM, ejecutar `GridSearchCV` dentro de entrenamiento y documentar cantidad de configuraciones.
    * *Qué presentar:* Código `GridSearchCV`; rejillas evaluadas; `best_params`, `best_score`, tabla top 5 de combinaciones y comparación antes/después.
12. **Decisión del mejor modelo**
    * *Actividades:* Fijar ganador considerando rendimiento CV, dispersión, costo de inferencia y explicabilidad; no emplear el test para decidir.
    * *Qué presentar:* Tabla final de candidatos, selección razonada (mínimo 150 palabras) y criterio de desempate declarado.
13. **Evaluación de prueba**
    * *Actividades:* Evaluar el pipeline ganador una sola vez en el test reservado; reportar unidades, incertidumbre y relación con baseline sin cambiar después el ganador.
    * *Qué presentar:* Resultados en test, tabla valor-real/predicción (10 filas) y gráficos de error o confusión con interpretación.
14. **Discusión y riesgos**
    * *Actividades:* Responder si el error es aceptable en el contexto; analizar subgrupos, casos difíciles, sesgos, cobertura, extrapolación y consecuencias de errores.
    * *Qué presentar:* Al menos tres hallazgos basados en métricas; dos limitaciones internas; dos externas; recomendaciones de uso responsable.
15. **Demostración de predicción**
    * *Actividades:* Preparar cinco entradas nuevas válidas no empleadas en fit; mantener idéntico pipeline; distinguir ejemplo hipotético de caso real y evitar mezclar columnas.
    * *Qué presentar:* Tabla con entradas, salida estimada y explicación de cada ejemplo; celda predict; evidencia de que no hay reentrenamiento.
16. **Conclusiones y recomendaciones**
    * *Actividades:* Redactar cinco conclusiones directamente respaldadas por tabla o figura; tres mejoras futuras concretas y priorizadas.
    * *Qué presentar:* Matriz conclusión-evidencia, tres acciones siguientes y decisión argumentada de uso experimental/no despliegue.

---

## ANEXO B. ACTIVIDADES PARTICULARES DEL GRUPO

* **Análisis específico requerido:** Comparar MAE entre fumadores/no fumadores sin convertir la asociación en causalidad.
* **Gráficos y evidencias recomendadas:** Distribución de `charges`, edad versus cargos, BMI versus cargos, boxplot por `smoker` y residuales.
* **Control metodológico obligatorio:** La variable objetivo suele ser asimétrica: analizar extremos y escala monetaria.

---

## ANEXO C. TABLAS QUE DEBEN APARECER EN EL INFORME

| Tabla | Nombre | Contenido mínimo |
| :--- | :--- | :--- |
| **T1** | Diccionario de variables | Variable, definición, tipo, unidad, disponible antes de predicción, incluir/excluir. |
| **T2** | Diagnóstico de calidad | Nulos, porcentajes, duplicados, anómalos, medida adoptada. |
| **T3** | Partición de datos | Registros y proporciones de train/test, método, semilla o fechas. |
| **T4** | Preprocesamiento | Columna o grupo, imputación, codificación, escalado, justificación. |
| **T5** | Comparación en validación | Modelo, fold1-fold5, media y desviación de la métrica primaria. |
| **T6** | Optimización | Algoritmo, valores ensayados, mejor configuración, CV antes/después. |
| **T7** | Decisión | Candidatos, CV, estabilidad, costo, argumentos del ganador. |
| **T8** | Prueba final | Métricas del ganador, baseline, interpretación cuantitativa. |
| **T9** | Predicciones nuevas | Cinco entradas, valores previstos y explicación del resultado. |

---

## ANEXO D. ESTRUCTURA DEL INFORME ACADÉMICO

* **Portada:** Universidad, asignatura, número de grupo, tema, integrantes, docente, fecha.
* **Resumen:** 150-200 palabras: problema, datos, algoritmos, selección, resultados y límite principal.
1. **Introducción:** Problema, contexto, relevancia; incluir fuentes de datos.
2. **Objetivos:** General, cuatro específicos y criterio previo de éxito.
3. **Metodología:** Descripción del dataset, división $X/y$, preprocesamiento, pipelines, cinco folds y búsqueda.
4. **Resultados:** Tablas T1-T9, gráficos y análisis: no sustituir resultados por texto genérico.
5. **Discusión:** Comparación crítica con baseline, costos de error, validez y riesgos de uso.
6. **Conclusiones:** Cinco conclusiones verificables y tres recomendaciones.
* **Referencias:** Fuente Kaggle y documentación técnica, en formato APA 7.
* **Anexos:** Repositorio o ZIP del proyecto Streamlit, instrucciones de instalación y ejecución, versiones y capturas de funcionamiento.

---

## ANEXO E. ESTRUCTURA OBLIGATORIA DE LA APLICACIÓN STREAMLIT

El aplicación Streamlit tendrá 16 secciones numeradas que coinciden con las etapas. Cada sección deberá incluir: propósito, código ejecutable, salida visible y una interpretación redactada por el equipo. Al inicio debe haber una celda con instrucciones para cargar el CSV original; al final una celda de predicción de cinco casos.

Debe ejecutarse desde *Entorno de ejecución / Ejecutar todas*, tras colocar los archivos en la ruta documentada, sin depender de variables no definidas o archivos privados del autor. Registrar la semilla, las versiones y las bibliotecas utilizadas.

---

## ANEXO F. ESTRUCTURA DE LA PRESENTACIÓN Y SUSTENTACIÓN

| Diapositiva | Contenido mínimo |
| :---: | :--- |
| **1** | Título, integrantes y problema |
| **2** | Contexto y pregunta predictiva |
| **3** | Objetivos y criterio de éxito |
| **4** | Dataset, objetivo y variables $X$ |
| **5** | EDA: nulos, distribución y hallazgos |
| **6** | Partición y diseño sin fuga de datos |
| **7** | Preprocesamiento con `Pipeline`/`ColumnTransformer` |
| **8** | Tres algoritmos y sus parámetros |
| **9** | Tabla de validación cruzada y optimización |
| **10** | Modelo elegido y justificación |
| **11** | Resultados en test y errores |
| **12** | Cinco predicciones y límites |
| **13** | Conclusiones y recomendaciones |

> **Nota para la defensa:** Durante la defensa cada integrante debe explicar una decisión de implementación y responder una pregunta metodológica. Se podrá solicitar ejecutar una predicción en vivo y localizar el bloque de código responsable del preprocesamiento.

---

## ANEXO G. LISTA DE VERIFICACIÓN ANTES DE ENTREGAR

- [ ] Se citó la URL exacta y se verificó el nombre de $y$ en el CSV.
- [ ] Se diferenciaron $X$ e $y$ y se justificaron variables excluidas.
- [ ] La división se efectuó antes de ajustar transformaciones.
- [ ] Los tres modelos usan datos, particiones y métrica primaria equivalentes.
- [ ] Se presentan resultados por fold, media y desviación.
- [ ] `GridSearchCV` solo usó entrenamiento y folds.
- [ ] El modelo ganador quedó decidido antes de evaluar test.
- [ ] Se reportaron métricas y gráficos propios del problema.
- [ ] Hay cinco predicciones nuevas, sin reentrenar el pipeline.
- [ ] Informe, aplicación Streamlit, dataset/instrucciones y PPT son consistentes; la aplicación inicia y permite predecir desde un navegador.

---

## ANEXO H. DESARROLLO OBLIGATORIO EN STREAMLIT

**Producto principal:** Aplicación web interactiva en Python con Streamlit. No se aceptará un notebook de Google Colab como sustituto. Los scripts auxiliares pueden utilizarse para preparar datos o entrenar, pero el usuario debe interactuar con la aplicación mediante un navegador.

### Pantallas / Módulos de Streamlit

| Módulo/Pantalla | Qué debe mostrar y permitir hacer |
| :--- | :--- |
| **1. Inicio y problema** | Título del grupo, propósito predictivo, fuente Kaggle, objetivo y advertencia sobre uso responsable. |
| **2. Datos y EDA** | Carga reproducible de CSV, vista previa, dimensión, tipos, nulos, duplicados, distribución del objetivo y por lo menos cinco gráficos con interpretación. |
| **3. Variables y preparación** | Selección o explicación de $X$ e $y$, exclusiones, partición train/test; descripción del `ColumnTransformer` sin ajustar sobre test. |
| **4. Modelos y comparación** | Tabla con los tres modelos, baseline, cinco folds, media, desviación estándar, mejores parámetros de `GridSearchCV` y gráfico comparativo. |
| **5. Evaluación** | Modelo ganador seleccionado con CV; métricas finales sobre test e interpretación y visualización de errores. |
| **6. Predicciones** | Formulario con valores de predictores, validación de entradas, botón *Predecir*, salida individual y posibilidad de demostrar cinco casos nuevos. |

### H.1. Arquitectura de archivos mínima

```text
├── app.py                     # Archivo de inicio de Streamlit; navegación o secciones.
├── src/
│   ├── data.py                # Carga, verificación y descripción del CSV (o módulo equivalente).
│   ├── ml.py                  # Preprocesamiento, entrenamiento, CV, selección y evaluación.
│   └── predict.py             # Carga de pipeline y predicción de observaciones nuevas.
├── models/
│   └── modelo_final.joblib    # Pipeline completo entrenado y guardado (o script reproducible).
├── requirements.txt           # Dependencias con versiones compatibles.
├── README.md                  # Instalación, estructura, fuente Kaggle, ejecución y capturas.
└── data/                      # CSV si la licencia lo permite, o instrucciones de descarga.