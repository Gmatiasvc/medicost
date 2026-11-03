# Documentación, Informe Académico y Presentación (Grupo 09)

Esta carpeta contiene la documentación académica, técnica y la presentación en diapositivas desarrollada según los requerimientos de `ejercicio.md`.

---

## 📁 Contenido del Directorio

1. **`informe_academico.md`**: Informe académico completo en formato Markdown (equivalente a 12-20 páginas) estructurado según el **Anexo D** y las **16 etapas** del proyecto. Incluye las tablas oficiales (**T1 a T9**), ecuaciones matemáticas, análisis de asimetría de `charges`, análisis de subgrupos (fumadores vs. no fumadores y niveles de gasto), precauciones metodológicas de no causalidad y referencias en formato APA 7.
2. **`informe_academico.tex`**: Versión completa del informe en código fuente **LaTeX**, listo para compilar con `pdflatex` o `xelatex`. Cuenta con maquetación profesional (`booktabs`, `amsmath`, `hyperref`, `tabularx`, paleta de colores institucional, encabezados y pies de página).
3. **`slides.md`**: Presentación de diapositivas interactiva desarrollada con **Slidev**, siguiendo exactamente las 13 diapositivas requeridas en el **Anexo F** más una sección para la defensa oral y preguntas de sustentación.
4. **`style.css`**: Tema visual personalizado, moderno y elegante para Slidev, con paleta de colores acorde a salud y Machine Learning (*Deep Slate, Emerald, Cyan y Rose*), tarjetas con efecto *glassmorphism*, insignias de estado y tipografía estilizada.
5. **`images/`**: 9 gráficos de alta resolución generados directamente a partir de los datos y modelos:
   - `eda_charges_distribution.png`: Distribución y bimodalidad de `charges`.
   - `eda_age_vs_charges.png`: Dispersión de edad y regímenes clínicos paralelos.
   - `eda_bmi_vs_charges.png`: Relación IMC vs. cargos y umbral OMS ($BMI = 30$).
   - `eda_smoker_boxplot.png`: Boxplots por condición de fumador y sexo.
   - `eda_correlation_multivariate.png`: Matriz de correlación y análisis regional.
   - `ml_cv_comparison.png`: Comparación de modelos en Validación Cruzada 5-Fold.
   - `ml_real_vs_pred.png`: Valores reales vs. predichos en prueba reservada ($n=268$).
   - `ml_residuals_analysis.png`: Diagnóstico de residuales y mediana de error ($40.14).
   - `ml_smoker_error_comparison.png`: Comparación de error absoluto en fumadores vs. no fumadores.
6. **`dist/`**: Compilación estática generada por `npm run build`, lista para previsualizar en cualquier navegador sin necesidad de iniciar un servidor local.

---

## 🚀 Cómo Ejecutar la Presentación Slidev

Desde la raíz del repositorio o dentro de `presentacion/`:

```bash
# 1. Ingresar a la carpeta
cd presentacion

# 2. Instalar dependencias (si es la primera vez)
npm install

# 3. Iniciar el servidor interactivo de diapositivas
npm run dev
```

La presentación estará disponible en `http://localhost:3030`.

### Para generar la versión web de producción:
```bash
npm run build
```
Los archivos optimizados se guardan en `presentacion/dist/`.
