---
theme: default
title: Estimación de Gastos de Seguros Médicos - Grupo 02
info: |
  Sistemas Inteligentes y Machine Learning
  Sustentación del Proyecto de Regresión Supervisada
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
css: ./style.css
---

# Estimación de Gastos de Seguros Médicos
### Sistema Predictivo Supervisado de Regresión sin Fuga de Información

<div class="mt-8 flex justify-center gap-3">
  <span class="badge badge-emerald">Grupo 02</span>
  <span class="badge badge-cyan">Regresión Supervisada</span>
  <span class="badge badge-rose">SVR Optimizado</span>
  <span class="badge badge-amber">Streamlit App</span>
</div>

<div class="grid-3 mt-10 text-left">
  <div class="card card-emerald">
    <div class="kpi-label">Problema Central</div>
    <div class="text-sm mt-1 text-slate-300">Incertidumbre en la predicción de costos médicos anuales (<code>charges</code>) al momento de suscripción de la póliza ($t=0$).</div>
  </div>
  <div class="card card-cyan">
    <div class="kpi-label">Modelo Ganador</div>
    <div class="text-sm mt-1 text-slate-300"><b>SVR (Kernel RBF)</b> optimizado vía <code>GridSearchCV</code> sobre 5 pliegues idénticos de entrenamiento.</div>
  </div>
  <div class="card card-rose">
    <div class="kpi-label">Resultado Final en Test</div>
    <div class="text-sm mt-1 text-slate-300"><b>MAE: $1,803.44 USD</b> (-81.2% vs. baseline) y <b>$R^2 = 0.8496$</b> sobre conjunto reservado ($n=268$).</div>
  </div>
</div>

<div class="mt-8 text-xs text-slate-400">
  <b>Integrantes:</b> Gerardo Matías Venegas Calizaya (Data & ML Engineer) & Equipo de Proyecto Grupo 02<br>
  <b>Curso:</b> Sistemas Inteligentes y Machine Learning | Ciclo 2026-II
</div>

---
layout: default
---

# 1. Contexto Operativo y Pregunta Predictiva
### ¿Quién usa la predicción, en qué momento y para qué decisión actuarial?

<div class="grid-2 mt-4">
  <div class="card card-cyan">
    <h3 class="text-base text-cyan-400 font-bold mb-2">Momento de Inferencia ($t=0$) y Decisor</h3>
    <ul class="text-xs space-y-2 text-slate-300">
      <li><b>Usuario Decisor:</b> Suscriptor médico de riesgos (<i>underwriter</i>) y comité actuarial de salud.</li>
      <li><b>Instante de Evaluación:</b> Al solicitar la póliza o renovación, <u>antes</u> de la ocurrencia de siniestros.</li>
      <li><b>Decisión Clave:</b> Fijación de prima pura individualizada, suficiencia de reserva técnica de siniestros no reportados (IBNR) y asignación a programas preventivos.</li>
      <li><b>Disponibilidad Ex-Ante:</b> Las 6 predictoras ($X$) se declaran al afiliarse; ninguna depende de diagnósticos o gastos futuros.</li>
    </ul>
  </div>

  <div class="card card-emerald">
    <h3 class="text-base text-emerald-400 font-bold mb-2">Esquema Predictivo sin Fuga</h3>
    <div class="text-xs text-slate-300 mb-2">
      $$\hat{y} = f(\mathbf{X}) \quad \text{donde } \hat{y} \in \mathbb{R}^+ \text{ (USD \$)}$$
    </div>
    <div class="space-y-1 text-xs">
      <div class="p-2 rounded bg-slate-800/60 border border-slate-700/50">
        <span class="text-cyan-400 font-semibold">Entradas ($X$):</span> <code>age</code>, <code>sex</code>, <code>bmi</code>, <code>children</code>, <code>smoker</code>, <code>region</code>
      </div>
      <div class="p-2 rounded bg-slate-800/60 border border-slate-700/50">
        <span class="text-emerald-400 font-semibold">Pipeline:</span> <code>ColumnTransformer</code> (Median + Mode Imputer, Scaler, OneHot)
      </div>
      <div class="p-2 rounded bg-slate-800/60 border border-slate-700/50">
        <span class="text-rose-400 font-semibold">Salida ($\hat{y}$):</span> Estimación de cargos médicos esperados para el año $[t, t+1]$
      </div>
    </div>
  </div>
</div>

<div class="card card-amber mt-4 py-2 px-3">
  <span class="text-xs text-amber-300"><b>Pregunta Predictiva Central:</b> ¿Qué precisión cuantitativa (MAE) alcanza un modelo supervisado al estimar el costo individual sin incurrir en fuga de información y evaluado una sola vez en datos de prueba reservados?</span>
</div>

---
layout: default
---

# 2. Objetivos y Criterio de Éxito Acordado
### Definición ex-ante de métricas de aceptación y costo del error

<div class="grid-3 mt-4">
  <div class="card card-emerald">
    <div class="kpi-label">Métrica Primaria</div>
    <div class="kpi-value text-emerald-400">MAE</div>
    <div class="text-xs text-slate-300 mt-1">
      <b>Menor es mejor.</b> Robusto frente a la asimetría de <code>charges</code>. Mide el desvío directo en dólares reales ($) sin sobredimensionar eventos fortuitos.
    </div>
  </div>

  <div class="card card-cyan">
    <div class="kpi-label">Métricas Secundarias</div>
    <div class="kpi-value text-cyan-400">RMSE & $R^2$</div>
    <div class="text-xs text-slate-300 mt-1">
      <b>$R^2 \ge 0.75$</b> para varianza explicada. RMSE cuantifica la dispersión cuadrática del riesgo de cola catastrófico.
    </div>
  </div>

  <div class="card card-rose">
    <div class="kpi-label">Línea Base Ingenua</div>
    <div class="kpi-value text-rose-400">$9,007</div>
    <div class="text-xs text-slate-300 mt-1">
      <code>DummyRegressor(mean)</code>. Umbral mínimo a superar: <b>MAE $< \$3,500$ USD</b> en prueba.
    </div>
  </div>
</div>

<div class="card card-rose mt-4">
  <h3 class="text-sm font-bold text-rose-400 mb-1">Costo Asimétrico del Error en Seguros de Salud</h3>
  <div class="grid-2 text-xs text-slate-300 gap-4">
    <div>
      <span class="text-rose-400 font-semibold">1. Subestimación ($\hat{y} \ll y$, error positivo):</span> Insuficiencia de prima y reservas técnicas; la aseguradora absorbe el déficit hospitalario afectando su margen de solvencia.
    </div>
    <div>
      <span class="text-amber-400 font-semibold">2. Sobreestimación ($\hat{y} \gg y$, error negativo):</span> Pérdida comercial de clientes saludables por tarifas infladas, induciendo selección adversa de cartera.
    </div>
  </div>
</div>

---
layout: default
---

# 3. Procedencia, Dataset y Diccionario de Variables
### Medical Cost Personal Datasets (Miri Choi / Brett Lantz) | Tablas T1 y T2

<div class="grid-2 mt-2">
  <div>
    <div class="text-xs text-slate-300 mb-2">
      <b>Fuente:</b> Kaggle (<code>mirichoi0218/insurance</code>) | $n = 1,338$ filas $\times 7$ columnas | Licencia DbCL v1.0
    </div>
    <table>
      <thead>
        <tr>
          <th>Variable</th>
          <th>Tipo</th>
          <th>Rol</th>
          <th>Unidad / Rango</th>
          <th>Decisión</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><code>age</code></td><td>int64</td><td>Predictora ($X$)</td><td>18 a 64 años</td><td><span class="badge badge-emerald">Incluir</span></td></tr>
        <tr><td><code>sex</code></td><td>object</td><td>Predictora ($X$)</td><td>female, male</td><td><span class="badge badge-emerald">Incluir</span></td></tr>
        <tr><td><code>bmi</code></td><td>float64</td><td>Predictora ($X$)</td><td>15.96 a 53.13 kg/m²</td><td><span class="badge badge-emerald">Incluir</span></td></tr>
        <tr><td><code>children</code></td><td>int64</td><td>Predictora ($X$)</td><td>0 a 5 hijos</td><td><span class="badge badge-emerald">Incluir</span></td></tr>
        <tr><td><code>smoker</code></td><td>object</td><td>Predictora ($X$)</td><td>yes, no</td><td><span class="badge badge-emerald">Incluir</span></td></tr>
        <tr><td><code>region</code></td><td>object</td><td>Predictora ($X$)</td><td>4 regiones cardinales</td><td><span class="badge badge-emerald">Incluir</span></td></tr>
        <tr><td><code>charges</code></td><td>float64</td><td><b>Objetivo ($y$)</b></td><td>\$1,121 a \$63,770</td><td><span class="badge badge-rose">Excluir de X</span></td></tr>
      </tbody>
    </table>
  </div>

  <div class="card card-emerald">
    <h3 class="text-sm font-bold text-emerald-400 mb-2">Auditoría de Calidad (Tabla T2)</h3>
    <ul class="text-xs space-y-2 text-slate-300">
      <li><b>Valores Nulos:</b> <code>0 celdas faltantes (0.00%)</code> en todo el dataset.</li>
      <li><b>Duplicados:</b> <code>1 fila exacta</code> (índices 195 y 581: asegurado de 19 años no fumador). Se conserva por ser demográficamente plausible.</li>
      <li><b>Extremos en BMI:</b> 9 casos ($0.67\%$) $> 47.29$ kg/m² (obesidad mórbida real).</li>
      <li><b>Extremos en Charges:</b> 139 registros ($10.39\%$) $> \$34,489.35$ USD. El <b>97.84% son fumadores</b> de alta siniestralidad.</li>
      <li><b>Medida Adoptada:</b> No eliminar outliers (destruiría el fenómeno actuarial); evaluar con MAE y pérdida $\epsilon$-insensible.</li>
    </ul>
  </div>
</div>

---
layout: default
---

# 4. Análisis Exploratorio (EDA) y Hallazgos Clave
### Asimetría, regímenes de costo y el umbral crítico de la OMS

<div class="grid-3 mt-2">
  <div class="card">
    <img src="/images/eda_charges_distribution.png" class="rounded h-42 object-contain mx-auto" />
    <div class="text-2xs text-slate-300 mt-1">
      <b>Charges:</b> Asimetría derecha ($\text{Skew} = 1.52$, $\text{Kurt} = 1.61$). Media (\$13,270) $\gg$ Mediana (\$9,382). Bimodalidad por tabaquismo.
    </div>
  </div>

  <div class="card">
    <img src="/images/eda_age_vs_charges.png" class="rounded h-42 object-contain mx-auto" />
    <div class="text-2xs text-slate-300 mt-1">
      <b>Edad vs. Cargos:</b> Tres franjas paralelas ascendentes: (1) no fumadores, (2) fumadores sin obesidad ($BMI < 30$), y (3) fumadores con obesidad ($BMI \ge 30$).
    </div>
  </div>

  <div class="card">
    <img src="/images/eda_bmi_vs_charges.png" class="rounded h-42 object-contain mx-auto" />
    <div class="text-2xs text-slate-300 mt-1">
      <b>BMI vs. Cargos:</b> Salto no lineal discontinuo en $BMI = 30$ kg/m² (corte OMS). En fumadores el costo salta de \$18k a \$40k+.
    </div>
  </div>
</div>

<div class="grid-2 mt-3 gap-3">
  <div class="card py-1.5 px-3">
    <span class="text-xs text-cyan-300"><b>Correlaciones:</b> <code>smoker_yes</code> lidera ($r = 0.79$), seguido de <code>age</code> ($0.30$) y <code>bmi</code> ($0.20$), sin colinealidad entre predictores ($|r| \le 0.11$).</span>
  </div>
  <div class="card py-1.5 px-3">
    <span class="text-xs text-emerald-300"><b>Implicación para Modelado:</b> Modelos lineales aditivos fracasan; se requieren algoritmos que resuelvan interacciones no lineales (SVR RBF y KNN).</span>
  </div>
</div>

---
layout: default
---

# 5. Partición Experimental y Protocolo Libre de Fuga
### Reserva estricta del 20% antes de transformaciones | Tabla T3

<div class="grid-2 mt-4">
  <div class="card card-emerald">
    <h3 class="text-sm font-bold text-emerald-400 mb-2">Partición 80/20 (Tabla T3)</h3>
    <table class="text-xs">
      <thead>
        <tr>
          <th>Subconjunto</th>
          <th>$n$</th>
          <th>\%</th>
          <th>Media ($)</th>
          <th>Mediana ($)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><b>Train</b></td><td>1,070</td><td>80.0%</td><td>\$13,346</td><td>\$9,413</td></tr>
        <tr><td><b>Test</b></td><td>268</td><td>20.0%</td><td>\$12,968</td><td>\$9,194</td></tr>
        <tr><td><b>Total</b></td><td>1,338</td><td>100.0%</td><td>\$13,270</td><td>\$9,382</td></tr>
      </tbody>
    </table>
    <div class="mt-2 text-xs text-slate-300">
      <code>random_state = 42</code> | <b>Filas Compartidas = 0</b> (conjuntos disjuntos).
    </div>
  </div>

  <div class="card card-cyan">
    <h3 class="text-sm font-bold text-cyan-400 mb-2">Protocolo Anti-Leakage Riguroso</h3>
    <ul class="text-xs space-y-2 text-slate-300">
      <li><b>Reserva Total:</b> El conjunto de prueba ($n=268$) se bloqueó inmediatamente tras la partición y no intervino en ninguna decisión de ajuste.</li>
      <li><b>Transformaciones Aisladas:</b> Nunca se calculó la media o varianza sobre todo el CSV. Todos los parámetros se aprendieron solo en Train.</li>
      <li><b>Validación 5-Fold Interna:</b> Dentro de cada pliegue, el preprocesador se ajusta solo en 4 folds y transforma el 5to fold sin fuga.</li>
    </ul>
  </div>
</div>

```python
# Código reproducible ejecutado en src/ml.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)
assert len(set(X_train.index).intersection(set(X_test.index))) == 0
```

---
layout: default
---

# 6. Preprocesamiento Modular con `ColumnTransformer`
### Tratamiento diferenciado de ramas numéricas y categóricas | Tabla T4

<div class="grid-2 mt-3">
  <div class="card card-cyan">
    <h3 class="text-sm font-bold text-cyan-400 mb-2">Estructura del Preprocesador</h3>
    <div class="text-xs space-y-2 text-slate-300">
      <div class="p-2 rounded bg-slate-800/70 border border-cyan-500/30">
        <b>Rama Numérica:</b> <code>['age', 'bmi', 'children']</code><br>
        1. <code>SimpleImputer(strategy='median')</code><br>
        2. <code>StandardScaler()</code> $(\mu=0, \sigma=1)$
      </div>
      <div class="p-2 rounded bg-slate-800/70 border border-emerald-500/30">
        <b>Rama Categórica:</b> <code>['sex', 'smoker', 'region']</code><br>
        1. <code>SimpleImputer(strategy='most_frequent')</code><br>
        2. <code>OneHotEncoder(handle_unknown='ignore')</code>
      </div>
    </div>
    <div class="mt-2 text-xs text-slate-400">
      El vector transformado genera <b>11 columnas ortogonales</b> normalizadas.
    </div>
  </div>

  <div class="card card-emerald">
    <h3 class="text-sm font-bold text-emerald-400 mb-2">Justificación Técnica (Tabla T4)</h3>
    <ul class="text-xs space-y-2 text-slate-300">
      <li><b>Por qué StandardScaler:</b> En KNN y SVR, las distancias euclidianas y los productos internos serían dominados por la edad (18-64) y el BMI (15-53) frente a los dependientes (0-5) si no se estandariza.</li>
      <li><b>Por qué OneHotEncoder con ignore:</b> Previene errores por categorías imprevistas en producción y evita asignar falsas jerarquías a las regiones.</li>
      <li><b>Escala de Charges:</b> Se mantiene en dólares reales (\$ USD) para conservar la interpretabilidad monetaria directa en los comités actuariales.</li>
    </ul>
  </div>
</div>

---
layout: default
---

# 7. Modelos Candidatos y Fundamentos Algorítmicos
### Tres familias de modelado supervisado frente a la línea base

<div class="grid-4 mt-4">
  <div class="card card-rose">
    <div class="kpi-label">Baseline</div>
    <div class="text-xs font-bold text-rose-400 mt-1">DummyRegressor</div>
    <div class="text-2xs text-slate-300 mt-2">
      <b>Mecanismo:</b> Predice la media de train ($\hat{y} = \bar{y}$).<br>
      <b>Fortaleza:</b> Piso mínimo referencial con costo cero.<br>
      <b>Límite:</b> No discrimina perfiles ($R^2 \approx 0$).
    </div>
  </div>

  <div class="card card-cyan">
    <div class="kpi-label">Paramétrico Lineal</div>
    <div class="text-xs font-bold text-cyan-400 mt-1">LinearRegression</div>
    <div class="text-2xs text-slate-300 mt-2">
      <b>Mecanismo:</b> Mínimos cuadrados ordinarios (OLS).<br>
      <b>Fortaleza:</b> Interpretable y computacionalmente instantáneo.<br>
      <b>Límite:</b> Supone aditividad; no modela la sinergia tabaco × BMI.
    </div>
  </div>

  <div class="card card-emerald">
    <div class="kpi-label">No Paramétrico</div>
    <div class="text-xs font-bold text-emerald-400 mt-1">KNeighborsRegressor</div>
    <div class="text-2xs text-slate-300 mt-2">
      <b>Mecanismo:</b> Promedio de $k$ vecinos locales.<br>
      <b>Fortaleza:</b> No asume distribución; flexible.<br>
      <b>Límite:</b> Sensible a la dimensionalidad y ruido en fronteras.
    </div>
  </div>

  <div class="card card-amber">
    <div class="kpi-label">Kernel No Lineal</div>
    <div class="text-xs font-bold text-amber-400 mt-1">SVR (Núcleo RBF)</div>
    <div class="text-2xs text-slate-300 mt-2">
      <b>Mecanismo:</b> Tubo $\epsilon$-insensible en espacio de Hilbert.<br>
      <b>Fortaleza:</b> Modela fronteras complejas y tolera outliers.<br>
      <b>Límite:</b> Requiere calibrar $C$, $\gamma$ y $\epsilon$.
    </div>
  </div>
</div>

<div class="card mt-4 py-2 px-3 text-xs text-slate-300">
  <b>Configuración de Pipelines:</b> Cada estimador fue acoplado al <code>ColumnTransformer</code> formando un <code>Pipeline</code> hermético donde el ajuste y la transformación se gestionan en una sola llamada <code>.fit()</code>.
</div>

---
layout: default
---

# 8. Validación Cruzada 5-Fold y Optimización GridSearchCV
### Búsqueda de hiperparámetros exclusivamente sobre entrenamiento | Tablas T5 y T6

<div class="grid-2 mt-2">
  <div class="card">
    <h3 class="text-xs font-bold text-cyan-400 mb-1">Validación Inicial 5-Fold (Tabla T5)</h3>
    <table class="text-2xs">
      <thead>
        <tr>
          <th>Modelo</th>
          <th>Media MAE ($)</th>
          <th>DE MAE ($)</th>
          <th>Media $R^2$</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Dummy Baseline</td><td>\$9,007.09</td><td>\$326.06</td><td>-0.0042</td></tr>
        <tr><td>LinearRegression</td><td>\$4,234.98</td><td>\$277.28</td><td>0.7389</td></tr>
        <tr><td>KNN (k=5)</td><td>\$3,654.66</td><td>\$184.48</td><td>0.7543</td></tr>
        <tr><td>SVR (C=1.0 defecto)</td><td>\$8,272.97</td><td>\$350.26</td><td>-0.0987</td></tr>
      </tbody>
    </table>
    <div class="text-3xs text-rose-300 mt-1">
      *SVR con C=1.0 subajusta severamente porque no puede cubrir la escala monetaria hasta \$60,000.
    </div>
  </div>

  <div class="card">
    <h3 class="text-xs font-bold text-emerald-400 mb-1">Optimización con GridSearchCV (Tabla T6)</h3>
    <table class="text-2xs">
      <thead>
        <tr>
          <th>Modelo</th>
          <th>Mejor Configuración</th>
          <th>MAE Antes</th>
          <th>MAE Después</th>
          <th>Mejora</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>KNN</b></td>
          <td>$k=5$, weights='distance', $p=1$</td>
          <td>\$3,654.66</td>
          <td><b>\$3,274.14</b></td>
          <td>-10.4%</td>
        </tr>
        <tr>
          <td><b>SVR</b></td>
          <td>kernel='rbf', $C=10^4$, $\gamma=\text{'scale'}$, $\epsilon=1$</td>
          <td>\$8,272.97</td>
          <td><b>\$2,014.67</b></td>
          <td><span class="text-emerald-400 font-bold">-75.6%</span></td>
        </tr>
      </tbody>
    </table>
    <div class="text-3xs text-emerald-300 mt-1">
      *Al elevar $C$ a 10,000, SVR adquiere la flexibilidad necesaria para resolver la sinergia tabaco × BMI.
    </div>
  </div>
</div>

<div class="mt-2">
  <img src="/images/ml_cv_comparison.png" class="rounded h-42 object-contain mx-auto" />
</div>

---
layout: default
---

# 9. Selección Razonada del Modelo Ganador (Pre-Test)
### Decisión transparente basada exclusivamente en Validación Cruzada | Tabla T7

<table class="text-xs mt-2">
  <thead>
    <tr>
      <th>Candidato Evaluado</th>
      <th>Configuración</th>
      <th>Media MAE CV ($)</th>
      <th>DE MAE ($)</th>
      <th>Media $R^2$ CV</th>
      <th>Estabilidad CV</th>
      <th>Decisión Pre-Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DummyRegressor</td>
      <td><code>strategy='mean'</code></td>
      <td>\$9,007.09</td>
      <td>\$326.06</td>
      <td>-0.0042</td>
      <td>3.62%</td>
      <td>Línea base referencial</td>
    </tr>
    <tr>
      <td>LinearRegression</td>
      <td>OLS ordinario</td>
      <td>\$4,234.98</td>
      <td>\$277.28</td>
      <td>0.7389</td>
      <td>6.55%</td>
      <td>Descartado (alto error aditivo)</td>
    </tr>
    <tr>
      <td>KNN Optimizado</td>
      <td>$k=5$, Distancia, Manhattan</td>
      <td>\$3,274.14</td>
      <td>\$263.34</td>
      <td>0.7860</td>
      <td>8.04%</td>
      <td>Descartado (segundo lugar)</td>
    </tr>
    <tr class="bg-emerald-950/40 border border-emerald-500/50">
      <td><b>SVR Optimizado</b></td>
      <td><b>RBF, $C=10^4$, $\gamma=\text{'scale'}$, $\epsilon=1$</b></td>
      <td><b>\$2,014.67</b></td>
      <td><b>\$322.28</b></td>
      <td><b>0.8289</b></td>
      <td><b>16.00%</b></td>
      <td><span class="badge badge-emerald">GANADOR BLOQUEADO</span></td>
    </tr>
  </tbody>
</table>

<div class="card card-emerald mt-3">
  <h3 class="text-xs font-bold text-emerald-400 mb-1">Fundamentos de la Elección Previa a Abrir Test</h3>
  <div class="grid-3 text-2xs text-slate-300 gap-2">
    <div>
      <b>1. Menor MAE Absoluto:</b> \$2,014.67 USD en 5-Fold CV, superando por \$1,259 a KNN y por \$2,220 a la regresión lineal.
    </div>
    <div>
      <b>2. Resolución No Lineal:</b> El núcleo RBF proyecta la discontinuidad tabaco-obesidad sin requerir ingeniería manual de polinomios.
    </div>
    <div>
      <b>3. Robustez $\epsilon$-insensible:</b> Tolera los 139 outliers de siniestralidad alta sin distorsionar el hiperplano óptimo.
    </div>
  </div>
</div>

---
layout: default
---

# 10. Evaluación Única en Test y Análisis de Residuales
### Evaluación final sobre las 268 observaciones reservadas | Tabla T8

<div class="grid-2 mt-2">
  <div class="card card-emerald">
    <h3 class="text-xs font-bold text-emerald-400 mb-1">Métricas en Test Reservado (Tabla T8)</h3>
    <table class="text-2xs">
      <thead>
        <tr>
          <th>Métrica</th>
          <th>SVR Ganador</th>
          <th>Baseline</th>
          <th>Mejora Absoluta</th>
          <th>Mejora (\%)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><b>MAE</b></td><td><b>\$1,803.44</b></td><td>\$9,593.34</td><td>-\$7,789.89</td><td><b>-81.20%</b></td></tr>
        <tr><td><b>RMSE</b></td><td><b>\$4,832.09</b></td><td>\$12,465.61</td><td>-\$7,633.52</td><td><b>-61.24%</b></td></tr>
        <tr><td><b>$R^2$</b></td><td><b>0.8496</b></td><td>-0.0009</td><td>+0.8505</td><td><b>84.96% exp.</b></td></tr>
      </tbody>
    </table>
    <div class="mt-2 text-2xs text-slate-300">
      <b>Estadísticas de Residuales ($y - \hat{y}$):</b><br>
      • Mediana de error absoluto: <b>\$40.14 USD</b> por póliza.<br>
      • El $50\%$ de los errores son $\le \$40$ USD (ajuste exacto en cartera sana).
    </div>
  </div>

  <div>
    <img src="/images/ml_real_vs_pred.png" class="rounded h-44 object-contain mx-auto" />
  </div>
</div>

<div class="card mt-2 py-1.5 px-3 text-xs text-slate-300">
  <b>Interpretación Actuarial:</b> La brecha entre RMSE (\$4,832) y MAE (\$1,803) evidencia la presencia de hospitalizaciones imprevistas en no fumadores (residuales de hasta +\$26,317 USD), las cuales penalizan la métrica cuadrática.
</div>

---
layout: default
---

# 11. Requisito Grupo 02: Fumadores vs. No Fumadores
### Comparación de MAE entre estratos y advertencia metodológica de no causalidad

<table class="text-2xs mt-2">
  <thead>
    <tr>
      <th>Subgrupo Analizado</th>
      <th>$n$ Test</th>
      <th>\%</th>
      <th>Gasto Medio ($)</th>
      <th>MAE en Test ($)</th>
      <th>Mediana Error ($)</th>
      <th>MAE Relativo</th>
      <th>$R^2$ Intra-Grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>No Fumadores (`smoker='no'`)</b></td>
      <td>214</td>
      <td>79.85%</td>
      <td>\$7,862.09</td>
      <td><b>\$1,219.08</b></td>
      <td><b>\$27.38</b></td>
      <td>15.51%</td>
      <td>0.4344</td>
    </tr>
    <tr>
      <td><b>Fumadores (`smoker='yes'`)</b></td>
      <td>54</td>
      <td>20.15%</td>
      <td>\$33,204.10</td>
      <td><b>\$4,119.27</b></td>
      <td><b>\$3,061.69</b></td>
      <td><b>12.41%</b></td>
      <td><b>0.6778</b></td>
    </tr>
  </tbody>
</table>

<div class="grid-2 mt-2 gap-3">
  <div class="card card-cyan">
    <h3 class="text-xs font-bold text-cyan-400 mb-1">Efecto Escala Monetaria vs. Precisión</h3>
    <p class="text-2xs text-slate-300">
      El MAE nominal de fumadores es mayor (\$4,119 vs. \$1,219) porque operan en una escala $4.2\times$ superior (\$33k vs. \$8k). <b>Porcentualmente, el modelo es más preciso en fumadores (12.4% vs. 15.5%)</b> y explica el $67.8\%$ de su varianza interna.
    </p>
  </div>

  <div class="card card-rose">
    <h3 class="text-xs font-bold text-rose-400 mb-1">⚠️ Precaución Metodológica: No Causalidad</h3>
    <p class="text-2xs text-slate-300">
      La elevada correlación entre <code>smoker='yes'</code> y costo médico refleja una <b>asociación observacional de facturación</b>, no un efecto causal directo. Existen variables confusoras no medidas (genética, antecedentes laborales, comorbilidades).
    </p>
  </div>
</div>

<div class="mt-2">
  <img src="/images/ml_smoker_error_comparison.png" class="rounded h-36 object-contain mx-auto" />
</div>

---
layout: default
---

# 12. Demostración de Nuevas Predicciones Plausibles
### Inferencia en tiempo real sobre perfiles no empleados en entrenamiento | Tabla T9

<table class="text-2xs mt-2">
  <thead>
    <tr>
      <th>Caso</th>
      <th>Edad</th>
      <th>Sexo</th>
      <th>BMI</th>
      <th>Hijos</th>
      <th>Fuma</th>
      <th>Región</th>
      <th>Cargo Estimado ($\hat{y}$)</th>
      <th>Perfil Actuarial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Caso 1</b></td><td>22</td><td>F</td><td>21.8</td><td>0</td><td>no</td><td>NW</td>
      <td><span class="text-emerald-400 font-bold">\$2,545.38</span></td>
      <td>Riesgo basal mínimo: juventud, normopeso, no fumadora.</td>
    </tr>
    <tr>
      <td><b>Caso 2</b></td><td>41</td><td>M</td><td>27.4</td><td>3</td><td>no</td><td>NE</td>
      <td><span class="text-cyan-400 font-bold">\$8,049.71</span></td>
      <td>Póliza familiar: gasto medio regular por edad y 3 hijos.</td>
    </tr>
    <tr>
      <td><b>Caso 3</b></td><td>27</td><td>F</td><td>23.5</td><td>0</td><td>yes</td><td>SW</td>
      <td><span class="text-amber-400 font-bold">\$16,556.97</span></td>
      <td>Fumadora joven: salto al régimen intermedio ($BMI < 30$).</td>
    </tr>
    <tr>
      <td><b>Caso 4</b></td><td>61</td><td>F</td><td>31.2</td><td>1</td><td>no</td><td>SE</td>
      <td><span class="text-cyan-400 font-bold">\$13,632.43</span></td>
      <td>Adulta mayor con sobrepeso, pero contenida por no fumar.</td>
    </tr>
    <tr>
      <td><b>Caso 5</b></td><td>52</td><td>M</td><td>36.8</td><td>2</td><td>yes</td><td>SE</td>
      <td><span class="text-rose-400 font-bold">\$47,134.61</span></td>
      <td>Máximo riesgo: sinergia edad $> 50$, tabaco y obesidad mórbida.</td>
    </tr>
  </tbody>
</table>

<div class="card card-emerald mt-3">
  <div class="grid-2 text-xs text-slate-300">
    <div>
      <b>Garantía de No Reentrenamiento:</b> Invocación directa de <code>predict_single()</code> y <code>predict_batch()</code> en <code>src/predict.py</code> consumiendo el pipeline serializado en <code>models/modelo_final.joblib</code>.
    </div>
    <div>
      <b>Trazabilidad Total:</b> El nuevo perfil ingresa como DataFrame crudo y es preprocesado con los transformadores ajustados exclusivamente en entrenamiento.
    </div>
  </div>
</div>

---
layout: default
---

# 13. Conclusiones y Mejoras Futuras Priorizadas
### Dictamen actuarial y hoja de ruta tecnológica

<div class="grid-2 mt-2">
  <div class="card card-emerald">
    <h3 class="text-xs font-bold text-emerald-400 mb-1">Cinco Conclusiones Clave</h3>
    <ol class="text-2xs space-y-1 text-slate-300 list-decimal pl-3">
      <li><b>Éxito Cuantitativo:</b> SVR superó las metas con $\text{MAE} = \$1,803$ USD y $R^2 = 0.8496$ (reducción del $81.2\%$ vs. baseline).</li>
      <li><b>Insuficiencia Lineal:</b> La sinergia tabaco × obesidad ($BMI \ge 30$) hace obligatorios algoritmos de proyección no lineal.</li>
      <li><b>Impacto de GridSearchCV:</b> SVR pasó de subajustar ($\text{MAE} = \$8,272$) a liderar ($\text{MAE} = \$2,014$) calibrando $C=10,000$.</li>
      <li><b>Tolerancia a Outliers:</b> La pérdida $\epsilon$-insensible logró una mediana de error absoluto de solo $\$40.14$ USD en test.</li>
      <li><b>Cero Fuga Certificada:</b> Arquitectura desacoplada en <code>Pipeline</code> y <code>ColumnTransformer</code> con 0 filas compartidas.</li>
    </ol>
  </div>

  <div class="card card-cyan">
    <h3 class="text-xs font-bold text-cyan-400 mb-1">Tres Mejoras Priorizadas</h3>
    <ul class="text-2xs space-y-2 text-slate-300">
      <li><b>P1. Variables Clínicas e Historial:</b> Agregar diagnósticos CIE-10 y siniestros del año $t-1$ para anticipar cirugías en no fumadores.</li>
      <li><b>P2. Intervalos Conformal Prediction:</b> Suministrar intervalos de confianza al $90\%$ para cálculo estocástico de solvencia.</li>
      <li><b>P3. Modelos Basados en Árboles (LightGBM):</b> Evaluar gradiente boosting con restricciones de monotonicidad positiva sobre la edad.</li>
    </ul>
    <div class="mt-2 p-1.5 rounded bg-emerald-950/60 border border-emerald-500/30 text-2xs text-emerald-300">
      <b>Dictamen de Despliegue:</b> <u>Aprobado para modo sombra (*shadow mode*)</u> en tarificación de apoyo actuarial; no autorizado para tarificación 100% desatendida sin supervisión humana.
    </div>
  </div>
</div>

---
layout: center
class: text-center
---

# ¡Gracias por su atención!
### Sustentación Oral y Defensa Técnica | Grupo 02

<div class="flex justify-center gap-4 mt-6">
  <div class="card card-emerald w-72 text-left">
    <h4 class="text-xs font-bold text-emerald-400 mb-1">Aplicación Interactiva</h4>
    <p class="text-2xs text-slate-300">
      <code>streamlit run app.py</code><br>
      6 módulos completos interactivos (Etapas 1 a 16).
    </p>
  </div>

  <div class="card card-cyan w-72 text-left">
    <h4 class="text-xs font-bold text-cyan-400 mb-1">Documentación Entregada</h4>
    <p class="text-2xs text-slate-300">
      • Informe Académico (Markdown & LaTeX)<br>
      • Presentación Interactiva (Slidev)
    </p>
  </div>
</div>

<div class="mt-8 text-xs text-slate-400">
  <b>Ronda de Preguntas de la Comisión Evaluadora</b><br>
  <i>Sistemas Inteligentes y Machine Learning | Ciclo 2026-II</i>
</div>
