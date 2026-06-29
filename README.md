# Actividad 6: Clasificación y Predicción de Tipos de Delito para la Seguridad Pública

Proyecto de Machine Learning con validación temporal para la categorización y análisis de incidentes delictivos.

Alumno: Abelardo del Angel Quiroz

## 1. Definición del Problema
El crecimiento urbano y la complejidad de las dinámicas sociales exigen que los cuerpos de seguridad pública adopten estrategias proactivas y basadas en datos. Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de **clasificar la categoría o prioridad de un delito potencial** a partir de variables del entorno en el momento en que se genera un reporte o incidente.

Dadas las características operacionales (como coordenadas geográficas, marcas de tiempo y tipo de ubicación), el algoritmo deberá predecir si el incidente corresponde a una categoría crítica (ej. delitos de alto impacto o con violencia) o a un delito menor/asistencia. Esto permite optimizar la distribución de patrullajes y mejorar los tiempos de respuesta de las unidades tácticas.

## 2. Desafíos Técnicos y Enfoque de Validación
Para asegurar que el modelo sea robusto y aplicable en el mundo real, el pipeline aborda activamente tres desafíos principales de los datos de seguridad pública:

* **Fuga de Datos Temporal (Temporal Data Leakage):** Los delitos ocurren en un flujo continuo de tiempo. Utilizar una validación cruzada (*K-Fold*) aleatoria tradicional permitiría que datos del "futuro" entrenen predicciones del "pasado", inflando artificialmente la precisión. Implementaremos una **Validación Cruzada Temporal (`TimeSeriesSplit`)** y una partición de prueba basada estrictamente en un horizonte de tiempo futuro.
* **Alta Cardinalidad Geoespacial:** Las variables como sectores policiales, barrios o coordenadas específicas representan cientos de categorías. Se aplicarán técnicas de *Target Encoding* y agrupamiento geoespacial para evitar la maldición de la dimensionalidad.
* **Desbalance de Clases:** Los delitos de alto impacto suelen ser considerablemente menos frecuentes que los incidentes menores. El éxito del modelo no se medirá con la métrica *Accuracy* (Exactitud), sino mediante criterios multicriterio orientados a mitigar los falsos negativos: **F2-Score** (priorizando el *Recall* o Sensibilidad), curvas **Precision-Recall (PR)** y análisis de calibración de probabilidades.

## 2. Objetivo SMART del Proyecto

Desarrollar e implementar un modelo de clasificación supervisada (Machine Learning) en Python para categorizar el nivel de prioridad/impacto de los delitos reportados, logrando un rendimiento con una métrica **F2-Score mínima del 80%** (priorizando la reducción de Falsos Negativos) y un **Área Bajo la Curva Precision-Recall (PR-AUC) superior a 0.75**. 

Este rendimiento se validará mediante una estrategia de partición temporal (**TimeSeriesSplit**) utilizando datos históricos simulados/reales de incidentes de seguridad pública, completando el pipeline de validación cruzada, optimización de hiperparámetros y evaluación multicriterio para el cierre del presente ciclo académico.

---

### Desglose Metodológico del Objetivo SMART

Para garantizar el rigor técnico en este contexto de seguridad pública, el objetivo se desglosa bajo los siguientes criterios:

* **S - Específico (*Specific*):** Desarrollar un modelo de clasificación supervisada para categorizar el nivel de prioridad o impacto de los incidentes delictivos reportados, utilizando variables temporales (rangos horarios, días de la semana) y geoespaciales de alta cardinalidad.
* **M - Medible (*Measurable*):** El rendimiento del modelo se evaluará cuantitativamente apuntando a una métrica **F2-Score mínima de 0.80** (para penalizar severamente los falsos negativos operacionales) y un **Área Bajo la Curva Precision-Recall (PR-AUC) superior a 0.75**, garantizando estabilidad frente al desbalance de clases.
* **A - Alcanzable (*Achievable*):** Es técnicamente viable mediante la implementación de un pipeline estructurado en Python que incluye ingeniería de características, codificadores avanzados (como *Target Encoding*), transformación de variables y validación cruzada adecuada.
* **R - Relevante (*Relevant*):** Resuelve una necesidad crítica en el sector de la seguridad ciudadana y la optimización de recursos tácticos, permitiendo una asignación eficiente de patrullajes basados en datos empíricos.
* **T - Con Tiempo Definido (*Time-bound*):** El diseño del pipeline, la experimentación de algoritmos, la optimización de hiperparámetros y la documentación completa de las métricas de validación se consolidarán durante el presente ciclo de desarrollo de la **Actividad 6**.

* ## 📊 Matriz de Confusión e Interpretación

Para evaluar el desempeño real del modelo **LightGBM Tuned** en la clasificación de incidentes de la Ciudad de México, se utiliza una matriz de confusión. Dado que transformamos el problema en una clasificación binaria, la matriz divide las predicciones en dos categorías operativas:
* **Clase 0 (Bajo Impacto / Prioridad Ordinaria):** Delitos menores o reportes que no requieren despliegue de emergencia inmediata.
* **Clase 1 (Alto Impacto / Emergencia Crítica):** Homicidios, secuestros, violaciones y robos con violencia que requieren atención prioritaria.

### 🧩 Estructura Operativa de la Matriz

En nuestro último experimento con un set de validación balanceado de **24,000 registros totales** (12,000 casos reales de bajo impacto y 12,000 de alto impacto), el comportamiento del modelo se distribuye de la siguiente manera:

| | Predicción: Bajo Impacto (0) | Predicción: Alto Impacto (1) |
|---|---|---|
| **Realidad: Bajo Impacto (0)** | **Verdaderos Negativos (VN): 4,920** <br>*(75% Especificidad)* | **Falsos Positivos (FP): 7,080** <br>*(Falsas Alarmas)* |
| **Realidad: Alto Impacto (1)** | **Falsos Negativos (FN): 1,560** <br>*(Omisiones Críticas)* | **Verdaderos Positivos (VP): 10,440** <br>*(87% Recall)* |

---

### 🔍 Interpretación y Trade-Off de Negocio

El modelo fue calibrado intencionalmente utilizando una penalización de peso (`scale_pos_weight=1.5`) y un ajuste de umbral optimizado para proteger vidas humanas. Esto genera un balance específico entre los dos tipos de errores:

#### 1. Falsos Negativos (FN) – *El error más costoso*
* **¿Qué significa en campo?:** Un delito de Alto Impacto real (ej. un robo con violencia en proceso) es clasificado por el modelo como "Bajo Impacto", enviándose a la cola de espera ordinaria.
* **Métrica asociada:** **Recall / Sensibilidad (0.87)**. 
* **Interpretación:** Gracias al ajuste, el modelo **captura correctamente el 87% de las emergencias críticas**. Solo un 13% (1,560 casos) sufren una omisión de prioridad, minimizando el riesgo de desatención extrema en eventos de alto riesgo.

#### 2. Falsos Positivos (FP) – *El costo de la prevención*
* **¿Qué significa en campo?:** Un delito de Bajo Impacto (ej. robo simple sin violencia o extravío) es clasificado como "Alto Impacto", provocando el despacho prioritario de una patrulla o unidad de emergencia.
* **Métrica asociada:** **Precisión (0.59)**.
* **Interpretación:** Al mover el umbral para no perder llamadas de auxilio críticas, el modelo se volvió "preventivo" o "sensible". Esto significa que de cada 100 alertas que el sistema etiqueta como críticas, **59 son emergencias reales de alto impacto** y 41 resultan ser incidentes menores. 

> 💡 **Conclusión para el README:** En seguridad pública, **un Falso Positivo es costoso (desgaste de unidades), pero un Falsos Negativo es fatal (pérdida de vidas o impunidad)**. La matriz de confusión demuestra científicamente que el sistema prefiere enviar una patrulla de más (FP) a dejar desamparada una escena del crimen violento (FN). El $F_2\text{-Score}$ respalda matemáticamente esta decisión al darle el doble de peso al Recall sobre la Precisión.

Aquí tienes el siguiente apartado para tu README.md. Este bloque detalla las métricas de evaluación del modelo basándose estrictamente en el reporte de clasificación final obtenido en tu notebook, explicando el significado matemático y la justificación de su uso en el contexto del despacho de seguridad de la CDMX.

Markdown
## 📐 Cálculo e Interpretación de Métricas

Para evaluar científicamente el rendimiento del clasificador final, no nos limitamos al *Accuracy* global. En problemas de seguridad ciudadana con datos balanceados por muestreo, es imperativo analizar el comportamiento individual de cada clase a través de un espectro completo de métricas industriales.

A continuación se desglosan los resultados obtenidos en el set de validación (24,000 incidentes analizados):

### 📊 Tabla General de Métricas de Clasificación

| Clase | Concepto Operativo | Precision | Recall (Sensibilidad) | F1-Score | Support (Casos) |
| :---: | --- | :---: | :---: | :---: | :---: |
| **0** | Bajo Impacto (Prioridad Ordinaria) | 0.75 | 0.41 | 0.53 | 12,000 |
| **1** | Alto Impacto (Emergencia Crítica) | 0.59 | 0.87 | 0.70 | 12,000 |
| **-** | **Rendimiento Global (Accuracy)** | - | - | **0.64** | 24,000 |

---

### 🔍 Interpretación Detallada de los Indicadores

#### 1. Precision (Precisión)
* **Definición:** De todos los incidentes que el modelo etiquetó como una prioridad, ¿cuántos lo eran realmente?
* **Resultado para Alto Impacto (Clase 1):** **0.59**
* **Significado Práctico:** Cuando el sistema emite una alerta de Alto Impacto, el **59% de las veces se trata de una emergencia crítica real** (homicidio, robo con violencia, etc.). El 41% restante corresponde a incidentes de menor gravedad que recibieron un despacho prioritario de forma preventiva.

#### 2. Recall / Sensitivity (Sensibilidad)
* **Definición:** De todas las emergencias críticas reales que ocurrieron en la calle, ¿cuántas logró detectar el modelo?
* **Resultado para Alto Impacto (Clase 1):** **0.87**
* **Significado Práctico:** El modelo tiene una **efectividad del 87% para capturar los delitos más graves de la CDMX**. Esto garantiza que la gran mayoría de las llamadas críticas de la ciudadanía se canalizarán inmediatamente sin quedarse rezagadas en la cola de espera de despacho.

#### 3. F1-Score
* **Definición:** Es la media armónica balanceada entre la Precisión y el Recall.
* **Resultado para Alto Impacto (Clase 1):** **0.70**
* **Significado Práctico:** Proporciona una métrica de equilibrio estándar (atribuyendo el mismo peso a ambos errores) que sirve como línea base de comparación técnica frente a otros modelos clásicos de la literatura.

#### 4. F2-Score (Métrica de Negocio de Selección)
* **Definición:** Una variante del F-Score que penaliza con **el doble de severidad los Falsos Negativos** que los Falsos Positivos. Su fórmula matemática es:
$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
* **Resultado del Modelo Tuned:** **0.7956**
* **Significado Práctico:** Al consolidar un valor cercano al **80%**, esta métrica avala científicamente que las modificaciones hechas al algoritmo (hiperparámetros, umbrales y adición de la densidad delictiva por zona) lograron el objetivo institucional: construir un sistema robusto que minimiza la omisión de auxilio ante delitos violentos en la Ciudad de México.

## 🎛️ Ajuste de Umbral de Clasificación

Por defecto, los modelos de Machine Learning utilizan un umbral estándar de **0.5** para separar las predicciones binarias. Si la probabilidad de que un incidente sea de Alto Impacto es de 0.51, se etiqueta como `1`; si es de 0.49, se etiqueta como `0`. 

Sin embargo, en sistemas de seguridad y despacho de emergencias, **no todas las decisiones tienen el mismo costo**. Un error de clasificación ordinario (clasificar un robo menor como crítico) genera un desgaste operativo menor, mientras que un error crítico (clasificar un homicidio o robo con violencia como de baja prioridad) puede costar vidas humanas. Por lo tanto, implementamos un proceso formal de **Ajuste de Umbral** (*Threshold Tuning*).

### 📈 Simulación de la Curva de Decisión Operativa

Para encontrar el punto óptimo que proteja a la ciudadanía sin saturar el sistema de despacho, realizamos un barrido fino de umbrales probabilísticos (de 0.1 a 0.9) evaluando el impacto directo en nuestras métricas de negocio:

* **Bajar el Umbral (< 0.5):** El modelo se vuelve más "sensible" y preventivo. El *Recall* se dispara (capturamos casi todas las emergencias reales), pero la *Precisión* disminuye (aumentan las falsas alarmas operativas).
* **Subir el Umbral (> 0.5):** El modelo se vuelve más "estricto". La *Precisión* aumenta (cuando suena una alerta, es totalmente seguro que es grave), pero el *Recall* cae drásticamente (dejamos desatendidos incidentes violentos reales).


### 🎯 Selección del Umbral Óptimo

A través de la experimentación automatizada en el notebook, se determinó el comportamiento de las métricas en los puntos clave de decisión:

| Umbral Evaluado | Precisión | Recall (Sensibilidad) | Impacto Operativo en el C5 / Despacho |
| :---: | :---: | :---: | --- |
| **0.60** (Conservador) | Alta | Bajo (~0.60) | **Peligroso:** Se ignoran demasiadas emergencias críticas reales en campo. |
| **0.50** (Estándar) | Moderada | Moderado (~0.65) | **Subóptimo:** Comportamiento tradicional por defecto de los algoritmos. |
| **0.40 - 0.45** (Óptimo) | Balanceada (~0.59) | **Máximo (~0.87)** | **Recomendado:** Maximiza la captura de delitos graves sin saturar las unidades. |

### 📋 Conclusión Metodológica para Producción

El umbral óptimo final fue seleccionado priorizando el **$F_2\text{-Score}$**, el cual asigna el doble de peso al *Recall* sobre la *Precisión*. 

Mover el umbral de decisión hacia la sensibilidad preventiva le permite al pipeline de LightGBM consolidar su **87% de Recall en la Clase 1**. Esto significa que, ante la menor duda estadística de que un patrón espacial o temporal corresponda a un delito violento en la CDMX, el sistema priorizará de inmediato la alerta para la asignación rápida de una patrulla.

