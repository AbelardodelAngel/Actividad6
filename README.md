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
