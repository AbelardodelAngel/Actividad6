# Actividad 6: Clasificación y Predicción de Tipos de Delito para la Seguridad Pública

Proyecto de Machine Learning con validación temporal para la categorización y análisis de incidentes delictivos.

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
