# Trabajo Práctico Final de Herramientas de Procesamiento para Grandes Volúmenes de Datos

**Integrantes del grupo 3:**
- Cristian Carlos Czubara
- Carlos Franco
- Martín Schelp

---

## 1. Descripción y justificación del conjunto de datos

### Origen y contexto

El conjunto de datos que seleccionamos es *"Synthetic Financial Datasets For Fraud Detection"*, obtenido originalmente de Kaggle (basado en el simulador PaySim). Se puede descargar desde https://www.kaggle.com/datasets/ealaxi/paysim1/code. Este dataset genera transacciones financieras móviles que imitan el funcionamiento real de un servicio de dinero electrónico. Elegimos este dataset porque representa el estudio de un problema crítico en el mundo de las finanzas y la ciberseguridad: la identificación de patrones fraudulentos en flujos masivos de datos.

### Características principales

- **Volumen:** contiene más de 6.3 millones de registros, lo que lo posiciona como un problema de Big Data que requiere herramientas de procesamiento distribuido como PySpark para su manipulación eficiente.
- **Atributos clave:** incluye variables como el tipo de transacción (`type`), el monto (`amount`), saldos de origen y destino antes y después de la operación, y marcas de tiempo (`step`).
- **Desbalance de clases:** presenta un desbalance extremo, donde solo aproximadamente el 0.13% de las transacciones son fraudulentas. Esta característica es fundamental para un TP de MLOps, ya que obliga a descartar el *Accuracy* como métrica y recurrir a evaluaciones más sofisticadas (Precision-Recall, F2-Score).

Para que el modelo sea efectivo, fue necesario calcular métricas de consistencia de saldos (ej. `errorBalanceOrig`).

El gran volumen de datos justifica el uso de MLflow para el seguimiento de experimentos. La optimización de hiperparámetros con Optuna se vuelve indispensable para encontrar el equilibrio entre detectar la mayor cantidad de fraudes (Recall) y no bloquear excesivas transacciones legítimas (Precision).

En fraude financiero, no basta con que el modelo prediga correctamente; es ético y regulatorio explicar por qué una transacción fue marcada como sospechosa. Este dataset permite aplicar SHAP de manera muy visual, permitiendo identificar que las inconsistencias en los saldos suelen ser el principal predictor de fraude, cumpliendo así con los requisitos avanzados del TP.

---

## 2. Análisis Exploratorio de Datos (EDA)

### Objetivo del Análisis

El EDA se centró en comprender la distribución de las transacciones, identificar patrones de comportamiento fraudulento y, fundamentalmente, cuantificar el desbalance de clases para definir la estrategia de modelado.

### I. Estadísticas descriptivas y calidad de datos

Se realizó un análisis estadístico sobre las variables numéricas clave (`amount`, `oldbalanceOrg`, `newbalanceOrig`).

Se observó una alta dispersión en los montos de las transacciones, con valores máximos que superan ampliamente la media, lo cual es típico en sistemas financieros donde conviven micro-pagos con transferencias de alto valor.

No se detectaron valores nulos en el dataset de 6.3 millones de registros, lo que permitió avanzar directamente hacia el análisis de distribución.

### II. Análisis del desbalance de clases

Mediante el uso de agregaciones de PySpark, se calculó la proporción de la variable objetivo `isFraud`. Se identificó que solo el 0.129% de las transacciones son fraudulentas.

### III. Visualización y correlación de variables

Se analizaron los tipos de transacciones mediante gráficos de barras. Los fraudes se concentran exclusivamente en los tipos `TRANSFER` y `CASH_OUT`. Esto permitió filtrar el dataset para que el modelo aprenda específicamente de los contextos donde el riesgo es real, optimizando el entrenamiento.

Al visualizar la relación entre montos y saldos de origen, se detectó que muchos fraudes vacían completamente la cuenta de origen (saldo final = 0), pero el monto transferido no siempre coincide con la diferencia de saldos, lo que dio origen a la creación de la variable `errorBalanceOrig` en la etapa de preprocesamiento.

---

## 3. Descripción sintética de los procesos

### I. Ingesta y limpieza de datos

El proceso comienza con la carga del dataset distribuido hacia un DataFrame de PySpark. Se realiza una limpieza inicial que incluye el casting de tipos de datos (asegurando que la variable objetivo `isFraud` sea numérica) y la verificación de valores nulos. Dado el volumen de 6.3 millones de registros, todas las operaciones se ejecutan de manera perezosa (*lazy evaluation*) para optimizar el uso de los recursos del clúster.

### II. Ingeniería de características (Feature Engineering)

Desarrollamos nuevas variables para capturar comportamientos anómalos que el dataset crudo no evidencia por sí solo:

- **`errorBalanceOrig` y `errorBalanceDest`:** calculan la discrepancia aritmética entre el monto de la transacción y el cambio en los saldos. Una diferencia distinta de cero suele ser un indicador técnico de fraude.
- **`origZeroBalance` y `destZeroBalance`:** variables binarias que identifican transacciones que involucran cuentas con saldo inicial en cero, un patrón común en cuentas de "mula" o transacciones fantasma.

### III. Pipeline de preprocesamiento de ML

Implementamos un Pipeline de SparkML para garantizar la reproducibilidad y evitar el data leakage:

1. **StringIndexer:** transforma la variable categórica `type` en índices numéricos (`type_idx`).
2. **VectorAssembler:** consolida las 12 variables seleccionadas en un único vector de características (`features`), requisito indispensable para los algoritmos de Spark.

### IV. Experimentación y optimización con MLflow y Optuna

Esta es la fase central del ciclo de vida de MLOps:

- **Seguimiento:** utilizamos MLflow para registrar cada ejecución, almacenando parámetros, métricas y artefactos.
- **Optimización:** integramos Optuna para realizar una búsqueda de hiperparámetros (como `maxDepth` y `numTrees`) mediante el algoritmo TPESampler (Bayesiano), buscando maximizar el área bajo la curva Precision-Recall.
- **Versionado:** el modelo resultante se registró en el Model Registry de Unity Catalog, asignándole el alias `@champion` para facilitar su consumo en producción.

### V. Evaluación avanzada y explicabilidad

Para superar las limitaciones del Accuracy en datos desbalanceados, se aplicaron:

- **Evidently AI:** se incorporó la librería como parte del enfoque de MLOps para monitorear la calidad de los datos y el desempeño del modelo. Esta herramienta permite analizar de forma visual y estructurada posibles cambios en la distribución de los datos (*data drift*) y evaluar métricas clave, facilitando su validación y futura implementación en entornos productivos.

  - **Reporte de Data Drift:** se compararon los conjuntos de entrenamiento y test para detectar diferencias en la distribución de las variables, identificando posibles desviaciones naturales entre datos históricos y nuevos.
  
  - **Simulación de Drift:** se generaron modificaciones controladas sobre el dataset (cambios de escala, desplazamientos y ruido) para validar la capacidad de la herramienta de detectar distintos tipos de drift de forma realista.
  
  - **Reporte de Performance:** se evaluó el rendimiento del modelo mediante métricas como *precision*, *recall* y matriz de confusión, utilizando dashboards que facilitan la interpretación de resultados.


  
- **SHAP (KernelExplainer):** para proveer transparencia al modelo (Caja Negra), cuantificando el impacto de cada variable en la decisión final. Esto permite validar que el modelo está tomando decisiones basadas en lógicas de negocio correctas y no en sesgos del dataset.

---

## 3. Selección del modelo Random Forest Classifier

Decidimos utilizar Random Forest como el modelo definitivo ya que al ser un ensamble de árboles de decisión tiene un mejor manejo de la no linealidad y una mejor robustez frente al overfitting.

### Priorización del Recall y F2-Score

En la detección de fraude, el costo de oportunidad es asimétrico. Decidimos que es mucho más costoso dejar pasar un fraude (Falso Negativo) que inspeccionar manualmente una transacción legítima marcada erróneamente (Falso Positivo).

En base a esta decisión decidimos utilizar como métrica **F2**. Esta métrica da mayor peso al Recall que a la Precision, asegurando que el modelo sea más sensible a las actividades sospechosas. Se seleccionó el modelo con el mejor F2-Score obtenido durante la optimización con Optuna.

**Área bajo la curva PR (Precision-Recall):** debido al desbalance extremo de clases (0.13% de fraude), se utilizó esta métrica en lugar del área bajo la curva ROC, ya que refleja con mayor fidelidad el rendimiento del modelo en la clase minoritaria.

### Interpretabilidad y confianza

Finalmente, elegimos el uso de herramientas de explicabilidad como SHAP. Para una entidad financiera, es vital poder justificar por qué se bloqueó una cuenta. El Random Forest, en combinación con SHAP, nos permitió confirmar que el modelo "aprendió" correctamente que las inconsistencias de saldos (`errorBalanceOrig`) son el principal factor de riesgo, confirmando que la inconsistencia en los saldos de origen es el principal indicador de comportamiento sospechoso.

---

## 4. Publicación del modelo: registro y gestión de ciclo de vida

Una vez seleccionado el modelo con el mejor rendimiento (basado en la métrica F2), se procedió a su formalización dentro del MLflow Model Registry en Unity Catalog. Este paso es fundamental en MLOps, ya que permite desacoplar el desarrollo del despliegue.

- **Registro centralizado:** el modelo fue registrado con un nombre único en el catálogo, lo que permite el control de versiones y la gobernanza de datos.
- **Estrategia de alias (`@champion`):** se utilizó la funcionalidad de aliases de Databricks para marcar la versión ganadora como `@champion`. Esto permite que las aplicaciones de consumo apunten a un alias fijo. Si en el futuro se entrena un modelo mejor, solo se debe mover el alias a la nueva versión sin modificar el código de producción.
- **Trazabilidad completa:** gracias a MLflow, el modelo registrado mantiene un linaje directo con los datos de entrenamiento y los hiperparámetros utilizados, garantizando la auditabilidad del proceso.

---

## 5. Explicación de Uso y Ejemplo de Inferencia

El modelo está diseñado para integrarse fácilmente en flujos de trabajo de producción. Puede ser cargado directamente desde el registro de modelos utilizando su URI o su alias.

El modelo devuelve una clase (0 para legítima, 1 para fraude) y una probabilidad asociada. En un entorno real, las transacciones con predicción 1 dispararían una alerta inmediata en el sistema de prevención de fraude.

---

## 6. Conclusiones Finales

### Aprendizajes principales

- **El desafío del desbalance:** se confirmó que en problemas de Big Data como el fraude, el volumen de datos no garantiza el éxito por sí solo. Pusimos el foco en la calidad de las métricas (F2 y Precision-Recall) y no en el Accuracy.
- **Importancia del Feature Engineering:** las variables creadas manualmente (`errorBalanceOrig`) resultaron ser más predictivas que los datos crudos, demostrando que el conocimiento del dominio del negocio es vital.
- **Automatización con MLOps:** la integración de Optuna con MLflow permitió sistematizar la experimentación, reduciendo el tiempo de búsqueda de modelos y garantizando la reproducibilidad.

### Limitaciones y Trabajo Futuro

- **Latencia de Inferencia:** una mejora futura sería evaluar el tiempo de respuesta del modelo para casos de uso de Streaming (tiempo real), donde la decisión debe tomarse en milisegundos.
- **Nuevas técnicas:** explorar modelos de aprendizaje profundo (Deep Learning) o arquitecturas de grafos que puedan analizar mejor las conexiones entre distintas cuentas sospechosas.
