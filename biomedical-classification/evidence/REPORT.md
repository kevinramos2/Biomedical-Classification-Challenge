# Clasificación multi‑etiqueta de Artículos Biomédicos con Modelos Baseline y Embeddings Semánticos

## Tabla de Contenido

1. [Introducción](#1-introducción)  
2. [Metodología](#2-metodología)  
   - [Dataset](#21-dataset)  
   - [Preprocesamiento](#22-preprocesamiento)  
   - [Modelos](#23-modelos)  
   - [Métricas de Evaluación](#24-métricas-de-evaluación)  
3. [Resultados](#3-resultados)  
   - [Modelo Baseline](#31-modelo-baseline)  
   - [Modelo con Embeddings](#32-modelo-con-embeddings)  
4. [Discusión](#4-discusión)  
5. [Conclusiones](#5-conclusiones)

---

## 1. Introducción

El acceso y la organización de la información científica en el campo biomédico constituyen un reto creciente debido al volumen cada vez mayor de publicaciones que aparecen en repositorios y bases de datos especializadas. En este contexto, los métodos de procesamiento de lenguaje natural (NLP) y aprendizaje automático ofrecen una alternativa eficiente para clasificar automáticamente los artículos de investigación en dominios específicos, facilitando así el trabajo de investigadores, clínicos y profesionales de la salud que necesitan acceder rápidamente a literatura relevante.

El presente informe documenta el desarrollo del proyecto **Biomedical Classification Challenge**, cuyo objetivo fue construir y evaluar modelos capaces de clasificar artículos biomédicos en cuatro dominios principales: _cardiovascular, neurological, hepatorenal y oncological_. El problema abordado corresponde a un escenario de **clasificación multi-etiqueta**, ya que un artículo puede pertenecer simultáneamente a más de un dominio.

A lo largo del trabajo se exploraron dos enfoques complementarios. Primero, se estableció un modelo baseline utilizando representaciones clásicas de texto basadas en **TF-IDF** y un clasificador **One-vs-Rest con Regresión Logística**. Posteriormente, se evaluó un enfoque más avanzado que emplea **embeddings pre-entrenados** mediante el modelo `sentence-transformers/all-MiniLM-L6-v2`, con el fin de capturar mejor las relaciones semánticas entre palabras y frases. El análisis comparativo de ambos enfoques permite no solo medir el impacto del uso de representaciones semánticas profundas, sino también entender las fortalezas y limitaciones de cada estrategia en el dominio biomédico.

---

## 2. Metodología


### 2.1 Dataset

El dataset utilizado fue provisto por la organización del challenge y estuvo compuesto por **3.565 artículos biomédicos**. Cada registro incluye dos campos textuales fundamentales: el **título** y el **resumen (abstract)**, junto con la columna `group` que contiene las etiquetas de clasificación correspondientes. Estas etiquetas reflejan la(s) categoría(s) médica(s) a las que pertenece el artículo, pudiendo ser una o varias simultáneamente.

El dataset combina información proveniente de diferentes fuentes, incluyendo **NCBI**, **BC5CDR** y datos sintéticos. Su naturaleza heterogénea lo convierte en un recurso valioso para probar modelos de NLP en contextos biomédicos reales, donde los textos presentan alta variabilidad tanto en estilo como en complejidad terminológica.

El conjunto de datos se dividió de manera **estratificada** en `train`, `validation` y `test`, respetando la distribución de clases en cada partición. La proporción utilizada fue de **70% entrenamiento, 10% validación y 20% prueba**.


### 2.2 Preprocesamiento

El preprocesamiento de los textos incluyó pasos básicos de normalización como:

- Conversión a minúsculas.  
- Eliminación de caracteres no alfabéticos.  
- Tokenización y limpieza de palabras irrelevantes.  

A partir de estos textos limpios se generaron las representaciones vectoriales de acuerdo con el enfoque seleccionado: **TF-IDF** para el baseline y **embeddings densos** en el segundo modelo.


### 2.3 Modelos

**2.3.1 Modelo Baseline (TF-IDF + Logistic Regression)**  
- Representación de los textos mediante TF-IDF con un máximo de **5.000 características**.  
- Clasificación multi-etiqueta mediante un esquema **One-vs-Rest** sobre un clasificador de **Regresión Logística**.  
- Este enfoque constituye una referencia inicial para evaluar el impacto de representaciones semánticas más avanzadas.  

**2.3.2 Modelo con Embeddings Semánticos**  
- Utilización del modelo pre-entrenado **all-MiniLM-L6-v2** de la librería `sentence-transformers`.  
- Cada artículo (título + abstract) se codificó como un vector denso de **384 dimensiones**.  
- Clasificación mediante **Regresión Logística One-vs-Rest**, optimizando además los umbrales de decisión por clase para maximizar el **F1-score**.  


### 2.4 Métricas de Evaluación

Se emplearon métricas estándar en clasificación multi-etiqueta, entre ellas:

- **Exact Match Ratio**: proporción de muestras en las que se predicen correctamente todas las etiquetas.  
- **F1-score Micro**: promedia el rendimiento ponderando por frecuencia de etiquetas.  
- **F1-score Macro**: promedia de manera uniforme el rendimiento en todas las clases.  
- **Reportes de clasificación por clase** (precisión, recall, F1).  
- **Matriz de confusión por clase** y **curvas precision-recall**, útiles para observar el comportamiento diferenciado en cada categoría.  

---

## 3. Resultados

### 3.1 Modelo Baseline

El modelo baseline alcanzó en el conjunto de validación un **Exact Match Ratio de 0.64**, un **F1 micro de 0.80** y un **F1 macro de 0.73**. Estos resultados muestran que, aunque el modelo logra un desempeño aceptable en promedio, presenta dificultades en ciertas categorías.

Al analizar el rendimiento por clase, se observa que la categoría **neurological** alcanzó el mejor desempeño con un **F1-score de 0.89**, mientras que la categoría **oncological** obtuvo un rendimiento considerablemente más bajo (**F1-score de 0.52**). Esto refleja que, aunque el modelo es capaz de captar patrones claros en dominios más frecuentes o con expresiones más específicas, enfrenta mayores dificultades en categorías menos representadas o con mayor ambigüedad terminológica.

![Gráfica 1 – F1-score por clase (Baseline validación)](/biomedical-classification/evidence/baseline_val/f1_per_class_baseline_val.png)

En el conjunto de prueba, el baseline mantuvo un rendimiento similar, con un **Exact Match Ratio de 0.65**, un **F1 micro de 0.81** y un **F1 macro de 0.77**, lo que indica que no hubo sobreajuste severo, pero sí limitaciones claras en la capacidad de generalización, especialmente para la clase oncológica.

![Gráfica 2 – Matriz de confusión (Baseline validación)](/biomedical-classification/evidence/baseline_test/confusion_matrices_baseline_test.png)
![Gráfica 3 – Curvas Precision-Recall (Baseline validación)](/biomedical-classification/evidence/baseline_test/precision_recall_baseline_test.png).

### 3.2 Modelo con Embeddings

El modelo basado en embeddings representó una mejora sustancial frente al baseline. En validación, alcanzó un **Exact Match Ratio de 0.70**, un **F1 micro de 0.86** y un **F1 macro de 0.85**. Esto muestra que las representaciones semánticas profundas lograron capturar relaciones entre palabras y expresiones médicas que TF-IDF no puede modelar.

Los resultados por clase confirman esta mejora:

- **Cardiovascular:** F1 = 0.88  
- **Hepatorenal:** F1 = 0.82  
- **Neurological:** F1 = 0.88  
- **Oncological:** F1 = 0.81  

A diferencia del baseline, la clase **oncological** dejó de ser un punto débil crítico y alcanzó un desempeño competitivo frente a las demás.

![Gráfica 4 – F1-score por clase (Embeddings validación)](/biomedical-classification/evidence/embeddings_val/f1_per_class_embeddings_val.png)

En el conjunto de prueba, los resultados se mantuvieron consistentes, con un **Exact Match Ratio de 0.68**, un **F1 micro de 0.85** y un **F1 macro de 0.85**. El modelo mostró especial estabilidad al mantener un equilibrio entre precisión y recall en las cuatro clases, lo cual es crucial en aplicaciones biomédicas, donde los falsos negativos y falsos positivos pueden tener un impacto significativo.

![Gráfica 5 – Matriz de confusión (Embeddings validación)](/biomedical-classification/evidence/embeddings_val/confusion_matrices_embeddings_val.png)  
![Gráfica 6 – Curvas Precision-Recall (Embeddings validación)](/biomedical-classification/evidence/embeddings_val/precision_recall_embeddings_val.png)

---

## 4. Discusión

La comparación entre ambos enfoques muestra una evolución clara en términos de capacidad predictiva. El modelo baseline, aunque simple y computacionalmente eficiente, presentó limitaciones importantes, especialmente en clases menos frecuentes como la **oncológica**. Su buen desempeño en categorías como la **neurológica** demuestra que TF-IDF aún es útil en contextos donde las palabras clave son suficientemente discriminativas, pero no logra capturar las sutilezas semánticas del lenguaje biomédico.

El modelo con embeddings, en contraste, aprovechó el conocimiento semántico aprendido previamente en grandes corpus para representar mejor los títulos y resúmenes. Esto se reflejó en una mejora sustancial de métricas globales y, sobre todo, en un rendimiento más equilibrado entre clases. El hecho de que la clase **oncológica** haya pasado de ser la más débil a estar al nivel de las demás demuestra que los embeddings aportan robustez frente a expresiones más ambiguas o menos frecuentes.

Es importante destacar, sin embargo, que los resultados aún pueden mejorar. En escenarios biomédicos reales, puede ser conveniente explorar arquitecturas más avanzadas, como redes neuronales finamente ajustadas (_fine-tuning_) sobre embeddings biomédicos específicos como **BioBERT** o **PubMedBERT**, que probablemente capturen aún mejor la terminología especializada.

---

## 5. Conclusiones

El proyecto **Biomedical Classification Challenge** permitió demostrar cómo diferentes representaciones de texto impactan significativamente en el rendimiento de modelos de clasificación multi-etiqueta en el dominio biomédico.

- El **baseline con TF-IDF y Regresión Logística** logró resultados iniciales aceptables (**F1-macro ≈ 0.73 en validación**), siendo útil como referencia, pero mostró limitaciones al abordar categorías con menor representación.  
- El **modelo con embeddings pre-entrenados** alcanzó un desempeño mucho más alto y balanceado (**F1-macro ≈ 0.85 en validación y prueba**), evidenciando la importancia de incorporar representaciones semánticas profundas en tareas de clasificación biomédica.  
- El análisis cualitativo de las métricas por clase mostró que el mayor beneficio de los embeddings fue **mejorar categorías difíciles como la oncológica**, reduciendo los desbalances en el rendimiento.  

En conjunto, los resultados refuerzan la idea de que, para aplicaciones críticas como la biomédica, es fundamental emplear representaciones semánticas ricas y modelos más sofisticados, pues el costo de errores de clasificación puede ser significativo en escenarios de soporte a la investigación o la práctica clínica.
