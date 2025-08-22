# Biomedical Classification Challenge 🧬🤖

## 📌 Contexto

Este proyecto fue desarrollado para el **Data Challenge de Clasificación Biomédica con IA**.  
El objetivo es construir una solución de Inteligencia Artificial que clasifique artículos de investigación médica en uno o varios dominios:

- **Cardiovascular**
- **Neurological**
- **Hepatorenal**
- **Oncological**

La clasificación se realiza a partir de dos campos de texto por artículo:

- `title` → título del artículo
- `abstract` → resumen científico

---

## 🎯 Objetivo

Dado un artículo médico (título + abstract), el sistema debe predecir correctamente su(s) categoría(s).  
Se trata de un **problema de clasificación multi-etiqueta**.

---

## 📊 Dataset

- Registros: **3,565 artículos**
- Columnas principales:
  - **title**: título del artículo
  - **abstract**: resumen científico
  - **group**: categoría(s) médica(s) a la que pertenece el artículo

El dataset fue provisto por la organización del challenge y combina datos de **NCBI**, **BC5CDR** y registros sintéticos.

---

## 🛠️ Tecnologías utilizadas

- **Python 3.10+**
- **Librerías de Ciencia de Datos**: pandas, numpy, scikit-learn
- **NLP (Procesamiento de Lenguaje Natural)**: NLTK, scikit-learn (TF-IDF), sentence-transformers (para embeddings)
- **Visualización**: matplotlib, seaborn, V0 (bonus)
- **Gestión del proyecto**: GitHub, notebooks (Jupyter)

---

## 📁 Estructura

```plaintext
biomedical-classification/
│
├── data/
│ └── dataset.csv
│
├── notebooks/
│ ├── 01_eda.ipynb -> análisis exploratorio
│ ├── 02_train_baseline.ipynb -> modelo baseline (TF-IDF + Logistic Regression)
│ └── 03_embeddings.ipynb -> mejora con embeddings
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── models/ # modelos entrenados (.joblib)
├── evidence/ # capturas, prompts y resultados de V0
│
├── requirements.txt # librerías necesarias
└── README.md # este archivo
```

---

## 🚀 Solución propuesta

1. **EDA (Exploratory Data Analysis)** → entender distribución de clases, textos y multi-etiquetas.
2. **Baseline con TF-IDF + Logistic Regression**
   - Transformar `title + abstract` en vectores numéricos con TF-IDF.
   - Entrenar un clasificador multi-etiqueta usando One-vs-Rest Logistic Regression.
   - Evaluar con métricas: **F1 ponderado (weighted F1)**, precisión y matriz de confusión.
3. **Mejora con embeddings semánticos**
   - Usar `sentence-transformers` (ej. `all-MiniLM-L6-v2`) para obtener representaciones vectoriales.
   - Entrenar un modelo clásico (Logistic Regression o XGBoost) sobre estos embeddings.
4. **Visualización de resultados**
   - Graficar métricas y matriz de confusión en Python.
   - Crear visualizaciones interactivas en **V0** (bonus).
5. **Entrega final**
   - Guardar modelo entrenado en `/models/`.
   - Documentar resultados y visualizaciones en `/evidence/`.

---

## 📈 Evaluación

El proyecto será evaluado principalmente con la métrica **Weighted F1-score**, además de precisión, recall y exactitud.  
También se incluye una **matriz de confusión** por etiqueta.

---

## 📦 Cómo usar este repositorio (Windows)

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/kevinramos2/Articulos-medicos-por-dominio.git
   cd biomedical-classification
   ```
2. Crear entorno virtual:
   ```bash
   python -m venv .venv
   ```
3. Activar entorno virtual (cada vez que abras el proyecto):
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```
4. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
5. Abrir los notebooks en orden:
   ```bash
   notebooks/01_eda.ipynb
   notebooks/02_train_baseline.ipynb
   notebooks/03_embeddings.ipynb (opcional)
   ```
6. Revisar modelos entrenados en /models/ y resultados en /evidence/.

## 👥 Equipo

Este proyecto fue desarrollado por:

- **Juan Felipe Miranda Arciniegas** — Estudiante de Ingeniería de Sistemas, Universidad Nacional de Colombia
- **Luis Alejandro Martínez Ramírez** — Estudiante de Ingeniería de Sistemas, Universidad Nacional de Colombia
- **Kevin Leandro Ramos Luna** — Estudiante de Ingeniería de Sistemas, Universidad Nacional de Colombia
