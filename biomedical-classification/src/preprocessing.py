# Limpieza de texto y features
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def preparar_texto(df, columna_title="title", columna_abs="abstract"):
    return (df[columna_title].fillna("") + " " + df[columna_abs].fillna("")).apply(limpiar_texto)
  # Une el título y el abstract en un solo campo de texto procesado

def preparar_labels(df):
  # Convierte las etiquetas a multilabel binario (estarán separadas por "|")
  multilabel = df["group"].apply(lambda x: x.split("|"))
  mlb = MultiLabelBinarizer()
  y = mlb.fit_transform(multilabel)
  return y, mlb


def vectorizar_texto(corpus, max_features=5000):
  # Genera matriz TF-IDF
  vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
  X = vectorizer.fit_transform(corpus)
  return X, vectorizer