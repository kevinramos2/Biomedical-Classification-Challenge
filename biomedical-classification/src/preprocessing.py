# Limpieza de texto y features
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def limpiar_texto():
  """
  Limpieza del texto:
  - Se pasará todo a minúsculas
  - Se eliminarán caracteres no alfanuméricos (se conservarán números y siglas)
  """
  texto = texto.lower()
  texto = re.sub(r"[â-z0-9\s]"," ", texto) # conservamos letras y números
  texto = re.sub(r"\s+"," ", texto).strip() 
  return texto


def preparar_texto(df):
  # Une el título y el abstract en un solo campo de texto procesado
  return (df["title"].fillna("") + " " + df["abstract"].fillna("")).apply(limpiar_texto)


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