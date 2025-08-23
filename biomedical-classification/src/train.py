# Entrenar Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

def baseline_train(X_train, y_train):
  modelo = OneVsRestClassifier(LogisticRegression(max_iter=200))
  modelo.fit(X_train, y_train)
  return modelo

def guardar_modelo(modelo, ruta):
  joblib.dump(modelo,ruta)

def cargar_modelo(ruta):
  return joblib.load(ruta)