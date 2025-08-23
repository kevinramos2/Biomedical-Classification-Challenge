# Métricas y Gráficas
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, multilabel_confusion_matrix

def evaluar_modelo(modelo, X_test, y_test, mlb, show_plots=True):
  #  Evalúa el modelo y devuelve métricas y gráficas

  # Predicciones
  y_pred = modelo.predict(X_test)

  # Métricas globales
  f1_micro = f1_score(y_test, y_pred, average="micro")
  f1_macro = f1_score(y_test,y_pred,average="macro")
  exact_match = accuracy_score(y_test, y_pred)

  print(f"Exact Match Ratio: {exact_match:.4f}")
  print(f"F1 Micro: {f1_micro:.4f}")
  print(f"F1 Macro: {f1_macro:.4f}")
  print("\nClassification Report:\n")
  reporte = classification_report(y_test, y_pred, target_names=mlb.classes_)
  print(reporte)

  # Gráficas
  if show_plots:
    # F1-Score por clase
    report_dict = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
    class_f1 = [report_dict[label]["f1-score"] for label in mlb.classes_]

    plt.figure(figsize=(8,5))
    sns.barplot(x=mlb.classes_, y=class_f1)
    plt.title("F1-score por clase")
    plt.ylabel("F1-score")
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.show()

    # Matriz de confusión multilabel (una por clase)
    cms = multilabel_confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, len(mlb.classes_), figsize=(15,4))

    for i, (ax, label) in enumerate(zip(axes, mlb.classes_)):
      sns.heatmap(cms[i], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
      ax.set_title(f"{label}")
      ax.set_xlabel("Predicho")
      ax.set_ylabel("Real")

    plt.tight_layout()
    plt.show()

  return {
    "exact_match": exact_match,
    "f1_micro": f1_micro,
    "f1_macro": f1_macro,
    "reporte": reporte
  }