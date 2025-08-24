import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    multilabel_confusion_matrix, precision_recall_curve, average_precision_score
)

def evaluar_modelo(modelo, X_test, y_test, mlb, nombre_modelo="modelo", split="test", show_plots=True, save_path=None):
    """
    Evalúa un modelo multilabel y guarda métricas + gráficas

    Argumentos:
        modelo: modelo entrenado
        X_test, y_test: datos de prueba
        mlb: MultiLabelBinarizer
        nombre_modelo: string para identificar al modelo (ej. 'baseline', 'embeddings')
        split: string indicando el conjunto (ej. 'train', 'val', 'test')
        show_plots: bool, mostrar gráficos en pantalla
        save_path: ruta donde guardar resultados (se crearán subcarpetas si no existen)
    """

    # ------------------------------
    # Predicciones
    # ------------------------------
    y_pred = modelo.predict(X_test)

    # ------------------------------
    # Métricas globales
    # ------------------------------
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    exact_match = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test, y_pred, target_names=mlb.classes_, output_dict=True
    )
    reporte_str = classification_report(
        y_test, y_pred, target_names=mlb.classes_
    )

    print(f"\n--- Resultados {nombre_modelo} ({split}) ---")
    print(f"Exact Match Ratio: {exact_match:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print("\nClassification Report:\n")
    print(reporte_str)

    # ------------------------------
    # Guardado en disco
    # ------------------------------
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Guardar métricas en txt
        with open(os.path.join(save_path, f"metrics_{nombre_modelo}_{split}.txt"), "w") as f:
            f.write(f"Exact Match Ratio: {exact_match:.4f}\n")
            f.write(f"F1 Micro: {f1_micro:.4f}\n")
            f.write(f"F1 Macro: {f1_macro:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(reporte_str)

        # Guardar métricas en JSON
        metrics_dict = {
            "model": nombre_modelo,
            "split": split,
            "exact_match": exact_match,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "per_class": {label: report_dict[label]["f1-score"] for label in mlb.classes_}
        }
        with open(os.path.join(save_path, f"metrics_{nombre_modelo}_{split}.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)

    # ------------------------------
    # Gráficas
    # ------------------------------
    if show_plots or save_path:
        # F1-score por clase
        class_f1 = [report_dict[label]["f1-score"] for label in mlb.classes_]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=mlb.classes_, y=class_f1)
        plt.title(f"F1-score por clase - {nombre_modelo} ({split})")
        plt.ylabel("F1-score")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)

        if save_path:
            plt.savefig(os.path.join(save_path, f"f1_per_class_{nombre_modelo}_{split}.png"))
        if show_plots:
            plt.show()
        plt.close()

        # Matrices de confusión
        cms = multilabel_confusion_matrix(y_test, y_pred)
        fig, axes = plt.subplots(1, len(mlb.classes_), figsize=(15, 4))
        for i, (ax, label) in enumerate(zip(axes, mlb.classes_)):
            sns.heatmap(cms[i], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{label}")
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")

        plt.suptitle(f"Matrices de confusión - {nombre_modelo} ({split})")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"confusion_matrices_{nombre_modelo}_{split}.png"))
        if show_plots:
            plt.show()
        plt.close()

        # Precision-Recall por clase
        try:
            y_prob = modelo.predict_proba(X_test)
            fig, axes = plt.subplots(1, len(mlb.classes_), figsize=(15, 4))
            for i, label in enumerate(mlb.classes_):
                precision, recall, _ = precision_recall_curve(y_test[:, i], y_prob[:, i])
                ap = average_precision_score(y_test[:, i], y_prob[:, i])
                axes[i].step(recall, precision, where="post")
                axes[i].set_title(f"{label} (AP={ap:.2f})")
                axes[i].set_xlabel("Recall")
                axes[i].set_ylabel("Precision")

            plt.suptitle(f"Curvas Precision-Recall - {nombre_modelo} ({split})")
            plt.tight_layout()
            if save_path:
                plt.savefig(os.path.join(save_path, f"precision_recall_{nombre_modelo}_{split}.png"))
            if show_plots:
                plt.show()
            plt.close()
        except Exception as e:
            print("No se pudo calcular curvas Precision-Recall:", e)

    # ------------------------------
    # Retorno
    # ------------------------------
    return {
        "exact_match": exact_match,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "reporte": reporte_str,
        "report_dict": report_dict
    }
