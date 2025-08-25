import pandas as pd
from sklearn.model_selection import train_test_split

def cargar_dataset(ruta, sep=";", **kwargs):
    """
    Carga un dataset desde una ruta local o URL remota.

    Args:
        ruta (str): Ruta local o URL del dataset en formato CSV
        sep (str): Separador de columnas (default=";")
        **kwargs: Parámetros adicionales para `pd.read_csv`

    Returns:
        pd.DataFrame
    """
    try:
        df = pd.read_csv(ruta, sep=sep, **kwargs)
        print(f"Dataset cargado desde: {ruta}")
        print(f"Tamaño: {df.shape}")
        return df
    except Exception as e:
        print(f"Error cargando dataset desde {ruta}: {e}")
        return None


def dividir_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Divide el dataset en train, validation y test.

    Args:
        df (pd.DataFrame): DataFrame con los datos
        test_size (float): proporción de test
        val_size (float): proporción de validación (del train)
        random_state (int): semilla para reproducibilidad

    Returns:
        (train_df, val_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["group"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state, stratify=train_df["group"]
    )
    return train_df, val_df, test_df
