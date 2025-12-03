import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Union, Dict

def completitud_datos(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el porcentaje de valores faltantes (completitud inversa) por columna.

    Parameters
    ----------
    df : pd.DataFrame
        El DataFrame de entrada.

    Returns
    -------
    pd.Series
        Serie con el porcentaje de nulos por columna, ordenado descendentemente.
    """
    # Basado en la lógica de cálculo de completitud [5]
    return round(df.isnull().sum().sort_values(ascending=False) / df.shape, 4)

def limpiar_nulos_imputer(df: pd.DataFrame, continuous_cols: List[str], discrete_cols: List[str]) -> pd.DataFrame:
    """
    Trata valores nulos en el DataFrame utilizando imputación basada en la media 
    para variables continuas y la moda (most_frequent) para discretas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos.
    continuous_cols : list of str
        Lista de nombres de columnas continuas.
    discrete_cols : list of str
        Lista de nombres de columnas discretas o categóricas.

    Returns
    -------
    pd.DataFrame
        DataFrame con los valores nulos imputados.
    """
    df_temp = df.copy()

    # Imputación para variables continuas (media o mediana) [6, 7]
    if continuous_cols:
        imputer_media = SimpleImputer(strategy='mean')
        df_temp[continuous_cols] = imputer_media.fit_transform(df_temp[continuous_cols])

    # Imputación para variables discretas (moda/most_frequent) [6-8]
    if discrete_cols:
        imputer_moda = SimpleImputer(strategy='most_frequent')
        # Utilizamos .ravel() para asegurar que el output se ajuste a una Serie de Pandas
        df_temp[discrete_cols] = imputer_moda.fit_transform(df_temp[discrete_cols]).astype(df[discrete_cols].dtypes)

    return df_temp

def tratar_outliers_iqr(
    df: pd.DataFrame, 
    col_name: str, 
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Identifica y elimina outliers en una columna específica utilizando el método IQR.

    El método IQR define límites a 1.5 veces el Rango Intercuartílico fuera de Q1 y Q3 [9].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col_name : str
        Nombre de la columna numérica a analizar.
    factor : float, optional
        Factor multiplicador del IQR (típicamente 1.5), por defecto es 1.5.

    Returns
    -------
    pd.DataFrame
        DataFrame con los valores atípicos eliminados para la columna dada.
    """
    # Cálculo de límites IQR [9, 10]
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - factor * IQR
    limite_superior = Q3 + factor * IQR

    # Filtrar el DataFrame
    mask_iqr = (df[col_name] >= limite_inferior) & (df[col_name] <= limite_superior)
    return df[mask_iqr]