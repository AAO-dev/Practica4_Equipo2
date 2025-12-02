import pandas as pd
import numpy as np

def procesado_dataset(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa los datos eliminando columnas con muchos nulos, imputando valores y tratando outliers.
    
    Pasos realizados:
    1. Eliminar columnas con más del 20% de valores nulos.
    2. Imputar valores faltantes:
       - Mediana para columnas numéricas.
       - Moda para columnas categóricas (o no numéricas).
    3. Tratar valores atípicos (outliers) en columnas numéricas usando el método IQR (clipping).
       - Se limitan los valores al rango [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Args:
        df_input (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame procesado.
    """
    # Hacemos una copia para no modificar el original
    df = df_input.copy()
    
    # 1. Eliminar columnas con más del 20% de valores nulos
    threshold = 0.2 * len(df)
    # Identificamos columnas que superan el umbral de nulos
    cols_to_drop = df.columns[df.isnull().sum() > threshold]
    if len(cols_to_drop) > 0:
        print(f"Eliminando columnas con > 20% nulos: {list(cols_to_drop)}")
        df.drop(columns=cols_to_drop, inplace=True)
    
    # Separamos columnas numéricas y categóricas para aplicar diferentes estrategias
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # 2. Imputación de valores faltantes
    
    # Para numéricas: imputar con la mediana
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    # Para categóricas: imputar con la moda
    for col in categorical_cols:
        if df[col].isnull().any():
            # mode() devuelve una Serie, tomamos el primer valor [0]
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            
    # 3. Tratamiento de valores atípicos (Outliers) con IQR Clipping
    # Solo aplicamos esto a columnas numéricas, EXCLUYENDO la variable objetivo
    # IMPORTANTE: No aplicar clipping a 'class' porque es la variable objetivo binaria
    target_col = 'class'
    numeric_cols_for_outliers = [col for col in numeric_cols if col != target_col]
    
    for col in numeric_cols_for_outliers:
        # Calculamos Q1 (percentil 25) y Q3 (percentil 75)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definimos los límites inferior y superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Aplicamos clipping (recorte)
        # Los valores menores al límite inferior se reemplazan por el límite inferior
        # Los valores mayores al límite superior se reemplazan por el límite superior
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df

