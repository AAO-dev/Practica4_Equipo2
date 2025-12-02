import pandas as pd
import numpy as np
from typing import Tuple, Union
from sklearn.tree import DecisionTreeClassifier

def agrupamiento_optimo(df: pd.DataFrame, feature: str, target: str, max_bins: int = 10, min_bins: int = 3) -> pd.Series:
    """
    Realiza un binning óptimo de una variable numérica basándose en un árbol de decisión.
    Esto asegura que los bins tengan una separación significativa en la variable target.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        feature (str): Nombre de la columna a discretizar.
        target (str): Nombre de la columna objetivo.
        max_bins (int): Número máximo de bins a crear.
        min_bins (int): Número mínimo de bins a crear.
        
    Returns:
        pd.Series: Serie con las categorías binned.
    """
    # Eliminar valores nulos temporalmente para el ajuste
    df_clean = df[[feature, target]].dropna()
    
    if len(df_clean) == 0:
        # Si todos son nulos, retornar categoría 'Missing'
        return pd.Series(['Missing'] * len(df), index=df.index)
    
    X = df_clean[[feature]].values
    y = df_clean[target].values
    
    # Determinar número óptimo de bins basado en la profundidad del árbol
    # Usamos max_depth para limitar el número de splits
    max_depth = int(np.log2(max_bins))
    
    try:
        # Entrenar árbol de decisión para encontrar puntos de corte óptimos
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=max(int(len(df_clean) * 0.05), 100),  # Al menos 5% o 100 muestras por hoja
            random_state=42
        )
        tree.fit(X, y)
        
        # Obtener los valores de corte del árbol
        thresholds = []
        def extract_thresholds(tree, node=0):
            if tree.tree_.feature[node] != -2:  # No es hoja
                thresholds.append(tree.tree_.threshold[node])
                extract_thresholds(tree, tree.tree_.children_left[node])
                extract_thresholds(tree, tree.tree_.children_right[node])
        
        extract_thresholds(tree)
        thresholds = sorted(set(thresholds))
        
        # Asegurar que tenemos al menos min_bins-1 cortes
        if len(thresholds) < min_bins - 1:
            # Usar percentiles si el árbol no genera suficientes cortes
            percentiles = np.linspace(0, 100, min_bins + 1)[1:-1]
            thresholds = np.percentile(df_clean[feature].values, percentiles).tolist()
            thresholds = sorted(set(thresholds))
        
        # Crear los bins usando los umbrales
        bins = [-np.inf] + thresholds + [np.inf]
        labels = [f'Bin_{i+1}' for i in range(len(bins)-1)]
        
        # Aplicar a todo el dataset (incluyendo nulos)
        result = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True, duplicates='drop')
        result = result.astype(str)
        result = result.replace('nan', 'Missing')
        
    except Exception as e:
        # Si falla el binning óptimo, usar qcut con manejo de duplicados
        try:
            result = pd.qcut(df[feature], q=max_bins, duplicates='drop')
            result = result.astype(str)
            result = result.replace('nan', 'Missing')
        except:
            # Último recurso: usar cut con bins equidistantes
            result = pd.cut(df[feature], bins=max_bins, duplicates='drop')
            result = result.astype(str)
            result = result.replace('nan', 'Missing')
    
    return result

