import pandas as pd
import numpy as np
from typing import Tuple, Union
from sklearn.tree import DecisionTreeClassifier
from .agrupamiento_optimo import agrupamiento_optimo

def woe_iv(df: pd.DataFrame, feature: str, target: str) -> Tuple[pd.DataFrame, float]:
    """
    Calcula el Peso de la Evidencia (WoE) y el Valor de Información (IV) para una variable categórica.
    
    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        feature (str): El nombre de la columna de la variable independiente (categórica o binned).
        target (str): El nombre de la columna de la variable objetivo (binaria 0/1).
        
    Returns:
        Tuple[pd.DataFrame, float]: 
            - Un DataFrame con las estadísticas detalladas por categoría (Count, Event, NonEvent, WoE, IV).
            - El valor total de IV de la variable.
    """
    
    # Asegurarse de que no haya nulos o tratarlos como una categoría
    df_temp = df[[feature, target]].copy()
    df_temp[feature] = df_temp[feature].fillna('Missing')

    # Calcular conteos de Eventos (1) y No Eventos (0) por categoría
    grouped = df_temp.groupby(feature)[target].value_counts().unstack(fill_value=0)
    
    # Si falta alguna columna (0 o 1), agregarla con ceros
    if 0 not in grouped.columns:
        grouped[0] = 0
    if 1 not in grouped.columns:
        grouped[1] = 0
        
    # Renombrar columnas para claridad
    grouped = grouped.rename(columns={0: 'NonEvent', 1: 'Event'})
    
    # Totales globales
    total_events = grouped['Event'].sum()
    total_non_events = grouped['NonEvent'].sum()
    
    # Evitar división por cero en totales (caso extremo)
    if total_events == 0 or total_non_events == 0:
        return grouped, 0.0

    # Calcular distribuciones
    grouped['DistEvent'] = grouped['Event'] / total_events
    grouped['DistNonEvent'] = grouped['NonEvent'] / total_non_events
    
    # Suavizado para evitar log(0) o división por cero en WoE
    # Se añade un pequeño epsilon si DistEvent o DistNonEvent es 0
    epsilon = 0.0001
    grouped['WoE'] = np.log((grouped['DistEvent'] + epsilon) / (grouped['DistNonEvent'] + epsilon))
    
    # Calcular IV por grupo
    grouped['IV_Group'] = (grouped['DistEvent'] - grouped['DistNonEvent']) * grouped['WoE']
    
    # Calcular IV total
    total_iv = grouped['IV_Group'].sum()
    
    # Ordenar y limpiar
    grouped = grouped.reset_index()
    
    return grouped, total_iv
