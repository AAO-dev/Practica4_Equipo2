import pandas as pd
from typing import List

def seleccionar_representantes_clustervers(rsquare_df: pd.DataFrame) -> List[str]:
    """
    Selecciona la variable representativa de cada clúster basándose en el ratio 1-R².
    
    El ratio 1-R² (RS_Ratio) indica qué tan bien la variable se correlaciona con su propio clúster
    versus los demás clústeres. Un valor bajo de 1-R² significa que la variable está bien representada
    por su clúster y no por otros, lo que la hace una buena candidata como representante.
    
    Args:
        rsquare_df (pd.DataFrame): DataFrame retornado por VarClusHi con columnas:
            - Cluster: ID del clúster
            - Variable: Nombre de la variable
            - RS_Own: R² con su propio clúster
            - RS_NC: R² con el siguiente clúster más cercano
            - RS_Ratio: Ratio 1-R² = (1 - RS_Own) / (1 - RS_NC)
            
    Returns:
        List[str]: Lista de variables seleccionadas (una por clúster).
    """
    # Agrupar por clúster y seleccionar la variable con el menor RS_Ratio
    # (mejor correlación con su propio clúster)
    selected_vars = []
    
    for cluster_id in rsquare_df['Cluster'].unique():
        cluster_vars = rsquare_df[rsquare_df['Cluster'] == cluster_id]
        
        # Seleccionar la variable con el menor RS_Ratio
        # (o alternativamente, la mayor RS_Own si RS_Ratio no está disponible)
        if 'RS_Ratio' in cluster_vars.columns:
            best_var = cluster_vars.loc[cluster_vars['RS_Ratio'].idxmin(), 'Variable']
        else:
            # Fallback: usar RS_Own (mayor es mejor)
            best_var = cluster_vars.loc[cluster_vars['RS_Own'].idxmax(), 'Variable']
        
        selected_vars.append(best_var)
    
    return selected_vars
