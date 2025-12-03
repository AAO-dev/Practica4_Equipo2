import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from varclushi import VarClusHi
# Importaciones necesarias para el Dendrograma
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Tuple, Dict



def ejecutar_varclus_analysis(df_input: pd.DataFrame) -> Tuple[VarClusHi, pd.DataFrame]:
    """
    Realiza un análisis de Clustering de Variables (VarClusHi) y genera un 
    dendrograma basado en la correlación para identificar grupos de variables.
    
    1. Escala (estandariza) los datos de entrada.
    2. Ejecuta el algoritmo VarClusHi.
    3. Genera el dendrograma de agrupación jerárquica.

    Args:
        df_input (pd.DataFrame): DataFrame con las variables predictoras a agrupar.

    Returns:
        Tuple[VarClusHi, pd.DataFrame]: 
            1. El objeto VarClusHi ajustado (para análisis detallado posterior).
            2. El DataFrame con la estructura de grupos finales (vc.info).
    """
    
    # 1. ESCALADO / ESTANDARIZACIÓN
    print("--- 1. Estandarización de Datos ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_input)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df_input.columns)
    print(f"Datos escalados. Variables: {X_scaled_df.shape[1]}")

    # 2. ANÁLISIS VARCLUSHI (Clustering)
    print("\n--- 2. Ejecutando VarClusHi (Grupos de Variables) ---")
    
    # maxeigval2=1: Criterio para detener la división de un cluster.
    vc = VarClusHi(
        X_scaled_df, 
        maxeigval2=1
    )
    vc.varclus() 
    print("Análisis VarClusHi completado. Grupos identificados.")

    # 3. VISUALIZACIÓN DEL DENDROGRAMA (Agrupación Jerárquica)
    
    print("\n--- 3. Generando Dendrograma de Agrupación Jerárquica ---")
    
    # Matriz de correlación absoluta
    corr_matrix = X_scaled_df.corr().abs()
    # Matriz de distancia (1 - |Correlación|)
    distance_matrix = 1 - corr_matrix
    # Vector de distancia condensada (requerido por scipy)
    # Ignoramos la diagonal y la parte superior del triángulo
    condensed_distance_vector = squareform(distance_matrix)
    
    # Aplicar el método de vinculación (Ward)
    # Ward intenta minimizar la varianza dentro de cada cluster
    linked = linkage(condensed_distance_vector, method='ward') 
    
    # Generar el Dendrograma
    labels = X_scaled_df.columns.tolist()
    plt.figure(figsize=(20, 10))
    plt.title('Dendrograma de Clustering de Variables (Distancia de Correlación)')
    plt.xlabel('Variables Predictoras')
    plt.ylabel('Distancia de Correlación (1 - |Correlación|)')
    dendrogram(
        linked, 
        orientation='top',
        labels=labels,
        distance_sort='descending',
        show_leaf_counts=False
    )
    plt.show() 
    

    # 4. RESULTADOS FINALES
    print("\n--- 4. Estructura de Grupos Finales (vc.info) ---")
    print(vc.info)
    
    return vc, vc.info


import pandas as pd
from varclushi import VarClusHi
from typing import List, Tuple



def seleccionar_representantes_varclus(vc_model: VarClusHi) -> Tuple[List[str], pd.DataFrame]:
    """
    Selecciona la variable representante de cada clúster generado por VarClusHi.
    
    El representante es la variable dentro de cada clúster que tiene el R-cuadrado 
    (RS_Own) más alto con respecto a su propio componente principal del clúster.
    
    Args:
        vc_model (VarClusHi): El objeto VarClusHi ya ajustado y con el clustering ejecutado.
        
    Returns:
        Tuple[List[str], pd.DataFrame]:
            1. Lista de nombres de las variables representantes seleccionadas.
            2. DataFrame con el detalle de las variables representantes (Cluster, Variable, RS_Own).
    """
    
    print("\n--- 1. Extrayendo Resultados y Ordenando por RS_Own ---")
    
    # Extraer el DataFrame con los resultados de R-cuadrado
    # Este DataFrame tiene las columnas 'Cluster', 'Variable', y 'RS_Own'
    rsquare_df = vc_model.rsquare.copy()
    
    # Ordenar: primero por Cluster, y luego por RS_Own descendente
    rsquare_df = rsquare_df.sort_values(by=['Cluster', 'RS_Own'], ascending=[True, False])

    # 2. SELECCIÓN DEL REPRESENTANTE
    print("--- 2. Filtrando la variable con el RS_Own más alto por cada Cluster ---")
    
    # Usar groupby('Cluster') y idxmax() en 'RS_Own' para obtener el índice de la 
    # fila con el valor máximo de RS_Own dentro de cada grupo.
    representatives = rsquare_df.loc[rsquare_df.groupby('Cluster')['RS_Own'].idxmax()]

    # 3. RESULTADOS FINALES
    
    # Filtrar solo las columnas de interés para la visualización
    representatives_output = representatives[['Cluster', 'Variable', 'RS_Own']].reset_index(drop=True)
    
    # Lista final de variables a usar en el modelo
    final_variables = representatives_output['Variable'].tolist()

    print("\n### 🎯 Variables Representantes por Grupo (RS_Own Máximo) ###")
    print(representatives_output.round(4))
    
    print("\n### Lista Final de Variables Representantes Seleccionadas ###")
    print(final_variables)
    
    return final_variables, representatives_output