import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def pca_analisis(df: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Realiza un Análisis de Componentes Principales (PCA) para reducir la dimensionalidad.
    
    PCA transforma las variables originales en un nuevo conjunto de variables no correlacionadas
    (componentes principales) que capturan la máxima varianza de los datos.
    
    Args:
        df (pd.DataFrame): DataFrame con variables numéricas.
        n_components (int): Número de componentes principales a retener (2 o 3 para visualización).
        
    Returns:
        Tuple[pd.DataFrame, PCA, np.ndarray]:
            - DataFrame con los componentes principales (PC1, PC2, ...)
            - Objeto PCA ajustado (contiene explained_variance_ratio_, components_, etc.)
            - Array con la varianza explicada por cada componente
    """
    # Filtrar solo columnas numéricas y eliminar target/year si existen
    df_numeric = df.select_dtypes(include=['number']).copy()
    
    # Guardar nombres de columnas para referencia
    original_columns = df_numeric.columns.tolist()
    
    # Eliminar columnas no deseadas (target, identificadores)
    cols_to_drop = [col for col in ['class', 'year', 'id', 'ID'] if col in df_numeric.columns]
    if cols_to_drop:
        df_numeric = df_numeric.drop(columns=cols_to_drop)
    
    # Estandarizar los datos (PCA requiere que las variables estén en la misma escala)
    # Media = 0, Desviación estándar = 1
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    print(f"Dimensiones del dataset antes de PCA: {df_numeric.shape} (Filas, Columnas)")
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_scaled)
    
    # Crear DataFrame con los componentes
    component_names = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(
        data=components,
        columns=component_names,
        index=df_numeric.index
    )
    
    print(f"Dimensiones del dataset después de PCA: {df_pca.shape} (Filas, Componentes)")
    print(f"Reducción: {df_numeric.shape[1]} variables -> {n_components} componentes")
    
    # Agregar la columna 'class' si existe en el DataFrame original
    if 'class' in df.columns:
        df_pca['class'] = df['class'].values
    
    # Varianza explicada por cada componente
    explained_variance = pca.explained_variance_ratio_
    
    return df_pca, pca, explained_variance
