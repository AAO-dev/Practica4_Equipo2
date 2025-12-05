import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from varclushi import VarClusHi
from typing import List, Tuple, Union

def escalar_datos(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Escala las columnas numéricas del DataFrame.

    El escalamiento es esencial antes de aplicar PCA [12, 13].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada con solo variables numéricas.
    method : str, optional
        Método de escalamiento a usar ('standard' para StandardScaler o 
        'minmax' para MinMaxScaler), por defecto es 'standard'.

    Returns
    -------
    pd.DataFrame
        DataFrame escalado.
    """
    if method == 'standard':
        scaler = StandardScaler() # StandardScaler es común para PCA [12]
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("El método debe ser 'standard' o 'minmax'.")

    X_scaled = scaler.fit_transform(df)
    return pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

def analisis_pca(df_scaled: pd.DataFrame, n_components: Union[int, float]) -> pd.DataFrame:
    """
    Aplica el Análisis de Componentes Principales (PCA).

    PCA transforma variables correlacionadas en componentes ortogonales que 
    capturan la máxima varianza [14, 15].

    Parameters
    ----------
    df_scaled : pd.DataFrame
        DataFrame escalado (media 0, varianza 1).
    n_components : int or float
        Número de componentes a retener (si es int) o varianza 
        acumulada mínima deseada (si es float, ej. 0.9 para 90%) [16, 17].

    Returns
    -------
    pd.DataFrame
        DataFrame con los nuevos componentes principales (PC).
    """
    # Inicializar y ajustar PCA
    pca = PCA(n_components=n_components)
    pca.fit(df_scaled)
    
    # Imprimir la varianza explicada acumulada
    print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.cumsum()}")

    # Transformar los datos
    Xp = pd.DataFrame(
        index=df_scaled.index, 
        data=pca.transform(df_scaled)
    )
    
    # Nombrar las columnas (PC1, PC2, ...)
    n_final_components = Xp.shape[1]
    Xp.columns = [f'PC{i+1}' for i in range(n_final_components)]
    
    return Xp

def analisis_varclushi(df_numeric: pd.DataFrame, sample_frac: float = 0.1) -> pd.DataFrame:
    """
    Ejecuta VarClusHi para agrupar variables altamente correlacionadas.

    Parameters
    ----------
    df_numeric : pd.DataFrame
        Solo columnas numéricas (idealmente escaladas).
    sample_frac : float
        Fracción a usar como muestra si el dataset es muy grande.

    Returns
    -------
    pd.DataFrame
        Cluster asignado a cada variable con métricas RSquared y RS_Ratio.
    """
    print("Iniciando VarClusHi...")

    # Selección de muestra para eficiencia
    if len(df_numeric) * sample_frac < 1000:
        numeric_sample = df_numeric
        print("Usando dataset completo.")
    else:
        numeric_sample = df_numeric.sample(frac=sample_frac, random_state=42)
        print(f"Usando muestra del {sample_frac * 100}%.")

    # Ejecutar VarClusHi
    vc = VarClusHi(numeric_sample)
    vc.varclus()

    clusters = vc.info

    # Ordenado por clúster y ratio
    if {'Cluster', 'RS_Ratio'}.issubset(clusters.columns):
        return clusters.sort_values(by=['Cluster', 'RS_Ratio'], ascending=[True, True])
    else:
        # Esto manejará el caso si solo retorna la columna 'Cluster'
        return clusters.sort_values(by=['Cluster'])

