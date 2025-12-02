import pandas as pd
from varclushi import VarClusHi

def varclushi_analisis(df: pd.DataFrame, max_eigval2: float = 1.0, max_pca_components: int = 20) -> pd.DataFrame:
    """
    Realiza un análisis VarClusHi para agrupar variables correlacionadas.
    
    VarClusHi requiere que solo se incluyan variables predictoras (features), no la variable objetivo
    ni identificadores. El algoritmo puede fallar si se incluyen columnas no apropiadas.
    
    Args:
        df (pd.DataFrame): DataFrame con variables numéricas.
        max_eigval2 (float): Umbral del segundo autovalor para detener la división (criterio de parada).
        max_pca_components (int): Número máximo de componentes principales a calcular.
        
    Returns:
        pd.DataFrame: DataFrame con la información de los clústeres y métricas (RS_Ratio, etc.).
    """
    # Filtrar solo columnas numéricas
    df_numeric = df.select_dtypes(include=['number']).copy()
    
    # Eliminar columnas que NO son features (target, identificadores, etc.)
    # Estas columnas pueden causar que VarClusHi falle o produzca NaN
    cols_to_exclude = ['class', 'year', 'id', 'ID', 'index']
    cols_to_drop = [col for col in cols_to_exclude if col in df_numeric.columns]
    
    if cols_to_drop:
        print(f"Excluyendo columnas no-feature: {cols_to_drop}")
        df_numeric = df_numeric.drop(columns=cols_to_drop)
    
    print(f"Ejecutando VarClusHi con {df_numeric.shape[1]} variables...")
    
    # Instanciar el modelo VarClusHi
    # maxeigval2: criterio de parada (si el segundo autovalor es < maxeigval2, no se divide más)
    # maxclus: número máximo de clústeres permitidos
    demo_vc = VarClusHi(df_numeric, maxeigval2=max_eigval2, maxclus=max_pca_components)
    
    # Entrenar
    demo_vc.varclus()
    
    # Obtener el resumen de información (rsquare)
    # Este dataframe contiene: Cluster, Variable, RS_Own, RS_NC, RS_Ratio
    # RS_Own: R² de la variable con su propio clúster
    # RS_NC: R² de la variable con el siguiente clúster más cercano
    # RS_Ratio: (1 - RS_Own) / (1 - RS_NC) - valores bajos indican buena representación
    rsquare = demo_vc.rsquare
    
    return rsquare
