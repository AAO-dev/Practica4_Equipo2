import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple

def seleccionar_kbest_features(
    df_predictoras: pd.DataFrame, 
    serie_target: pd.Series, 
    k: int = 7
) -> Tuple[List[str], pd.DataFrame]:
    """
    Aplica el método SelectKBest con f_classif (ANOVA F-test) para seleccionar
    las K mejores variables predictoras con respecto a una variable objetivo binaria.

    Args:
        df_predictoras (pd.DataFrame): DataFrame con las variables explicativas (features X).
        serie_target (pd.Series): Serie de Pandas con la variable objetivo binaria (Y).
        k (int): Número de variables que se desean seleccionar (K).

    Returns:
        Tuple[List[str], pd.DataFrame]: 
            1. Lista de nombres de las K variables seleccionadas.
            2. DataFrame completo con el F-Score y P-Value de todas las variables.
    """
    
    # 1. ESTANDARIZACIÓN
    print("--- 1. Estandarizando las variables predictoras (X) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_predictoras)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df_predictoras.columns)

    # 2. APLICAR SELECTKBEST
    print(f"--- 2. Aplicando SelectKBest con K={k} y F-test ---")
    
    # Inicializar el selector
    selector = SelectKBest(score_func=f_classif, k=k)

    # Ajustar y transformar (entrenar el selector)
    selector.fit(X_scaled_df, serie_target)

    # 3. EXTRACCIÓN Y ORGANIZACIÓN DE RESULTADOS
    
    # Obtener las puntuaciones (F-scores) y los p-valores
    scores = pd.Series(selector.scores_, index=X_scaled_df.columns)
    p_values = pd.Series(selector.pvalues_, index=X_scaled_df.columns)

    # Crear el DataFrame con los resultados completos
    results_df = pd.DataFrame({
        'F_Score': scores,
        'P_Value': p_values
    }).sort_values(by='F_Score', ascending=False)
    
    # Obtener los nombres de las K mejores variables
    selected_features = results_df.head(k).index.tolist()

    # 4. VISUALIZACIÓN
    print(f"\n### 🎯 Top {k} Variables Seleccionadas por F-Score ###")
    print(results_df.head(k).round(4))

    print("\n--- Lista Final de Variables Seleccionadas ---")
    print(selected_features)
    
    return selected_features, results_df