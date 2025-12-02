import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Tuple

def select_mejor_k(X: pd.DataFrame, y: pd.Series, k: int = 7) -> pd.DataFrame:

    """
    Selecciona las 'K' mejores variables (features) de un DataFrame basándose en el criterio estadístico f_classif.
    Este criterio (ANOVA F-value) es adecuado para problemas de clasificación donde las variables de entrada son numéricas.
    Se eligió f_classif sobre chi2 porque el dataset contiene valores negativos, lo cual hace que chi2 no sea válido
    sin un escalado previo (MinMaxScaler). f_classif es robusto a valores negativos en las features.

    Args:
        X (pd.DataFrame): DataFrame que contiene las variables independientes (features).
        y (pd.Series): Serie que contiene la variable objetivo (target/clase).
        k (int, optional): El número de mejores variables a seleccionar. Por defecto es 7.
    Returns:
        pd.DataFrame: Un nuevo DataFrame que contiene solo las 'k' mejores variables seleccionadas.

    """
    

    # Inicializar el selector SelectKBest con la función de puntuación f_classif y el número de features k
    # f_classif calcula el valor F de ANOVA para la muestra proporcionada.

    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Ajustar el selector a los datos (X, y) y transformar X para reducirlo a las k mejores features
    # fit_transform devuelve un array de numpy, por lo que perderemos los nombres de las columnas

    X_new_array = selector.fit_transform(X, y)
    
    # Obtener los índices de las columnas seleccionadas (una máscara booleana)

    selected_indices = selector.get_support(indices=True)

    # Obtener los nombres de las columnas seleccionadas usando los índices

    selected_features = X.columns[selected_indices]
    
    # Crear un nuevo DataFrame con las features seleccionadas, preservando el índice original de X

    X_new = pd.DataFrame(X_new_array, columns=selected_features, index=X.index)
    
    # Imprimir información sobre las variables seleccionadas para retroalimentación

    print(f"Se han seleccionado las siguientes {k} variables usando f_classif:")
    for feature in selected_features:
        print(f" - {feature}")

    return X_new

