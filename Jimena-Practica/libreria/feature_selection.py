import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Tuple, Union, Literal

def seleccionar_kbest(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int = 7, 
    score_func: Literal['f_classif', 'f_regression'] = 'f_classif'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selecciona las 'K' mejores características (variables) con mayor poder predictivo 
    utilizando SelectKBest.

    SelectKBest es un método de filtro que evalúa la relevancia de las características 
    de forma independiente al modelo [3].

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame de características (variables independientes). Debe contener solo valores numéricos no negativos.
    y : pd.Series
        Serie de la variable objetivo (target).
    k : int, optional
        Número de características principales a seleccionar, por defecto es 7.
    score_func : {'f_classif', 'f_regression'}, optional
        Función de puntuación a utilizar. 'f_classif' es para clasificación (problemas binarios como el de bancarrota) 
        y 'f_regression' para regresión [4]. Por defecto es 'f_classif'.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        DataFrame con las K características seleccionadas y un DataFrame con todas las 
        puntuaciones y nombres de las variables, ordenadas de mayor a menor poder predictivo.
    """
    
    # Mapear string a función real de Scikit-Learn
    if score_func == 'f_classif':
        scoring = f_classif
    elif score_func == 'f_regression':
        scoring = f_regression
    else:
        raise ValueError("score_func debe ser 'f_classif' o 'f_regression'.")

    # Asegurar que X y Y tengan el mismo índice
    X_aligned, y_aligned = X.align(y, join='inner', axis=0)
    
    # Inicializar y ajustar SelectKBest
    kb = SelectKBest(k=k, score_func=scoring)
    kb.fit(X_aligned, y_aligned)

    # Crear el DataFrame de resultados (scores)
    scores_df = pd.DataFrame({
        'Variable': X.columns,
        'Puntuación': kb.scores_,
        'p-valor': kb.pvalues_
    }).sort_values(by='Puntuación', ascending=False).reset_index(drop=True)
    
    # Obtener las K mejores variables
    selected_features = kb.get_feature_names_out()
    X_selected = X_aligned[selected_features]
    
    # Una puntuación alta indica una fuerte asociación y alto poder predictivo [5].
    return X_selected, scores_df

def calcular_woe_iv_variable(df_temp: pd.DataFrame, feature: str, target: str) -> Tuple[pd.DataFrame, float]:
    """
    Calcula el Peso de la Evidencia (WoE) y el Valor de Información (IV) para una 
    variable (asumida discreta o ya binned) contra una variable objetivo binaria.

    WoE/IV se basa en la Teoría de la Información para medir el poder predictivo en 
    problemas de clasificación binaria [1, 4, 6].

    Parameters
    ----------
    df_temp : pd.DataFrame
        DataFrame que contiene la característica y el target.
    feature : str
        Nombre de la variable característica (ya discretizada/categórica).
    target : str
        Nombre de la variable objetivo (binaria 0/1).

    Returns
    -------
    tuple of (pd.DataFrame, float)
        DataFrame de resultados (WoE por categoría) y el Valor de Información (IV) total.
    """
    
    # Paso 1: Conteo de Eventos (1) y No Eventos (0) por categoría
    agg = df_temp.groupby(feature)[target].agg(
        total_count='count', 
        evento_count='sum'
    ).reset_index()
    
    agg['no_evento_count'] = agg['total_count'] - agg['evento_count']
    
    # Evitar divisiones por cero (WoE indefinido) [7]
    agg['evento_count'] = agg['evento_count'].apply(lambda x: x if x > 0 else 1)
    agg['no_evento_count'] = agg['no_evento_count'].apply(lambda x: x if x > 0 else 1)

    # Totales de Eventos/No Eventos en la población
    total_eventos = agg['evento_count'].sum()
    total_no_eventos = agg['no_evento_count'].sum()

    # Paso 2: Cálculo de Proporciones y WoE
    agg['% Evento'] = agg['evento_count'] / total_eventos
    agg['% No Evento'] = agg['no_evento_count'] / total_no_eventos
    
    # WoE = ln(P(No Evento) / P(Evento)) [8]
    agg['WoE'] = np.log(agg['% No Evento'] / agg['% Evento'])
    
    # Paso 3: Cálculo de IV
    # IV = Σ [(% No Eventos - % Eventos) * WoE] [9]
    agg['IV_individual'] = (agg['% No Evento'] - agg['% Evento']) * agg['WoE']
    
    iv_total = agg['IV_individual'].sum()
    
    return agg.set_index(feature), iv_total

def discretizar_variable_para_woe(
    df: pd.DataFrame, 
    feature: str, 
    n_bins: int = 10
) -> pd.Series:
    """
    Discretiza una variable continua utilizando quantile binning (qcut), un paso 
    necesario antes de calcular WoE/IV [7].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    feature : str
        Nombre de la variable continua a discretizar.
    n_bins : int, optional
        Número de bins (segmentos) a crear.

    Returns
    -------
    pd.Series
        Serie de datos discretizados (categorías de rango).
    """
    # Usamos qcut (binning por cuantiles) para intentar tener igual número de observaciones en cada bin
    # 'duplicates="drop"' maneja casos de valores repetidos donde no se pueden formar 'n_bins' únicos.
    return pd.qcut(df[feature], q=n_bins, duplicates="drop").astype(str)