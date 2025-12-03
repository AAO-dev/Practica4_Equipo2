#### - Implementar una función para calcular IV y WoE para las variables categóricas del conjunto de datos.

import pandas as pd
import numpy as np

def calcular_woe_iv(df, feature, target, num_bins=10):
    """
    Calcula el Peso de la Evidencia (WoE) y el Valor de Información (IV)
    para una característica continua dada una variable objetivo binaria.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        feature (str): Nombre de la columna predictiva (continua).
        target (str): Nombre de la columna objetivo (binaria: 0 y 1).
        num_bins (int): Número de cuantiles (bins) a crear para la discretización.

    Returns:
        tuple: (pd.DataFrame con detalles de WoE/IV por bin, float con IV total)
    """

    # --- 1. Discretización (Binning) usando cuantiles (qcut) ---
    try:
        # Crea 'num_bins' contenedores (bins) con frecuencias de datos similares.
        # 'duplicates=drop' maneja los casos con muchos valores repetidos.
        df['Bin'] = pd.qcut(df[feature], q=num_bins, duplicates='drop', retbins=True)[0]
    except Exception as e:
        # Si qcut falla (ej. muchos NaNs o valores idénticos), lo trata como una sola categoría
        print(f"Advertencia: Fallo en qcut para {feature}. Se agrupará la variable completa. Error: {e}")
        df['Bin'] = 'Única Categoría'

    # --- 2. Agrupación y Conteo (Cálculo de Proporciones) ---
    df_agg = (
        df.groupby('Bin', observed= True)[target]
        .agg(['count', 'sum'])
        .rename(columns={'count': 'Total', 'sum': 'Malos_Count'})
    )
    df_agg['Buenos_Count'] = df_agg['Total'] - df_agg['Malos_Count']

    # Manejar el caso de división por cero (si no hay Buenos o Malos totales)
    if df_agg['Malos_Count'].sum() == 0 or df_agg['Buenos_Count'].sum() == 0:
        return pd.DataFrame(), 0.0

    # Totales para calcular la proporción (%)
    Total_Malos = df_agg['Malos_Count'].sum()
    Total_Buenos = df_agg['Buenos_Count'].sum()

    df_agg['%_Malos'] = df_agg['Malos_Count'] / Total_Malos
    df_agg['%_Buenos'] = df_agg['Buenos_Count'] / Total_Buenos

    # --- 3. Cálculo del WoE (Peso de la Evidencia) ---
    # Para evitar log(0), sumamos una constante pequeña (epsilon) si el porcentaje es cero.
    epsilon = 0.0000001
    df_agg['WoE'] = np.log(
        (df_agg['%_Buenos'] + epsilon) / (df_agg['%_Malos'] + epsilon)
    )

    # --- 4. Cálculo del IV (Valor de Información) ---
    df_agg['IV_Contribucion'] = (df_agg['%_Buenos'] - df_agg['%_Malos']) * df_agg['WoE']
    IV_Total = df_agg['IV_Contribucion'].sum()

    # Limpieza: Eliminar la columna temporal 'Bin' del DataFrame original si es necesario
    # Nota: Aquí no se elimina para mantener la función simple y autocontenida.

    return df_agg.reset_index(), IV_Total