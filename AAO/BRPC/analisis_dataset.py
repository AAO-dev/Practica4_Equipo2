import pandas as pd
import numpy as np

def analisis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función general para verificar la completitud y calidad de los datos.
    
    Esta función analiza un DataFrame de pandas y genera un reporte con estadísticas
    clave sobre la calidad de los datos, incluyendo valores nulos, tipos de datos,
    y estadísticas de dispersión para variables numéricas.
    
    Args:
        df (pd.DataFrame): El DataFrame de entrada a analizar.
        
    Returns:
        pd.DataFrame: Un DataFrame resumen con las siguientes columnas:
            - 'Null Count': Cantidad de valores nulos.
            - 'Completeness (%)': Porcentaje de datos no nulos.
            - 'Data Type': Tipo de dato de la columna.
            - 'Std': Desviación estándar (solo para numéricos).
            - 'Variance': Varianza (solo para numéricos).
            - 'Type': Clasificación automática (Continua/Discreta).
    """
    
    # Inicializamos una lista para guardar los resultados de cada columna
    summary_data = []
    
    for col in df.columns:
        # Obtenemos la serie de datos de la columna actual
        series = df[col]
        
        # 1. Conteo de Nulos
        null_count = series.isnull().sum()
        
        # 2. Porcentaje de completitud
        # Calculamos (Total - Nulos) / Total * 100
        completeness = ((len(df) - null_count) / len(df)) * 100
        
        # 3. Tipo de Datos
        dtype = series.dtype
        
        # Inicializamos estadísticos de dispersión
        std_dev = np.nan
        variance = np.nan
        data_class = "Unknown"
        
        # Verificamos si la columna es numérica para calcular estadísticas
        if np.issubdtype(dtype, np.number):
            # 4. Estadísticos de Dispersión
            std_dev = series.std()
            variance = series.var()
            
            # 5. Clasificación automática (Continua/Discreta)
            # Heurística: Si es float o tiene muchos valores únicos (>20), asumimos continua.
            # Si son pocos valores únicos, asumimos discreta (o categórica numérica).
            # Para este dataset financiero, la mayoría serán continuas.
            if pd.api.types.is_float_dtype(dtype) or series.nunique() > 20:
                data_class = "Continua"
            else:
                data_class = "Discreta"
        else:
            # Si no es numérica, la tratamos como Discreta (Categórica)
            data_class = "Discreta"
            
        # Agregamos la fila al resumen
        summary_data.append({
            'Column': col,
            'Null Count': null_count,
            'Completeness (%)': round(completeness, 2),
            'Data Type': str(dtype),
            'Std': round(std_dev, 4) if not np.isnan(std_dev) else np.nan,
            'Variance': round(variance, 4) if not np.isnan(variance) else np.nan,
            'Type': data_class
        })
    
    # Convertimos la lista de diccionarios a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Establecemos la columna 'Column' como índice para mejor legibilidad
    summary_df.set_index('Column', inplace=True)
    
    return summary_df
