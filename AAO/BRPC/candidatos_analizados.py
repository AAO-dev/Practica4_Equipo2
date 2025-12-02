import pandas as pd
import numpy as np

from .analisis_dataset import analisis_dataset

def candidatos_analizados(df: pd.DataFrame = None):
    if df is None:
        print("Cargando datos...")
        try:
            df = pd.read_csv('../dataset/bankruptcy_polish_companies.csv')
        except FileNotFoundError:
            print("Dataset no encontrado en ../dataset/, intentando ruta local")
            return

    # Procesamiento de datos para manejar NaNs
    df_clean = analisis_dataset(df)
    
    # Eliminacion de variables no numericas y objetivo
    numeric_df = df_clean.select_dtypes(include=['number'])
    if 'class' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['class'])
    if 'year' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['year'])

    print(f"Analizando {numeric_df.shape[1]} variablesnumericas...")

    # 1. Analisis de varianza
    variances = numeric_df.var().sort_values()
    low_variance = variances[variances < 0.01]
    print(f"\nVariables con varianza muy baja (< 0.01): {len(low_variance)}")
    print(low_variance.head())

    # 2. Analisis de correlacion
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Buscar variables con correlacion > 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    print(f"\nVariables con correlacion > 0.9: {len(to_drop)}")
    print(to_drop[:10]) # Show first 10

    # 3. Pares específicos
    print("\nTop 10 pares correlacionados:")
    pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
             .stack()
             .sort_values(ascending=False))
    print(pairs.head(10))

if __name__ == "__main__":
    candidatos_analizados()
