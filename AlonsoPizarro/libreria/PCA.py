import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Necesario para la escalabilidad

def ejecutar_pca_analisis(df_variables_predictoras: pd.DataFrame, n_components_final: int = 0):
    """
    Realiza un Análisis de Componentes Principales (PCA) completo:
    1. Escala (estandariza) las variables predictoras.
    2. Evalúa la varianza explicada con un PCA completo.
    3. Genera el gráfico de Varianza Acumulada.
    4. Proyecta los datos a 2 y 3 componentes principales.

    Args:
        df_variables_predictoras (pd.DataFrame): DataFrame con las variables predictoras.
        n_components_final (int): Número de componentes a seleccionar según el umbral.
                                  Si es > 0, realiza una proyección adicional con ese número.

    Returns:
        tuple: (DataFrame PCA de 2 CP, DataFrame PCA de 3 CP, DataFrame PCA n_final)
    """
    
    print("--- 1. Preparación de Datos: Estandarización ---")
    
    # 1. ESCALADO / ESTANDARIZACIÓN
    # Es crucial estandarizar los datos antes de aplicar PCA.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_variables_predictoras)
    columnas_X = df_variables_predictoras.columns
    X_scaled_df = pd.DataFrame(X_scaled, columns=columnas_X)
    num_vars = len(X_scaled_df.columns)

    print(f"Datos escalados. Variables originales: {num_vars}")

    # --- 2. EVALUACIÓN Y SELECCIÓN DEL NÚMERO DE COMPONENTES ---
    print("\n--- 2. Evaluación PCA Completo ---")
    
    pca_full = PCA(n_components=None) # n_components=None usa todas las variables
    pca_full.fit(X_scaled_df)

    # Varianza Explicada Acumulada
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Graficar Varianza Acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Varianza Acumulada Explicada por Componentes Principales')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada Explicada')
    plt.grid(True)
    plt.axhline(y=0.90, color='r', linestyle='-', label='Umbral del 90%')
    plt.legend(loc='best')
    plt.show()
    
    # Imprimir interpretación
    print(f"CP1 explica: {pca_full.explained_variance_ratio_[0]*100:.2f}% de la varianza total.")
    print(f"CP1 y CP2 juntos explican: {cumulative_variance[1]*100:.2f}% de la varianza total.")

    
    # --- 3. PROYECCIÓN DE COMPONENTES SELECCIONADOS ---
    resultados_pca = {}
    
    # A. PCA de 2 Componentes (Para Visualización)
    print("\n--- 3A. PCA con 2 Componentes Principales ---")
    pca_2 = PCA(n_components=2)
    X_pca_2 = pca_2.fit_transform(X_scaled_df)
    X_pca_2_df = pd.DataFrame(data = X_pca_2, columns = ['PC1', 'PC2'])
    
    print(f"Varianza total explicada (2 CP): {pca_2.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Visualización 2D
    plt.figure(figsize=(8,8))
    plt.scatter(X_pca_2_df['PC1'], X_pca_2_df['PC2'], alpha=0.6, s=10)
    plt.xlabel('Componente Principal 1 (PC1)')
    plt.ylabel('Componente Principal 2 (PC2)')
    plt.title('Conjunto de Datos Proyectado en 2 Componentes Principales')
    plt.grid()
    plt.show()
    
    resultados_pca['pca_2'] = X_pca_2_df

    # B. PCA de 3 Componentes
    print("\n--- 3B. PCA con 3 Componentes Principales ---")
    pca_3 = PCA(n_components=3)
    X_pca_3 = pca_3.fit_transform(X_scaled_df)
    X_pca_3_df = pd.DataFrame(data = X_pca_3, columns = ['PC1', 'PC2', 'PC3'])
    
    print(f"Varianza total explicada (3 CP): {pca_3.explained_variance_ratio_.sum()*100:.2f}%")
    resultados_pca['pca_3'] = X_pca_3_df

    # C. PCA Final (Si se especifica)
    if n_components_final > 0:
        print(f"\n--- 3C. PCA con {n_components_final} Componentes (Selección Final) ---")
        pca_final = PCA(n_components=n_components_final)
        X_pca_final = pca_final.fit_transform(X_scaled_df)
        
        columnas_final = [f'PC{i+1}' for i in range(n_components_final)]
        X_pca_final_df = pd.DataFrame(data = X_pca_final, columns = columnas_final)
        
        print(f"Varianza total explicada ({n_components_final} CP): {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
        resultados_pca['pca_final'] = X_pca_final_df
    else:
        resultados_pca['pca_final'] = None
        
    print("\nAnálisis PCA completado.")
    return resultados_pca['pca_2'], resultados_pca['pca_3'], resultados_pca['pca_final']


import pandas as pd
import plotly.express as px
# Nota: La función ya asume que los DataFrames de PCA fueron creados previamente.

def graficar_pca_plotly(X_pca_2_df: pd.DataFrame, X_pca_3_df: pd.DataFrame, template: str = 'plotly_white'):
    """
    Genera y muestra gráficos de dispersión interactivos 2D y 3D
    de los resultados del Análisis de Componentes Principales (PCA) usando Plotly Express.

    Args:
        X_pca_2_df (pd.DataFrame): DataFrame con 2 columnas (PC1, PC2).
        X_pca_3_df (pd.DataFrame): DataFrame con 3 columnas (PC1, PC2, PC3).
        template (str): Tema de Plotly a usar (ej: 'plotly_white', 'plotly_dark').
        
    Returns:
        tuple: (figura 2D de Plotly, figura 3D de Plotly)
    """
    
    print("\n--- Generando Gráficos Interactivos con Plotly Express ---")

    ### 1. Gráfico de Dispersión 2D (PC1 vs PC2)
    print("Generando Scatter Plot 2D (PC1 vs PC2)...")
    fig2D = px.scatter(
        X_pca_2_df, 
        x='PC1', 
        y='PC2', 
        title='Proyección de Datos en PC1 vs PC2 (PCA 2 Componentes)',
        template=template
    )
    fig2D.show()

    ### 2. Gráfico de Dispersión 3D (PC1 vs PC2 vs PC3)
    print("Generando Scatter Plot 3D (PC1 vs PC2 vs PC3)...")
    fig3D = px.scatter_3d(
        X_pca_3_df, 
        x='PC1', 
        y='PC2', 
        z='PC3', 
        title='Proyección de Datos en 3 Componentes Principales (PCA 3 Componentes)',
        template=template,
        opacity=0.6, # Suaviza la visualización de muchos puntos
        height=700 # Altura de la figura
    )
    fig3D.show()

    print("\nGráficos generados.")
    return fig2D, fig3D