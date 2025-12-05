import pandas as pd
import cufflinks as cf
import plotly.express as px
from plotly.offline import iplot
from plotly.graph_objects import Figure
from typing import List, Union
import plotly.graph_objects as go



def generar_histograma_cf(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Genera histogramas interactivos usando Plotly.
    """

    for col in columns:

        if col not in df.columns:
            print(f"⚠ La columna '{col}' no existe en el DataFrame.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠ La columna '{col}' no es numérica.")
            continue

        print(f"Generando histograma para: {col}")

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df[col],
                    nbinsx=50,
                    marker=dict(color="rgba(31,119,180,1)")
                )
            ]
        )

        fig.update_layout(
            title=f"Distribución de {col}",
            xaxis_title=col,
            yaxis_title="Frecuencia"
        )

        fig.show()


        
def generar_boxplot_cf(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Genera boxplots interactivos sin usar Cufflinks, evitando errores
    derivados de colores inválidos en Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    columns : list of str
        Lista de columnas numéricas para graficar.
    """

    for col in columns:

        # Validar columna existente y numérica
        if col not in df.columns:
            print(f"⚠ La columna '{col}' no existe en el DataFrame.")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠ La columna '{col}' no es numérica.")
            continue

        print(f"Generando Boxplot para: {col}")

        fig = go.Figure(
            data=[
                go.Box(
                    y=df[col],
                    name=col,
                    marker=dict(color="rgba(255,153,51,1)")  # color válido
                )
            ]
        )

        fig.update_layout(
            title=f"Boxplot de {col}",
            yaxis_title=col
        )

        fig.show()

def visualizar_pca_componentes(df_pca: pd.DataFrame, dim: int = 2) -> Figure:
    """
    Genera un scatter plot interactivo (2D o 3D) de los componentes principales 
    utilizando plotly.express.

    La reducción a 2D/3D facilita la visualización de patrones de datos complejos.

    Parameters
    ----------
    df_pca : pd.DataFrame
        DataFrame resultante de PCA, con columnas 'PC1', 'PC2', etc.
    dim : int, optional
        Dimensionalidad de la visualización (2 o 3), por defecto es 2.

    Returns
    -------
    plotly.graph_objects.Figure
        Objeto Figure de Plotly.
    """
    # Usamos plotly.express que es más estable y evita el error de formato de color
    
    if dim == 3:
        # Aseguramos que existan las 3 columnas
        required_cols = ['PC1', 'PC2', 'PC3']
        if not all(col in df_pca.columns for col in required_cols):
             raise ValueError("Se requieren las columnas 'PC1', 'PC2' y 'PC3' para la visualización 3D.")
        
        # Para PCA, a menudo se desea colorear por el índice o un grupo
        # Usamos el índice como un identificador único si no hay otra columna de color
        df_pca['ID_Unico'] = df_pca.index.astype(str)

        fig = px.scatter_3d(
            df_pca,
            x='PC1', 
            y='PC2', 
            z='PC3',
            color='ID_Unico',  # Colorea por el ID único (índice)
            opacity=0.7,
            title='Visualización PCA 3D', 
            template='plotly_white'
        )
        
    elif dim == 2:
        # Aseguramos que existan las 2 columnas
        required_cols = ['PC1', 'PC2']
        if not all(col in df_pca.columns for col in required_cols):
             raise ValueError("Se requieren las columnas 'PC1' y 'PC2' para la visualización 2D.")
        
        # Usamos el índice como un identificador único si no hay otra columna de color
        df_pca['ID_Unico'] = df_pca.index.astype(str)

        fig = px.scatter(
            df_pca,
            x='PC1', 
            y='PC2', 
            color='ID_Unico', # Colorea por el ID único (índice)
            opacity=0.8,
            title='Visualización PCA 2D', 
            template='plotly_white'
        )
        
    else:
        raise ValueError("La dimensión debe ser 2 o 3.")
        
    # Eliminamos la columna temporal que creamos
    if 'ID_Unico' in df_pca.columns:
        df_pca.drop(columns=['ID_Unico'], inplace=True)
        
    fig.show()
    return fig