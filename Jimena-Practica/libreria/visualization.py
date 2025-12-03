import pandas as pd
import cufflinks as cf
import plotly.express as px
from plotly.offline import iplot
from plotly.graph_objects import Figure
from typing import List, Union
import plotly.offline as pyo
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
    Genera un scatter plot interactivo (2D o 3D) de los componentes principales.

    La reducción a 2D/3D facilita la visualización de patrones de datos complejos [30, 31].

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
    if dim == 3:
        if df_pca.shape[19] < 3:
            raise ValueError("Se requieren al menos 3 componentes (PC1, PC2, PC3) para la visualización 3D.")
        fig = df_pca.iplot(
            kind='scatter3d', 
            mode='markers', 
            x='PC1', 
            y='PC2', 
            z='PC3',
            title='Visualización PCA 3D', 
            opacity=0.8, 
            asFigure=True
        )
    elif dim == 2:
        fig = df_pca.iplot(
            kind='scatter', 
            mode='markers', 
            x='PC1', 
            y='PC2', 
            title='Visualización PCA 2D', 
            asFigure=True
        )
    else:
        raise ValueError("La dimensión debe ser 2 o 3.")
        
    fig.show()
    return fig
