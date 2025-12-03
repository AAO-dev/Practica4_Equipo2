import pandas as pd
import cufflinks as cf
import plotly.express as px
from plotly.offline import iplot
from plotly.graph_objects import Figure
from typing import List, Union

cf.go_offline()

def generar_histograma_cf(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Genera histogramas interactivos para las columnas continuas usando Cufflinks.

    Ayuda a identificar la distribución y el comportamiento de los datos [28].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    columns : list of str
        Lista de columnas numéricas para graficar.
    """
    for col in columns:
        print(f"Generando histograma para: {col}")
        # Uso de iplot con 'hist' como tipo de gráfico (kind)
        df[col].iplot(
            kind='hist', 
            title=f'Distribución de {col}', 
            xTitle=col, 
            yTitle='Frecuencia', 
            bins=50
        )
        
def generar_boxplot_cf(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Genera boxplots interactivos para las columnas continuas usando Cufflinks.

    Útil para visualizar la dispersión y la detección de outliers (método IQR) [9, 29].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    columns : list of str
        Lista de columnas numéricas para graficar.
    """
    for col in columns:
        print(f"Generando Boxplot para: {col}")
        # Uso de iplot con 'box' como tipo de gráfico (kind)
        df[col].iplot(
            kind='box', 
            title=f'Boxplot de {col}'
        )

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
