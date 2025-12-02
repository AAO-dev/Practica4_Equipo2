import pandas as pd
import plotly.graph_objects as go
from typing import Optional

def plot_pca_2d_cufflinks(df_pca: pd.DataFrame, hue: Optional[str] = 'class', 
                          explained_variance: Optional[list] = None) -> go.Figure:
    """
    Genera un gráfico de dispersión 2D de los componentes principales.
    
    Args:
        df_pca: DataFrame con columnas PC1, PC2 y opcionalmente la columna para colorear
        hue: Nombre de la columna para colorear los puntos
        explained_variance: Lista con la varianza explicada por cada componente
        
    Returns:
        Figura de Plotly lista para mostrar
    """
    # Verificar columnas requeridas
    if 'PC1' not in df_pca.columns or 'PC2' not in df_pca.columns:
        raise ValueError("Se requieren columnas PC1 y PC2")
    
    # Construir título
    if explained_variance is not None and len(explained_variance) >= 2:
        title = f'PCA 2D - PC1 ({explained_variance[0]*100:.2f}%) vs PC2 ({explained_variance[1]*100:.2f}%)'
    else:
        title = 'PCA 2D - Componentes Principales'
    
    # Crear figura
    fig = go.Figure()
    
    if hue and hue in df_pca.columns:
        # Gráfico con categorías
        for category in df_pca[hue].unique():
            mask = df_pca[hue] == category
            fig.add_trace(go.Scatter(
                x=df_pca.loc[mask, 'PC1'],
                y=df_pca.loc[mask, 'PC2'],
                mode='markers',
                name=str(category),
                marker=dict(size=8, opacity=0.7)
            ))
    else:
        # Gráfico sin categorías
        fig.add_trace(go.Scatter(
            x=df_pca['PC1'],
            y=df_pca['PC2'],
            mode='markers',
            marker=dict(size=8, opacity=0.7, color='teal')
        ))
    
    # Configurar diseño
    fig.update_layout(
        title=title,
        xaxis_title='Componente Principal 1',
        yaxis_title='Componente Principal 2',
        template='plotly_white',
        width=800,
        height=600,
        font=dict(size=12),
        title_font=dict(size=16, family='Arial')
    )
    
    return fig
