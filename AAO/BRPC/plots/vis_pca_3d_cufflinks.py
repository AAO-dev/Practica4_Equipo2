import pandas as pd
import plotly.graph_objects as go
from typing import Optional

def plot_pca_3d_cufflinks(df_pca: pd.DataFrame, hue: Optional[str] = 'class',
                          explained_variance: Optional[list] = None) -> go.Figure:
    """
    Genera un gráfico de dispersión 3D de los componentes principales.
    
    Args:
        df_pca: DataFrame con columnas PC1, PC2, PC3 y opcionalmente la columna para colorear
        hue: Nombre de la columna para colorear los puntos
        explained_variance: Lista con la varianza explicada por cada componente
        
    Returns:
        Figura de Plotly 3D lista para mostrar
    """
    # Verificar columnas requeridas
    if 'PC1' not in df_pca.columns or 'PC2' not in df_pca.columns or 'PC3' not in df_pca.columns:
        raise ValueError("Se requieren columnas PC1, PC2 y PC3")
    
    # Construir título
    if explained_variance is not None and len(explained_variance) >= 3:
        title = (f'PCA 3D - PC1 ({explained_variance[0]*100:.2f}%), '
                f'PC2 ({explained_variance[1]*100:.2f}%), '
                f'PC3 ({explained_variance[2]*100:.2f}%)')
    else:
        title = 'PCA 3D - Componentes Principales'
    
    # Crear figura 3D
    fig = go.Figure()
    
    if hue and hue in df_pca.columns:
        # Gráfico con categorías
        for category in df_pca[hue].unique():
            mask = df_pca[hue] == category
            fig.add_trace(go.Scatter3d(
                x=df_pca.loc[mask, 'PC1'],
                y=df_pca.loc[mask, 'PC2'],
                z=df_pca.loc[mask, 'PC3'],
                mode='markers',
                name=str(category),
                marker=dict(size=5, opacity=0.7)
            ))
    else:
        # Gráfico sin categorías
        fig.add_trace(go.Scatter3d(
            x=df_pca['PC1'],
            y=df_pca['PC2'],
            z=df_pca['PC3'],
            mode='markers',
            marker=dict(size=5, opacity=0.7, color='teal')
        ))
    
    # Configurar diseño
    fig.update_layout(
        title=title,
        template='plotly_white',
        width=900,
        height=700,
        font=dict(size=12),
        title_font=dict(size=16, family='Arial'),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        )
    )
    
    return fig
