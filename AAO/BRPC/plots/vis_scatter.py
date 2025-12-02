import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue: str = None):
    """
    Genera un gráfico de dispersión (scatter plot) estático usando Seaborn.
    Rediseñado con una estética moderna y limpia para mayor legibilidad.
    
    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        x_col (str): Nombre de la columna para el eje X.
        y_col (str): Nombre de la columna para el eje Y.
        hue (str, optional): Nombre de la columna para agrupar por colores (categoría).
    """
    try:
        # Configurar tema moderno y limpio
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
        
        # Tamaño de figura generoso
        plt.figure(figsize=(12, 8))
        
        # Configurar estilo de puntos
        # Usamos 'edgecolor' blanco para separar puntos solapados ligeramente
        scatter_kws = {'alpha': 0.7, 's': 40, 'edgecolor': 'white', 'linewidth': 0.5}
        
        if hue:
            # Usar una paleta de alto contraste para categorías
            # 'deep' es buena por defecto, 'bright' si se quiere más intensidad.
            # Si hue es numérico, seaborn usará automáticamente una secuencial (e.g. rocket/mako)
            ax = sns.scatterplot(
                data=df, 
                x=x_col, 
                y=y_col, 
                hue=hue, 
                palette='bright', # Paleta vibrante y distinguible
                **scatter_kws
            )
            plt.title(f'{x_col} vs {y_col} por {hue}', fontsize=16, fontweight='bold', pad=20)
            
            # Mover la leyenda fuera si es necesario, o dejarla "best"
            # Para "simple", seaborn suele colocarla bien, pero podemos ajustarla
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=hue)
        else:
            # Color sólido pero moderno si no hay hue
            ax = sns.scatterplot(
                data=df, 
                x=x_col, 
                y=y_col, 
                color='#2E86C1', # Un azul profesional y agradable
                **scatter_kws
            )
            plt.title(f'{x_col} vs {y_col}', fontsize=16, fontweight='bold', pad=20)
            
        # Etiquetas de ejes más claras
        plt.xlabel(x_col, fontsize=13, fontweight='medium')
        plt.ylabel(y_col, fontsize=13, fontweight='medium')
        
        # Eliminar bordes innecesarios (arriba y derecha) para un look más limpio
        sns.despine(trim=True, offset=10)
        
        # Ajustar márgenes para que los puntos no toquen los bordes
        plt.margins(0.05)
        
        plt.tight_layout()
        
        return ax
    except Exception as e:
        print(f"Error al generar scatter plot: {e}")
        return None
