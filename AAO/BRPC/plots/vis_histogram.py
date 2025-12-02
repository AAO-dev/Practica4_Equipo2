import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograms(df: pd.DataFrame, columns: list[str] = None):
    """
    Genera histogramas estáticos para las columnas especificadas usando Seaborn.
    
    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        columns (list[str], optional): Lista de nombres de columnas a graficar. 
                                       Si es None, se grafican todas las numéricas.
    """
    if columns is None:
        # Seleccionamos solo columnas numéricas si no se especifican
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Limitamos a las primeras 5 para no saturar si son muchas
        if len(columns) > 5:
            print("Nota: Se graficarán solo las primeras 5 columnas numéricas por defecto.")
            columns = columns[:5]
            
    if not columns:
        print("Advertencia: No hay columnas numéricas para graficar.")
        return None

    try:
        # Configurar el tamaño de la figura
        plt.figure(figsize=(12, 8))
        
        # Usamos melt para facilitar el ploteo con FacetGrid o simplemente iteramos
        # Para simplificar, si son pocas columnas, podemos usar subplots o un solo plot superpuesto
        # Como el usuario quiere "simple", haremos un FacetGrid o subplots.
        # Pero para mantenerlo muy simple y compatible con la idea de "ver distribuciones",
        # haremos un loop para crear subplots.
        
        num_plots = len(columns)
        rows = (num_plots // 2) + (num_plots % 2)
        
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Frecuencia')
            axes[i].grid(True, alpha=0.3)
            
        # Ocultar ejes vacíos si hay número impar de plots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error al generar histogramas: {e}")
        return None
