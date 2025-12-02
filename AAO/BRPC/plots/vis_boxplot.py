import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplots(df: pd.DataFrame, columns: list[str] = None):
    """
    Genera diagramas de caja (boxplots) estáticos usando Seaborn.
    
    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        columns (list[str], optional): Lista de nombres de columnas a graficar.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        if len(columns) > 5:
            print("Nota: Se graficarán solo las primeras 5 columnas numéricas por defecto.")
            columns = columns[:5]
            
    if not columns:
        print("Advertencia: No hay columnas numéricas para graficar.")
        return None

    try:
        # Configurar el tamaño de la figura
        plt.figure(figsize=(10, 6))
        
        # Crear el boxplot
        # Usamos melt para poner los datos en formato largo para seaborn
        df_melted = df[columns].melt(var_name='Variable', value_name='Valor')
        
        ax = sns.boxplot(x='Variable', y='Valor', data=df_melted, hue='Variable', palette='Set2', legend=False)
        
        plt.title('Distribución y Outliers (Boxplots)')
        plt.grid(True, alpha=0.3)
        
        # Retornamos el objeto Axes para que se pueda mostrar en el notebook
        return ax
    except Exception as e:
        print(f"Error al generar boxplots: {e}")
        return None
