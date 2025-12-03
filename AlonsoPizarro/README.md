# sobre la estructura de este repo
## Práctica 4 

**Carpetas**:
- *Datos*: contiene 5 archivos (1year.arff, 2year.arff, 3year.arff, 4year.arff y 5year.arff) que corresponden al dataset *Polish Companies Bankruptcy* proveniente del repositorio UC Irvine Machine Learning Repository.
    - Puede consultar el link de descarga en https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data.

    - Sobre el dataSe y los alcances de la investigación:

        - Son 5 archivos que corresponden a 5 conjuntos de datos distintos. Estos archivos se recolectaron en 5 momentos diferentes  correspondiente a los años fiscales anteriores a la posible quiebra de las empresas. Se reacabaron reportes financieros de empresas de la Union Europea, específicamente de Polonia.

        - Este conjunto de datos es como una bola de cristal financiera que usamos para ver si una empresa bancaria va a quebrar. El truco es que no solo intentamos predecir si quiebra en el próximo año, sino en una ventana de hasta cinco años en el futuro. Para lograr esto, tenemos cinco archivos diferentes, donde cada uno representa un "horizonte de tiempo" distinto: el archivo más importante (el más "actualizado") usa la información de la empresa de solo un año antes de la quiebra real, mientras que el más antiguo usa los datos de cinco años antes. El objetivo es encontrar patrones que te ayuden a predecir con mayor o menor probabilidad si dado los estados financieros de la compañía a distintos horizontes de tiempo la empresa es más propicia a qiebrar.

        - Problema de Clasificación Binaria. 

        - El dataset tiene 66 columnas: 64 columnas con características (ratios financieros), 1 columna es el target (columna 'class') y finalmente, la columna de año (columna year).

        - El data set contiene 43405 registros.

        - El desafío principal es el desbalance de clases ya que solo el 4.8% (class = 1) de las compañías del data set quebraron mientras el el 95.2% sobrevivieron (class = 0).

        - Por otro lado, el exceso de variables y la redundacia en la info supone graves problemas de multicolinealidad.


- *libreria*: esta carpeta contiene 6 archivos .py.
    - __init__.py : archivo vacío que sirve para convertir un directorio (en este caso la carpeta libreria/) en un paquete python.

    - PCA: módulo que contiene las funciones para ejecutar el Análisis de componentes principales y graficar en 2D como en 3D.
    
    - SelectBest: Módulo que contiene la función que ejecuta el análisis de seleccionar los mejores predictores.
    
    - Varclus: es el módulo que contiene la función que realiza un análisis de Clustering de Variables (VarClusHi) y genera un dendrograma basado en la correlación para identificar grupos de variables. Además, contiene la función que selecciona el mejor representante.

    - WoE_iV: este módulo contiene las funciones que determinan el *Peso de la evidencia* y el *valor de información*


- Metodología y Análisis:
    - Se analizó la información y se encontró que - Con excepción de dos atributos (el atributo 21 y 37) la completitud supera el 93%
    - El Atributo 37 tiene un 43.7% de valores nulos.
    - No hay valores infinitos.
    - Todas las columnas tienen valores fuera del IQR.
    - Nota que la varible class está desbalanceada. esto es, 95.2% de las empresas sovbrevivieron mientras que el 4.8% se declararon en quiebra.

    Siguiente paso:
    
    - Imputar mediana a valores faltantes. Se revisaron que ya no hubieran valores nulos.
    - Hacer una winsorización para tratar valores extremos. Se craeron boxplots para revisar el éxito de la winsorización.
    - Se aplicaron Heatmap y Análisis de Variation Inflation Factor (VIF) y se encontró un fuerte problema de multicolinealidad.

    **Métodos de reducción de Dimensiones**
    - Se Realizó un análisis de Clustering de Variables (VarClusHi) y generó un dendrograma basado en la correlación para identificar grupos de variables. El ejercicios arrojó que 13 fueron los grupos resultantes con un representante por grupo.
 
    - Se realizó PCA. Se encontró que cerca del 51% de la varianza es explicada por dos PCs y que llegamos a explicar el 61% con tres CPs. Se muestran gráficas de los 2 y 3 Componentes principales, respectivamente.
 
    
    **Métodos de selección de variables y transformación entrópica**

    - Se explicó el método SelectKBest para seleccionar los mejores predictores. 7 predictores fueron elegidos por su poder predictivo sobre la variable class.
    - El peso de la evidencia (WoE) y valor de información (IV) fueron calculados para cada una de las métricas con tal de elegir a los mejores predictores.Cerca de 29 predictores     resultaron con un valor mayor a 0.3 (considerado alto poder predictivo).
    

    **Creación de una Biblioteca con funciones Personalizadas**

    - En la carpeta *libreria* se creraon los módulos para cada una de las funciones que se describen arriba. Tanto para los métodos de reducción de dimensiones como los de selección de características.

    **Documentación**
    - Todo el código, definiciones y gráficas se encuentran en Notebook llamado *Practica4.ipynb*.




    
