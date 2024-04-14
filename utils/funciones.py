import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def cardinalidad(df: pd.DataFrame):
    """Funciòn cardinalidad

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        df (pd.DataFrame) : incluye -> valores únicos, la cardinalidad, el tipo de dato y por último un input que deberá rellenarse de forma manual para calificar nuestro dato.
    """
    cardi = pd.DataFrame(columns=['cardinalidad', 'porcentaje_cardinalidad', 'tipo_de_dato', 'valores_unicos', 'tipo_de_variable'],
                         index=df.columns)

    cardi['cardinalidad'] = [df[col].nunique() for col in df.columns]
    cardi['porcentaje_cardinalidad'] = cardi['cardinalidad'] / len(df) * 100
    cardi['tipo_de_dato'] = df.dtypes
    cardi['valores_unicos'] = [valor if valor <= 15 else 'valores unicos no representativos' for valor in
                               [df[columna].nunique() for columna in df.columns]]

    valores_tipo_variable = [input(f'Para la columna {columna} ingrese el valor "Tipo_de_variable": ') for columna in df.columns]
    cardi['tipo_de_variable'] = valores_tipo_variable

    return cardi

def extended_describe(column : str, df : pd.DataFrame) -> pd.DataFrame:
    """ DataFrame con medidas de tendencias central y de distribución incluído la kurtosis y asimetría.

    Args:
        column (str)
        df (pd.DataFrame)

    Returns:
        DataFrame
    """

    describe_df = df[column].describe()
    # Crear un nuevo DataFrame con las estadísticas extendidas
    extended_describe_df = pd.DataFrame({
        'count': describe_df['count'],
        'mean': describe_df['mean'],
        'median':df[column].median(),
        'mode' : df[column].mode()[0],
        'std' : round(describe_df['std'],2),
        'min': describe_df['min'],
        '25%': describe_df['25%'],
        '50%': describe_df['50%'],
        '75%': describe_df['75%'],
        'max': describe_df['max'],
        'kurtosis': round(kurtosis(df[column]), 2),
        'skewness': round(skew(df[column]),2)
    }, index=[column])
    
    if kurtosis(df[column]) > 0 :
        print(f"La distribución es leptocúrtica con una curtosis de {round(kurtosis(df[column]), 2)}. Los datos se encuentran concentrados alrededor de la media.")
    elif kurtosis(df[column]) < 0 :
        print(f"La distribución es platicúrtica con una curtosis de {round(kurtosis(df[column]), 2)}. Los datos se encuentran dispersos.")
    elif kurtosis(df[column]) == 0 :
        print(f"La distribución es mesocúrtica con una curtosis de {round(kurtosis(df[column]), 2)}. Los datos se comportan de manera normal")
        
    if skew(df[column]) > 0 :
        print(f"La distribución se encuentra sesgada hacia la izquierda {round(skew(df[column]),2)}.")
    else:
        print(f"La distribución se encuentra sesgada hacia la derecha {round(skew(df[column]),2)}.")
        
    return extended_describe_df

class CategoricalAnalysis:
    """
    Análisis categórico de Variables, con el fin de dar un pantallazo más amplio.
    
    """
    def __init__(self, df):
        self.df = df
        
    def plot_top_categories(self, title : str, column_name : str, labely : str, n=5):
        """Incluyen dos gráficos, uno de brras con máximo de 5 valores y los mismos dentro de un gráfico de pie.

        Args:
            title (str)
            column_name (str)
            labely (str): label del eje y.
            n (int, optional): Defaults to 5.
        """
        # Obtener recuento de valores y nombres de las primeras n categorías
        top_categories = self.df[column_name].value_counts().nlargest(n)
        top_category_names = top_categories.index.tolist()
        top_category_counts = top_categories.values.tolist()
        
        # Colores para los gráficos
        branch_col = ['navy', 'crimson', 'forestgreen', 'orange', 'purple', 'darkblue', 'darkred', 'darkgreen', 'yellow', 'darkviolet']
        
        # Crear gráficos
        with plt.style.context('fivethirtyeight'):
            plt.rcParams.update({'font.size': 12})
            fig, ax = plt.subplots(1, 2, figsize=(13, 6))
            plt.subplots_adjust(wspace=0.3)
            
            # Gráfico de barras
            if len(top_category_names) >= 5:  # Si hay cinco o menos categorías
                ax[0].bar(top_category_names, 
                          top_category_counts, 
                          color=branch_col[:n])
                ax[0].tick_params(axis='x', rotation=65)  # Rotar el eje x 45 grados
            else:
                ax[0].bar(top_category_names, 
                          top_category_counts, 
                          color=branch_col[:n], width=0.7)  # Ajustar el ancho de las barras si hay más de cinco categorías
            for x , y, col in zip(top_category_names, 
                             top_category_counts, branch_col[:n]):
                ax[0].text(x, y/2, y, 
                           ha='center',color='white', 
                           bbox=dict(facecolor=col, edgecolor='white', boxstyle='circle'))
            ax[0].set_ylabel(labely)
            # Agregar leyenda para el gráfico de barras
   
            # Gráfico de pastel
            pie = ax[1].pie(x=top_category_counts, 
                            labels=top_category_names,
                            colors=branch_col[:n],  
                            autopct='%1.1f%%',
                            textprops={'color': 'darkgray'}) 
            
            ax[1].legend(loc='upper left', fontsize="xx-small")
            plt.title(title, loc= 'center', fontsize=12)

            plt.show()


    def plot_distribution(self, title : str, column_name: str, alpha : float, color: str, cant_bins : int, rotation : int):
        """histplot con estilo uniforme al igual que la función plot_top_categories

        Args:
            title (str)
            column_name (str)
            alpha (float): entre 0 y 1.
            color (str)
            cant_bins (int)
            rotation (int): de labels sobre el eje X.

        Returns:
            Gráficos
        """
        plt.rcParams['axes.labelpad'] = 5  # Ajustar el espaciado entre las etiquetas y los ejes
        plt.rcParams['xtick.bottom'] = True  # Colocar las etiquetas del eje X en la parte inferior
        plt.rcParams['ytick.left'] = True  # Colocar las etiquetas del eje Y en la izquierda
        plt.rcParams['xtick.top'] = False  # Deshabilitar las etiquetas del eje X en la parte superior
        plt.rcParams['ytick.right'] = False  # Deshabilitar las etiquetas del eje Y en la derecha
        
        # Crear el histograma sin KDE
        with plt.style.context('fivethirtyeight'):
            plt.rcParams.update({'font.size': 10})
            plt.figure(figsize=(10,7))
            plt.grid(True, alpha=0.3)  # Establecer la transparencia del grid
            sns.histplot(data=self.df, x=column_name, color=color, bins=cant_bins, alpha=alpha, edgecolor='white', linewidth=0)
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.xticks(rotation=rotation)
            plt.title(title, loc='center', fontsize=12)
            plt.show()

        answer_df = input('¿Deseas un dataframe con las medidas centrales, de distribución y asimetría del gráfico? : si / no')
        
        if answer_df.lower() == 'si':
            return extended_describe(column_name, self.df)
           

# Preparación de datos 
from sklearn.model_selection import train_test_split, cross_validate
# Modelos
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import  RandomForestClassifier,  GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
# from catboost import CatBoostClassifier
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# # Ignorar warnings
# import warnings
# warnings.filterwarnings('ignore')

# def BaseLine(x_train, y_train, cv: int, metricas_cross_validate: list):
#     """ Crea la validación cruzada para los modelos que deseemos.

#     Args:
#         x_train (_type_): _description_
#         y_train (_type_): _description_
#         cv (int): folds.
#         metricas_cross_validate (list): solo Clasificación.

#     Returns:
#         df: DtaFrame con las métricas obtenidas de la validación cruzada.
#     """
#     try:
#         # Definir modelos disponibles
#         modelos = {
#             "1":  LogisticRegression(),
#             "2":  RandomForestClassifier(),
#             "3":  AdaBoostClassifier(),
#             "4":  GradientBoostingClassifier(),
#             "5":  ExtraTreeClassifier(),
#             "6":  DecisionTreeClassifier(),
#             "7":  CatBoostClassifier(silent=True),
#             "8":  lgb.LGBMClassifier(verbosity= -1),
#             "9":  xgb.XGBClassifier(),
#             "10": KNeighborsClassifier(),
#             "11": SVC(), 
#             "12": HistGradientBoostingClassifier()
#         }

#         # Pedir al usuario que seleccione los modelos
#         answer_modelos = input('¿Cuáles son los modelos que desea utilizar? (seleccione números separados por comas o escriba "todos" para seleccionar todos los modelos): 1: Logistic Regression, 2: Random Forest, 3: ADABoosting, 4: GradientBoosting, 5: ExtraTrees, 6: DecisionTree, 7: CatBoost, 8: LGBM, 9: XGBoost, 10: KNN, 11: SVC, 12: HistGradientBoost')

#         # Seleccionar modelos según la entrada del usuario
#         if answer_modelos.lower() == 'todos':
#             modelos_seleccionados = modelos
#         else:
#             selected_models_indices = [int(x.strip()) for x in answer_modelos.split(',')]
#             modelos_seleccionados = {key: modelos[key] for key in map(str, selected_models_indices)}

#         # Realizar la validación cruzada y calcular las métricas
#         metricas = metricas_cross_validate
#         resultados_dict = {}

#         for nombre_modelo, modelo in modelos_seleccionados.items():
#             if cv:
#                 cv_resultados = cross_validate(modelo, x_train, y_train, cv=cv, scoring=metricas)
#             else:
#                 cv_resultados = cross_validate(modelo, x_train, y_train, cv=5, scoring=metricas)

#             for metrica in metricas:
#                 clave = f"{nombre_modelo}_{metrica}"
#                 resultados_dict[clave] = cv_resultados[f"test_{metrica}"].mean()

#         # Mapear claves numéricas a nombres de modelos más descriptivos
#         nombres_descriptivos = ['Logistic Regression', 'Random Forest', 'ADABoosting', 'Gradient Boosting', 'Extra Trees', 'Decision Tree', 'CatBoost', 'LGBM', 'XGBoost', 'KNN', 'SVC', 'HistGradientBoost']
#         diccionario_nombres = {clave: nombres_descriptivos[int(clave) - 1] for clave in modelos_seleccionados}

#         index = [diccionario_nombres.get(clave.split('_')[0]) for clave in resultados_dict.keys()]
#         metrica = [clave.split('_')[1] for clave in resultados_dict.keys()]

#        # Crear DataFrame con nombres descriptivos y métricas como columnas
#         resultados_df = pd.DataFrame({'Modelo': index, 'Metrica': metrica, 'Score': resultados_dict.values()})

#         # Crear una tabla pivote para tener las métricas como columnas
#         resultados_pivot = resultados_df.pivot(index='Modelo', columns='Metrica', values='Score')

#         return resultados_pivot

#     except Exception as e:
#         print('Surgió un error:', e)

# from sklearn.model_selection import train_test_split, cross_validate
# Modelos

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

def BaseLineRegressor(x_train, y_train, cv: int, metricas_cross_validate: list):
    """ Crea la validación cruzada para los modelos que deseemos.

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        cv (int): folds.
        metricas_cross_validate (list): solo Regresión.

    Returns:
        df: DataFrame con las métricas obtenidas de la validación cruzada.
    """
    try:
        # Definir modelos disponibles
        modelos = {
            "1":  LinearRegression(),
            "2":  RandomForestRegressor(),
            "3":  AdaBoostRegressor(),
            "4":  GradientBoostingRegressor(),
            "5":  ExtraTreeRegressor(),
            "6":  DecisionTreeRegressor(),
            "8":  xgb.XGBRegressor(),
            "9": KNeighborsRegressor(),
            "10": SVR(), 
            "11": HistGradientBoostingRegressor()
        }

        # Pedir al usuario que seleccione los modelos
        answer_modelos = input('¿Cuáles son los modelos que desea utilizar? (seleccione números separados por comas o escriba "todos" para seleccionar todos los modelos): 1: Linear Regression, 2: Random Forest, 3: ADABoosting, 4: GradientBoosting, 5: ExtraTrees, 6: DecisionTree, 8: XGBoost, 9: KNN, 10: SVR, 11: HistGradientBoost')

        # Seleccionar modelos según la entrada del usuario
        if answer_modelos.lower() == 'todos':
            modelos_seleccionados = modelos
        else:
            selected_models_indices = [int(x.strip()) for x in answer_modelos.split(',')]
            modelos_seleccionados = {key: modelos[key] for key in map(str, selected_models_indices)}

        # Realizar la validación cruzada y calcular las métricas
        metricas = metricas_cross_validate
        resultados_dict = {}

        for nombre_modelo, modelo in modelos_seleccionados.items():
            if cv:
                cv_resultados = cross_validate(modelo, x_train, y_train, cv=cv, scoring=metricas)
            else:
                cv_resultados = cross_validate(modelo, x_train, y_train, cv=5, scoring=metricas)

            for metrica in metricas:
                clave = f"{nombre_modelo}_{metrica}"
                resultados_dict[clave] = cv_resultados[f"test_{metrica}"].mean()

        # Mapear claves numéricas a nombres de modelos más descriptivos
        nombres_descriptivos = ['Linear Regression', 'Random Forest', 'ADABoosting', 'Gradient Boosting', 'Extra Trees', 'Decision Tree', 'CatBoost', 'LGBM', 'XGBoost', 'KNN', 'SVR', 'HistGradientBoost']
        diccionario_nombres = {clave: nombres_descriptivos[int(clave) - 1] for clave in modelos_seleccionados}

        index = [diccionario_nombres.get(clave.split('_')[0]) for clave in resultados_dict.keys()]
        metrica = [clave.split('_')[1] for clave in resultados_dict.keys()]

       # Crear DataFrame con nombres descriptivos y métricas como columnas
        resultados_df = pd.DataFrame({'Modelo': index, 'Metrica': metrica, 'Score': resultados_dict.values()})

        # Crear una tabla pivote para tener las métricas como columnas
        resultados_pivot = resultados_df.pivot(index='Modelo', columns='Metrica', values='Score')

        return resultados_pivot

    except Exception as e:
        print('Surgió un error:', e)
