import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

from sklearn.ensemble import ExtraTreesRegressor

import warnings
warnings.filterwarnings('ignore')
import pickle


# Carga los datos del archivo 'datos_procesados.csv' en un DataFrame
df = pd.read_csv("../data/datos_procesados.csv")

# Filtra las filas donde 'donation_type' es igual a 1
df = df[df['donation_type'] == 1]

# Agrupa los datos por fecha de donación ('donation_date') y suma los valores de cada grupo
df = df.groupby('donation_date').sum()

# Convierte el índice del DataFrame (que contiene las fechas de donación) en tipo datetime
df['fecha'] = pd.to_datetime(df.index)

# Encuentra la fecha de inicio y la fecha de fin del rango de fechas
fecha_inicio = df.index.min()
fecha_fin = df.index.max()

# Crea un rango de fechas diarias desde la fecha de inicio hasta la fecha de fin
rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')

# Crear un DataFrame con todas las fechas posibles
serie_temporal = pd.DataFrame()
serie_temporal['fecha'] = rango_fechas

# Unir los datos de donaciones al DataFrame de fechas
serie_temporal = pd.merge(serie_temporal, df, on='fecha', how='left')

# Rellenar los valores faltantes con 0
serie_temporal['amount'].fillna(0, inplace=True)

serie_temporal['fecha'] = pd.to_datetime(serie_temporal['fecha'])
serie_temporal['año'] = serie_temporal['fecha'].dt.year
serie_temporal['mes'] = serie_temporal['fecha'].dt.month
serie_temporal['dia'] = serie_temporal['fecha'].dt.day

serie_temporal = serie_temporal.drop(['name', 'fecha', 'email', 'date', 'is_partner', 'company', 'role_company', 'donation_frecuency', 'suscription_status', 'donation_type', 'method_pay'], axis=1)

# Dividir los datos en características (X) y la variable objetivo (y)
X = serie_temporal[['año', 'mes', 'dia']]
y = serie_temporal['amount']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo ExtraTreesRegressor
model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Guarda el modelo en un archivo pickle
with open('../model/extra_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)

df_vacio = X_train.drop(X_train.index)
# Guarda un dataframe vacío
with open('../model/dataframe.pkl', 'wb') as f:
    pickle.dump(df_vacio, f)