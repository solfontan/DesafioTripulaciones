from flask import Flask
from flask import request
from flask import render_template
import pandas as pd
# import numpy as np
import pickle
import os
from datetime import datetime

os.chdir(os.path.dirname(__file__))
print(os.getcwd())

with open('../model/extra_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../model/dataframe.pkl', 'rb') as f:
    df = pickle.load(f)

# Crea una fila de datos con valores cero
fila_cero = pd.Series([0] * len(df.columns), index=df.columns)

# Agrega la fila de datos al DataFrame
df.loc[0] = fila_cero

app = Flask(__name__)

def obtener_prediccion_dia():
    # Obtener la fecha y hora actual del sistema
    fecha_actual = datetime.now()
    diccionario = df.to_dict()

    diccionario['año'] = fecha_actual.year
    diccionario['mes'] = fecha_actual.month
    diccionario['dia'] = fecha_actual.day+1

    df_model = pd.DataFrame(diccionario, index=[0])

    valor = model.predict(df_model)
    print(valor)

    return valor

def obtener_prediccion_semana():
    # Obtener la fecha y hora actual del sistema
    fecha_actual = datetime.now()
    diccionario = df.to_dict()

    suma = 0
    for i in range(7):
        diccionario['año'] = fecha_actual.year
        diccionario['mes'] = fecha_actual.month
        diccionario['dia'] = fecha_actual.day+1+i

        df_model = pd.DataFrame(diccionario, index=[0])

        valor = model.predict(df_model)

        suma = suma + valor

    return suma

def obtener_prediccion_2_semanas():
    # Obtener la fecha y hora actual del sistema
    fecha_actual = datetime.now()
    diccionario = df.to_dict()

    suma = 0
    subirMes = 0
    subir_dia = 0

    for i in range(14):
        if (fecha_actual.month == 12 and subirMes == 1):
            diccionario['año'] = fecha_actual.year+1

        if (fecha_actual.month == 12 and subir_dia):
            subirMes = 1
            diccionario['mes'] = 1

        if (fecha_actual.day == 30):
            subir_dia = 1
            diccionario['mes'] = fecha_actual.month+1
            diccionario['dia'] = 1+i

        df_model = pd.DataFrame(diccionario, index=[0])

        valor = model.predict(df_model)

        suma = suma + valor

    return suma


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.form

    print()

    donacion = 0

    if (int(datos['prediction']) == 1):
        donacion = obtener_prediccion_dia()
    elif (int(datos['prediction']) == 2):
        donacion = obtener_prediccion_semana()
    elif (int(datos['prediction']) == 3):
        donacion = obtener_prediccion_2_semanas()


    print('Donación estimada: {} rupias'.format(round(donacion[0], 2)))
    print(donacion)
    return 'Donación estimada: {} rupias'.format(donacion)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)