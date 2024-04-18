# Análisis de Serie Temporal con Extra Tree Regressor

Este proyecto utiliza un modelo de regresión basado en Extra Trees para analizar series temporales. El objetivo es predecir valores futuros en la serie temporal proporcionada. Además, se implementa una API para mostrar visualmente las predicciones generadas por el modelo. 📈🌳

## Requisitos

- Python 3.11
- Bibliotecas Python: numpy, pandas, scikit-learn, Flask, matplotlib

## Instalación

1. Clona el repositorio:

    ```bash
    git clone https://github.com/tu_usuario/DesafioTripulaciones.git
    ```

2. Accede al directorio del proyecto:

    ```bash
    cd DesafioTripulaciones
    ```

3. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Asegúrate de tener los datos de la serie temporal en un formato compatible. Puedes cargar los datos en un DataFrame de pandas.

2. Entrena el modelo de regresión utilizando el script `train_model.py`:

    ```bash
    python train_model.py
    ```

3. Una vez entrenado el modelo, inicia la API utilizando el script `app.py`:

    ```bash
    python app.py
    ```

4. Accede a la interfaz de la API en tu navegador web utilizando la URL proporcionada por Flask. Desde allí, podrás cargar los datos de la serie temporal y visualizar las predicciones generadas por el modelo. 🖥️


## Estructura del Proyecto

- `data/`: Carpeta para almacenar los datos de la serie temporal.
- `app/`: Carpeta para almacenar los requisitos de nuestra API.
1. `model/`: Carpeta para almacenar los modelos entrenados.
2. `static/`: Carpeta para archivos estáticos de la API (por ejemplo, archivos .css).
3. `templates/`: Carpeta para plantillas HTML de la API.
4. `app.py`: Script principal para iniciar la API.
5. `train_model.py`: Script para entrenar el modelo de regresión.
6. `requirements.txt`: Lista de dependencias del proyecto.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas mejorar este proyecto, por favor, abre un issue o envía una pull request.
