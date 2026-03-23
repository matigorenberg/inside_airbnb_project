# Identificación de Listings de Alta Demanda en Airbnb CABA mediante Modelos de Clasificación – Proyecto Final Integrador

Este repositorio contiene el código y los recursos para el Proyecto Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado. El objetivo es clasificar el potencial de éxito de los alojamientos de Airbnb en la Ciudad Autónoma de Buenos Aires (CABA) utilizando técnicas de Machine Learning, siguiendo la metodología [CRISP-DM](https://es.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining).

## Resultados Clave
- **Modelo Seleccionado:** LightGBM con optimización de hiperparámetros.
- **Métrica Principal (Recall):** 86% en la identificación de casos de éxito.
- **Hallazgo Principal:** La gestión operativa (tasa de respuesta y aceptación) tiene un impacto predictivo mayor que la ubicación geográfica en el mercado de CABA.

## Estructura del Proyecto
```bash
├── data/                           # Datos crudos (listings.zip incluido para reproducibilidad)
├── inside_airbnb_ml_project.ipynb  # Notebook principal con el ciclo de vida del dato
├── README.md                       # Documentación del repositorio
└── .gitignore                      # Archivos excluidos (entornos virtuales y temporales)
```

## Contenido
- `data/`: Carpeta destinada a alojar el dataset crudo `listings.zip` (ver aclaraciones e instrucciones más abajo).
- `inside_airbnb_ml_project.ipynb`: Notebook Jupyter con todo el análisis: carga de datos, limpieza general, ingeniería de variables, modelado (comparativa de modelos lineales y ensamble, incluyendo boosting), evaluación de métricas y explicación de características mediante valores SHAP.
- `README.md`: Este archivo describe el proyecto, los requisitos y cómo ejecutar el notebook.

## Requisitos
- Python 3.11 o superior.
- Librerías de Python: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `shap`, `folium`.

## Ejecución del Notebook
Este proyecto puede ejecutarse de dos maneras:

### Opción A: Google Colab (Recomendado)
1. Suba el archivo `inside_airbnb_ml_project.ipynb` a su Google Drive.
2. Cargue el archivo `listings.zip` en la sección de "Archivos" (ícono de carpeta a la izquierda) dentro de una carpeta llamada `data/`, o ajuste la ruta de carga en el notebook.
3. Ejecuta las celdas en orden. Al abrir el notebook, las celdas iniciales instalarán automáticamente las librerías faltantes (como `shap` o `lightgbm`) si no estuvieran presentes en el ambiente.
  
### Opción B: Local (Jupyter Notebook o o JupyterLab)
1. Clona o descarga este repositorio.
2. Crea un entorno virtual (opcional pero recomendado):
```bash
python -m venv env
source env/bin/activate
```
3. Instala las dependencias necesarias:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap folium
```
4. Ejecuta Jupyter Notebook o JupyterLab y abra el archivo `.ipynb`.
5. Asegúrese de que la ruta en la que se encuentra el archivo `listings.zip` sea consistente con la carga de datos tal cual estipulada para la sección correspondiente del código.
6. Ejecuta las celdas en orden.
  
El notebook realiza los siguientes pasos:
- Importación saneada de librerías estandarizadas y configuración del entorno visual.
- Carga del dataset.
- Limpieza general inicial e ingeniería de variables.
- Análisis exploratorio de datos (EDA).
- Preprocesamiento de datos.
- Entrenamiento y validación cruzada estratificada (Regresión Logística, Random Forest, LightGBM, XGBoost).
- Optimización de hiperparámetros mediante GridSearchCV, priorizando la maximización del Recall.
- Análisis de interpretabilidad profundo sobre el modelo ganador (LightGBM) con valores SHAP.

## Datos y Reproducibilidad
El dataset utilizado para este proyecto corresponde a la extracción de Inside Airbnb para la Ciudad Autónoma de Buenos Aires con fecha de corte al 30 de enero de 2025.
El notebook el archivo `.zip` sin necesidad de descomprimirlo, puesto que `pandas` permite leer tal extensión de manera directa.

***Nota Crítica sobre los Datos:*** *Debido a que Inside Airbnb actualiza periódicamente su estructura de archivos (versiones más recientes pueden presentar cambios significativos en nombres de columnas y tipos de datos), se incluye en este repositorio el archivo `listings.zip` original utilizado para el entrenamiento y validación del modelo. Esto garantiza la total reproducibilidad de los resultados presentados.*

**¿Cómo proceder con datos nuevos? Si desea utilizar una versión más reciente de los datos:**

- Accede a http://insideairbnb.com/get-the-data/.
- Busca la ciudad de Buenos Aires y descarga el archivo `listings.csv.gz` (Detailed Listings data).
- Descomprime, de ser necesario, el archivo descargado para obtener la carpeta que contiene el archivo `listings.csv`.
- Mueve el archivo a la carpeta `data/` dentro de tu repositorio local.
Importante: Realice un mapeo de columnas y ajuste las funciones de limpieza (como initial_data_cleaning) y feature engineering en el notebook para atender los cambios en la estructura de la fuente.
Recuerde ajustar la ruta del archivo.

## Citaciones
- Fuente de Datos: Extraídos de [Inside Airbnb](http://insideairbnb.com/).
- Algoritmo de Predicción Seleccionado: [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/).
- Framework de Interpretabilidad: [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations).
