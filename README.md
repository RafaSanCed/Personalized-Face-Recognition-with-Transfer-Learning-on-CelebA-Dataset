## Clasificación imágenes CelebA para clasificación de imágenes propias utilizando Transfer Learning

En este proyecto se usan de base modelos de redes neuonrales convolucionales para clasificar la base de datos CelebA para que mediante la técnica de Transfer Learning poder hacer un clasificador de imágenes propias. 

- El proyecto está desarrllado en Jupyter Notebooks
- El proyecto está diseñado para su reproducción total
- En algunos casos será necesario cambiar las rutas de acceso principales, por ejemplo, en la notebook llamado *Moving_CelebA_photos.py* será necesario cambiar: 
```python
mypath='../../../Data/Processed/img_align_celeba/'
```
por la dirección completa de la ruta de acceso.

**Tabla de contenido**



[TOC]

#Data
Aquí colocamos todas las bases de datos. A partir de bases de datos enteras se van generando separaciones de test y train en las notebooks de la sección *Creating Dataset*.
##Raw
En esta sección se encuentran los atributos de las imágenes CelebA en formato .txt
##Processed
En esta sección se encuentran las imágnees CelebA (img_align_celeba) al igual que imágenes de prueba para el testeo de modelos (img_prop). De igual forma se encuentran los atributos en formato .csv con nombre *atributes.csv* y los mismos atributos trabajados en la notebook *Exploratory Data Analisis.ipynb*.

Esto último se hizo para disminuir el número de atributos de 41 a 32 además de pasar los -1 a 1 para aplicar la métrica de *binary_accuracy* a los modelos.
### img_align_celeba
Aquí se encuentra la base de datos completa de CelebA.
### img_prop
Se encuentran imágenes que se usaron de prueba para los modelos.
##Recognize_me
En esta sección se encuentran las bases de datos necesarias para entrenar, validar y testear los modelos del reconocimiento de mis propias imágenes.
### Others
En esta sección se encuentra una pequeña sección tomada aleatoriamente de la base de datos CelebA. Aunque se tomó de manera aleatoria, se pueden obtener las mismas imágenes de todo el dataset con la notebook *Moving_CelebA_photos.ipynb*.
### Total_Me
Aquí se guardaron unas cuantas imágenes mías para el uso de los modelos de clasificación de mis imágenes. 
### Test
Aquí se guardaron imágenes tomadas de forma aleatorias de la base de datos *Total_Me* para ser los datos de testeo de los modelos. Se pueden obtener los mismos resultados con la notebook *Moving_photos_of_me.ipynb*
#### Me 
Imágenes propias de test
#### Others
Imágenes de otras personas de test

### Train
Aquí se guardaron imágenes tomadas de forma aleatorias de la base de datos *Total_Me* para ser los datos de entrenamiento de los modelos. Se pueden obtener los mismos resultados con la notebook *Moving_photos_of_me.ipynb*
#### Me 
Imágenes propias de entrenamiento
#### Others
Imágenes de otras personas de entrenamiento

### Val
Aquí se guardaron imágenes tomadas de forma aleatorias de la base de datos *Total_Me* para ser los datos de validación de los modelos. Se pueden obtener los mismos resultados con la notebook *Moving_photos_of_me.ipynb*
#### Me 
Imágenes propias de validación

#### Others
Imágenes de otras personas de validación

#Notebooks
En esta sección se encuentran todas las notebooks para este proyecto. Si lo queremos ver en orden cronológico, primero se hizo el *Data cleaning* de los datos de CelebA para que sean los datos de entrenamiento, validación y testeo en Primer, Segundo, Tercer y Cuarto Modelo. 

Después se 
## Data cleaning
## Primer Modelo 
## Segundo Modelo
## Tercer Modelo
## Cuarto Modelo
## Reconocimiento propio
### Creating Dataset
### Models
#### FE_model2
#### FT_model2
