## Clasificación imágenes CelebA para clasificación de imágenes propias utilizando Transfer Learning

En este proyecto se usan de base modelos de redes neuonrales convolucionales para clasificar la base de datos CelebA para que mediante la técnica de Transfer Learning poder hacer un clasificador de imágenes propias. 

- El proyecto está desarrllado en Jupyter Notebooks
- El proyecto está diseñado para su reproducción total
- En algunos casos será necesario cambiar las rutas de acceso principales, por ejemplo, en la notebook llamado *Moving_CelebA_photos.py* será necesario cambiar: 
```python
mypath = '../../../Data/Processed/img_align_celeba/'
```
por la dirección completa de la ruta de acceso.

**Tabla de contenido**

- [Data](#Data)
  * [Raw](#Raw)
    + [img_align_celeba](#img_align_celeba)
    + [img_prop](#img_prop)
  * [Recognize_me](#Recognize_me)
    + [Others](#Others)
    + [Total_Me](#Total_Me)
    + [Test](#Test)
    + [Train](#Train)
    + [Val](#Val)
    
- [Notebooks](#Notebooks)
  * [Data cleaning](#Data cleaning)
  * [Primer_Modelo](#Primer_Modelo)
  * [Segundo_Modelo](#Segundo_Modelo)
  * [Tercer_Modelo](#Tercer_Modelo)
  * [Cuarto_Modelo](#Cuarto_Modelo)
  * [Quinto_Modelo](#Quinto_Modelo)
  * [Reconocimiento_Propio](#Reconocimiento_Propio)
    + [Creating_Dataset](#Creating_Dataset)
    + [Models](#Models)
      * [FE_model2](#FE_model2)
      * [FT_model2](#FT_model2)
    


# Data
Aquí colocamos todas las bases de datos. A partir de bases de datos enteras se van generando separaciones de test y train en las notebooks de la sección *Creating Dataset*.
##Raw
En esta sección se encuentran los atributos de las imágenes CelebA en formato .txt
##Processed
En esta sección se encuentran las imágnees CelebA (img_align_celeba) al igual que imágenes de prueba para el testeo de modelos (img_prop). De igual forma se encuentran los atributos en formato .csv con nombre *atributes.csv* y los mismos atributos trabajados en la notebook *Exploratory Data Analisis.ipynb*.

Esto último se hizo para disminuir el número de atributos de 41 a 32 además de pasar los -1 a 1 para aplicar la métrica de *binary_accuracy* a los modelos.
### img_align_celeba
Aquí se encuentra la base de datos completa de CelebA.
### img_prop
Se encuentran imágenes que se usaron como prueba para los modelos.
##Recognize_me
En esta sección se encuentran las bases de datos necesarias para entrenar, validar y testear los modelos del reconocimiento de mis propias imágenes.
### Others
En esta sección se encuentran fotos tomada aleatoriamente de la base de datos CelebA. Aunque se tomó de manera aleatoria, se pueden obtener las mismas imágenes con la notebook *Moving_CelebA_photos.ipynb*.
### Total_Me
Aquí se guardaron unas cuantas imágenes mías para su uso en los modelos de clasificación de mis imágenes. 
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

# Notebooks
En esta sección se encuentran todas las notebooks para este proyecto. Primero se hizo el *Data cleaning* de los datos de CelebA para que sean los datos de entrenamiento, validación y testeo en Primer, Segundo, Tercer y Cuarto Modelo. 

Después en *Reconcomiento Propio* se encuentra todo lo necesario para hacer la clasificació de imágenes de mi persona con la de otras personas
## Data cleaning
Aquí se crean las notebooks para modificar los datos de los atributos de las imágenes de CelebA. Primero se pasa del archivo .txt de los atributos de CelebA a uno .csv para poder alimentar a los modelos en la notebook llamada *Conversion_csv.ipynb*. 

En la notebook llamada *Exploratory Data Analisis.ipynb* se cambian los -1 por 0 para poder utilizar la métrica *binary_crossentropy* además de que se disminuyen los atributos de 41 a 32. 
## Primer Modelo
Se crea, guarda y testea el primer modelo usando redes convolucionales para la clasificación de las imágenes de CelebA.
## Segundo Modelo
Se mejora el primer modelo usando callbacks para disminuir el *learning rate* a través de las épocas y se crea un *early stop* para evitar el sobreajuste.
## Tercer Modelo
Se separan los datos de testeo del modelo y se itroduce un treshold a la métrica *BinaryCrossentropy* ya que se percató que los datos estaban sesgados, siendo el 78% ceros.
## Cuarto Modelo
Se utiliza la estructura de las resnet para poder aumentar los parámetros entrenables sin que el modelo sobreajuste:
```python
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu",**kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
                            keras.layers.Conv2D(filters, 3, strides=strides,
                                            padding="same",use_bias=False),
                            self.activation,
                            keras.layers.Conv2D(filters, 3, strides=1,
                                            padding="same",use_bias=False)]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                          keras.layers.Conv2D(filters, 1, strides=strides,
                                              padding="same",use_bias=False)]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation' : self.activation,
            'main_layers' : self.main_layers,
            'skip_layers' : self.skip_layers,})
        return config

    def call(self, inputs):
        skip_x = inputs
        x = inputs
        for layer in self.main_layers:
            x = layer(x)
        for layer in self.skip_layers:
            skip_x = layer(skip_x)
    
        return self.activation(keras.layers.add([x, skip_x]))
```
## Quinto Modelo
Se utiliza la técnica de Fine Tuning para el modelo de CelebA, utilizando la red neuronal [VGG16](https://keras.io/api/applications/vgg/ "VGG16") previamente entrenada. El model VGG16 es una red de 16 capas propuesta por Karen Simonyan y Andrew Zisserman en su artículo *[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://ui.adsabs.harvard.edu/abs/2014arXiv1409.1556S/abstract "Very Deep Convolutional Networks for Large-Scale Image Recognition")*.
## Reconocimiento_propio
En esta sección se crean los modelos y se separan los datos para la clasificación de imágenes de mi persona. Cabe resaltar que está hecho para que siempre las imágenes se separen de la misma manera cada vez que se corren las notebook, esto para su reproducibilidad.
### Creating_Dataset
Aquí se crean todas las bases de datos (train, test, validation) para la clasificación de imágenes mías. 
### Models
Se crean dos modelos, uno que utiliza la técina de Feature Extraction y otro que utiliza la técnica de Fine Tuning:
[![](https://miro.medium.com/max/896/1*01aITXnKNAleWAd-lcSxLQ.png)](http://https://miro.medium.com/max/896/1*01aITXnKNAleWAd-lcSxLQ.png)
#### FE_model2
Se crea y se evalúa el modelo creado utilizando Feature Extraction
#### FT_model2
Se crea y se evalúa el modelo creado utilizando Fine Tuning

