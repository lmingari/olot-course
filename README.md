[![Binder][binder-badge]](https://mybinder.org/v2/gh/lmingari/olot-course/master)
[![Google Colab: Launch][colab-launch]](https://colab.research.google.com/github/lmingari/olot-course/blob/master/)

# Modelización numérica, IA y machine learning aplicado a la volcanología

Material para el módulo de _modelización numérica_ del [curso][web]:

| __Curso internacional de volcanología__    | This site |
| :----------------------------------------: | :-------: |
| Olot (La Garrotxa) - La Palma              | ![](figs/qr.svg) |
| 20 de octubre al 1 de noviembre de 2025    | |
| _1a edición_                               | |

## Teóricas

| Teórica | Enlace |
| :------ | ------ |
| Introducción                        | [![PDF][pdf-icon]][teorica-intro] |
| 1. Modelos                          | [![PDF][pdf-icon]][teorica1] |
| 2. El modelo FALL3D                 | [![PDF][pdf-icon]][teorica2] |
| 3. Machine Learning en volcanología | [![PDF][pdf-icon]][teorica3] |
| 4. Redes neuronales                 | [![PDF][pdf-icon]][teorica4] |

## Prácticas

| Práctica | Colab | Kaggle | Binder |
| :------- | ----- | ------ | ------ |
| 1: Loading an ensemble of FALL3D forecasts             | [![Colab][colab-badge]][s1-colab]  | [![Kaggle][kaggle-badge]][s1-kaggle]  | [![Binder][binder-badge]][s1-binder]  |
| 2.1: Training a Neural Network with PyTorch            | [![Colab][colab-badge]][s21-colab] | [![Kaggle][kaggle-badge]][s21-kaggle] | [![Binder][binder-badge]][s21-binder] |
| 2.2: A multilayer perceptron (MLP) for classification  | [![Colab][colab-badge]][s22-colab] | [![Kaggle][kaggle-badge]][s22-kaggle] | [![Binder][binder-badge]][s22-binder] |
| 3.1: Convolutional neural networks (CNN)               | [![Colab][colab-badge]][s31-colab] | [![Kaggle][kaggle-badge]][s31-kaggle] | [![Binder][binder-badge]][s31-binder] |
| 3.2: Autoencoder (I): Training a CNN-based Autoencoder | [![Colab][colab-badge]][s32-colab] | [![Kaggle][kaggle-badge]][s32-kaggle] | [![Binder][binder-badge]][s32-binder] |
| 3.3: Autoencoder (II): Loading a pre-trained model     | [![Colab][colab-badge]][s33-colab] | [![Kaggle][kaggle-badge]][s33-kaggle] | [![Binder][binder-badge]][s33-binder] |

## Contenido del repositorio

En este repositorio se encuentra el material para las 
prácticas acorde a la siguiente estructura:

#### 1. Loading an ensemble of FALL3D forecasts

En esta primera sección aprenderemos como leer los datos 
de una simulación del modelo de dispersión FALL3D y cómo 
generar mapas a partir de estos datos.

#### 2.1: Training a Neural Network with PyTorch

Aquí se cubren las características generales de la 
biblioteca PyTorch para _deep learning_, así como los 
pasos básicos para construir una red neuronal sencilla.

#### 2.2: A multilayer perceptron (MLP) for classification

Un perceptrón multicapa (MLP) es un tipo de red neuronal 
artificial que puede estar formada por varias capas de neuronas. 
En esta sección se entrena un sencillo MLP con unas pocas capas 
ocultas para predecir el impacto de la caída de tefra en 
La Palma tras la erupción de 2021.

#### 3.1: Convolutional neural networks (CNN)

Una red neuronal convolucional (CNN) es un tipo de red neuronal artificial 
diseñada para procesar datos estructurados, como imágenes o series temporales. 
Utiliza capas convolucionales para extraer características relevantes 
(como bordes o texturas en imágenes) mediante la aplicación de filtros, 
seguidas de capas que reducen la dimensionalidad al tiempo que preservan información relevante.

#### 3.2: Autoencoder (I): Training a CNN-based Autoencoder

Un autoencoder basado en CNN es una red neuronal que combina capas convolucionales (CNN) 
para comprimir (codificar) datos en una representación de baja dimensión y 
luego reconstruirlos (decodificar) para replicar la entrada original. 

En esta sección utilizaremos un ensemble de simulaciones de dispersión de
plumas volcánicas generado por el modelo FALL3D para entrenar un autoencoder
basados en CNN.

#### 3.3: Autoencoder (II): Loading a pre-trained model

En esta sección aprenderemos a cargar un modelo pre-entrenado para inferencia. 
Luego veremos cómo es posible generar nuevos datos similares a los originales 
que permiten extender el ensamble original sirviendo como base para el desarrollo 
de modelos de IA generativa.

## Running on your machine

To run the notebooks on your own machine:

1. Clone the repository and navigate to it:

```bash
$ git clone https://github.com/lmingari/olot-course.git
$ cd olot-course
```

2. Ensure Python and pip are installed: Verify with `python --version` or `python3 --version` and `pip --version` or `pip3 --version`.

3. Use a virtual environment to avoid conflicts:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

[web]: https://espaicrater.com/es/cursovolcanologia/
[teorica-intro]: https://saco.csic.es/s/82DMHtD9Kt2LAXd/download/Introducci%C3%B3n.pdf
[teorica1]: https://saco.csic.es/s/W4Jf7Zc35bKDDoL/download/Teoria_1_Modelos.pdf
[teorica2]: https://saco.csic.es/s/sqsf7JJxbL9BocE/download/Teoria_2_FALL3D%20model.pdf
[teorica3]: https://saco.csic.es/s/JS8k3GAkHR95qf5/download/Teoria_3_ML%20en%20volcanologia.pdf
[teorica4]: https://saco.csic.es/s/ZzqMCwCpoGZnQak/download/Teoria_4_Redes%20neuronales.pdf
[pdf-icon]: figs/PDF_icon.svg
[colab-launch]: https://img.shields.io/badge/Google%20Colab-Launch-blue.svg
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[kaggle-badge]: figs/kaggle_badge.svg
[binder-badge]: figs/binder_badge.svg
[s1-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/1-FALL3D-loading-data.ipynb
[s21-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/2.1-MLP-introduction.ipynb
[s22-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/2.2-multiclass-classifier.ipynb
[s31-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/3.1-CNN-introduction.ipynb
[s32-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/3.2-autoencoder-training.ipynb
[s33-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/3.3-autoencoder-loading.ipynb
[s1-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/1-FALL3D-loading-data.ipynb
[s21-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/2.1-MLP-introduction.ipynb
[s22-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/2.2-multiclass-classifier.ipynb
[s31-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/3.1-CNN-introduction.ipynb
[s32-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/3.2-autoencoder-training.ipynb
[s33-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/3.3-autoencoder-loading.ipynb
[s1-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F1-FALL3D-loading-data.ipynb
[s21-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F2.1-MLP-introduction.ipynb
[s22-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F2.2-multiclass-classifier.ipynb
[s31-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F3.1-CNN-introduction.ipynb
[s32-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F3.2-autoencoder-training.ipynb
[s33-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F3.3-autoencoder-loading.ipynb
