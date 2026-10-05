[![Binder][binder-badge]](https://mybinder.org/v2/gh/lmingari/olot-course/master)
[![Google Colab: Launch][colab-launch]](https://colab.research.google.com/github/lmingari/olot-course/blob/master/)

# Modelización numérica, IA y *machine learning* aplicado a la volcanología

Material para el módulo de *modelización numérica* del [Curso internacional de volcanología][web].

| __Curso internacional de volcanología__ |
| :-------------------------------------: |
| Olot (La Garrotxa) - La Palma |
| 12 Oct - 24 Oct 2026 |
| _2.ª edición_ |
| <img src="figs/qr.svg" width=240 alt="Código QR del curso"> |

## Teóricas

| Sesión | Slides |
| :------ | ------ |
| Introducción | [![PDF][pdf-icon]][teorica-intro] |
| 1.1 Modelos numéricos | [![PDF][pdf-icon]][teorica11] |
| 1.2 *Machine learning* en volcanología | [![PDF][pdf-icon]][teorica12] |
| 2. El modelo FALL3D | [![PDF][pdf-icon]][teorica2] |
| 3. Introducción a las redes neuronales | [![PDF][pdf-icon]][teorica3] |

## Prácticas

| Sesión | Colab | Kaggle | Binder |
| :------- | :---: | :----: | :----: |
| 1. Exploring a FALL3D output | [![Colab][colab-badge]][s1-colab] | [![Kaggle][kaggle-badge]][s1-kaggle] | [![Binder][binder-badge]][s1-binder] |
| 2.1 Training a Neural Network with PyTorch | [![Colab][colab-badge]][s21-colab] | [![Kaggle][kaggle-badge]][s21-kaggle] | [![Binder][binder-badge]][s21-binder] |
| 2.2 A multilayer perceptron (MLP) for classification | [![Colab][colab-badge]][s22-colab] | [![Kaggle][kaggle-badge]][s22-kaggle] | [![Binder][binder-badge]][s22-binder] |
| 3.1 Convolutional neural networks (CNN) | [![Colab][colab-badge]][s31-colab] | [![Kaggle][kaggle-badge]][s31-kaggle] | [![Binder][binder-badge]][s31-binder] |
| 3.2 Super-Resolution with U-Net | [![Colab][colab-badge]][s32-colab] | [![Kaggle][kaggle-badge]][s32-kaggle] | [![Binder][binder-badge]][s32-binder] |
| 4 Generative AI | [![Colab][colab-badge]][s4-colab] | [![Kaggle][kaggle-badge]][s4-kaggle] | [![Binder][binder-badge]][s4-binder] |

## Contenido del repositorio

### 1. Exploring a FALL3D output

Introducción al procesamiento de resultados del modelo FALL3D. Se usa `xarray` para abrir, explorar y representar campos de una simulación y de un conjunto (*ensemble*) de pronósticos. La práctica termina con el cálculo de probabilidades de excedencia.

### 2.1 Training a Neural Network with PyTorch

Introducción al flujo de trabajo de aprendizaje supervisado con PyTorch: preparación y división de datos, modelo, función de pérdida, descenso de gradiente, entrenamiento y validación. Se entrena un perceptrón multicapa para aproximar una función, se realiza inferencia y se introduce la codificación de Fourier.

### 2.2 A multilayer perceptron (MLP) for classification

Clasificación del impacto de la caída de tefra de la erupción de Tajogaite de 2021 en La Palma. Se definen clases de impacto a partir del espesor del depósito, se prepara el conjunto de datos y se entrena un MLP para visualizar sus regiones de decisión.

### 3.1 Convolutional neural networks (CNN)

Introducción a las redes neuronales convolucionales y a los filtros de convolución 2-D. Incluye la implementación de una CNN sencilla, aplicaciones en volcanología, autoencoders convolucionales y la arquitectura U-Net.

### 3.2 Super-Resolution with U-Net

Reconstrucción de campos simulados de plumas volcánicas de alta resolución a partir de entradas de baja resolución. Se construye un conjunto de datos de simulaciones FALL3D, se define y entrena una U-Net, y se evalúa su capacidad de reconstrucción en datos no vistos.

### 4. Generative AI
Introducción a los modelos generativos como herramienta para aprender la distribución de un ensamble de simulaciones y generar nuevos campos plausibles. Se presenta la interpretación de la generación como un flujo dinámico y se introduce Flow Matching. La práctica utiliza un modelo preentrenado para transformar muestras de ruido gaussiano en nuevos campos de plumas volcánicas y se comparan los resultados con simulaciones de FALL3D.

## Ejecución local

```bash
git clone https://github.com/lmingari/olot-course.git
cd olot-course
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib torch torchsummary xarray netCDF4 jupyter
jupyter notebook
```

Abre la `notebook` que quieras ejecutar desde la interfaz de Jupyter. Los datos necesarios están incluidos en `data/`.

[web]: https://espaicrater.com/es/cursovolcanologia/
[teorica-intro]: https://saco.csic.es/s/82DMHtD9Kt2LAXd
[teorica11]: https://saco.csic.es/s/Kodfn2bnXmWjAY9
[teorica12]: https://saco.csic.es/s/xY2Cynfw6sztRSe
[teorica2]: https://saco.csic.es/s/LqKEF3FDsKszWBf
[teorica3]: https://saco.csic.es/s/qoar9dr3pNqnqS3
[pdf-icon]: figs/PDF_icon.svg
[colab-launch]: https://img.shields.io/badge/Google%20Colab-Launch-blue.svg
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[kaggle-badge]: figs/kaggle_badge.svg
[binder-badge]: figs/binder_badge.svg
[s1-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/1-FALL3D.ipynb
[s21-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/2.1-MLP-introduction.ipynb
[s22-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/2.2-MLP-classification.ipynb
[s31-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/3.1-CNN-introduction.ipynb
[s32-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/3.2-CNN-unet.ipynb
[s4-colab]: https://colab.research.google.com/github/lmingari/olot-course/blob/master/4-Generative-AI.ipynb
[s1-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/1-FALL3D.ipynb
[s21-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/2.1-MLP-introduction.ipynb
[s22-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/2.2-MLP-classification.ipynb
[s31-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/3.1-CNN-introduction.ipynb
[s32-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/3.2-CNN-unet.ipynb
[s4-kaggle]: https://kaggle.com/kernels/welcome?src=https://github.com/lmingari/olot-course/blob/master/4-Generative-AI.ipynb
[s1-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F1-FALL3D.ipynb
[s21-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F2.1-MLP-introduction.ipynb
[s22-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F2.2-MLP-classification.ipynb
[s31-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F3.1-CNN-introduction.ipynb
[s32-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F3.2-CNN-unet.ipynb
[s4-binder]: https://mybinder.org/v2/gh/lmingari/olot-course/master?urlpath=%2Fdoc%2Ftree%2F4-Generative-AI.ipynb
