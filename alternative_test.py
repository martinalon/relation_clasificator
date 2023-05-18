import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
import pathlib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random as rn
from image_processing import obtener_extension_archivos, construir_base_datos, train_validation_test_sets
from tensorflow import keras


Imagenes=[]
Etiquetas=[]
IMG_TAM = 150
DIR_BASE =os.path.abspath(os.path.dirname(__file__))
DIR_no_relation = DIR_BASE+'/scatter_plots/alternative_test/no_relation'
DIR_relation = DIR_BASE+'/scatter_plots/alternative_test/relation'


Imagenes, Etiquetas = construir_base_datos([], [], 0, DIR_no_relation, "grayscale", IMG_TAM)
Imagenes, Etiquetas = construir_base_datos(Imagenes, Etiquetas, 1, DIR_relation, "grayscale", IMG_TAM)

Imagenes = np.array(Imagenes)
Etiquetas = np.array(Etiquetas)


my_model = tf.keras.models.load_model('modelo_hecho_desde_cero.h5')
tasa_aprendizaje_base = 0.0000005
my_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=tasa_aprendizaje_base), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

perdida_prueba_pre_entrenado, exactitud_prueba_pre_entrenado =  my_model.evaluate(
                                                                                      x=Imagenes,
                                                                                      y = Etiquetas,
                                                                                      batch_size=200,
                                                                                      verbose=1)