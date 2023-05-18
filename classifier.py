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
DIR_no_relation = DIR_BASE+'/scatter_plots/no_relation'
DIR_relation = DIR_BASE+'/scatter_plots/relation'


Imagenes, Etiquetas = construir_base_datos([], [], 0, DIR_no_relation, "grayscale", IMG_TAM)
Imagenes, Etiquetas = construir_base_datos(Imagenes, Etiquetas, 1, DIR_relation, "grayscale", IMG_TAM)
clases=["No Relation", "Relation"]


img_train, etq_train, img_val, etq_val, img_test, etq_test = train_validation_test_sets(Imagenes, Etiquetas, 0.05, 0.05)

print(len(etq_train))
print(len(etq_val))
print(len(etq_test))


#
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=0,  # rotación aleatoria de la imágenes en un rango de (0 a 30)
            zoom_range = 0.05, # acercamiento aleatorio dentro de la imagen
            width_shift_range=0,  # desplazamiento  horizontal aleatorio
            height_shift_range=0,  # desplazamiento vertical aleatorio
            horizontal_flip=True,  # volteo horizontal de las imágenes
            vertical_flip=True,  # volteo vertical de las imágenes
            channel_shift_range=0.5   #modificación aleatorio de los valores RGB de la imagen
            )


redConv1 = tf.keras.Sequential([
  tf.keras.layers.Conv2D(6, (5,5),activation='relu',padding='Same', input_shape = (150,150,1)),  
  tf.keras.layers.MaxPooling2D((2,2)),  
  tf.keras.layers.Conv2D(9,(3,3),activation='relu',padding='Same'),  
  tf.keras.layers.MaxPooling2D((2,2)), 
  tf.keras.layers.Conv2D(18,(3,3),activation='relu',padding='Same'),  
  tf.keras.layers.MaxPooling2D((2,2)),  
  tf.keras.layers.Conv2D(36,(3,3),activation='relu',padding='Same'),  
  tf.keras.layers.MaxPooling2D((2,2)), 
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1000, activation = "relu"),
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(50, activation="relu"),
  tf.keras.layers.Dense(2,activation='relu')
  ])
print(redConv1.summary())

opt = keras.optimizers.Adam(learning_rate=0.0000005)
redConv1.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
evolucion = redConv1.fit(datagen.flow(img_train, etq_train, batch_size=200),validation_data=(img_val, etq_val,), epochs=400, batch_size=200)


plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(evolucion.history['loss'])
plt.plot(evolucion.history['val_loss'])
plt.title('Modelo hecho desde cero')
plt.ylabel('Perdida')
plt.xlabel('Epocas')
plt.legend(['entrenamiento', 'prueba'])

plt.subplot(1, 2, 2)
plt.plot(evolucion.history['accuracy'])
plt.plot(evolucion.history['val_accuracy'])
plt.title('Modelo hecho desde cero')
plt.ylabel('Exactitud')
plt.xlabel('Epocas')
plt.legend(['entrenamiento', 'prueba'])


redConv1.save("model2.h5")

my_model = tf.keras.models.load_model('model2.h5')
tasa_aprendizaje_base = 0.0000005
my_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=tasa_aprendizaje_base), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

perdida_prueba_pre_entrenado, exactitud_prueba_pre_entrenado =  my_model.evaluate(
                                                                                      x=img_prueba,
                                                                                      y = etq_prueba,
                                                                                      batch_size=200,
                                                                                      verbose=1)

print(perdida_prueba_pre_entrenado, exactitud_prueba_pre_entrenado)




