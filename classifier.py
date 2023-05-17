import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
import pathlib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random as rn




Imagenes=[]
Etiquetas=[]
IMG_TAM = 150
DIR_BASE =os.path.abspath(os.path.dirname(__file__))
DIR_no_relation = DIR_BASE+'/scatter_plots/no_relation'
DIR_relation = DIR_BASE+'/scatter_plots/relation'




def obtener_extension_archivos(DIR):
    """
    This function allow us to know the extensions
    of all files in one directory.
    Parameters.
    DIR: The Directory path that contain all documents
    
    Return.
    A dictionary with the different extensions into the
    directory.
    """
    ext=[]
    for archivo in os.listdir(DIR):
        nom_archivo, ext_archivo = os.path.splitext(archivo)
        ext.append(ext_archivo)
    extensiones=set(ext)
    return(extensiones)


def imagen_valida(archivo,extensiones=[".jpg"]): # Note, the format is .jpg by defoult
    """
    This function has two purposes:
    1.- See if the file exists
    2.- Check if the extension is inte the allowed
    If the two conditions above are fulfilled, then the
    function returns True, and False otherwise
    
    Parameters.
    archivo: The file pathe to the file
    extensiones: A list with the estensions allowed
    """
    nom_archivo, ext_archivo = os.path.splitext(archivo)
    es_archivo = os.path.isfile(archivo)
    es_imagen = ext_archivo.lower() in extensiones
    return(es_archivo and es_imagen)


def construir_base_datos(Imagenes_list, Etiquetas_list, etiqueta, DIR, color_mode, IMG_TAM):
    """
    This function uses the function imagen_valida(ruta) to see if a file exist and if its
    extension is valid. It the function give us a True, then the file is processed as follows
    
    1.- Loads the image (file) into PIL format.
    2.- The PIL format is converted into an array
    3.- The array is normalized and resized by the [IMG_TAM].
    4.- Finally it is converted into a numpy.array
    
    Each valid file is processed as above ant stored in a list named Imagenes and the corresponding
    tag is also stored in a list named Etiquetas.
    
    Parameters.
    Imagenes_list: This is the list that will stored each file after processing
    Etiquetas_list: This is the list that will stored each label after processing
    etiqueta: This is the label corrsponding to the Directory loaded with the files
    DIR: The directory path with the files
    color_mode: This is the color mode of the imagenes could be one of "grayscale", "rgb" and "rgba"
    IMG_TAM: This is the new size for the imagenes
    
    """
    Imagenes = Imagenes_list
    Etiquetas = Etiquetas_list
    for archivo in tqdm(os.listdir(DIR)):
        ruta = os.path.join(DIR,archivo)
        if imagen_valida(ruta):
            img = tf.keras.preprocessing.image.load_img(ruta,color_mode=color_mode)
            matriz_img = tf.keras.preprocessing.image.img_to_array(img)
            matriz_img = tf.image.resize(matriz_img/255,[IMG_TAM,IMG_TAM])
            Imagenes.append(matriz_img.numpy())                                # Esta lista se define al inicio
            Etiquetas.append(etiqueta)
    return(Imagenes, Etiquetas)

Imagenes, Etiquetas = construir_base_datos([], [], 0, DIR_no_relation, "grayscale", IMG_TAM)
Imagenes, Etiquetas = construir_base_datos(Imagenes, Etiquetas, 1, DIR_relation, "grayscale", IMG_TAM)


clases=["No Relation", "Relation"]
fig,ax=plt.subplots(4,4)
fig.set_size_inches(10,10)
for i in range(4):
    for j in range (4):
        l=rn.randint(0,len(Etiquetas))
        ax[i,j].imshow(Imagenes[l])
        ax[i,j].set_title('Clase: '+ clases[Etiquetas[l]])        
plt.tight_layout()
plt.show()
























