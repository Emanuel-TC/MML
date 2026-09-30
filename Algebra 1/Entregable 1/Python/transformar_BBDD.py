#!/Users/ramirezgd/.misc/micromamba/bin/python
# -*- coding: utf-8 -*-

#%% Importación de módulos

import numpy as np
import matplotlib.image as mpimg

import requests
import io
import zipfile
import os

from IPython import get_ipython
get_ipython().run_line_magic("reset","-sf")
get_ipython().run_line_magic("clear","")

#%% Parámetros

urldb = 'http://datasets.openimaj.org/imm_face_db.zip'
filetosave = 'database.npz'
tmp_folder = '/tmp/imm_face_db/'

#%% Funciones auxiliares

def getstructure(filename):

    # Lectura del fichero asf por líneas y eliminar el caracter de salto de línea
    lines = [line.rstrip('\n') for line in open(filename + '.asf')]
    
    # Obtener el número de puntos
    npoints = int(lines[lines.index('# number of model points') + 2])
    
    # Obtener el número de línea del primer punto
    index = [i for i, item in enumerate(lines) if item.startswith('# format: <path#> <type> <x rel.> <y rel.>')][-1] + 2
    
    # Obtener los puntos como una lista de una lista de strings
    lines = [lines[i].split("\t")[:7] for i in range(index,index+npoints)]
    
    # Número de paths
    path_number = [int(line[0]) for line in lines]
    npaths = max(path_number) + 1
                
    # Landmarks
    landmarks = [np.asarray([[lines[i][2], lines[i][3]] for i, item in enumerate(path_number) if item == index], dtype=np.float32) for index in range(npaths)]
    
    # Están los paths cerrados
    closed_paths = [not any([lines[i][4]==lines[i][5] or lines[i][4]==lines[i][6] for i, item in enumerate(path_number) if item == index]) for index in range(npaths)]
    
    # Lectura de la imágen
    img = mpimg.imread(filename + '.jpg')
    
    # Se devuelven los datos como un diccionario
    return {'landmarks': landmarks, 'closed_path': closed_paths, 'image': img, 'resolution': img.shape[0:2]}

#%% Descargar y descomprimir el fichero zip

r = requests.get(urldb, stream=True)

zip_ref = zipfile.ZipFile(io.BytesIO(r.content))
zip_ref.extractall(tmp_folder)
zip_ref.close()

#%% Leer la base de datos y guardarlo en un array de diccionarios

filenames = [f[:-4] for f in os.listdir(tmp_folder) if f.endswith(".jpg")]

structs = [getstructure(tmp_folder + filename) for filename in filenames]

#%% Guardar la base de datos

np.savez_compressed(tmp_folder + filetosave, structs=structs)