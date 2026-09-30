#!/Users/ramirezgd/.misc/micromamba/bin/python
# -*- coding: utf-8 -*-

#%% Importación de módulos

import numpy as np
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic("reset","-sf")
get_ipython().run_line_magic("clear","")
# get_ipython().run_line_magic("matplotlib","inline")
get_ipython().run_line_magic("matplotlib","qt5")

#%% Parámetros

filetoload = 'database.npz'
tmp_folder = '/tmp/imm_face_db/'
epsilon = 1e-5

#%% Funciones auxiliares

def align_shapes(shape1,shape2):

    dummy = np.trace(np.dot(shape1.T,shape1))
    a = np.trace(np.dot(shape1.T,shape2))/dummy
    b = (np.dot(shape1[:,0].T,shape2[:,1]) - np.dot(shape1[:,1].T,shape2[:,0]))/dummy
        
    scale = np.sqrt(a**2 + b**2)
    rotation = np.arctan(b/a)

    R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])

    return scale*np.dot(shape1,R.T)

#%% Carga del fichero

loaded = np.load(tmp_folder + filetoload, allow_pickle=True)

structs = loaded['structs']

#%% Pintamos una cara aleatoria y los paths

index = np.random.randint(len(structs))

plt.figure(0)
plt.imshow(structs[index]['image'])

for kk in range(len(structs[index]['closed_path'])):
    indexes = [i for i in range(len(structs[index]['landmarks'][kk][:,0]))]
    if structs[index]['closed_path'][kk]:
        indexes.append(0)  # Add the first index to close the path
    plt.plot(structs[index]['landmarks'][kk][indexes,0]*structs[index]['resolution'][1],structs[index]['landmarks'][kk][indexes,1]*structs[index]['resolution'][0])
    
#%% Matriz con todos los landmarks

landmarks = [np.vstack(struct['landmarks']) for struct in structs]

#%% Pintamos todos los landmarks

plt.figure(1)

for landmark in landmarks:
    plt.plot(landmark[:,0],-landmark[:,1],'.') # Se necesita el signo menos para la segunda componente porque y = 0 en una imágen corresponde con su parte superior

#%% Como los landmarks no están alineados, tenemos que alinearlos

# Centramos los landmarks
landmarks_centered = [landmark - np.mean(landmark,axis=0) for landmark in landmarks]

# Normalizamos la forma promedio
x_bar0 = landmarks_centered[0]/np.sqrt(np.sum(landmarks_centered[0]**2))

# Algoritmo iternativo para alinearlos

flag = 1
landmarks_aligned = landmarks_centered

while flag:
    
    # Align the shapes
    landmarks_aligned = [align_shapes(landmark_aligned,x_bar0) for landmark_aligned in landmarks_aligned]
    
    # Obtenemos la nueva forma promedio
    x_bar = np.mean(np.asarray(landmarks_aligned),axis=0)

    # Alineamos la forma promedio
    x_bar = align_shapes(x_bar,x_bar0)
    
    # Normalizamos la forma promedio
    x_bar = x_bar/np.sqrt(np.sum(x_bar[0]**2))

    if np.sqrt(np.sum((x_bar - x_bar0)**2))>epsilon:
        # No ha convergido
        x_bar0 = x_bar       
    else:
        # Ha convergido
        flag = 0
        
#%% Pintamos todos los landmarks alineados

plt.figure(2)

for landmark_aligned in landmarks_aligned:
    plt.plot(landmark_aligned[:,0],-landmark_aligned[:,1],'.')  # Se necesita el signo menos para la segunda componente porque y = 0 en una imágen corresponde con su parte superior

#%% Guardamos los landmarks alineados

save_structs = [{'landmarks': structs[index]['landmarks'], 'landmarks_aligned': landmarks_aligned[index], 'closed_path': structs[index]['closed_path'], 'image': structs[index]['image'], 'resolution': structs[index]['resolution']} for index in range(len(structs))]

np.savez_compressed(tmp_folder + filetoload, structs=save_structs)
