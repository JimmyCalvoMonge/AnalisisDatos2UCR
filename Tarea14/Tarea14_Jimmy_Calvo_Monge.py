#!/usr/bin/env python
# coding: utf-8

# <style>
#     .title_container {
#         margin: auto;
#         background: rgb(81,92,103);
#         background: linear-gradient(90deg, rgba(81,92,103,1) 36%, rgba(12,35,66,1) 62%);
#         border-radius: 7px;
#         color: white;
#         text-align:center;
#         width:75%;
#         padding-top:2%;
#         padding-bottom:2%;
#     }
#     
#     .question_container {
#         margin: auto;
#         background: rgb(84,138,142);
#         background: linear-gradient(90deg, rgba(84,138,142,1) 41%, rgba(145,201,73,1) 81%);
#         border-radius: 7px;
#         color: white;
#         text-align:left;
#         width:75%;
#         padding-top:1%;
#         padding-bottom:1%;
#         padding-left: 2%;
#         margin-top:2%;
#     }
#     
#     .question_container p {
#         font-size: 16px;
#     }
#     
#     .alert_container {
#         margin: auto;
#         background: rgb(142,94,84);
#         background: linear-gradient(128deg, rgba(142,94,84,1) 13%, rgba(201,103,73,1) 69%);
#         border-radius: 7px;
#         color: white;
#         text-align:left;
#         width:75%;
#         padding-top:1%;
#         padding-bottom:1%;
#         padding-left: 2%;
#         margin-top:2%;
#     }
#     
#     .alert_container p {
#         font-size: 16px;
#     }
#     
#     .code_span {
#         background-color: #E2E7EC;
#         padding:2px;
#         border-radius:1px;
#         font-family: Consolas,monaco,monospace;
#         color:black;
#     }
# </style>

# <div class ='title_container'>
#     <h1> Análisis de Datos II </h1>
#     <h2> Profesor: Oldemar Rodríguez </h2>
#     <h3> Estudiante: Jimmy Calvo Monge </h3>
#     <h3> Carné: B31281 </h3>
#     <hr style='color:white; width:80%;'>
#     <h4> TAREA 14 </h4>
#     <h4> Fecha de entrega: 3 de Diciembre de 2022 </h4>
# </div>

# Importamos los módulos necesarios para resolver esta tarea.

# In[1]:


### Basicos
import numpy as np
import pandas as pd
from pandas import DataFrame

### Utilidades/Varios
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle
from sklearn.tree import export_graphviz
from sklearn import tree
import seaborn as sns
import time
import graphviz
import os

### Training/Testing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#CV2
import cv2

import warnings
warnings.filterwarnings('ignore')


# <div class='question_container'>
#     <h2> Pregunta 1 </h2>
#     <p> Dada una Red Neuronal <code>RNNM</code> tipo <code>LTSM</code>, donde $\tau(x)$ se define como sigue: </p>
#     $$
#     \begin{align*}
#     \tau(x) = \begin{cases}
#     1 - \frac{1}{1+x} & \text{ si } x \geq 0 \\
#     -1 + \frac{-1}{1-x} & \text{ si } x < 0. 
#     \end{cases}
#     \end{align*}
#     $$
#     <p> Además </p>
#     $$
#     \begin{align*}
#     &f_t = \tau(W_{hf}h_{t-1} + W_{xf}x_t) \\
#     &i_t = \tau(W_{hi}h_{t-1} + W_{xi}x_t) \\
#     &o_t = \tau(W_{ho}h_{t-1} + W_{xo}x_t) \\
#     &g_t = \tanh(W_{hg}h_{t-1} + W_{xg}x_t) \\
#     &h_0 = \overrightarrow{0} \\
#     &c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
#     &h_t = o_t \odot \tanh (c_t) \\
#     &c_0 = \overrightarrow{0}
#     \end{align*}
#     $$
#     <p>Además, se tiene que:</p>
#     $$
#     \begin{align*}
#     &x_1 = \begin{pmatrix}
#     2 \\
#     7 \\
#     3 \\
#     1 \\
#     1 \\
#     \end{pmatrix}, \quad x_2 = \begin{pmatrix}
#     0 \\
#     1 \\
#     0 \\
#     1 \\
#     1 \\
#     \end{pmatrix}, \\
#     &
#     W_{xf} = \begin{pmatrix}
#     2&5&0&3 \\
#     0&3&8&2 \\
#     3&0&1&1 \\
#     0&1&2&0 \\
#     1&1&2&1 \\
#     \end{pmatrix}, \quad W_{hf} = \begin{pmatrix}
#     1&0&1&1&2 \\
#     1&1&4&1&2 \\
#     3&0&1&1&1 \\
#     1&1&2&0&9 \\
#     1&1&1&1&1 \\
#     \end{pmatrix}, \\
#     & W_{xi} = \begin{pmatrix}
#     2&5&0&3 \\
#     0&3&8&2 \\
#     3&0&1&1 \\
#     0&1&2&0 \\
#     1&1&2&1 \\
#     \end{pmatrix}, \quad W_{hi} = \begin{pmatrix}
#     1&0&1&1&2 \\
#     1&1&4&1&2 \\
#     3&0&1&1&1 \\
#     1&1&2&0&9 \\
#     1&1&1&1&1 \\
#     \end{pmatrix}, \\
#     &W_{xo} = \begin{pmatrix}
#     2&0&0&3 \\
#     0&3&0&2 \\
#     3&0&1&1 \\
#     0&1&0&0 \\
#     1&1&0&1 \\
#     \end{pmatrix}, \quad W_{ho} =
#     \begin{pmatrix}
#     1&0&1&1&0 \\
#     1&1&0&1&2 \\
#     3&0&1&1&1 \\
#     1&1&0&0&9 \\
#     1&1&0&1&1 \\
#     \end{pmatrix}, \\
#     & W_{xg} = \begin{pmatrix}
#     2&0&0&0 \\
#     0&0&0&2 \\
#     0&0&1&1 \\
#     0&1&0&0 \\
#     1&1&0&1 \\
#     \end{pmatrix}, \quad W_{hg} = \begin{pmatrix}
#     1&0&0&1&2 \\
#     1&1&0&1&2 \\
#     0&0&1&1&1 \\
#     1&1&0&0&0 \\
#     1&1&0&1&0 \\
#     \end{pmatrix}.
#     \end{align*}
#     $$
#     <ul>
#     <li> Calcule $h_1$ y $h_2$.</li>
#     <li> Prediga $y_2 = W_{hy}h_2$ para 
#         $$
#         \begin{align*}
#         W_{hy} = \begin{pmatrix}
#         1&0&1&0&1 \\
#         0&1&0&1&1 \\
#         \end{pmatrix}.
#         \end{align*}
#         $$
#     </li>
#     </ul>
# </div>

# **Respuesta**
# 
# 1. En la primera iteración calculamos $f_1 = \tau(W_{hf}h_{0} + W_{xf}x_1) = \tau(W_{xf}x_1)$ ya que $h_0 = \overrightarrow{0}$.
# 
# Calculamos entonces $W_{xf}x_1$ que viene dado por
# 
# $$
# \begin{align*}
# W_{xf}x_1 = \begin{pmatrix}
#     2&5&0&3 \\
#     0&3&8&2 \\
#     3&0&1&1 \\
#     0&1&2&0 \\
#     1&1&2&1
#     \end{pmatrix}\begin{pmatrix}
#     2 \\
#     7 \\
#     3 \\
#     1
#     \end{pmatrix} = \begin{pmatrix}42\\ 47\\ 10\\ 13\\ 16\end{pmatrix}
# \end{align*}
# $$
# Al aplicarle $\tau$ tenemos que
# 
# $$
# f_1 = \tau \begin{pmatrix}42\\ 47\\ 10\\ 13\\ 16\end{pmatrix} = \begin{pmatrix}0.9767\\ 0.9792\\ 0.9091\\ 0.9286\\ 0.9412 \end{pmatrix}
# $$

# Similarmente tenemos que $i_1$ viene dado por
# 
# $$
# \begin{align*}
#  i_1 = \tau(W_{hi}h_{0} + W_{xi}x_1) &= \tau( W_{xi}x_1 ) \\
#  &=\tau \left(\begin{pmatrix}
#     2&5&0&3 \\
#     0&3&8&2 \\
#     3&0&1&1 \\
#     0&1&2&0 \\
#     1&1&2&1
#     \end{pmatrix}\begin{pmatrix}
#     2 \\
#     7 \\
#     3 \\
#     1
#     \end{pmatrix} \right)\\
#     &=\tau \begin{pmatrix}42\\ 47\\ 10\\ 13\\ 16\end{pmatrix} \\
#     &=\begin{pmatrix}0.9767\\ 0.9792\\ 0.9091\\ 0.9286\\ 0.9412 \end{pmatrix}
# \end{align*}
# $$
# El mismo resultado de arriba ya que $W_{xf} = W_{xi}$

# Similarmente tenemos que $o_1$ viene dado por
# 
# $$
# \begin{align*}
#  o_1 = \tau(W_{ho}h_{0} + W_{xo}x_1) &= \tau( W_{xo}x_1 ) \\
#  &=\tau \left(\begin{pmatrix}
#     2&0&0&3 \\
#     0&3&0&2 \\
#     3&0&1&1 \\
#     0&1&0&0 \\
#     1&1&0&1
#     \end{pmatrix}\begin{pmatrix}
#     2 \\
#     7 \\
#     3 \\
#     1
#     \end{pmatrix} \right)\\
#     &=\tau \begin{pmatrix}7\\ 23\\ 10\\ 7\\ 10\end{pmatrix} \\
#     &=\begin{pmatrix}0.8750 \\ 0.9583 \\ 0.9091 \\ 0.8750 \\ 0.9091 \end{pmatrix}
# \end{align*}
# $$

# Similarmente tenemos que $g_1$ viene dado por
# 
# $$
# \begin{align*}
#  g_1 = \tau(W_{hg}h_{0} + W_{xg}x_1) &= \tau( W_{xg}x_1 ) \\
#  &=\tau \left(\begin{pmatrix}
#     2&0&0&0 \\
#     0&0&0&2 \\
#     0&0&1&1 \\
#     0&1&0&0 \\
#     1&1&0&1
#     \end{pmatrix}\begin{pmatrix}
#     2 \\
#     7 \\
#     3 \\
#     1
#     \end{pmatrix} \right)\\
#     &=\tau \begin{pmatrix}4\\ 2\\ 4\\ 7\\ 10\end{pmatrix} \\
#     &=\begin{pmatrix}0.8\\ 0.6667\\ 0.8 \\0.8750\\ 0.9091 \end{pmatrix}
# \end{align*}
# $$

# Luego usamos que 
# 
# $$
# \begin{align*}
# c_1 = f_1 \odot c_{0} + i_1 \odot g_1 &= i_1 \odot g_1 = \begin{pmatrix}0.9767\\ 0.9792\\ 0.9091\\ 0.9286\\ 0.9412 \end{pmatrix} \odot \begin{pmatrix}0.8\\ 0.6667\\ 0.8 \\0.8750\\ 0.9091 \end{pmatrix} = \begin{pmatrix} 0.78139 \\ 0.65278 \\ 0.72727 \\0.8125\\ 0.85561\end{pmatrix}.
# \end{align*}
# $$
# 
# Hemos truncado los números decimales. Los cálculos están abajo, se hicieron con `numpy`. Por otro lado:
# 
# $$
# \begin{align*}
# h_1 = o_1 \odot \tanh (c_1) = \begin{pmatrix}0.8750 \\ 0.9583 \\ 0.9091 \\ 0.8750 \\ 0.9091 \end{pmatrix} \odot \tanh \begin{pmatrix} 0.78139 \\ 0.65278 \\ 0.72727 \\0.8125\\ 0.85561\end{pmatrix} = \begin{pmatrix}0.5718\\ 0.5496\\ 0.5649\\ 0.5871\\ 0.6309\end{pmatrix}
# \end{align*}
# $$

# In[3]:


def tau(x):
    if x>=0:
        return 1 - 1/(1+x)
    else:
        return -1 -1/(1-x)
    
tau_vect = np.vectorize(tau)


# In[11]:


f_1 = tau_vect(np.array([42,47,10,13,16])) # = i_1
i_1 = f_1
f_1


# In[9]:


o_1 = tau_vect(np.array([7,23,10,7,10]))
o_1


# In[10]:


g_1 = tau_vect(np.array([4,2,4,7,10]))
g_1


# In[12]:


c_1 = np.multiply(np.array([0.97674419, 0.97916667, 0.90909091, 0.92857143, 0.94117647]),
            np.array([0.8       , 0.66666667, 0.8       , 0.875     , 0.90909091]))
c_1


# In[13]:


np.multiply(i_1, g_1)


# In[18]:


h_1 = np.multiply(o_1, np.tanh(c_1))
h_1


# Esto es el valor de $h_1$. Ahora calculamos $h_2$. Lo haremos con `python` por comodidad.

# In[37]:


x_1 = np.array([2,7,3,1])
x_2 = np.array([0,1,0,1])
W_xf = np.array([
[2,5,0,3 ],
[0,3,8,2 ],
[3,0,1,1 ],
[0,1,2,0 ],
[1,1,2,1 ]
])
W_hf = np.array([
[1,0,1,1,2 ],
[1,1,4,1,2 ],
[3,0,1,1,1 ],
[1,1,2,0,9 ],
[1,1,1,1,1 ]
])
W_xi = np.array([
[2,5,0,3 ],
[0,3,8,2 ],
[3,0,1,1 ],
[0,1,2,0 ],
[1,1,2,1 ]
])
W_hi = np.array([
[1,0,1,1,2 ],
[1,1,4,1,2 ],
[3,0,1,1,1 ],
[1,1,2,0,9 ],
[1,1,1,1,1 ]
])
W_xo = np.array([
[2,0,0,3 ],
[0,3,0,2 ],
[3,0,1,1 ],
[0,1,0,0 ],
[1,1,0,1 ]
])
W_ho = np.array([
[1,0,1,1,0 ],
[1,1,0,1,2 ],
[3,0,1,1,1 ],
[1,1,0,0,9 ],
[1,1,0,1,1 ]
])
W_xg = np.array([
[2,0,0,0 ],
[0,0,0,2 ],
[0,0,1,1 ],
[0,1,0,0 ],
[1,1,0,1 ]
])
W_hg = np.array([
[1,0,0,1,2 ],
[1,1,0,1,2 ],
[0,0,1,1,1 ],
[1,1,0,0,0 ],
[1,1,0,1,0 ]
])


# In[38]:


f_2 = tau_vect(np.matmul(W_hf,h_1) + np.matmul(W_xf,x_2))
f_2


# In[39]:


i_2 = tau_vect(np.matmul(W_hi,h_1) + np.matmul(W_xi,x_2))
i_2


# In[40]:


o_2 = tau_vect(np.matmul(W_ho,h_1) + np.matmul(W_xo,x_2))
o_2


# In[41]:


g_2 = tau_vect(np.matmul(W_hg,h_1) + np.matmul(W_xg,x_2))
g_2


# In[42]:


c_2 = np.multiply(i_2, g_2)
c_2


# In[43]:


h_2 = np.multiply(o_2, np.tanh(c_2))
h_2


# Aquí entonces tenemos el valor de $h_2$. Finalmente, nuestra predicción para $y_2 = W_{hy}y_2$ viene dada por lo siguiente:

# In[44]:


W_hy = np.array([[1,0,1,0,1],[0,1,0,1,1]])
y_2 = np.matmul(W_hy, h_2)
y_2


# <div class='question_container'>
#     <h2> Pregunta 2 </h2>
#     <p> Descomprima el archivo <code>emociones.zip</code>. Este conjunto de audios tiene 7 clases: <code>Disgustado, Enojado, Feliz, Neutral, Sorprendido, Temeroso</code> y <code>Triste</code>. El objetivo de este ejercicio es identificar la emoción que expresa la persona cuando habla. Con la tabla de datos realice lo siguiente: </p>
#     <ul>
#         <li> Cargue los audios contenidos en el archivo <code>emociones.zip</code> y obtenga la etiqueta correspondiente para cada audio. </li>
#         <li> Realice un preprocesamiento de los datos con ayuda de la técnica de <code>MFCC</code>. </li>
#         <li> Divida la tabla utilizando un 90% para entrenamiento y un 10% para pruebas. Aplique el one-hot-encoding a las etiquetas. </li>
#         <li> Genere una estructura de Redes Neuronales <code>LSTM</code>. La estructura puede definirla a su conveniencia.</li>
#     <li> Configure el modelo con la función de costo <code>categorical_crossentropy</code>, la función de optimización <code>adam</code> y la métrica <code>accuracy</code>.</li>
#         <li> Entrene el modelo con los siguientes parámetros <code>epochs=100</code> y <code>batch_size=32</code>. Esto puede tardar un rato. Establezca el valor de 0 para el parámetro <code>verbose</code> para omitir la salida en consola.</li>
#         <li> Utilice el método <code>evaluate()</code> para evaluar la precisión del modelo. </li>
#         <li> Obtenga y cargue 10 audios de una persona hablando, pueden ser de personas conocidas o tomados de internet. Luego con ayuda del modelo realice una predicción de esos 10 audios. Muestre y comente los resultados.</li>
#     </ul>
# </div>

# In[143]:


import re
import librosa
from tqdm import tqdm
labels = []
audios = []

for carpeta in next(os.walk('./emociones/emociones'))[1]:
    print("Leyendo "+ carpeta +"...")
    for nombrearchivo in tqdm(next(os.walk('./emociones/emociones' + '/' + carpeta))[2]):
        if re.search("\\.(mp3|wav|m4a|wma|aiff)$", nombrearchivo):
            try:
                audio, sample_rate = librosa.load('./emociones/emociones' + '/' + carpeta + '/' + nombrearchivo, sr = 1600)
                audio = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 128)
                audio = np.mean(audio.T, axis=0)
                audios.append(audio)
                labels.append(carpeta)
            except Exception as e:
                print(e)
                print("No se pudo cargar el audio: " + nombrearchivo + " en la carpeta: " + carpeta)


# In[144]:


X = np.array(audios, dtype = np.float32)
y = np.array(labels)


# In[145]:


print(
  'Total de individuos: ', len(X),
  '\nNúmero total de salidas: ', len(np.unique(y)), 
  '\nClases de salida: ', np.unique(y))


# ##### Preparamos los datos:

# In[146]:


# Dividir en train y test
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.1)

# Normalizar
scaler = StandardScaler()
sc_training = scaler.fit_transform(train_X)
sc_testing  = scaler.fit_transform(test_X)

# Cambiamos las etiquetas de categoricas a one-hot encoding
train_Y_one_hot_pd = pd.get_dummies(train_Y)
train_Y_one_hot = train_Y_one_hot_pd.to_numpy()

test_Y_one_hot_pd = pd.get_dummies(test_Y)
test_Y_one_hot = test_Y_one_hot_pd.to_numpy()


# In[147]:


train_Y_one_hot_pd


# In[178]:


test_X.shape


# Creamos la arquitectura del modelo de RNN a utilizar:

# In[149]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical, plot_model


# In[150]:


modelo_emociones = Sequential()
modelo_emociones.add(LSTM(units = 32, input_shape = (128, 1)))
modelo_emociones.add(Dense(15, activation = "relu"))
modelo_emociones.add(Dense(128, activation = "relu"))
modelo_emociones.add(Dense(64, activation = "relu"))
modelo_emociones.add(Dense(7, activation = "softmax"))
modelo_emociones.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo_emociones.summary()


# In[180]:


import time
start = time.time()
modelo_emociones.fit(train_X, train_Y_one_hot, epochs=100, batch_size=32, verbose=0)
end = time.time()
print(f"Entrenar el modelo tomó {(end-start)/60} minutos.")


# In[181]:


modelo_emociones.save("lstm_emociones.h5py")


# In[182]:


precision = modelo_emociones.evaluate(test_X, test_Y_one_hot, verbose=0)
print(precision[1])


# He intentado con varias arquitecturas, pero no logro obtener una precisión más alta. Lamentablemente cada modelo dura algunos minutos en entrenar.

# El siguiente código fue tomado de este repositorio de github: [alexmuhr/Voice_Emotion](https://github.com/alexmuhr/Voice_Emotion)
# en donde se entrenan CNNs con audios obtenidos con webscraping a partir de ciertas librerías públicas de audio.

# In[155]:


import requests
from bs4 import BeautifulSoup


# In[156]:


url = "https://tspace.library.utoronto.ca/handle/1807/24487"
response = requests.get(url)
response.status_code


# In[157]:


def make_soup(url):
    response = requests.get(url)
    code = response.status_code
    assert ((code >= 200) & (code < 300))
    page = response.text
    soup = BeautifulSoup(page, 'lxml')
    return soup


# In[ ]:


soup = make_soup(url)
strongs = soup.find_all('strong')[1:]
hrefs = [x.find('a', href = True)['href'] for x in strongs]


# In[161]:


link_list = []
for href in hrefs:
    url = "https://tspace.library.utoronto.ca" + href
    soup = make_soup(url)
    div = soup.find('div', {'class': 'item-files'})
    a_tags = div.find_all('a')
    links = [x['href'] for x in a_tags]
    link_list += links
link_list[0:5]


# In[162]:


url_list = ['https://tspace.library.utoronto.ca' + x for x in link_list]
with open('url_list.txt', 'w') as file:
    for url in url_list:
        file.write(url + '\n')
    file.close()


# El archivo `url_list.txt` tiene una lista de direcciones web con los archivos `.wav` de la biblioteca TESS, de la Universidad de Toronto. Hemos descargado 10 archivos de esta biblioteca que utilizaremos como prueba.

# In[168]:


labels_nuevos = []
audios_nuevos = []

for nombrearchivo in tqdm(next(os.walk('./emociones_nuevas'))[2]):
    if re.search("\\.(mp3|wav|m4a|wma|aiff)$", nombrearchivo):
        try:
            audio, sample_rate = librosa.load('./emociones_nuevas/' + nombrearchivo, sr = 1600)
            audio = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 128)
            audio = np.mean(audio.T, axis=0)
            audios_nuevos.append(audio)
            labels_nuevos.append(nombrearchivo.split('_')[2].replace('.wav',''))
        except Exception as e:
            print(e)
            print("No se pudo cargar el audio: " + nombrearchivo)


# In[169]:


labels_nuevos


# In[ ]:





# In[177]:


audios_validar = np.array(audios_nuevos)
audios_validar.shape


# In[183]:


pred = modelo_emociones.predict(audios_validar)
pred


# In[186]:


preds_vals = np.argmax(pred, axis = 1)
[train_Y_one_hot_pd.columns[pred_val] for pred_val in preds_vals]


# Repito que nuestra arquitectura no tiene una precisión muy alta.

# <div class='question_container'>
#     <h2> Pregunta 3 </h2>
#     <p> En este ejercicio vamos a intentar predecir la letra que sigue del alfabeto griego. Cabe resaltar que no existe un mapeo exacto del abecedario al nuestro y que el número de letras es inferior al nuestro también. Copie y pegue el siguiente código y utilicelo cómo set de datos.</p>
#     <code>alfabetoGriego</code> = $\alpha\beta\gamma\delta\epsilon\zeta\eta\theta\iota\kappa\lambda\mu\nu\xi\phi\pi\rho\sigma\tau\upsilon\chi\psi\omega$
#     <p>Para esto realice lo siguiente:</p>
#     <ol>
#     <li>Defina un conjunto con los diferentes caracteres que conforman el set.</li>
#     <li>Cree un diccionario que permita definir la equivalencia entre caracter e índice, así como su inverso.</li>
#     <li>Haga los pares de entrada y salida para entrenar el modelo y transforme al formato esperado por la red.</li>
#     <li>Normalice los datos y trasforme la variable a predecir a formato One-Hot.</li>
#     <li>Haga el modelo usando la función de activación <code>softmax</code> y con las capas que considere necesarias. Utilice la función de optimización <code>RMSProp</code>, la función de costo <code>categorical_crossentropy</code> y las métricas <code>accuracy</code>.</li>
#     <li>Haga un resumen del modelo.</li>
#     <li>Haga una predicción entrenando el modelo con 500 epochs y un tamaño de lote igual a 1.</li>
#     <li>Genere la matriz de confusión.</li>
#     <li>Calcule la precisión global. Interprete la calidad de los resultados.</li>
#     <li>Repita el ejercicio 5 pero esta vez utilice como optimizador la función <code>adam</code>. Compare resultados.</li>
#     </ol>
# </div>

# Tomamos los caracteres unicode para las letras griegas de [Unicode characters for engineers in Python](https://pythonforundergradengineers.com/unicode-characters-in-python.html).

# In[187]:


alfabeto = ["\u03B1","\u03B2","\u03B3","\u03B4","\u03B5","\u03B6",
            "\u03B7","\u03B8","\u03B9","\u03BA","\u03BB",
            "\u03BC","\u03BD","\u03BE","\u03BF","\u03C0",
            "\u03C1","\u03C2","\u03C3","\u03C4","\u03C5",
            "\u03C6","\u03C7","\u03C8","\u03C9"]
# crear mapeo de caracteres a números enteros (0-25) y viceversa
char_to_int = dict((c, i) for i, c in enumerate(alfabeto))
int_to_char = dict((i, c) for i, c in enumerate(alfabeto))

print("char_to_int:\n",char_to_int,"\n")
print("int_to_char:\n",int_to_char)


# #### Creamos los datos de entrada

# In[188]:


# creamos nuestros pares de entrada y salida para entrenar nuestra red neuronal
#seq_length = 2
seq_length = 1
dataX = []
dataY = []

for i in range(0, len(alfabeto) - seq_length):
    seq_in = alfabeto[i:i + seq_length]
    seq_out = alfabeto[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)


# In[189]:


X = np.reshape(dataX, (len(dataX), seq_length, 1))
X = X / float(len(alfabeto))
print("Shape: ", X.shape)


# #### Pasamos a formato One-Hot los datos de salida

# In[190]:


y = to_categorical(dataY)
print(y[0:5])


# #### Creación del modelo

# In[191]:


modelo = Sequential()
modelo.add(LSTM(64, input_shape = (X.shape[1], X.shape[2])))
modelo.add(Dense(15, activation = 'relu'))
modelo.add(Dense(y.shape[1], activation = 'softmax'))
modelo.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])


# #### Resumen del modelo

# In[192]:


modelo.summary()


# #### Entrenamiento del modelo

# In[193]:


start = time.time()
modelo.fit(X, y, epochs=500, batch_size=1, verbose=0)
end= time.time()
print(f"Entrenar este modelo tomó {(end-start)/60} minutos.")


# #### Predicción del modelo

# In[194]:


modelo_pred = modelo.predict(X,verbose=0)
rnn_modelo_predicted = np.argmax(modelo_pred, axis=1)
rnn_modelo_predicted


# In[195]:


rnn_modelo_cm = confusion_matrix(np.argmax(y, axis=1), rnn_modelo_predicted)

# Visualizamos la matriz de confusión
import seaborn as sn
rnn_modelo_df_cm = pd.DataFrame(rnn_modelo_cm, range(len(rnn_modelo_predicted)), range(len(rnn_modelo_predicted)))  
plt.figure(figsize = (8,10))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(rnn_modelo_df_cm, annot=True, annot_kws={"size": 12}) # font size  
plt.show()


# In[196]:


# demuestra la predicciones del modelo
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alfabeto))
    prediction = modelo.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)


# In[197]:


# sumariza el rendimiento del modelo
scores = modelo.evaluate(X, y, verbose=0)
print("Precisión del modelo: %.2f%%" % (scores[1]*100))


# Obtuvimos una precisión moderadamente para este modelo, con la arquitectura que hemos utilizado. Podríamos buscar otra arquitectura para lograr mejorar la predicción. Ahora utilizamos el optimizador `adam` y comparamos.

# In[198]:


print("Configurando modelo ...")
modelo = Sequential()
modelo.add(LSTM(64, input_shape = (X.shape[1], X.shape[2])))
modelo.add(Dense(15, activation = 'relu'))
modelo.add(Dense(y.shape[1], activation = 'softmax'))
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Entrenando modelo ...")
start = time.time()
modelo.fit(X, y, epochs=500, batch_size=1, verbose=0)
end= time.time()
print(f"Entrenar este modelo tomó {(end-start)/60} minutos.")

print("Efecuando predicciones...")
modelo_pred = modelo.predict(X,verbose=0)
rnn_modelo_predicted = np.argmax(modelo_pred, axis=1)
rnn_modelo_df_cm = pd.DataFrame(rnn_modelo_cm, range(len(rnn_modelo_predicted)), range(len(rnn_modelo_predicted))) 
print(rnn_modelo_df_cm)

print("Precisión...")
# demuestra la predicciones del modelo
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alfabeto))
    prediction = modelo.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)
    
# sumariza el rendimiento del modelo
scores = modelo.evaluate(X, y, verbose=0)
print("Precisión del modelo: %.2f%%" % (scores[1]*100))


# Con este optimizador logramos incrementar sustancialmente la precisión del modelo, utilizando la misma arquitectura.
