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
#     <h4> TAREA 12 </h4>
#     <h4> Fecha de entrega: 20 de Noviembre de 2022 </h4>
# </div>

# Importamos los módulos necesarios para resolver esta tarea.

# In[1]:


### Basicos
import numpy as np
import pandas as pd
from pandas import DataFrame
import math

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
import itertools
from tqdm import tqdm
import time

### Training/Testing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

### predictPy
from predictPy import Analisis_Predictivo

### Validacion Cruzada
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

### Modelos:
# MLPClassifier:
from sklearn.neural_network import MLPClassifier

### Clase MatConf de la Tarea 2
class MatConf:
    
    def __init__(self,matriz):
        
        self.mat_conf = matriz
        
        VN = self.mat_conf[0,0]
        VP = self.mat_conf[1,1]
        FP = self.mat_conf[0,1]
        FN = self.mat_conf[1,0]
        
        dict_medidas = {
            'Precisión Global' : (VN+VP)/(VN+FP+FN+VP),
            'Error Global' : (FN+FP)/(VN+FP+FN+VP),
            'Precisión Positiva (PP)' : VP/(FN+VP),
            'Precisión Negativa (PN)' : VN/(VN+FP),
            'Proporción de Falsos Positivos (PFP)' : FP/(VN+FP),
            'Proporción de Falsos Negativos (PFN)' : FN/(FN+VP),
            'Asertividad Positiva (AP)' : VP/(FP+VP),
            'Asertividad Negativa (AN)' : VN/(VN+FN)
        }
        self.dict_medidas = dict_medidas
        
    def __str__(self):
        mensaje="Estos son los resultados para esta matriz de confusion:"
        for key in list(self.dict_medidas.keys()):
            mensaje = mensaje + f"\n - {key}: {self.dict_medidas[key]}"
        return mensaje

import warnings
warnings.filterwarnings('ignore')


# <div class='question_container'>
#     <h2> Pregunta 1 </h2>
#     <p> La tabla de datos <code>novatosNBA.csv</code> contiene diferentes métricas de desempeño de novatos de la NBA en su primera temporada. Para esta tabla, las 21 primeras columnas corresponden a las variables predictoras y la variable <code>Permanencia</code> es la variable a predecir, la cual indica si el jugador permanece en la NBA luego de 5 años. La tabla contiene 1340 filas (individuos) y 21 columnas (variables), con la tabla realice lo siguiente: </p>
#     <ul>
#     <li> Usando el paquete <code>MLPClassifier</code> en <code>Python</code> genere modelos predictivos usando un 75% de los datos para tabla aprendizaje y un 25% para la tabla testing. Genere al menos 2 modelos con configuraciones diferentes en los parámetros vistos en clase (<code>hidden_layer_sizes</code>, <code>activation</code>, <code>solver</code>). Realice lo anterior sin estandarizar los datos y luego con los datos estandarizados, es decir, al menos 4 modelos. Para estandarizar los datos utilice la clase <code>StandardScaler</code> de <code>sklearn.preprocessing</code> </li>
#     <li> Para cada modelo obtenga los índices de precisión, compare e interprete los resultados y
# las diferencias entre los modelos con datos estandarizados y los que no. </li>
#     </ul>
# </div>

# In[2]:


datos_novatos = pd.read_csv("novatosNBA.csv",sep=";",index_col=0)
datos_novatos.head(5)


# In[3]:


### Hay datos faltantes:
nas_dict={}
for col in datos_novatos.columns:
    nas_dict[col]=datos_novatos[col].isna().sum()
nas_dict


# In[4]:


datos_novatos = datos_novatos.dropna() ### Eliminamos los fatos faltantes por ahora.


# In[5]:


X = datos_novatos.drop(['Permanencia'],axis=1)
y = datos_novatos['Permanencia']


# Estandarizamos los datos para estos dos primeros modelos.

# In[9]:


datos_novatos_std = datos_novatos.copy()
datos_novatos_std.iloc[:,0:19] = StandardScaler().fit_transform(datos_novatos_std.iloc[:,0:19])
datos_novatos_std.head(5)


# In[23]:


datos_novatos.head(5)


# In[28]:


nnet1 = MLPClassifier(hidden_layer_sizes = (5,5,5),
                      activation = "logistic",
                      solver = "sgd",
                      random_state = 0)

nnet2 = MLPClassifier(hidden_layer_sizes = (50,25,15,10,5),
                      activation = "tanh",
                      solver = "lbfgs",
                      random_state = 0)

### 2 modelos con datos estandarizados.

analisis_1_std = Analisis_Predictivo(datos_novatos_std,
                                 predecir = "Permanencia",
                                 modelo = nnet1, 
                                 train_size = 0.7,
                                 random_state = 0)

analisis_2_std = Analisis_Predictivo(datos_novatos_std,
                                 predecir = "Permanencia",
                                 modelo = nnet2, 
                                 train_size = 0.7,
                                 random_state = 0)

resultados_1 = analisis_1_std.fit_predict_resultados(imprimir = False)

resultados_2 = analisis_2_std.fit_predict_resultados(imprimir = False)


# Ahora, hacemos lo mismo con los datos sin estandarizar.

# In[29]:


nnet3 = MLPClassifier(hidden_layer_sizes = (5,5,5),
                      activation = "logistic",
                      solver = "sgd",
                      random_state = 0)

nnet4 = MLPClassifier(hidden_layer_sizes = (50,25,15,10,5),
                      activation = "tanh",
                      solver = "lbfgs",
                      random_state = 0)

### 2 modelos con datos NO estandarizados.

analisis_3 = Analisis_Predictivo(datos_novatos,
                                 predecir = "Permanencia",
                                 modelo = nnet3, 
                                 train_size = 0.75,
                                 random_state = 0)

analisis_4 = Analisis_Predictivo(datos_novatos,
                                 predecir = "Permanencia",
                                 modelo = nnet4, 
                                 train_size = 0.75,
                                 random_state = 0)

resultados_3 = analisis_3.fit_predict_resultados(imprimir = False)

resultados_4 = analisis_4.fit_predict_resultados(imprimir = False)


# In[30]:


comparacion_df = pd.DataFrame({})
resultados = [resultados_1, resultados_2, resultados_3, resultados_4]

for res in resultados:
    
    medidas = MatConf(res['Matriz de Confusión']).dict_medidas
    comp_res = pd.DataFrame({})

    for key in list(medidas.keys()):
        comp_res[key] = [medidas[key]]
    comparacion_df = comparacion_df.append(comp_res, ignore_index=True)

comparacion_df.index = ['Nnet 1 estandarizado', 'Nnet 2 estandarizado',
                       'Nnet 1 NO estandarizado', 'Nnet 2 NO estandarizado']
comparacion_df


# En este caso, vemos que los modelos tienen un comportamiento similar, pero esto es quizá por una inapropiada escogencia de arquitecturas para nuestras redes neuronales. 
# Sí podemos notar, sin embargo, que la precisión global es ligeramente mayor en los casos que utilizan los datos estandarizados, para cada modelo correspondiente. Lo cual era de esperar.

# <div class='question_container'>
#     <h2> Pregunta 2 </h2>
#     <p>Este conjunto de datos es originalmente del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales. El objetivo del conjunto de datos es predecir de forma diagnóstica si un paciente tiene diabetes o no, basándose en determinadas medidas de diagnóstico incluidas en el conjunto de datos. El conjunto de datos tiene 390 filas y 16 columnas: </p>
#     <ul>
#         <li><code>X</code>: Id del paciente.</li>
#         <li><code>colesterol</code>: Colesterol en mg/dL.</li>
#         <li><code>glucosa</code>: Glucosa en mg/dL.</li>
#         <li><code>hdl_col</code>: Lipoproteínas (colesterol bueno).</li>
#         <li><code>prop_col_hdl</code>: Proporción del colesterol entre el hdl.</li>
#         <li><code>edad</code>: Edad del paciente.</li>
#         <li><code>genero</code>: Género del paciente.</li>
#         <li><code>altura</code>: Altura en pulgadas del paciente.</li>
#         <li><code>peso</code>: Peso en libras del paciente.</li>
#         <li><code>IMC</code>: índice de masa corporal.</li>
#         <li><code>ps_sistolica</code>: Presión arterial sistólica.</li>
#         <li><code>ps_diastolica</code>: Presión arterial diastólica.</li>
#         <li><code>cintura</code>: Longitud de la cintura en pulgadas.</li>
#         <li><code>cadera</code>: Longitud de la cadera en pulgadas.</li>
#         <li><code>prop_cin_cad</code>: Proporción de la longitud de la cintura entre la longitud de la cadera.</li>
#         <li><code>diabetes</code>: Diagnóstico de la diabetes.</li>
#     </ul>
#     <p>Realice lo siguiente:</p>
#     <ul>
#     <li> Cargue en <code>Python</code> la tabla de datos <code>diabetes.csv</code>.</li>
#     <li> Usando el paquete <code>MLPClassifier</code> en <code>Python</code> genere modelos predictivos usando un 75% de los datos para tabla aprendizaje y un 25% para la tabla testing. Genere al menos 2 modelos con configuraciones diferentes en los parámetros vistos en clase (<code>hidden_layer_sizes</code>, <code>activation</code>, <code>solver</code>). Realice lo anterior sin estandarizar los datos y luego con los datos estandarizados, es decir, al menos 4 modelos. Para estandarizar los datos utilice la clase <code>StandardScaler</code> de <code>sklearn.preprocessing</code> </li>
#         <li>Para cada modelo obtenga los índices de precisión, compare e interprete los resultados y las diferencias entre los modelos con datos estandarizados y los que no.</li>
#     </ul>
# </div>

# In[39]:


datos_diabetes = pd.read_csv("diabetes.csv",index_col=0)
datos_diabetes['genero'].value_counts()


# In[33]:


datos_diabetes.dtypes


# La variable `genero` es categórica, así que la convertiremos a dummy.

# In[36]:


#Convertimos a Dummy algunas de las variables predictoras
datos_diabetes_dum = pd.get_dummies(datos_diabetes, columns=['genero'])
col_diat = datos_diabetes_dum.pop("diabetes")
datos_diabetes_dum.insert(15, "diabetes", col_diat )
datos_diabetes_dum.head(5)


# In[46]:


datos_diabetes_dum_std = datos_diabetes_dum.copy()
datos_diabetes_dum_std.iloc[:,0:15] = StandardScaler().fit_transform(datos_diabetes_dum_std.iloc[:,0:15])
datos_diabetes_dum_std.head(5)


# Procedemos a hacer lo mismo que en el ejercicio anterior.

# In[51]:


nnet1 = MLPClassifier(hidden_layer_sizes = (3,)*4, activation = "relu",
                      solver = "adam", random_state = 0)

nnet2 = MLPClassifier(hidden_layer_sizes = (100,50,30,10), activation = "identity",
                      solver = "lbfgs", random_state = 0)

### 2 modelos con datos estandarizados.
analisis_1 = Analisis_Predictivo(datos_diabetes_dum_std, predecir = "diabetes",
                                 modelo = nnet1, train_size = 0.75, random_state = 40)

analisis_2 = Analisis_Predictivo(datos_diabetes_dum_std, predecir = "diabetes",
                                 modelo = nnet2, train_size = 0.75, random_state = 40)

resultados_1 = analisis_1.fit_predict_resultados(imprimir = False)
resultados_2 = analisis_2.fit_predict_resultados(imprimir = False)

nnet3 = MLPClassifier(hidden_layer_sizes = (3,)*4, activation = "relu", 
                      solver = "adam", random_state = 0)

nnet4 = MLPClassifier(hidden_layer_sizes = (100,50,30,10), activation = "identity",
                      solver = "lbfgs", random_state = 0)


analisis_3 = Analisis_Predictivo(datos_diabetes_dum, predecir = "diabetes",
                                 modelo = nnet3, train_size = 0.75, random_state = 40)

analisis_4 = Analisis_Predictivo(datos_diabetes_dum, predecir = "diabetes",
                                 modelo = nnet4, train_size = 0.75, random_state = 40)

resultados_3 = analisis_3.fit_predict_resultados(imprimir = False)
resultados_4 = analisis_4.fit_predict_resultados(imprimir = False)


# In[52]:


comparacion_df = pd.DataFrame({})
resultados = [resultados_1, resultados_2, resultados_3, resultados_4]

for res in resultados:
    
    medidas = MatConf(res['Matriz de Confusión']).dict_medidas
    comp_res = pd.DataFrame({})

    for key in list(medidas.keys()):
        comp_res[key] = [medidas[key]]
    comparacion_df = comparacion_df.append(comp_res, ignore_index=True)

comparacion_df.index = ['Nnet 1 estandarizado', 'Nnet 2 estandarizado',
                       'Nnet 1 NO estandarizado', 'Nnet 2 NO estandarizado']
comparacion_df


# Observe que para esta selección particular de parámetros y arquitecturas tuvimos un comportamiento superior en con los datos estandarizados. Algún fenómeno sucedió con la segunda arquitectura en los datos no estandarizados. Posiblemente de naturaleza numérica al emplear el solver y la función de activación seleccionados.

# <div class='question_container'>
#     <h2> Pregunta 3 </h2>
#     <p>En este ejercicio vamos a predecir números escritos a mano (Hand Written Digit Recognition), la tabla de aprendizaje está en el archivo `ZipDataTrainCod.csv` y la tabla de testing está en el archivo `ZipDataTestCod.csv`. En la figura siguiente se ilustran los datos: </p>
#     <p> Los datos de este ejemplo vienen de los códigos postales escritos a mano en sobres del correo postal de EE.UU. Las imágenes son de 16 $\times$ 16 en escala de grises, cada pixel va de intensidad de -1 a 1 (de blanco a negro). Las imágenes se han normalizado para tener aproximadamente el mismo tamaño y orientación. La tarea consiste en predecir, a partir de la matriz de 16 $\times$ 16 de intensidades de cada pixel, la identidad de cada imagen (0, 1, ...,  9) de forma rápida y precisa. Si es lo suficientemente precisa, el algoritmo resultante se utiliza como parte de un procedimiento de selección automática para sobres. Este es un problema de clasificación para el cual la tasa de error debe mantenerse muy baja para evitar la mala dirección de correo. La columna 1 tiene la variable a predecir Número codificada como sigue: 0='cero'; 1='uno'; 2='dos'; 3='tres'; 4='cuatro'; 5='cinco';6='seis'; 7='siete'; 8='ocho' y 9='nueve', las demás columnas son las variables predictivas, además cada fila de la tabla representa un bloque 16 $\times$ 16 por lo que la matriz tiene 256 variables predictoras. </p>
#     <ol>
#         <li>Usando el paquete <code>MLPClassifier</code> en <code>Python</code> genere un modelo predictivo de redes neuronales para estos datos. Utilice los siguientes parámetros: <code>hidden_layer_sizes = (250,100,50,25)</code>, <code>max_iter = 50000</code>, <code>activation = 'relu'</code>, <code>solver = 'adam'</code>, <code>random_state=0</code>. Interprete los resultados. </li>
#         <li>Genere un modelo de redes neuronales con los mismos parámetros del ítem anterior, pero esta vez reemplace cada bloque $4\times4$ de píxeles por su promedio. ¿Mejora la predicción? ¿Qué ventaja tiene estos datos respecto a los anteriores? Recuerde que cada bloque $16\times16$ está representado por una fila en las matrices de aprendizaje y testing. Despliegue la matriz de confusión resultante. La matriz de confusión obtenida debería ser igual o muy similar a ésta (ver enunciado). </li>
#         <li>Repita el item anterior pero esta vez reemplace cada bloque $4\times 4$ de píxeles por el máximo. ¿Mejoran resultados respecto a usar el promedio de cada bloque?</li>
#     </ol>
# </div>

# In[53]:


zipdata_train = pd.read_csv("ZipDataTrainCod.csv",sep=';')
zipdata_test = pd.read_csv("ZipDataTestCod.csv",sep=';')


# In[54]:


X_train = zipdata_train.drop(['Numero'],axis=1)
y_train = zipdata_train['Numero']
X_test = zipdata_test.drop(['Numero'],axis=1)
y_test = zipdata_test['Numero']


# Ajustamos ahora la red neuronal solicitada sobre estos datos. Por el momento, no vamos a estandarizar los datos, aunque sería posible hacerlo en este ejercicio.

# In[68]:


nnet = MLPClassifier(
    hidden_layer_sizes = (250,100,50,25),
    max_iter = 50000,
    activation = 'relu',
    solver = 'adam', 
    random_state=0) #Parametros indicados en la tarea

nnet.fit(X_train, y_train)


# Obtenemos nuestra matriz de confusión.

# In[69]:


prediccion = nnet.predict(X_test)
mat_cfn = confusion_matrix(y_test, prediccion)
mat_cfn


# Encontramos la precisión por cada dígito. Para esto reciclamos una función de la Tarea 3

# In[80]:


def get_prec_multi(mat_cfn, labels):
    suma_total = sum(sum(mat_cfn))
    suma_diag = sum([mat_cfn[i,i] for i in range(mat_cfn.shape[0])])
    prec_global = suma_diag/suma_total
    err_global = 1- prec_global
    prec_digitos={} ### Creamos un diccionario con la precisión de cada dígito.
    prec_digitos['Precisión Global']=prec_global
    prec_digitos['Error Global']=err_global
    for i in range(mat_cfn.shape[0]):
        prec_este_digito = mat_cfn[i,i]/sum([mat_cfn[i,j] for j in range(mat_cfn.shape[0])])
        prec_digitos[f'Precisión "{labels[i]}"']= prec_este_digito
    return prec_digitos


# In[81]:


labels_dig=['cero','cinco','cuatro','dos','nueve','ocho','seis','siete','tres','uno']
get_prec_multi(mat_cfn, labels_dig)


# Tenemos precisiones altas para la mayoría de los dígitos acá.

# Seguidamente, haremos algo similar pero haciendo bloques de $4 \times 4$. Para lograr esto usamos el mismo código empleado en la Tarea 3.

# In[70]:


def hacer_bloques(data,p):
    
    data_dim = int(math.sqrt(data.shape[1]))
    
    if data_dim % p == 0:
        
        q = int(data_dim / p)
        data_blocked = {}
        for i in range(q):
            for j in range(q):
                data_blocked[f'V{i}_{j}']=[]
        data_blocked = pd.DataFrame({})
        
        for r in range(data.shape[0]):
            ### Hacer un bloque de cada fila ###
            fila=data.iloc[r].tolist()
            x = np.array(fila)
            x = x.reshape(data_dim, data_dim)
            data_blocked_fila={}
            for i in range(q):
                for j in range(q):
                    
                    bloque=[]
                    for k in range(p):
                        bloque.append(x[p*i+k][p*j:p*(j+1)])
                    bloque=np.array([bloque])
                    
                    mean_bloque = np.mean(bloque)
                    data_blocked_fila[f'V_{i}_{j}']=[mean_bloque]
                    
            data_blocked_fila = pd.DataFrame(data_blocked_fila)
            data_blocked = data_blocked.append(data_blocked_fila, ignore_index=True)
                
        return data_blocked
    else:
        raise Exception("No se pueden hacer bloques de este tamaño")


# In[71]:


data_bloques=hacer_bloques(data = X_train, p = 4)
data_bloques.head()


# In[67]:


X_train_p4 = hacer_bloques(data = X_train, p = 4)
X_test_p4 = hacer_bloques(data = X_test, p = 4)
y_train_p4 = zipdata_train['Numero']
y_test_p4 = zipdata_test['Numero']


# Aqui no utilizaré datos estandarizados.

# In[73]:


nnet_p4 = MLPClassifier(
    hidden_layer_sizes = (250,100,50,25),
    max_iter = 50000,
    activation = 'relu',
    solver = 'adam', 
    random_state=0) #Parametros indicados en la tarea

nnet_p4.fit(X_train_p4.values, y_train_p4)
prediccion_p4 = nnet_p4.predict(X_test_p4.values)

mat_cfn_p4 = confusion_matrix(y_test, prediccion_p4)
mat_cfn_p4


# Esta es la matriz de confusión que obtenemos ahora con los datos agrupados en bloques de $4\times 4$. Tiene una precisión menor para varias clases.

# In[76]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(nnet_p4, X_test_p4, y_test)


# In[82]:


get_prec_multi(mat_cfn_p4, labels_dig)


# En general vemos que las precisiones por dígito bajaron en este caso.

# Ahora hacemos una agrupación por bloques, pero utilizando el máximo en cada bloque, veamos lo que sucede con la matriz de confusión.

# In[85]:


def hacer_bloques_max(data,p):
    
    data_dim = int(math.sqrt(data.shape[1]))
    
    if data_dim % p == 0:
        
        q = int(data_dim / p)
        data_blocked = {}
        for i in range(q):
            for j in range(q):
                data_blocked[f'V{i}_{j}']=[]
        data_blocked = pd.DataFrame({})
        
        for r in range(data.shape[0]):
            ### Hacer un bloque de cada fila ###
            fila=data.iloc[r].tolist()
            x = np.array(fila)
            x = x.reshape(data_dim, data_dim)
            data_blocked_fila={}
            for i in range(q):
                for j in range(q):
                    
                    bloque=[]
                    for k in range(p):
                        bloque.append(x[p*i+k][p*j:p*(j+1)])
                    bloque=np.array([bloque])
                    
                    mean_bloque = np.max(bloque)  ### AQUI CAMBIAMOS A USAR EL MAXIMO !!
                    data_blocked_fila[f'V_{i}_{j}']=[mean_bloque]
                    
            data_blocked_fila = pd.DataFrame(data_blocked_fila)
            data_blocked = data_blocked.append(data_blocked_fila, ignore_index=True)
                
        return data_blocked
    else:
        raise Exception("No se pueden hacer bloques de este tamaño")


# In[78]:


X_train_p4m = hacer_bloques_max(data = X_train, p = 4)
X_test_p4m = hacer_bloques_max(data = X_test, p = 4)
y_train_p4m = zipdata_train['Numero']
y_test_p4m = zipdata_test['Numero']


# In[79]:


nnet_p4m = MLPClassifier(
    hidden_layer_sizes = (250,100,50,25),
    max_iter = 50000,
    activation = 'relu',
    solver = 'adam', 
    random_state=0) #Parametros indicados en la tarea

nnet_p4m.fit(X_train_p4m.values, y_train_p4m)
prediccion_p4m = nnet_p4m.predict(X_test_p4m.values)

mat_cfn_p4m = confusion_matrix(y_test, prediccion_p4m)
mat_cfn_p4m


# In[84]:


get_prec_multi(mat_cfn_p4m, labels_dig)


# Aquí podemos observar que la agrupación en bloques de $4 \times 4$ tomando el máximo, en realidad bajó mucho más el desempeño del clasificador. Por eso nos quedamos con el original.

# <div class='question_container'>
#     <h2> Pregunta 4 </h2>
#     <p> Represente en un grafo dirigido la Red Neuronal que tiene la siguiente entrada:</p>
#     $$
#     x = \begin{pmatrix}
#     -2 \\
#     1 \\
#     1 \\
#     3
#     \end{pmatrix}.
#     $$
#     <p> Tiene las siguientes matrices de pesos </p>
#     $$
#     W_1 = \begin{pmatrix}
#     1 & 2 & 0 & 4 \\
#     5 & 3 & 1 & 2 \\
#     2 & 3 & 0 & 2
#     \end{pmatrix} \quad W_2 = \begin{pmatrix}
#     4 & 2 & 3 \\
#     1 & 3 & 6
#     \end{pmatrix},
#     $$
#     tiene el siguiente bias:
#     $$
#     b = \begin{pmatrix}
#     -6 \\
#     -2
#     \end{pmatrix}.
#     $$
#     Además use una función de activación tipo <strong> Tangente Hiperbólica </strong>, es decir:
#     $$
#     f(x) = \frac{2}{1+ e^{-2x}} + 1.
#     $$
# </div>

# La imagen del grafo dirigido para esta red neuronal se encuentra adjunta en la entrega de la tarea como el archivo `Tarea12_Jimmy_Calvo_grafo_dirigido.png`. 

# Justificación con cálculos.

# In[93]:


primera_capa = np.matmul(
    np.array([ [1,2,0,4], [5,3,1,2], [2,3,0,2] ]),
    np.array([ [-2], [1], [1], [3] ])
)

segunda_capa = np.matmul(
    np.array([ [4,2,3], [1,3,6] ]),
    primera_capa
) + np.array([ [-6], [-2] ])

print(primera_capa)
print(segunda_capa)


# In[100]:


2/(1+math.exp(-1*57)) + 1.0


# In[99]:


2/(1+math.exp(-1*40)) + 1.0


# <div class='question_container'>
#     <h2> Pregunta 5 </h2>
#     <p> [no usar <code>MLPClassifier</code>] Para la Tabla de Datos que se muestra seguidamente donde $x^j$ para $j = 1, 2, 3$ son las variables predictoras y la variable a predecir es $z$. Diseñe y programe a pie una Red Neuronal de una capa (Perceptron):</p>
#     <table style = "width:20%;">
#         <thead>
#         <tr>
#             <th> $x^1$ </th>
#             <th> $x^2$ </th>
#             <th> $x^3$ </th>
#             <th> $z$ </th>
#          </tr>
#         </thead>
#         <tbody>
#         <tr>
#             <td> 1 </td>
#             <td> 0 </td>
#             <td> 0 </td>
#             <td> 1 </td>
#          </tr>
#             <tr>
#             <td> 1 </td>
#             <td> 0 </td>
#             <td> 1 </td>
#             <td> 1 </td>
#          </tr>
#             <tr>
#             <td> 1 </td>
#             <td> 1 </td>
#             <td> 0 </td>
#             <td> 0 </td>
#          </tr>
#             <tr>
#             <td> 1 </td>
#             <td> 1 </td>
#             <td> 1 </td>
#             <td> 0 </td>
#          </tr>
#         </tbody>
#     </table>
#     <p> Es decir, encuentre todos los posibles pesos $w_1, w_2, w_3$ y umbrales $\theta$ para la Red Neuronal que se muestra en el siguiente gráfico (ver enunciado de la tarea).</p>
#     <p> Use una función de activación tipo <strong>Sigmoidea</strong>, es decir:</p>
#     $$
#     f(x) = \frac{1}{1 + e^{-x}}.
#     $$
#     <p> Para esto escriba una Clase en <code>Python</code> que incluya los métodos necesarios pra implementar esta Red Neuronal. </p>
#     <p> Se deben hacer variar los pesos $w_j$ con $j = 1,2,3$ en los siguientes valores $v=(-1,-0.9,-0.8,...,0,...,0.8,0.9,1)$ y haga variar $\theta$ en $u=(0,0.1,...,0.8,0.9,1)$. Escoja los pesos $w_j$ con $j = 1, 2, 3$ y el umbral $\theta$ de manera que se minimiza el error cuadrático medio: </p>
#     $$
#         E(w_1,w_2,w_3) = \frac{1}{4}\sum_{i=1}^4 \left[ I \left[ f\left( \sum_{j=1}^3 w_j\cdot x_i^j - \theta \right)\right] - z_i\right]^2
#     $$
#     <p> donde $x_i^j$ es la entrada en la fila $i$ de la variable $x^j$ e $I(t)$ se define como sigue:</p>
#     $$
#         I(t) = \begin{cases}
#         1 & \text{ si } t \geq \frac{1}{2} \\
#         0 & \text{ si } t < \frac{1}{2} \\
#         \end{cases}
#     $$
# </div>

# **Respuesta** Nuestra matriz tomará una matriz de tamaño $m \times n$ y un vector de respuestas $z$ para calcular este Perceptrón.

# In[76]:


class Perceptron:
    
    def __init__(self, X, z):
        
        self.X = X
        self.z = y
        self.m = X.shape[0]
        self.n = X.shape[1]
        
    def f(self,x):
        return 1/(1+math.exp(-1*x))
        
    def I(self,t):
        
        if t < 0.5:
            return 0
        else:
            return 1

    def evaluar_MSE(self, w, theta):
        
        MSE = 0
        
        for i in range(self.m):
            
            eval_pt = sum([w[j]*self.X[i,j] for j in range(len(w))]) - theta
            eval_val = self.f(eval_pt)
            MSE = MSE + (self.I(eval_val) - self.z[i])**2
        
        return MSE
    
    def encontrar_arquitectura(self):
        
        valores = pd.DataFrame({})
        
        ### Aquí creamos todas las combinaciones posibles para los valores de theta y los pesos w.
        v = [round(vt, 1) for vt in np.linspace(-1,1,21)]
        u = [round(ut, 1) for ut in np.linspace(0,1,11)]
        combs = [v]*self.n
        combs.append(u)
        all_combs = list(itertools.product(*combs))
        
        start = time.time()
        
        for combo in tqdm(all_combs):
            # Para cada combinacion, evaluamos el MSE, todo quedará en un dataframe.
            MSE_tupla = self.evaluar_MSE(w = combo[0:self.n], theta = combo[-1])
            valores_tupla = pd.DataFrame({
                'MSE': [MSE_tupla],
                'theta': [combo[-1]]
            })
            for r in range(self.n):
                valores_tupla[f'w_{r+1}'] = [combo[r]]
            
            valores = valores.append(valores_tupla, ignore_index = True)
            
        end = time.time()
        print(f"Buscar las arquitecturas que minimizan el MSE duró {end - start} segundos.")
                           
        self.resultados_grid = valores
        
    def __str__(self):
        return f"""
        
        X: {self.X}
        x: {self.z}
        
        Mejores valores para los pesos y el bias:
        
        {self.resultados_grid.sort_values(by=['MSE']).head(5)}
        
        =========================================
        
        """


# In[77]:


X = np.array([ [ 1, 0, 0] , [1, 0, 1], [1, 1, 0], [1, 1, 1] ])
z = np.array([1, 1, 0, 0])

mi_perceptron = Perceptron(X, z)
mi_perceptron.encontrar_arquitectura()
print(mi_perceptron.__str__())


# In[78]:


# Si usamos la primera fila del resultado que vemos arriba.
mi_perceptron.evaluar_MSE(w = [1,-0.9, 0], theta=0.6)


# Verificamos que obtuvimos el ajuste perfecto en este caso, porque obtuvimos un MSE de $0$. Para eso usamos la primera fila, que dió este MSE.

# In[79]:


prediccion = []
for i in range(X.shape[0]):
    val = [X[i,j] for j in range(X.shape[1])]
    evval = 1*val[0] -0.9*val[1] + 0*val[2] - 0.6
    sigm = 1/(1+math.exp(-1*evval))
    if sigm < 0.5:
        prediccion.append(0)
    else:
        prediccion.append(1)
prediccion


# Es igual al $z$ dado en el ejercicio. Con esto pudimos encontrar una arquitectura que se ajusta a este vector.
