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
#     <h4> TAREA 3 </h4>
#     <h4> Fecha de entrega: 11 de Setiembre de 2022 </h4>
# </div>

# Importamos los módulos necesarios para resolver esta tarea.

# In[108]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn
import math


# In[2]:


def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global     = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":   precision_global, 
            "Error Global":       error_global, 
            "Precisión por categoría":precision_categoria}

import matplotlib.ticker as mticker

def distribucion_variable_predecir(data:pd.DataFrame,variable_predict:str, ax = None):
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize = (15,10), dpi = 200)
    colors = list(dict(**mcolors.CSS4_COLORS))
    df = pd.crosstab(index = data[variable_predict],columns = "valor") / data[variable_predict].count()
    countv = 0
    titulo = "Distribución de la variable %s" % variable_predict
    
    for i in range(df.shape[0]):
        ax.barh(1, df.iloc[i], left = countv, align = 'center', color = colors[11 + i], label = df.iloc[i].name)
        countv = countv + df.iloc[i]
        
    ax.set_xlim(0,1)
    ax.set_yticklabels("")
    ax.set_ylabel(variable_predict)
    ax.set_title(titulo)
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
    
    countv = 0
    for v in df.iloc[:,0]:
        ax.text(np.mean([countv, countv + v]) - 0.03, 1 , '{:.1%}'.format(v), color = 'black', fontweight = 'bold')
        countv = countv + v
    ax.legend(loc = 'upper center', bbox_to_anchor = (1.08, 1), shadow = True, ncol = 1)
    
def poder_predictivo_categorica(data: pd.DataFrame, var: str, variable_predict: str, ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (15, 10), dpi = 200)
    df = pd.crosstab(index = data[var], columns = data[variable_predict])
    df = df.div(df.sum(axis = 1), axis = 0)
    titulo = "Distribución de la variable %s según la variable %s" % (var, variable_predict)
    df.plot(kind = 'barh', stacked = True,   legend = True, ax = ax,
            xlim = (0, 1), title   = titulo, width = 0.8)
            
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
    ax.legend(loc = 'upper center', bbox_to_anchor = (1.08, 1), 
              shadow = True, ncol = 1)
              
    for bars in ax.containers:
        plt.setp(bars, width = .9)
    for i in range(df.shape[0]):
        countv = 0
        for v in df.iloc[i]:
            ax.text(np.mean([countv, countv+v]) - 0.03, i,
                    '{:.1%}'.format(v), color = 'black', fontweight = 'bold')
            countv = countv + v
            
def poder_predictivo_numerica(data:pd.DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue = variable_predict, height = 8, aspect = 1.8).map(sns.kdeplot, var, shade = True).add_legend()


# <div class='question_container'>
#     <h2> Pregunta 1 </h2>
#     <p> Dada la siguiente Tabla de Testing de un Scoring de Crédito: </p>
#     <ol>
#         <li> Usando la columna BuenPagador en donde aparece el verdadero valor de la variable a predecir y la columna Predicción KNN en donde aparece la predicción del Método KNN para esta tabla de Testing, calcule la Matriz de Confusión. </li>
#         <li> Con la Matriz de Confusión anterior calcule 'a mano' la Precisión Global, el Error Global, la Precisión Positiva (PP), la Precisión Negativa (PN), la Proporción de Falsos Positivos (PFP), la Proporción de Falsos Negativos (PFN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). </li>  
#     </ol>
# </div>

# **Respuesta:** Voy a guardar las columnas `BuenPagador` y `PrediccionKNN` para calcular la matriz de confusión de este modelo. (Son las únicas columnas que se necesitan).

# In[3]:


Datos=pd.DataFrame({
    'BuenPagador':['Si']*19 + ['No']*6,
    'PrediccionKNN':['Si','No']+['Si']*6 + ['No'] + ['Si']*10 + ['No']*2+ ['Si']*2 + ['No']*2
})
Datos


# Podemos calcular la matriz de confusion en `Python` utilizando la biblioteca `sklearn`.

# In[4]:


mat_cfn = metrics.confusion_matrix(Datos['BuenPagador'], Datos['PrediccionKNN'])


# In[5]:


mat_cfn


# En efecto, revisamos que los cálculos son correctos:

# In[6]:


VP = Datos[(Datos['BuenPagador']=='Si') & (Datos['PrediccionKNN']=='Si')].shape[0]
VN = Datos[(Datos['BuenPagador']=='No') & (Datos['PrediccionKNN']=='No')].shape[0]
FP = Datos[(Datos['BuenPagador']=='No') & (Datos['PrediccionKNN']=='Si')].shape[0]
FN = Datos[(Datos['BuenPagador']=='Si') & (Datos['PrediccionKNN']=='No')].shape[0]

dict_prec = {
    'VP':VP,
    'VN':VN,
    'FP':FP,
    'FN':FN
}
print(dict_prec)


# Los cálculos que nos solicita el ejercicio entonces son los siguientes:

# In[7]:


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

print("Estos son los resultados para la pregunta 1:")

for key in list(dict_medidas.keys()):
    print(f" - {key}: {dict_medidas[key]}")


# <div class='question_container'>
#     <h2> Pregunta 2 </h2>
#     <p> Programe en lenguaje `Python` una clase que contenga un método que reciba como entrada la Matriz de Confusión (para el caso 2 $\times$ 2) que calcule y retorne en un diccionario: la Precisión Global, el Error Global, la Precisión Positiva (PP), la Precisión Negativa (PN), la Proporción de Falsos Positivos (PFP), la Proporción de Falsos Negativos (PFN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). </p>
#     <p> Supongamos que tenemos un modelo predictivo para detectar Fraude en Tarjetas de Crédito, la variable a predecir es Fraude con dos posibles valores Sí (para el caso en que sí fue fraude) y No (para el caso en que no fue fraude). Supongamos que la matriz de confusión es: </p>
#     <table>
#         <tr>
#             <th></th>
#             <th>No</th>
#             <th>Sí</th>
#         </tr>
#         <tr>
#             <td>No</td>
#             <td>892254</td>
#             <td>252</td>
#         </tr>
#         <tr>
#             <td>Sí</td>
#             <td>9993</td>
#             <td>270</td>
#         </tr>
#     </table>
#     <ul>
#         <li> Con ayuda de la clase programada anteriormente calcule la Precisión Global, el Error Global, la Precisión Positiva (PP), la Precisión Negativa (PN), la Proporción de Falsos Positivos (PFP), la Proporción de Falsos Negativos (PFN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). </li>
#         <li> ¿Es bueno o malo el modelo predictivo? Justifique su respuesta. </li>
#     </ul>
# </div>

# In[8]:


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


# In[9]:


cfn_mat =np.matrix([[892254,252],[9993,270]])
cfn_mat_obj = MatConf(cfn_mat)
print(cfn_mat_obj.__str__())


# ##### Observaciones: 
# 
# - Note que se trata de un problema no balanceado, con una gran cantidad de Negativos.
# - En mi opinión personal diría que este modelo es no es bueno. Claramente sobre los Negativos tenemos un muy buen comportamiento, pero esto se debe al desbalance de las clases. Sin embargo las notas bajan sobre los Positivos, vea que hay una alta proporción de falsos negativos, y la precisión negativa es muy baja, y la asertividad negativa es cercana al 50%.

# <div class='question_container'>
#     <h2> Pregunta 3 </h2>
#     <p> En este ejercicio usaremos la tabla de datos abandono clientes.csv, que contiene los detalles de los clientes de un banco. </p>
#     <p> La tabla contiene 11 columnas (variables), las cuales se explican a continuación. </p>
#     <ul>
#         <li> <span class='code_span'> CreditScore </span>: Indica el puntaje de crédito. </li>
#         <li> <span class='code_span'>Geography</span>: País al que pertenece.</li>
#         <li> <span class='code_span'>Gender</span>: Género del empleado.</li>
#         <li> <span class='code_span'>Age</span>: Edad del empleado.</li>
#         <li> <span class='code_span'>Tenure</span>: El tiempo del vínculo con la empresa.</li>
#         <li> <span class='code_span'>Balance</span>: La cantidad que les queda.</li>
#         <li> <span class='code_span'>NumOfProducts</span>: Los productos que posee.</li>
#         <li> <span class='code_span'>HasCrCard</span>: Tienen tarjeta de crédito o no.</li>
#         <li> <span class='code_span'>IsActiveMember</span>: Es un miembro activo o no.</li>
#         <li> <span class='code_span'>EstimatedSalary</span>: Salario estimado.</li>
#         <li> <span class='code_span'>Exited</span>: Indica si el cliente se queda o se va.</li>
#     </ul>
#     <p> Realice lo siguiente: </p>
#     <ol>
#         <li> Cargue en <span class='code_span'>Python</span> la tabla de datos <span class='code_span'>abandono_clientes.csv</span>.</li>
#         <li> ¿Es este problema equilibrado o desequilibrado? Justifique su respuesta.</li>
#         <li> Use el método de K vecinos más cercanos en <span class='code_span'>Python</span> para generar un modelo predictivo para la tabla abandono clientes.csv usando el 75% de los datos para la tabla aprendizaje y un 25% para la tabla testing. Intente con varios valores de K e indique cuál fue la mejor opción.</li>
#         <li> Genere un Modelo Predictivo usando K vecinos más cercanos para cada uno de los siguientes núcleos <span class='code_span'>ball_tree</span>, <span class='code_span'>kd_tree</span> y <span class='code_span'>brute</span> ¿Cuál produce los mejores resultados en el sentido de que predice mejor?</li>
#     </ol>
# </div>

# In[10]:


df_clientes = pd.read_csv("abandono_clientes.csv")
df_clientes = df_clientes.drop(['Unnamed: 0'], axis=1)
df_clientes


# In[11]:


df_clientes.dtypes


# Con el siguiente gráfico nos damos cuenta que en efecto estamos tratando con un problema moderadamente desbalanceado, ya que hay 80% de 'No's y un minoritario 20% de 'Si's.

# In[12]:


distribucion_variable_predecir(df_clientes,"Exited")
plt.show()


# Preparación de los datos

# In[13]:


# Convierte las variables a categórica
columnas_cat= [col for col in df_clientes.columns if str(df_clientes.dtypes[col]) =='object' and col!='Exited' ] ### Columnas predictivas y string
for col in columnas_cat:
    df_clientes[col] = df_clientes[col].astype('category')
    
# Variable a predecir
y = df_clientes["Exited"].ravel()

#Convertimos a Dummy algunas de las variables predictoras
X = pd.get_dummies(df_clientes.drop(columns=["Exited"]), columns=columnas_cat)

#Estandarizamos los datos
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=[X.columns])
X.head()


# Split de Train y Testing

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75)


# Para cada `k` entre 1 y 10 vamos a ajustar un modelo de `k` vecinos e imprimiremos la evaluación sobre los datos de prueba.

# In[15]:


for k in range(1,11):
    
    print(f" Modelo de k vecinos para k = {k} :")
    
    instancia_knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    instancia_knn.fit(X_train.values,y_train)
    
    prediccion = instancia_knn.predict(X_test.values)

    labels = ["No","Si"]
    MC = confusion_matrix(y_test, prediccion, labels=labels)

    indices = indices_general(MC,labels)
    for r in indices:
        print("\n%s:\n%s"%(k,str(indices[r])))
        
    print("----------------------------------------------")


# #### Observaciones:
# Note que la mayoría de modelos se comportan similar, y todos tienen una buena predicción sobre la clase del 'No'. Debido al desbalance de las clases esto es esperado. Para juzgarlos más bien deberíamos ver a la clase del 'Si', sobre la cual se genera una mejor precisión si $k=1$ (por lo menos para esta simulación y para esta partición de los datos).

# Intentemos ahora probar con los núcleos que nos da el enunciado. Para ver qué sucede haremos un rango de `k` y de los núcleos propuestos. Como el problema es desbalanceado voy a a calcular la precisión sobre los Si's para cada combinación.

# In[16]:


datos_nucleos = pd.DataFrame({'k':[],'nucleo':[],'precision_positivos':[]})
nucleos = ['auto','ball_tree', 'kd_tree', 'brute']

for k in range(1,11):
    for nucleo in nucleos:
        
        instancia_knn = KNeighborsClassifier(n_neighbors=k,algorithm=nucleo)
        instancia_knn.fit(X_train.values,y_train)
        
        prediccion = instancia_knn.predict(X_test.values)
        MC = confusion_matrix(y_test, prediccion, labels=labels)
        
        indices = indices_general(MC,labels)
        prec_pos = indices['Precisión por categoría']['Si'][0]
         
        datos_nucleos = datos_nucleos.append(pd.DataFrame({'k':[k],'nucleo':[nucleo],'precision_positivos':[prec_pos]}))
        
datos_nucleos


# Por alguna razón no estoy teniendo resultados diferentes al utilizar los núcleos. He revisado mi código y parece que todo está bien, pero apreciaría mucho una observación!

# Aquí una verificación con dos variables distintas.

# In[17]:


instancia_knn1 = KNeighborsClassifier(n_neighbors=5)
instancia_knn1.fit(X_train.values,y_train)
prediccion1 = instancia_knn1.predict(X_test.values)
MC1 = confusion_matrix(y_test, prediccion1, labels=labels)
indices1 = indices_general(MC1,labels)
print(indices1)


# In[18]:


instancia_knn2 = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
instancia_knn2.fit(X_train.values,y_train)
prediccion2 = instancia_knn2.predict(X_test.values)
MC2 = confusion_matrix(y_test, prediccion2, labels=labels)
indices2 = indices_general(MC2,labels)
print(indices2)


# Me da igual en ambos casos.

# <div class='question_container'>
#     <h2> Pregunta 4 </h2>
#     <p> En este ejercicio vamos a usar la tabla de datos <span class='code_span'>raisin.csv</span>, que contiene el resultado de un sistema de visión artificial para distinguir entre dos variedades diferentes de pasas (Kecimen y Besni) cultivadas en Turquía. Estas imágenes se sometieron a varios pasos de preprocesamiento y se realizaron 7 operaciones de extracción de características morfológicas utilizando técnicas de procesamiento de imágenes. </p>
#     <p>El conjunto de datos tiene 900 filas y 8 columnas las cuales se explican a continuación:</p>
#     <ul>
#         <li><span class='code_span'>Area</span> El número de píxeles dentro de los límites de la pasa. </li>
#         <li><span class='code_span'>MajorAxisLength</span> La longitud del eje principal, que es la línea más larga que se puede dibujar en la pasa. </li>
#         <li><span class='code_span'>MinorAxisLength</span> La longitud del eje pequeño, que es la línea más corta que se puededibujar en la pasa. </li>
#         <li><span class='code_span'>Eccentricityl</span> Una medida de la excentricidad de la elipse, que tiene los mismos momentos que las pasas. </li>
#         <li><span class='code_span'>ConvexArea</span> El número de píxeles de la capa convexa más pequeña de la región formada por la pasa. </li>
#         <li><span class='code_span'>Extent</span> La proporción de la región formada por la pasa al total de píxeles en el cuadro delimitador. </li>
#         <li><span class='code_span'>Perimeter</span> Mide el entorno calculando la distancia entre los límites de la pasa y los píxeles que la rodean. </li>
#         <li><span class='code_span'>Class</span> Tipo de pasa Kecimen y Besni (Variable a predecir). </li>
#     </ul>
#     <p> Realice lo siguiente: </p>
#     <ol>
#         <li> Cargue en <span class='code_span'>Python</span> la tabla de datos <span class='code_span'>raisin.csv</span>. </li>
#         <li> Realice un análisis exploratorio (estadísticas básicas) que incluya: el resumen numérico (media, desviación estándar, etc.), la correlación entre las variables, el poder predictivo de las variables predictoras. Interprete los resultados. </li>
#         <li> ¿Es este problema equilibrado o desequilibrado? Justifique su respuesta. </li>
#         <li> Use el método de K vecinos más cercanos en <span class='code_span'>Python</span> (con los parámetros por defecto) para generar un modelo predictivo para la tabla raisin.csv usando el 75% de los datos para la tabla aprendizaje y un 25% para la tabla testing, luego calcule para los datos de testing la matriz de confusión, la precisión global y la precisión para cada una de las dos categorías. ¿Son buenos los resultados? Explique. </li>
#         <li> Repita el item 4), pero esta vez, seleccione las 4 variables que, según su criterio, tienen mejor poder predictivo. </li>
#         <li> Usando la función programada en el ejercicio 1 y los modelos generados arriba, construya un DataFrame de manera que en cada una de las filas aparezca un modelo predictivo y en las columnas aparezcan los índices Precisión Global, Error Global, Precisión Positiva (PP), Precisión Negativa (PN), Falsos Positivos (FP), los Falsos Negativos (FN), la Asertividad Positiva (AP) y la Asertividad Negativa (AN). ¿Cuál de los modelos es mejor para estos datos? </li>
#         <li> Repita el item 4), pero esta vez en el método KNeighborsClassifier utilice los 3 diferentes algoritmos <span class='code_span'>ball_tree</span>, <span class='code_span'>kd_tree</span> y <span class='code_span'>brute</span>. ¿Cuál da mejores resultados? </li>
#     </ol>
# </div>

# In[19]:


df_raisin = pd.read_csv('raisin.csv')
df_raisin


# In[20]:


df_raisin.dtypes  ### Todas las variables predictivas son numéricas.


# ### Análisis Exploratorio de los datos

# In[21]:


### Distribución de la variable de respuesta:
### Es un problema BALANCEADO
distribucion_variable_predecir(df_raisin,"Class")
plt.show()


# In[22]:


### Calculamos los estadísticos básicos sobre cada columna numérica. Esto se puede hacer en pandas con un describe.
df_raisin.describe()


# Observemos que las escalas de los datos son diferentes para todas las variables.

# In[23]:


df_raisin_num = df_raisin.drop(['Class'],axis=1)
df_raisin_num.corr()


# In[24]:


plt.rcParams["figure.figsize"] = (10,10)  ### Cambiar las dimensiones de la figura


# In[27]:


sn.heatmap(df_raisin_num.corr(), annot=True)
plt.show()


# Cuidado! 
# - Hay algunas variables que tienen una alta correlación, vea por ejemplo `ConvexArea` y `Area`, o `MajorAxisLength` y `Perimeter`.

# **Veamos los gráficos de poder predictivo para darnos una idea de la relación de cada variable con la respuesta.**

# In[28]:


for col in df_raisin_num.columns:
    poder_predictivo_numerica(df_raisin,col,"Class")
    plt.show()


# Algunas variables que discriminan mejor las clases de la respuesta podrían ser `Area`, `MajorAxisLength`, `ConvexArea` y `Perimeter`. Una en particular que no parece tener mucho poder predictivo es `Extent`.

# #### Ajuste de KNN
# A continuación ajustaremos varios modelos de KNN, variando el `k` y observando el comportamiento sobre los datos de prueba.

# In[29]:


# Variable a predecir
y = df_raisin["Class"].ravel()

#Convertimos a Dummy algunas de las variables predictoras
X = df_raisin_num

#Estandarizamos los datos
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=[X.columns])

#Partimos los datos en training-testing
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75)

prec_globals = []

for k in range(1,11):
    
    print(f" Modelo de k vecinos para k = {k} :")
    
    instancia_knn = KNeighborsClassifier(n_neighbors=k)
    instancia_knn.fit(X_train.values,y_train)
    
    prediccion = instancia_knn.predict(X_test.values)

    labels = ["Kecimen","Besni"]
    MC = confusion_matrix(y_test, prediccion, labels=labels)

    indices = indices_general(MC,labels)
    prec_globals.append(indices['Precisión Global'])
    
    for r in indices:
        
        print("\n%s:\n%s"%(k,str(indices[r])))
        
    print("----------------------------------------------")


# Con `k=10` obtuvimos la mayor precisión global y la mayor precisión en cada clase. \
# Como es un problema balanceado, la precisión global es un criterio suficiente para evaluar al modelo. \
# Finalmente nuestra precisión global es de 0.8622222, y en cada clase la precisión está entre 0.8 y 0.9 aproximadamente. \
# Podríamos decir que es un modelo relativamente bueno 😀

# In[30]:


prec_globals


# In[230]:


max(prec_globals)


# Vamos a repetir este procedimiento, pero para ello vamos a escoger variables con el mejor poder predictivo. \
# De los gráficos anteriores podríamos proponer intuitivamente que estas variables son `Area`, `MajorAxisLength`, `ConvexArea` y `Perimeter`.

# In[31]:


### Solo seleccionamos las variables con alto poder predictivo.
### Lo ideal sería utilizar la misma partición del inciso anterior, para compararlos.

X_train2 = X_train[['Area', 'MajorAxisLength', 'ConvexArea', 'Perimeter']]
X_test2 = X_test[['Area', 'MajorAxisLength', 'ConvexArea', 'Perimeter']]
X_train2.head()


# In[32]:


prec_globals2 = []

for k in range(1,11):
    
    print(f" Modelo de k vecinos para k = {k} :")
    
    instancia_knn = KNeighborsClassifier(n_neighbors=k)
    instancia_knn.fit(X_train2.values,y_train)
    
    prediccion = instancia_knn.predict(X_test2.values)

    labels = ["Kecimen","Besni"]
    MC = confusion_matrix(y_test, prediccion, labels=labels)

    indices = indices_general(MC,labels)
    prec_globals2.append(indices['Precisión Global'])
    
    for r in indices:
        
        print("\n%s:\n%s"%(k,str(indices[r])))
        
    print("----------------------------------------------")


# In[33]:


prec_globals2


# In[34]:


max(prec_globals2)


# Al hacer esta selección de variables mejoramos un poco nuestra precisión global, ahora siendo lo óptimo seleccionar un `k=10`

# En lo que sigue calculamos el dataframe del inciso 6. Tendrá dos filas, una para cada modelo seleccionado de los dos incisos anteriores. En ambos lo óptimo fue seleccionar `k=5`.

# In[35]:


df_inc_6 = pd.DataFrame({})

### modelo inciso 4. 

instancia_knn1 = KNeighborsClassifier(n_neighbors=9)
instancia_knn1.fit(X_train.values,y_train)
prediccion1 = instancia_knn1.predict(X_test.values)
MC1 = confusion_matrix(y_test, prediccion1, labels=labels)
indices1 = indices_general(MC1,labels)

### Uso la clase del ejercicio 2.
medidas1= MatConf(indices1['Matriz de Confusión']).dict_medidas
df1=pd.DataFrame({'Modelo':['Inciso 4']})
for key in list(medidas1.keys()):
    df1[key]=medidas1[key]
df_inc_6 = df_inc_6.append(df1)

### modelo inciso 5.

instancia_knn2 = KNeighborsClassifier(n_neighbors=9)
instancia_knn2.fit(X_train2.values,y_train)
prediccion2 = instancia_knn2.predict(X_test2.values)
MC2 = confusion_matrix(y_test, prediccion2, labels=labels)
indices2 = indices_general(MC2,labels)

### Uso la clase del ejercicio 2.
medidas2= MatConf(indices2['Matriz de Confusión']).dict_medidas
df2=pd.DataFrame({'Modelo':['Inciso 5']})
for key in list(medidas2.keys()):
    df2[key]=medidas2[key]
df_inc_6 = df_inc_6.append(df2)

df_inc_6


# En general vemos que el segundo modelo se ha comportado mejor en todos los indicadores. Esto, claro, puede depender de la partición que hemos hecho. No se ha realizado validación cruzada (no estaba en los enunciados) para eliminar la incertidumbre de la partición.

# Finalmente, hacemos una comparación de los modelos con los tres núcleos posibles.

# In[36]:


datos_nucleos = pd.DataFrame({'k':[],'nucleo':[],'precision_global':[]})
nucleos = ['auto','ball_tree', 'kd_tree', 'brute']

for k in range(1,11):
    for nucleo in nucleos:
        
        instancia_knn = KNeighborsClassifier(n_neighbors=k,algorithm=nucleo)
        instancia_knn.fit(X_train.values,y_train)
        
        prediccion = instancia_knn.predict(X_test.values)
        MC = confusion_matrix(y_test, prediccion, labels=labels)
        
        indices = indices_general(MC,labels)
        prec = indices['Precisión Global']
         
        datos_nucleos = datos_nucleos.append(pd.DataFrame({'k':[k],'nucleo':[nucleo],'precision_global':[prec]}))
        
datos_nucleos


# <div class='question_container'>
#     <h2> Pregunta 5 </h2>
#     <p>En este ejercicio vamos a predecir números escritos a mano (Hand Written Digit Recognition), la tabla de aprendizaje está en el archivo `ZipDataTrainCod.csv` y la tabla de testing está en el archivo `ZipDataTestCod.csv`. En la figura siguiente se ilustran los datos: </p>
#     <p> Los datos de este ejemplo vienen de los códigos postales escritos a mano en sobres del correo postal de EE.UU. Las imágenes son de 16 $\times$ 16 en escala de grises, cada pixel va de intensidad de -1 a 1 (de blanco a negro). Las imágenes se han normalizado para tener aproximadamente el mismo tamaño y orientación. La tarea consiste en predecir, a partir de la matriz de 16 $\times$ 16 de intensidades de cada pixel, la identidad de cada imagen (0, 1, ...,  9) de forma rápida y precisa. Si es lo suficientemente precisa, el algoritmo resultante se utiliza como parte de un procedimiento de selección automática para sobres. Este es un problema de clasificación para el cual la tasa de error debe mantenerse muy baja para evitar la mala dirección de correo. La columna 1 tiene la variable a predecir Número codificada como sigue: 0='cero'; 1='uno'; 2='dos'; 3='tres'; 4='cuatro'; 5='cinco';6='seis'; 7='siete'; 8='ocho' y 9='nueve', las demás columnas son las variables predictivas, además cada fila de la tabla representa un bloque 16 $\times$ 16 por lo que la matriz tiene 256 variables predictoras. </p>
#     <ol>
#         <li>Usando K vecinos más cercanos genere un modelo predictivo para la tabla de aprendizaje, con los parámetros que usted estime más convenientes.</li>
#         <li>Con la tabla de testing calcule la matriz de confusión, la precisión global, el error global y la precisión en cada unos de los dígitos. ¿Son buenos los resultados?</li>
#         <li>Repita los items 1) y 2) pero usando solamente los 1s, 6s y los 9s. ¿Mejora la predicción?</li>
#         <li>Repita los items 1) y 2) utilizando n neighbors=5 y algorithm=`auto` (parámetros por defecto) pero reemplazando cada bloque 4 $\times$ 4 de píxeles por su promedio. ¿Mejora la predicción? Recuerde que cada bloque 16 $\times$ 16 está representado por una fila en las matrices de aprendizaje y testing. Despliegue la matriz de confusión resultante. La matriz de confusión obtenida debería ser: (Mostrada en la Tarea). No es necesario que las categorías se muestren en orden. </li>
#         <li>Repita los items 1) y 2) pero reemplazando cada bloque p$\times$p de píxeles por su promedio. ¿Mejora la predicción? (pruebe con algunos valores de p). Despliegue las matrices de confusión resultantes.</li>
#     </ol>
# </div>

# In[3]:


zipdata_train = pd.read_csv("ZipDataTrainCod.csv",sep=';')
zipdata_test = pd.read_csv("ZipDataTestCod.csv",sep=';')


# In[4]:


zipdata_train.head(3)


# In[5]:


zipdata_test.head(3)


# In[6]:


X_train = zipdata_train.drop(['Numero'],axis=1)
y_train = zipdata_train['Numero']
X_test = zipdata_test.drop(['Numero'],axis=1)
y_test = zipdata_test['Numero']


# In[7]:


instancia_knn = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
instancia_knn.fit(X_train.values,y_train)
prediccion = instancia_knn.predict(X_test.values)


# In[10]:


mat_cfn = confusion_matrix(y_test, prediccion)
mat_cfn


# Esta es la matriz de confusión para el modelo.

# In[12]:


plt.rcParams["figure.figsize"] = (10,10)  ### Cambiar las dimensiones de la figura


# In[13]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(instancia_knn, X_test, y_test)


# Para esta matriz de confusión vamos a calcular la precisión global, el error global y la precisión para cada uno de los dígitos

# In[39]:


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


# In[40]:


labels_dig=['cero','cinco','cuatro','dos','nueve','ocho','seis','siete','tres','uno']
get_prec_multi(mat_cfn, labels_dig)


# Inciso 3: solo con los 1's, 6's y 9's.

# In[30]:


zipdata_train3 = zipdata_train[zipdata_train['Numero'].isin(['uno','seis','nueve'])]
zipdata_test3 = zipdata_test[zipdata_test['Numero'].isin(['uno','seis','nueve'])]
zipdata_train3.head(10)


# In[31]:


X_train3 = zipdata_train3.drop(['Numero'],axis=1)
y_train3 = zipdata_train3['Numero']
X_test3 = zipdata_test3.drop(['Numero'],axis=1)
y_test3 = zipdata_test3['Numero']


# In[32]:


instancia_knn3 = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
instancia_knn3.fit(X_train3.values,y_train3)
prediccion3 = instancia_knn3.predict(X_test3.values)


# In[33]:


mat_cfn3 = confusion_matrix(y_test3, prediccion3)
mat_cfn3


# In[41]:


plot_confusion_matrix(instancia_knn3, X_test3, y_test3)


# In[43]:


get_prec_multi(mat_cfn3, labels=['nueve','seis','uno'])


# En este caso vemos que la predicción ha mejorado, por ejemplo, los 'nueve' han sido predichos perfectamente.

# Ahora queremos hacer bloques de $4\times 4$ para correr el algoritmo de knn sobre estos nuevos tipos de datos. Lo que debemos hacer es reagrupar los datos y crear nuevas variables, que sean los promedios de los valores de los pixeles en cada bloque.

# In[98]:


primer_numero=X_train.iloc[0].tolist()
x = np.array(primer_numero)
x = x.reshape(16, 16)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(x)
plt.show()


# Lo que sucede acá es que cada fila del dataframe, es la concatenación de las 16 filas que conforman al número.
# Para agrupar por bloques de $4 \times 4$ debemos hacer un recorrido especial. Para ello utilizaré la función que transforma una fila en una matriz (El `reshape` de numpy, y luego crearé los bloques, todo está en esta función que sigue).

# In[172]:


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


# Hagámoslo con bloques de $4 \times 4$

# In[173]:


data_bloques=hacer_bloques(data=X_train,p=4)
data_bloques.head()


# In[174]:


data_bloques.shape


# In[161]:


primer_numero=data_bloques.iloc[0].tolist()
x = np.array(primer_numero)
x = x.reshape(4, 4)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(x)
plt.show()


# Parece que los bloques de 4 no son muy informativos. Vamos a ver qué sucede con el clasificador en este caso.

# In[182]:


X_train_p4= hacer_bloques(data=X_train,p=4)
X_test_p4= hacer_bloques(data=X_test,p=4)
y_train_p4= zipdata_train['Numero']
y_test_p4= zipdata_test['Numero']


# In[183]:


X_train_p4


# In[184]:


instancia_knn_p4 = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
instancia_knn_p4.fit(X_train_p4.values,y_train_p4)
prediccion_p4 = instancia_knn_p4.predict(X_test_p4.values)


# In[185]:


mat_cfn_p4 = confusion_matrix(y_test_p4, prediccion_p4)
mat_cfn_p4


# In[186]:


plot_confusion_matrix(instancia_knn_p4, X_test_p4, y_test_p4)


# Observe que es la misma matriz que se muestra en la tarea, con un orden distinto de los dígitos. Veamos la predicción:

# In[187]:


get_prec_multi(mat_cfn_p4, labels_dig)


# Como lo suponíamos, la precisión global ha disminuido. Algo intermedio se puede obtener con otro p.

# Ahora intentemos con bloques de $2\times 2$. Estos se deberían de ver así:

# In[189]:


X_train_p2= hacer_bloques(data=X_train,p=2)
X_test_p2= hacer_bloques(data=X_test,p=2)
y_train_p2= zipdata_train['Numero']
y_test_p2= zipdata_test['Numero']


# In[190]:


X_train_p2.head()


# In[191]:


instancia_knn_p2 = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
instancia_knn_p2.fit(X_train_p2.values,y_train_p2)
prediccion_p2 = instancia_knn_p2.predict(X_test_p2.values)


# In[192]:


mat_cfn_p2 = confusion_matrix(y_test_p2, prediccion_p2)
mat_cfn_p2


# In[193]:


plot_confusion_matrix(instancia_knn_p2, X_test_p2, y_test_p2)


# In[194]:


get_prec_multi(mat_cfn_p2, labels_dig)


# Tenemos una precisión parecida a la inicial. Vea que aquí los dígitos tienen una forma más definida.

# In[195]:


primer_numero=X_train_p2.iloc[0].tolist()
x = np.array(primer_numero)
x = x.reshape(8, 8)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(x)
plt.show()

