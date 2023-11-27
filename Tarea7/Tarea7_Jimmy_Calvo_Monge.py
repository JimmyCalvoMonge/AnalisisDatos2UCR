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
#     <h4> TAREA 7 </h4>
#     <h4> Fecha de entrega: 9 de Octubre de 2022 </h4>
# </div>

# Importamos los módulos necesarios para resolver esta tarea.

# In[2]:


### Basicos
import numpy as np
import pandas as pd
from pandas import DataFrame

### Utilidades/Varios
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
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

### predictPy
from predictPy import Analisis_Predictivo

### Modelos:
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')


# In[3]:


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


# <div class='question_container'>
#     <h2> Pregunta 1 </h2>
#     <p> La tabla de datos novatosNBA.csv contiene diferentes métricas de desempeño de novatos de la NBA en su primera temporada. Para esta tabla, las 21 primeras columnas corresponden a las variables predictoras y la variable Permanencia es la variable a predecir, la cual indica si el jugador permanece en la NBA luego de 5 años. La tabla contiene 1340 filas (individuos) y 21 columnas (variables), con la tabla realice lo siguiente:</p>
#     <ul>
#         <li> Use Bayes en Python para generar un modelo predictivo para la tabla novatosNBA.csv usando el 80% de los datos para la tabla aprendizaje y un 20% para la tabla testing. Obtenga los índices de precisión e interprete los resultados.</li>
#         <li>Construya un DataFrame que compare el modelo generado en el ítem anterior contra los modelos vistos en las clases anteriores para la tabla novatosNBA.csv. Para esto en cada una de las filas debe aparecer un modelo predictivo y en las columnas aparezcan los índices Precisión Global, Error Global, Precisión Positiva (PP) y Precisión Negativa (PN). ¿Cuál de los modelos es mejor para estos datos?</li>
#     </ul>
# </div>

# In[8]:


datos_novatos=pd.read_csv("novatosNBA.csv",sep=";")
datos_novatos


# In[17]:


datos_novatos.dtypes


# In[22]:


### Hay datos faltantes:
nas_dict={}
for col in datos_novatos.columns:
    nas_dict[col]=datos_novatos[col].isna().sum()
nas_dict


# El modelo de Naive Bayes dará error si hay datos faltantes. Son 11 observaciones que tienen datos faltantes en la columna `Puntos3Porcentaje`. Por el momento serán eliminadas.

# In[23]:


datos_novatos=datos_novatos.dropna()


# Creamos un objeto de la clase `Analisis_Predictivo` y vemos la distribución de la variable a predecir. Notamos que se trata de un problema ligeramente desbalanceado.

# In[24]:


analisis_Novatos = Analisis_Predictivo(datos_novatos, predecir = "Permanencia")
analisis_Novatos.distribucion_variable_predecir()
plt.show()


# Ahora ajustamos un modelo de Naive Bayes sobre estos datos usando las especificaciones del ejercicio.

# In[25]:


# Usamos los parámetros por defecto
bayes = GaussianNB()

analisis_Novatos_pred = Analisis_Predictivo(
    datos_novatos,
    predecir = "Permanencia",
    modelo = bayes, 
    train_size = 0.8,
    random_state = 12
)


# In[26]:


resultados = analisis_Novatos_pred.fit_predict_resultados(imprimir=False)


# In[29]:


for indice in resultados:
    print(indice)
    print(resultados[indice])


# Hasta ahora no hemos usado esta tabla (novatosNBA.csv) con los modelos de las clases anteriores, o por lo menos no he podido encontrarla en el material, entonces voy a ajustar todos los modelos estudiados y haré la comparación final. Por simplicidad no haré la búsqueda de hiperparámetros para cada modelo, es decir efectuaré las comparaciones utilizando los parámetros por defecto.
# 
# Modelos a ajustar:
# 
# - KNN
# - Árboles de Decisión
# - Bosques Aleatorios
# - AdaBoost
# - XGBoost
# - SVM's
# - NaiveBayes

# In[30]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
# Arboles de Decision
from sklearn.tree import DecisionTreeClassifier
# Bosques Aleatorios
from sklearn.ensemble import RandomForestClassifier
# Ada Boost
from sklearn.ensemble import AdaBoostClassifier
# XG Boost
from sklearn.ensemble import GradientBoostingClassifier
# SVM
from sklearn.svm import SVC


# Para todos estos modelos vamos a crear los mismos datos train-test, los escalamos y mediremos las precisiones.

# In[32]:


# Variables Predictoras
X = datos_novatos.drop(['Permanencia'],axis=1)
# Variable a predecir
y = datos_novatos['Permanencia'].ravel()

#Partimos los datos en training-testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#Estandarizamos los datos para aplicar SVM
X_train_scaled=X_train.copy()
X_test_scaled=X_test.copy()

for col in X_train.columns:
    sd_col=np.std(X_train[col])
    mean_col=np.mean(X_train[col])
    X_train_scaled[col]=[(obs-mean_col)/sd_col for obs in X_train[col]]
    X_test_scaled[col]=[(obs-mean_col)/sd_col for obs in X_test[col]]


# In[39]:


modelos_ajustar=[]

instancia_knn = KNeighborsClassifier()
modelos_ajustar.append(instancia_knn)


instancia_arbol = DecisionTreeClassifier()
modelos_ajustar.append(instancia_arbol)


instancia_bosques = RandomForestClassifier()
modelos_ajustar.append(instancia_bosques)


instancia_tree = DecisionTreeClassifier(criterion="gini")
instancia_ada = AdaBoostClassifier(base_estimator=instancia_tree)
modelos_ajustar.append(instancia_ada)

instancia_xgb = GradientBoostingClassifier()
modelos_ajustar.append(instancia_xgb)

instancia_svm = SVC()
modelos_ajustar.append(instancia_svm)

instancia_bayes=GaussianNB()
modelos_ajustar.append(instancia_bayes)


# In[40]:


df_comp=pd.DataFrame({})

start=time.time()

for modelo in modelos_ajustar:
    
        modelo.fit(X_train_scaled.values,y_train)
        prediccion = modelo.predict(X_test_scaled.values)
        MC = confusion_matrix(y_test, prediccion, labels=list(np.unique(y_train)))
        medidas=MatConf(MC).dict_medidas
        df_este_modelo=pd.DataFrame({})
        for key in list(medidas.keys()):
            df_este_modelo[key]=[medidas[key]]
        df_comp= df_comp.append(df_este_modelo,ignore_index=True)
    
end=time.time()
print(f"Esta comparación de modelos tomó {end-start} segundos.")

df_comp.index=['KNN','Árbol de Decisión','Bosque Aleatorio','ADA Boost','XG Boost','SVM','Naive Bayes']


# In[41]:


df_comp=df_comp.sort_values(by=['Precisión Global'],ascending=False)
df_comp


# Logramos alcanzar una mejor precisión global utilizando el método de Máquinas de Vectores de Soporte, sin embargo Naive Bayes nos dió la mayor precisión en la clase minoritaria.

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
#         <li>Cargue en Python la tabla de datos diabetes.csv.</li>
#         <li> Use Bayes en Python para generar un modelo predictivo para la tabla diabetes.csv usando el 75% de los datos para la tabla aprendizaje y un 25% para la tabla testing, luego calcule para los datos de testing la matriz de confusión, la precisión global y la precisión para cada una de las dos categorías. ¿Son buenos los resultados? Explique.</li>
#         <li>Construya un DataFrame que compare el modelo generado en el ítem anterior contra los modelos vistos vistos en las clases anteriores para la tabla diabetes.csv. Para esto en cada una de las filas debe aparecer un modelo predictivo y en las columnas aparezcan los índices Precisión Global, Error Global, Precisión Positiva (PP) y Precisión Negativa (PN). ¿Cuál de los modelos es mejor para estos datos?</li>
#         <li>Repita el ítem 2, pero esta vez seleccione 6 variables predictoras ¿Mejora la predicción?</li>
#     </ul>
# </div>

# In[54]:


datos_diabetes=pd.read_csv("diabetes.csv",index_col=0)
datos_diabetes


# In[55]:


datos_diabetes.dtypes


# La variable `genero` es categórica, así que la convertiremos a dummy.

# In[56]:


#Convertimos a Dummy algunas de las variables predictoras
datos_diabetes_dum = pd.get_dummies(datos_diabetes, columns=['genero'])
datos_diabetes_dum.head(5)


# In[57]:


# Usamos los parámetros por defecto
bayes = GaussianNB()

analisis_diabetes_pred = Analisis_Predictivo(
    datos_diabetes_dum,
    predecir = "diabetes",
    modelo = bayes, 
    train_size = 0.75,
    random_state = 45
)


# In[58]:


resultados = analisis_diabetes_pred.fit_predict_resultados(imprimir=True)


# Tuvimos una alta precisión global, y una buena precisión en la clase de No Diabetes, sin embargo en la clase de Diabetes tenemos una precisión menor. Note que este es un problema desbalanceado, hay muy pocas personas con diabetes, lo cual puede explicar este fenómeno.

# In[59]:


analisis_diabetes_pred.distribucion_variable_predecir()
plt.show()


# Para este conjunto de datos sucede algo similar al ejercicio anterior, y es que no he visto dónde aplicamos los otros modelos al ejemplo de diabetes. Voy a ajustar los modelos estudiados antes y hacer la comparación.

# In[60]:


# Variables Predictoras
X = datos_diabetes.drop(['diabetes'],axis=1)
X = pd.get_dummies(X, columns=['genero'])

# Variable a predecir
y = datos_diabetes['diabetes'].ravel()

#Partimos los datos en training-testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#Estandarizamos los datos para aplicar SVM
X_train_scaled=X_train.copy()
X_test_scaled=X_test.copy()

for col in X_train.columns:
    sd_col=np.std(X_train[col])
    mean_col=np.mean(X_train[col])
    X_train_scaled[col]=[(obs-mean_col)/sd_col for obs in X_train[col]]
    X_test_scaled[col]=[(obs-mean_col)/sd_col for obs in X_test[col]]


# In[61]:


modelos_ajustar=[]

instancia_knn = KNeighborsClassifier()
modelos_ajustar.append(instancia_knn)


instancia_arbol = DecisionTreeClassifier()
modelos_ajustar.append(instancia_arbol)


instancia_bosques = RandomForestClassifier()
modelos_ajustar.append(instancia_bosques)


instancia_tree = DecisionTreeClassifier(criterion="gini")
instancia_ada = AdaBoostClassifier(base_estimator=instancia_tree)
modelos_ajustar.append(instancia_ada)

instancia_xgb = GradientBoostingClassifier()
modelos_ajustar.append(instancia_xgb)

instancia_svm = SVC()
modelos_ajustar.append(instancia_svm)

instancia_bayes=GaussianNB()
modelos_ajustar.append(instancia_bayes)


# In[68]:


df_comp=pd.DataFrame({})

start=time.time()

for modelo in modelos_ajustar:
    
        modelo.fit(X_train_scaled.values,y_train)
        prediccion = modelo.predict(X_test_scaled.values)
        MC = confusion_matrix(y_test, prediccion, labels=list(np.unique(y_train)))
        print(f"Matriz de confusión para {modelo}:")
        print(MC)
        medidas=MatConf(MC).dict_medidas
        df_este_modelo=pd.DataFrame({})
        for key in list(medidas.keys()):
            df_este_modelo[key]=[medidas[key]]
        df_comp= df_comp.append(df_este_modelo,ignore_index=True)
    
end=time.time()
print(f"Esta comparación de modelos tomó {end-start} segundos.")

df_comp.index=['KNN','Árbol de Decisión','Bosque Aleatorio','ADA Boost','XG Boost','SVM','Naive Bayes']


# In[69]:


df_comp=df_comp.sort_values(by=['Precisión Global'],ascending=False)
df_comp


# Otra vez, SVM nos da el resultado superior en precisión global y en este caso Naive Bayes fue el clasificador que peor se desempeñó de todos los clasificadores testeados.

# Finalmente, observamos si al hacer una selección de variables podemos mejorar el desempeño del clasificador de Naive Bayes. Voy a realizar esta selección de variables usando un bosque aleatorio.

# In[70]:


### Para Bosques Aleatorios ###
etiquetas=np.array(X_train_scaled.columns.tolist())

instancia_bosques = RandomForestClassifier()
instancia_bosques.fit(X_train_scaled.values,y_train)
importancia_bosques = np.array(instancia_bosques.feature_importances_)

orden = np.argsort(importancia_bosques)
importancia_bosques = importancia_bosques[orden]
etiquetas = etiquetas[orden]

fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
ax.barh(etiquetas, importancia_bosques)
plt.show()


# Para este modelo particular de bosques aleatorios, se obtuvo que las siguientes son las variables más importantes:
# 
# - `glucosa`
# - `prop_col_hdl`
# - `edad`
# - `IMC`
# - `colesterol`
# - `ps_sistolica`
# 
# Procedemos a realizar un clasificador de Naive Bayes con estas variables solamente.

# In[72]:


# Usamos los parámetros por defecto
bayes = GaussianNB()

analisis_diabetes_pred_6 = Analisis_Predictivo(
    datos_diabetes[['glucosa','prop_col_hdl','edad','IMC','colesterol','ps_sistolica','diabetes']],
    predecir = "diabetes",
    modelo = bayes, 
    train_size = 0.75,
    random_state = 45
)

resultados_6 = analisis_diabetes_pred_6.fit_predict_resultados()


# Observamos que la predicción sí mejoró ligeramente al sólo considerar estas variables.

# <div class='question_container'>
#     <h2> Pregunta 3 </h2>
#     <p>Para la siguiente tabla, la cual se vio en clase, suponga que se tiene una nueva fila o registro de la base de datos <code> t = (Isabel, F, 4, ?)</code>, prediga (a mano) si Isabel corresponde a la clase pequeño, mediano o alto.</p>
#     <table>
#       <tr>
#         <th>Nombre</th>
#         <th>Género</th>
#         <th>Altura</th>
#         <th>Clase</th>
#       </tr>
#         <tr>
#         <td>Kristina</td> <td>F</td> <td> 1</td> <td> P</td>
#         </tr>
#         <tr>
#          <td>Jim</td> <td>M</td> <td> 5</td> <td> A</td>
#         </tr>
#         <tr>
#          <td>Maggi</td> <td>F</td> <td> 4</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Martha</td> <td>F</td> <td> 4</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Stephanie</td> <td>F</td> <td> 2</td> <td> P</td>
#         </tr>
#         <tr>
#          <td>Bob</td> <td>M</td> <td> 4</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Kathy</td> <td>F</td> <td> 1</td> <td> P</td>
#         </tr>
#         <tr>
#          <td>Dave</td> <td>M</td> <td> 2</td> <td> P</td>
#         </tr>
#         <tr>
#          <td>Worth</td> <td>M</td> <td> 6</td> <td> A</td>
#         </tr>
#         <tr>
#          <td>Steven</td> <td>M</td> <td> 6</td> <td> A</td>
#         </tr>
#         <tr>
#          <td>Debbie</td> <td>F</td> <td> 3</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Todd</td> <td>M</td> <td> 5</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Kim</td> <td>F</td> <td> 5</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Amy</td> <td>F</td> <td> 3</td> <td> M</td>
#         </tr>
#         <tr>
#          <td>Wynette</td> <td>F</td> <td>3</td> <td> M</td>
#         </tr>
#     </table>
# </div>

# **Respuesta:** Lo que tenemos que calcular son las tres probabilidades condicionales:
# $$
# \begin{align*}
# P\bigl(\text{Clase = P} | X=(F,4)\bigr), \quad P\bigl(\text{Clase = M} | X=(F,4)\bigr) \quad \text{ y } \quad P\bigl(\text{Clase = A} | X=(F,4)\bigr)
# \end{align*}
# $$
# Para eso utilizamos la fórmula de Bayes que, en cada caso, nos dice que
# $$
# P(\text{Clase = P} | X=(F,4)) = \frac{P\bigl(X=(F,4) |\text{Clase = P} \bigr)P\bigl(\text{Clase = P}\bigr)}{P\bigl(X=(F,4) |\text{Clase = P} \bigr)P\bigl(\text{Clase = P}\bigr)+P\bigl(X=(F,4) |\text{Clase = M} \bigr)P\bigl(\text{Clase = M}\bigr)+P\bigl(X=(F,4) |\text{Clase = A} \bigr)P\bigl(\text{Clase = A}\bigr)} \quad (1)
# $$
# 
# $$
# P(\text{Clase = M} | X=(F,4)) = \frac{P\bigl(X=(F,4) |\text{Clase = M} \bigr)P\bigl(\text{Clase = M}\bigr)}{P\bigl(X=(F,4) |\text{Clase = P} \bigr)P\bigl(\text{Clase = P}\bigr)+P\bigl(X=(F,4) |\text{Clase = M} \bigr)P\bigl(\text{Clase = M}\bigr)+P\bigl(X=(F,4) |\text{Clase = A} \bigr)P\bigl(\text{Clase = A}\bigr)} \quad (2)
# $$
# 
# $$
# P(\text{Clase = A} | X=(F,4)) = \frac{P\bigl(X=(F,4) |\text{Clase = A} \bigr)P\bigl(\text{Clase = A}\bigr)}{P\bigl(X=(F,4) |\text{Clase = P} \bigr)P\bigl(\text{Clase = P}\bigr)+P\bigl(X=(F,4) |\text{Clase = M} \bigr)P\bigl(\text{Clase = M}\bigr)+P\bigl(X=(F,4) |\text{Clase = A} \bigr)P\bigl(\text{Clase = A}\bigr)} \quad (3)
# $$

# Primero calculamos las probabilidades absolutas de las clases:
# 
# $$
# \begin{align*}
# &P(\text{Clase=P})=\frac{4}{15}\\
# &P(\text{Clase=A})=\frac{3}{15}\\
# &P(\text{Clase=M})=\frac{8}{15}
# \end{align*}
# $$
# 
# Ahora tenemos que calcular las probabilidades condicionales:
# 
# $$
# \begin{align*}
# P\bigl(X=(F,4) |\text{Clase = P} \bigr) &=P\bigl(\text{Genero=F} |\text{Clase = P} \bigr) \cdot P\bigl(\text{Altura=4} |\text{Clase = P} \bigr)\\
# &= \frac{3}{4} \cdot \frac{0}{4}\\
# &=0.
# \end{align*}
# $$
# 
# $$
# \begin{align*}
# P\bigl(X=(F,4) |\text{Clase = M} \bigr) &=P\bigl(\text{Genero=F} |\text{Clase = M} \bigr) \cdot P\bigl(\text{Altura=4} |\text{Clase = M} \bigr)\\
# &= \frac{6}{8}\cdot \frac{3}{8} \\
# &= \frac{9}{4}.
# \end{align*}
# $$
# 
# $$
# \begin{align*}
# P\bigl(X=(F,4) |\text{Clase = A} \bigr) &=P\bigl(\text{Genero=F} |\text{Clase = A} \bigr) \cdot P\bigl(\text{Altura=4} |\text{Clase = A} \bigr)\\
# &=\frac{0}{3}\cdot\frac{0}{3} \\
# &=0.
# \end{align*}
# $$
# 
# De éstos cálculos es inmediato ver que la probabilidad máxima se alcanzará con la clase M, ya que es la única que no dará cero (de hecho dará igual a $1$). Así que ésta es la clase en la que se clasificará Isabel.

# <div class='question_container'>
#     <h2> Pregunta 4 </h2>
#     <p>Para la siguiente tabla, la cual se vio en clase, suponga que se tiene una nueva fila o registro <code>12 = (1, 3, 2, 4, ?)</code> en la base de datos, prediga (a mano) si el individuo corresponde a un buen pagador o a un mal pagador.</p>
#     <table>
#         <tr>
#             <th>Id</th>
#             <th>Monto.Crédito</th>
#             <th>Ingreso.Neto</th>
#             <th>Monto.Cuota</th>
#             <th>Grado.Academico</th>
#             <th>Buen.Pagador</th>
#         </tr>
#             <tr>
#             <td>1</td> <td>2</td> <td> 4</td> <td> 1</td> <td> 4</td> <td> Sí</td>
#             </tr>
#             <tr>
#             <td>2</td> <td> 2</td> <td> 3</td> <td> 1</td> <td> 4</td> <td> Sí</td>
#             </tr>
#             <tr>
#             <td>3</td> <td> 4</td> <td> 1</td> <td> 4</td> <td> 2</td> <td> No</td>
#             </tr>
#             <tr>
#             <td>4</td> <td> 1</td> <td> 4</td> <td> 1</td> <td> 4</td> <td> Sí</td>
#             </tr>
#             <tr>
#             <td>5</td> <td> 3</td> <td> 3</td> <td> 3</td> <td> 2</td> <td> No</td>
#             </tr>
#             <tr>
#             <td>6</td> <td> 3</td> <td> 4</td> <td> 1</td> <td> 4</td> <td> Sí</td>
#             </tr>
#             <tr>
#             <td>7</td> <td> 4</td> <td> 2</td> <td> 3</td> <td> 2</td> <td> No</td>
#             </tr>
#             <tr>
#             <td>8</td> <td> 4</td> <td> 1</td> <td> 3</td> <td> 2</td> <td> No</td>
#             </tr>
#             <tr>
#             <td>9</td> <td> 3</td> <td> 4</td> <td> 1</td> <td> 3</td> <td> Sí</td>
#             </tr>
#             <tr>
#             <td>10</td> <td> 1</td> <td> 3</td> <td> 2</td> <td> 4</td> <td> Sí</td>
#             </tr>
#             <tr>
#             <td>11</td> <td> 1</td> <td> 4</td> <td> 2</td> <td> 4</td> <td> Sí</td>
#             </tr>
#         </table>
# </div>

# **Respuesta:** Lo que tenemos que calcular son las dos probabilidades condicionales:
# $$
# \begin{align*}
# P\bigl(\text{Buen.Pagador=Si} | X=(1,3,2,4)\bigr)\quad \text{ y } \quad P\bigl(\text{Buen.Pagador=No} | X=(1,3,2,4)\bigr) 
# \end{align*}
# $$
# Para eso utilizamos la fórmula de Bayes que, en cada caso, nos dice que
# $$
# \begin{align*}
# &P(\text{Buen.Pagador=Si} | X=(1,3,2,4)) \\
# &= \frac{P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=Si} \bigr)P\bigl(\text{Buen.Pagador=Si}\bigr)}{P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=Si} \bigr)P\bigl(\text{Buen.Pagador=Si}\bigr)+P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=No} \bigr)P\bigl(\text{Buen.Pagador=No}\bigr)} \quad (1)
# \end{align*}
# $$
# 
# $$
# \begin{align*}
# &P(\text{Buen.Pagador=No} | X=(1,3,2,4))\\
# &= \frac{P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=No} \bigr)P\bigl(\text{Buen.Pagador=No}\bigr)}{P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=Si} \bigr)P\bigl(\text{Buen.Pagador=Si}\bigr)+P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=No} \bigr)P\bigl(\text{Buen.Pagador=No}\bigr)} \quad (2)
# \end{align*}
# $$

# Primero calculamos entonces las probabilidades absolutas de ambas clases:
# 
# $$
# P\bigl(\text{Buen.Pagador=Si}\bigr)=\frac{7}{11}.
# $$
# 
# $$
# P\bigl(\text{Buen.Pagador=No}\bigr)=\frac{4}{11}.
# $$
# 
# Ahora calculamos las otras probabilidades condicionales:
# 
# $$
# \begin{align*}
# &P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=Si} \bigr) = \\
# &P\bigl(X_1=1 |\text{Buen.Pagador=Si} \bigr)\cdot P\bigl(X_2=3 |\text{Buen.Pagador=Si} \bigr) \cdot P\bigl(X_3=2 |\text{Buen.Pagador=Si} \bigr) \cdot P\bigl(X_4=4 |\text{Buen.Pagador=Si} \bigr) \\
# &=\frac{3}{7}\cdot\frac{2}{7}\cdot\frac{2}{7}\cdot\frac{6}{7}\\
# &=\frac{72}{2401}.
# \end{align*}
# $$
# 
# Por otro lado:
# 
# $$
# \begin{align*}
# &P\bigl(X=(1,3,2,4) |\text{Buen.Pagador=No} \bigr) = \\
# &P\bigl(X_1=1 |\text{Buen.Pagador=No} \bigr)\cdot P\bigl(X_2=3 |\text{Buen.Pagador=No} \bigr) \cdot P\bigl(X_3=2 |\text{Buen.Pagador=No} \bigr) \cdot P\bigl(X_4=4 |\text{Buen.Pagador=No} \bigr) \\
# &=\frac{0}{7}\cdot\frac{1}{7}\cdot\frac{0}{7}\cdot\frac{0}{7}\\
# &=0
# \end{align*}
# $$
# 
# Otra vez sucede lo mismo que en el ejercicio anterior, en donde claramente al sustituir en las probabilidades tendremos que este individuo será clasificado como <code>Buen.Pagador=Si</code>.

# *Comentario:* Tengo entendido que algunas veces se usa el corrector de Laplace para enfrentar el problema de probabilidad cero en este tipo de clasificadores: [Laplace Smoothing in Naive Bayes](https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece#:~:text=Laplace%20smoothing%20is%20a%20smoothing%20technique%20that%20helps%20tackle%20the,the%20positive%20and%20negative%20reviews.), no sé si lo vimos en clase pero quería comentarlo. Gracias!
