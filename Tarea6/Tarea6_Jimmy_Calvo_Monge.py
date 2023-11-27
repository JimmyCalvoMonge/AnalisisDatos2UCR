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
#     <h4> TAREA 6 </h4>
#     <h4> Fecha de entrega: 2 de Octubre de 2022 </h4>
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
from sklearn.tree import export_graphviz
from sklearn import tree
import seaborn as sns
import time
import graphviz

### Training/Testing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

### predictPy
from predictPy import Analisis_Predictivo

### Modelos:
from sklearn.svm import SVC


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
#     <p> En este ejercicio vamos a usar la tabla de datos <span class='code_span'>raisin.csv</span>, que contiene el resultado de un sistema de visión artificial para distinguir entre dos variedades diferentes de pasas (Kecimen y Besni) cultivadas en Turquía. Estas imágenes se sometieron a varios pasos de preprocesamiento y se realizaron 7 operaciones de extracción de características morfológicas utilizando técnicas de procesamiento de imágenes. </p>
#     <p>El conjunto de datos tiene 900 filas y 8 columnas las cuales se explican a continuación:</p>
#     <ul>
#         <li><span class='code_span'>Area</span> El número de píxeles dentro de los límites de la pasa. </li>
#         <li><span class='code_span'>MajorAxisLength</span> La longitud del eje principal, que es la línea más larga que se puede dibujar en la pasa. </li>
#         <li><span class='code_span'>MinorAxisLength</span> La longitud del eje pequeño, que es la línea más corta que se puededibujar en la pasa. </li>
#         <li><span class='code_span'>Eccentricity</span> Una medida de la excentricidad de la elipse, que tiene los mismos momentos que las pasas. </li>
#         <li><span class='code_span'>ConvexArea</span> El número de píxeles de la capa convexa más pequeña de la región formada por la pasa. </li>
#         <li><span class='code_span'>Extent</span> La proporción de la región formada por la pasa al total de píxeles en el cuadro delimitador. </li>
#         <li><span class='code_span'>Perimeter</span> Mide el entorno calculando la distancia entre los límites de la pasa y los píxeles que la rodean. </li>
#         <li><span class='code_span'>Class</span> Tipo de pasa Kecimen y Besni (Variable a predecir). </li>
#     </ul>
#     <p> Realice lo siguiente: </p>
#     <ol>
#         <li> Use Máquinas de Soporte Vectorial en Python para generar un modelo predictivo para la tabla <code>raisin.csv</code usando el 80% de los datos para la tabla aprendizaje y un 20% para la tabla testing. Obtenga los índices de precisión e interprete los resultados.<li>
#         <li> Repita el ítem anterior pero intente identificar el mejor núcleo (Kernel) y valor para el parámetro de regularización C. ¿Mejora la predicción? </li>
#         <li> Construya un <span class='code_span'>DataFrame</span> que compare los modelos construidos arriba con los mejores modelos construidos en tareas anteriores para la tabla <span class='code_span'>raisin.csv</span>. Para esto en cada una de las filas debe aparecer un modelo predictivo y en las columnas aparezcan los índices Precisión Global, Error Global, Precisión Positiva (PP) y Precisión Negativa (PN). ¿Cuál de los modelos es mejor para estos datos? Guarde los datos de este DataFrame, ya que se irá modificando en próximas tareas.</li>
#     </ol>
# </div>

# In[12]:


### Leer datos
df_raisin = pd.read_csv('raisin.csv')
### Training-Testing

# Variable a predecir
y = df_raisin["Class"].ravel()

#Convertimos a Dummy algunas de las variables predictoras
df_raisin_num = df_raisin.drop(['Class'],axis=1)
X = df_raisin_num

#Partimos los datos en training-testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

#Estandarizamos los datos para aplicar SVM
X_train_scaled=X_train.copy()
X_test_scaled=X_test.copy()

for col in X_train.columns:
    sd_col=np.std(X_train[col])
    mean_col=np.mean(X_train[col])
    X_train_scaled[col]=[(obs-mean_col)/sd_col for obs in X_train[col]]
    X_test_scaled[col]=[(obs-mean_col)/sd_col for obs in X_test[col]]


# Primero ajustamos una SVM con los parámetros por defecto. Medimos la precisión en los datos de prueba.

# In[13]:


instancia_svm = SVC()
analisis_raisin = Analisis_Predictivo(df_raisin, predecir= "Class", modelo=instancia_svm, train_size= 0.75)


# In[14]:


resultados = analisis_raisin.fit_predict_resultados()


# Tenemos resultados similares a los de las tareas anteriores. Una precisión global del 0.86222, con relativamente buenas precisiones en cada clase. Ahora, para tratar de identificar el mejor núcleo y parámetro de regularización C vamos a hacer una malla de hiperparámetros y tratar de obtener el modelo con la mejor precisión global.

# In[15]:


df_svm=pd.DataFrame({})

Cs=np.linspace(0.01,50,100)
kernels=['linear', 'poly', 'rbf', 'sigmoid']

start=time.time()
for kernel in kernels:
    for c in Cs:
        instancia_svm = SVC(C=c,kernel=kernel)
        instancia_svm.fit(X_train_scaled.values,y_train)
        prediccion_svm = instancia_svm.predict(X_test_scaled.values)
        MC_svm = confusion_matrix(y_test, prediccion_svm, labels=list(np.unique(y_train)))
        medidas_este_svm=MatConf(MC_svm).dict_medidas
        df_este_svm=pd.DataFrame({'kernel':[kernel], 'C':[c]})
        for key in list(medidas_este_svm.keys()):
            df_este_svm[key]=[medidas_este_svm[key]]
        df_svm= df_svm.append(df_este_svm,ignore_index=True)
    
end=time.time()
print(f"Esta búsqueda de hiperparámetros para SVM's tomó {end-start} segundos.")


# In[16]:


df_svm=df_svm.sort_values(by=['Precisión Global'],ascending=False)
df_svm.head(1)


# Note que con este modelo logramos incrementar nuestras precisiones, así que la predicción mejoró en general. Comparamos con la Tarea Pasada.

# En la tarea pasada teníamos los siguientes resultados:

# In[17]:


comparacion_T5=pd.DataFrame({
    'Precisión Global': [0.875556,0.861111,0.866667,0.840000,0.888889,0.861111],
    'Error Global':[0.124444,0.138889,0.133333,0.160000,0.111111,0.138889],
    'Precisión Positiva (PP)':[0.842975,0.908163,0.957265,0.888889,0.948718,0.845238],
    'Precisión Negativa (PN)':[0.913462,0.804878,0.768519,0.787037,0.824074,0.875000],
    'Proporción de Falsos Positivos (PFP)':[0.086538,0.195122,0.231481,0.212963,0.175926,0.125000],
    'Proporción de Falsos Negativos (PFN)':[0.157025,0.091837,0.042735,0.111111,0.051282,0.154762],
    'Asertividad Positiva (AP)':[0.918919,0.847619,0.817518,0.818898,0.853846,0.855422],
    'Asertividad Negativa (AN)':[0.833333,0.880000,0.943182,0.867347,0.936842,0.865979]
})

comparacion_T6=comparacion_T5.append(df_svm.head(1).drop(['kernel','C'],axis=1),ignore_index=True)
comparacion_T6.index=['KNN','Árbol Decisión','Bosque Aleatorio','ADA Boost','XG Boost','Consenso Propio','SVM']
print("Los resultados finales de todos los modelos fueron: ")
comparacion_T6


# In[18]:


comparacion_T6.to_csv("comparacion_T6_raisin.csv",index=False)


# Observación: Hasta ahora, la mayor precisión global se ha obtenido con el modelo de SVM. Sin embargo este modelo no da el mejor resultado en cada categoría. La precisión positiva alcanzó su máximo con un Bosque Aleatorio y la precisión negativa en un KNN. Esto, repito, se puede deber a la variación en los testing training sets que hacemos en cada tarea, pero da una mejor idea de la comparación entre cada modelo. Una forma para reducir la variación del testint-training split y la semilla aleatoria que se elija puede ser utilzar validación cruzada en cada ajuste.

# <div class='question_container'>
#     <h2> Pregunta 2 </h2>
#     <p> En este ejercicio usaremos la tabla de datos <code>abandono_clientes.csv</code>, que contiene los detalles de los clientes de un banco. </p>
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
#         <li> Cargue en <span class='code_span'>Python</span> la tabla de datos <span class='code_span'>abandono_clientes.csv</span>. </li>
#         <li> Use Máquinas de Soporte Vectorial en Python (con los parámetros por defecto) para generar un modelo predictivo para la tabla abandono_clientes.csv usando el 75% de los datos para la tabla aprendizaje y un 25% para la tabla testing, luego calcule para los datos de testing la matriz de confusión, la precisión global y la precisión para cada una de las dos categorías. ¿Son buenos los resultados? Explique. </li>
#         <li> Repita el ítem anterior pero intente identificar el mejor núcleo (Kernel) y valor para el parámetro de regularización C. ¿Mejora la predicción?. </li>
#         <li> Con los mejores parámetros identificados en el ítem anterior realice un nuevo modelo pero haciendo selección de 6 variables. >Mejoran los resultados?</li>
#         <li> Construya un <span class='code_span'>DataFrame</span> que compare los mejores modelos construidos arriba con los mejores modelos generados en tareas anteriores para la tabla <span class='code_span'>abandono_clientes.csv</span>. Para esto en cada una de las filas debe aparecer un modelo predictivo y en las columnas aparezcan los índices Precisión Global, Error Global, Precisión Positiva (PP) y Precisión Negativa (PN). ¿Cuál de los modelos es mejor para estos datos? Guarde los datos de este DataFrame, ya que se irá modificando en próximas tareas. </li>
#         <li> Utilizando el mejor modelo construido prediga los nuevos individuos que se encuentran en el archivo nuevos abandono clientes.csv. Recuerde que si estandarizó los datos para entrenar el modelo debe guardar valores como la media y desviación estándar para estandarizar los nuevos individuos.</li>
#     </ol>
# </div>

# In[6]:


# Leemos los datos

df_clientes = pd.read_csv("abandono_clientes.csv")
df_clientes = df_clientes.drop(['Unnamed: 0'], axis=1)

# Convierte las variables a categórica
columnas_cat= [col for col in df_clientes.columns if str(df_clientes.dtypes[col]) =='object' and col!='Exited' ] ### Columnas predictivas y string
for col in columnas_cat:
    df_clientes[col] = df_clientes[col].astype('category')
    
# Variable a predecir
y = df_clientes["Exited"].ravel()

#Convertimos a Dummy algunas de las variables predictoras
X = pd.get_dummies(df_clientes.drop(columns=["Exited"]), columns=columnas_cat)

X.head()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)


# In[8]:


X_train.head()


# In[9]:


medias_train={}
sds_train={}
for col in X_train.columns:
    medias_train[col]=np.mean(X_train[col])
    sds_train[col]=np.std(X_train[col])
    
#Estandarizamos los datos para aplicar SVM
X_train_scaled=X_train.copy()
X_test_scaled=X_test.copy()

for col in X_train.columns:
    X_train_scaled[col]=[(obs-medias_train[col])/sds_train[col] for obs in X_train[col]]
    X_test_scaled[col]=[(obs-medias_train[col])/sds_train[col] for obs in X_test[col]]


# In[10]:


X_train_scaled.head(5)


# Ajustamos y medimos la precisión de una SVM con los parámetros por defecto.

# In[11]:


### Con los parámetros por defecto.
instancia_svm = SVC()
instancia_svm.fit(X_train_scaled.values,y_train)
prediccion_svm = instancia_svm.predict(X_test_scaled.values)
MC_svm = confusion_matrix(y_test, prediccion_svm, labels=list(np.unique(y_train)))
medidas_svm=MatConf(MC_svm).dict_medidas
medidas_svm


# In[12]:


MC_svm


# Ahora, tratamos de buscar una combinación de hiperparámetros que nos de una mejor precisión global. Note que el problema es no balanceado así que la precisión global puede no significar un buen modelo, como vimos en tareas anteriores.

# In[13]:


df_svm=pd.DataFrame({})

Cs=np.linspace(1,20,15)  ### No podemos elevar mucho el C ya que sino la búsqueda tomaría más tiempo.
kernels=['linear', 'poly', 'rbf', 'sigmoid']

start=time.time()
for kernel in kernels:
    for c in Cs:
        
        instancia_svm = SVC(C=c,kernel=kernel)
        instancia_svm.fit(X_train_scaled.values,y_train)
        prediccion_svm = instancia_svm.predict(X_test_scaled.values)
        MC_svm = confusion_matrix(y_test, prediccion_svm, labels=list(np.unique(y_train)))
        
        ### Ver si no tenemos una columna de ceros
        if sum(MC_svm[:,1])!=0 and sum(MC_svm[:,0])!=0:
            medidas_este_svm=MatConf(MC_svm).dict_medidas
            df_este_svm=pd.DataFrame({'kernel':[kernel], 'C':[c]})
            for key in list(medidas_este_svm.keys()):
                df_este_svm[key]=[medidas_este_svm[key]]
            df_svm= df_svm.append(df_este_svm,ignore_index=True)
    
end=time.time()
print(f"Esta búsqueda de hiperparámetros para SVM's tomó {end-start} segundos.")


# In[14]:


df_svm=df_svm.sort_values(by=['Precisión Global'],ascending=False)
df_svm.head(5)


# Note que elevamos un poco la precisión global al cambiar el hiperparámetro de regularización. Utilicemos el mejor modelo del ítem anterior para hacer una selección de variables utilizando SVM's. En la tarea pasada, habíamos seleccionado las siguientes variables utilizando un modelo de bosques aleatorios.

# In[15]:


vars_select=['Age','NumOfProducts','Balance','CreditScore','EstimatedSalary','Tenure']

### Creamos datasets solo con esas 6 variables:
X_train_scaled_6var=X_train_scaled[vars_select]
X_test_scaled_6var=X_test_scaled[vars_select]


# In[16]:


X_train_scaled_6var.head(2)


# In[17]:


X_test_scaled_6var.head(2)


# In[18]:


df_svm_6var=pd.DataFrame({})

Cs=np.linspace(1,20,15)  ### No podemos elevar mucho el C ya que sino la búsqueda tomaría más tiempo.
kernels=['linear', 'poly', 'rbf', 'sigmoid']

start=time.time()
for kernel in kernels:
    for c in Cs:
        
        instancia_svm = SVC(C=c,kernel=kernel)
        instancia_svm.fit(X_train_scaled_6var.values,y_train)
        prediccion_svm = instancia_svm.predict(X_test_scaled_6var.values)
        MC_svm = confusion_matrix(y_test, prediccion_svm, labels=list(np.unique(y_train)))
        
        ### Ver si no tenemos una columna de ceros
        if sum(MC_svm[:,1])!=0 and sum(MC_svm[:,0])!=0:
            medidas_este_svm=MatConf(MC_svm).dict_medidas
            df_este_svm=pd.DataFrame({'kernel':[kernel], 'C':[c]})
            for key in list(medidas_este_svm.keys()):
                df_este_svm[key]=[medidas_este_svm[key]]
            df_svm_6var= df_svm_6var.append(df_este_svm,ignore_index=True)
    
end=time.time()
print(f"Esta búsqueda de hiperparámetros para SVM's tomó {end-start} segundos.")

df_svm_6var=df_svm_6var.sort_values(by=['Precisión Global'],ascending=False)
df_svm_6var.head(5)


# Vemos que en este caso la predicción no mejoró al reducir a 6 variables.

# Ahora guardamos los resultados del mejor modelo que obtuvimos acá y lo comparamos con los de la tarea anterior. En esa tarea habíamos obtenido los siguientes resultados:

# In[19]:


comparacion_T4=pd.DataFrame({
    'Precisión Global':	[0.830116,0.861004],
    'Error Global':	[0.169884,0.138996],
    'Precisión Positiva (PP)': [0.231939,0.323770],
    'Precisión Negativa (PN)': [0.982558,0.985728],
    'Proporción de Falsos Positivos (PFP)': [0.017442,0.014272],
    'Proporción de Falsos Negativos (PFN)': [0.768061,0.676230],
    'Asertividad Positiva (AP)': [0.772151,0.840426],
    'Asertividad Negativa (AN)': [0.833882,0.862614],
})
comparacion_T4.index=['Mejor KNN','Mejor AD']
comparacion_T5=pd.read_csv("Comp_Abandono_Clientes_T5.csv",index_col=0)
comparacion_T4=comparacion_T4.append(comparacion_T5)


df_svm_best=df_svm.head(1).drop(['C','kernel'],axis=1)
df_svm_best.index=['Mejor SVM']
comparacion_T6=comparacion_T4.append(df_svm_best)

comparacion_T6.to_csv("Comparacion_Abandono_Clientes_T6.csv")

comparacion_T6


# Vemos que la mejor precisión global, hasta ahora, la hemos obtenido con un XG Boost para este conjunto de datos. Las predicciones han sido buenas para la clase negativa, pero en general no muy buenas para la clase positiva, debido al desbalance de los datos.

# Finalmente, vamos a aplicar una predicción sobre los individuos nuevos que hemos recibido para esta tarea.

# In[20]:


df_clientes_nuevo=pd.read_csv("nuevos_abandono_clientes_V2.csv",index_col=0)
df_clientes_nuevo.head(5)


# In[32]:


# Creamos las variables dummy:

# Convierte las variables a categórica
columnas_cat= [col for col in df_clientes_nuevo.columns if str(df_clientes_nuevo.dtypes[col]) =='object' and col!='Exited' ] ### Columnas predictivas y string
for col in columnas_cat:
    df_clientes_nuevo[col] = df_clientes_nuevo[col].astype('category')
    
# Variable a predecir
y_nuevo = df_clientes_nuevo["Exited"].ravel()

#Convertimos a Dummy algunas de las variables predictoras
X_nuevo = pd.get_dummies(df_clientes_nuevo.drop(columns=["Exited"]), columns=columnas_cat)

X_nuevo.head()


# In[25]:


### Escalar los datos nuevos

X_nuevo_scaled=X_nuevo.copy()
for col in X_train.columns:
    X_nuevo_scaled[col]=[(obs-medias_train[col])/sds_train[col] for obs in X_nuevo[col]]


# In[28]:


### Ajustamos el mejor modelo seleccionado.

instancia_svm = SVC(C=df_svm['C'].tolist()[0],kernel=df_svm['kernel'].tolist()[0])
instancia_svm.fit(X_train_scaled.values,y_train)
prediccion_svm = instancia_svm.predict(X_nuevo_scaled.values)
MC_svm = confusion_matrix(y_nuevo, prediccion_svm, labels=list(np.unique(y_train)))
medidas_este_svm=MatConf(MC_svm).dict_medidas
medidas_este_svm


# In[29]:


MC_svm


# In[30]:


prediccion_svm


# In[31]:


y_nuevo


# El modelo predijo perfectamente todos los nuevos individuos.

# <div class='question_container'>
#     <h2> Pregunta 3 </h2>
#     <p>Según el ejemplo de los hiperplanos visto en clase realice lo siguiente: </p>
#     <ol>
#         <li> Escriba la regla de clasificación para el clasificador con margen máximo. Debe ser algo como lo siguiente: $w = (w_1,w_2,w_3)$ se clasifica como <code>Rojo</code> si $ax+by+cz+d>0$, de otra manera se clasifica como <code>Azul</code>. </li>
#         <li> Indique la medida del margen entre el hiperplano óptimo de separación y los vectores de soporte. </li> 
#         <li> Explique por qué un ligero movimiento de la octava observación no afectaría el hiperplano de margen máximo.</li>
#     </ol>
# </div>

# Para esto vamos a aplicar una SVM con un núcleo lineal y sin regularización.

# In[102]:


d = {'X': [1, 1, 1, 3, 1, 3, 1, 3, 1], 'Y': [0, 0, 1, 1, 1, 2, 2, 2, 1], 
  'Z': [1, 2, 2, 4, 3, 3, 1, 1, 0], 
  'Clase': ['Rojo', 'Rojo', 'Rojo', 'Rojo', 'Rojo', 'Azul', 'Azul', 'Azul', 'Azul']}
df = pd.DataFrame(data = d)
df


# In[105]:


instancia_svm = SVC(kernel='linear')
instancia_svm.fit(df[['X','Y','Z']].values,df['Clase'])


# In[108]:


instancia_svm._get_coef()


# In[111]:


instancia_svm._intercept_


# Estos coeficientes en realidad son una aproximación al verdadero valor que vimos en la clase, dado por $(-1,-1, 1)$. Esto se debe a los algoritmos numéricos dentro del modelo. La ecuación del plano entonces es
# 
# $$
# -1-x-y+z=0
# $$
# 
# o lo que es lo mismo
# 
# $$
# x+y-z -1=0.
# $$

# Vimos en las figuras de la clase que lo que está encima del hiperplano son los puntos rojos, y lo que está por debajo son los puntos azules. Esto se traduce en la siguiente regla de decisión:
# 
# $$
# \begin{align*}
# \text{Clase}(w_1,w_2,w_3)=\begin{cases}
# \text{Rojo} & \text{ si } w_1+w_2-w_3-1> 0 \\
# \text{Azul} & \text{ si } w_1+w_2-w_3-1\leq 0 \\
# \end{cases} 
# \end{align*}
# $$
# 
# Los dos hiperplanos márgenes tienen las ecuaciones
# 
# $$
# x+y-z-2=0, \quad x+y-z=0
# $$
# 
# El margen se puede calcular como el doble de la distancia de uno de los vectores de soporte al hiperplano óptimo. Recordemos que del álgebra lineal, la distancia entre un punto $P=(x_0,y_0,z_0)$ y un hiperplano $Ax+By+Cz+D=0$ se puede calcular con la siguiente fórmula:
# 
# $$
# d=\frac{Ax_0+By_0+Cz_0+D}{\sqrt{A^2+B^2+C^2}},
# $$
# 
# En este caso vimos que uno de los puntos vectores de soporte era $P=(3,2,3)$ así que dicha distancia viene dada por lo que sigue (Aquí $A=1,B=1,C=-1,D=-1$):
# 
# $$
# d=\frac{1\cdot 3+1\cdot2+ (-1)\cdot 3 + (-1)}{\sqrt{1^2+1^2+(-1)^2}} = \frac{\sqrt{3}}{3},
# $$
# 
# por lo que el margen es $m=\frac{2\sqrt{3}}{3}$.
# 
# Finalmente, notamos que la octava observación del conjunto de datos se encuentra completamente de un lado del hiperplano (ya que no es un vector de soporte y no se encuentra en el área comprendida entre los márgenes). Variar un poco esta observación (y por un poco nos referimos a un $\epsilon$ menor a su distancia al hiperplano óptimo), no va a afectar la lista de vectores de soporte y los cálculos seguirán como antes.

# <div class='question_container'>
#     <h2> Pregunta 4 </h2>
#     <p>Pruebe que si la función objetivo a minimizar es:
#     $$
#     \begin{align*}
#     f(w)=\frac{||w||^2}{2} + C \left(  \sum_{i=1}^n \xi_i \right)^2
#     \end{align*}
#     $$
#     donde $C$ es un parámetro del modelo, entonces Lagrangiano Dual para la Máquina Vectorial de Soporte lineal con datos no separables es:
#     $$
#     \begin{align*}
#     L_D =  \sum_{i=1}^n \lambda_i -\frac{1}{2}\sum_{i,j}\lambda_i\lambda_j y_iy_j x_i \cdot x_j -  C\left(\sum_{i=1}^n  \xi_i\right)^2.
#     \end{align*}
#     $$
#     </p>
# </div>

# #### Respuesta:
# Como vimos en clase, esto corresponde a resolver el siguiente problema de optimización:
# 
# $$
# \begin{align*}
# \min L(w) = \frac{||w||^2}{2} + C\left(\sum_{i=1}^n \xi_i \right)^2
# \end{align*}
# $$
# sujeto a 
# $$
# \begin{align*}
# y_i(w\cdot x_i + b) \geq 1-\xi_i \quad \text{ para todo } i=1,\cdots,n.
# \end{align*}
# $$
# 
# En este caso el Lagrangiano queda así:
# 
# $$
# L_P = \frac{||w||^2}{2} + C\left(\sum_{i=1}^n \xi_i \right)^2 - \sum_{i=1}^n \lambda_i \{y_i(w\cdot x_i + b) - 1 + \xi_i\} - \sum_{i=1}^n \xi_i
# $$

# De manera similar a los cálculos que hemos hecho en clase tenemos que
# $$
# \begin{align*}
# &\frac{\partial L_P}{\partial w_j} = w_j - \sum_{i=1}^n\lambda_iy_ix_{ij}=0 \Rightarrow w_j=\sum_{i=1}^n\lambda_iy_ix_{ij}\\
# &\frac{\partial L_P}{\partial b} = -\sum_{i=1}^n \lambda_iy_i \Rightarrow b=\sum_{i=1}^n \lambda_iy_i.
# \end{align*}
# $$
# Sin embargo, aquí tenemos que
# 
# $$
# \begin{align*}
# \frac{\partial L_P}{\partial \xi_i} = 2C\left(\sum_{j=1}^n \xi_j \right) -\lambda_i - \mu_i =0 \Rightarrow \mu_i=2C\left(\sum_{j=1}^n \xi_j \right) - \lambda_i.
# \end{align*}
# $$

# Aplicando las condiciones de KKT entonces el Lagrangiano Dual en este caso queda así:
# 
# $$
# \begin{align*}
# L_D = &\frac{1}{2}\sum_{i,j} \lambda_i\lambda_j y_iy_j x_i \cdot x_j +C\left(\sum_{i=1}^n\xi_i\right)^2 \\
# &-\sum_{i=1}^n\lambda_i \left\{y_i \left(\sum_{j=1}^n \lambda_jy_j x_i \cdot x_j + b \right) -1+\xi_i\right\}\\
# & -\sum_i \left[2C\left(\sum_{j=1}^n\xi_j\right)-\lambda_i\right]\xi_i\\
# & =\frac{-1}{2}\sum_{i,j} \lambda_i\lambda_j y_iy_j x_i \cdot x_j + \sum_{i=1}^n\lambda_i +C\left(\sum_{i=1}^n\xi_i\right)^2 - \sum_i 2C\left(\sum_{j=1}^n\xi_j\right)\xi_i \\
# & =\frac{-1}{2}\sum_{i,j} \lambda_i\lambda_j y_iy_j x_i \cdot x_j + \sum_{i=1}^n\lambda_i +C\left(\sum_{i=1}^n\xi_i\right)^2 - 2C\left(\sum_{j=1}^n\xi_j\right)\left(\sum_i \xi_i\right) \\
# & =\frac{-1}{2}\sum_{i,j} \lambda_i\lambda_j y_iy_j x_i \cdot x_j + \sum_{i=1}^n\lambda_i +C\left(\sum_{i=1}^n\xi_i\right)^2 - 2C\left(\sum_{i=1}^n\xi_j\right)^2 \\
# & =\frac{-1}{2}\sum_{i,j} \lambda_i\lambda_j y_iy_j x_i \cdot x_j + \sum_{i=1}^n\lambda_i -C\left(\sum_{i=1}^n\xi_i\right)^2.
# \end{align*}
# $$
# Que era la expresión a la que queríamos llegar.
