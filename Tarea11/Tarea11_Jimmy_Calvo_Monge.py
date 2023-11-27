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
#     <h4> TAREA 11 </h4>
#     <h4> Fecha de entrega: 13 de Noviembre de 2022 </h4>
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

import warnings
warnings.filterwarnings('ignore')


# <div class='question_container'>
#     <h2> Pregunta 1 </h2>
#     <p> Cree en <code>Python</code> usando <code>numpy</code> un tensor de 0D, 1D, 2D y uno 3D. </p>
# </div>

# In[2]:


print("Un tensor de 0D:")
x0 = np.array(1)
print(x0)
print(f"Dim: {x0.ndim}")

print("Un tensor de 1D:")
x1 = np.array([1,1,2,3,5,8])
print(x1)
print(f"Dim: {x1.ndim}")

print("Un tensor de 2D:")
x2 = np.array([[1,1,2,3,5,8], [13,21,34,55,89,144]])
print(x2)
print(f"Dim: {x2.ndim}")

print("Un tensor de 3D:")
x3 = np.array([ [[1,1,2], [3,5,8], [13,21,34]], [[1,2,5],[14,42,132],[429,1430,4862]] ])
print(x3)
print(f"Dim: {x3.ndim}")


# <div class='question_container'>
#     <h2> Pregunta 2 </h2>
#     <p> Imprima las dimensiones (shape) de la tabla de entrenamiento de <code>MNIST</code>. Esta es una tabla de dígitos escritos a mano y se usan para problemas de clasificación (problemas predictivos). Esta tabla viene con el paquete <code>tensorflow</code> y se obtiene como sigue:</p>
# </div>

# In[3]:


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Las dimensiones de ambas tablas son las siguientes:

# In[4]:


x_train.shape


# In[5]:


x_test.shape


# Como vemos, ambas son en realidad tensores (tienen 3 dimensiones).

# <div class='question_container'>
#     <h2> Pregunta 3 </h2>
#     <p> ¿Qué tipo de datos contiene <code>MNIST</code>? </p>
# </div>

# In[10]:


x_test


# In[9]:


x_test[0].shape


# Este conjunto de datos está formado por dos tensores de 3 dimensiones (uno para entrenamiento y otro para prueba). Cada elemento en un tensor se puede ver como una matriz, de tamaño $28 \times 28$. Cada matriz en realidad representa un dígito escrito a mano.

# <div class='question_container'>
#     <h2> Pregunta 4 </h2>
#     <p> La operación <code>relu()</code> es una operación que se aplica entrada por entrada de un vector, esta devuelve el máximo entre cada entrada del vector y 0 (<code>relu(x) = max(x,0)</code>). Reprograme esta función en <code>Python</code> y después pruébela con el siguiente vector <code>x</code>.</p>
# </div>

# In[12]:


x = np.array([1, -9, -0.9, 45])


# In[13]:


def relu(x):
    
    def relu_entrada(t):
        return max(t,0)
    
    # Usamos el método vectorize de numpy para aplicar una función entrada por entrada.
    # Otras opciones y su velocidad de ejecución se discuten aquí:
    
    # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
    return np.vectorize(relu_entrada)(x)
    


# In[14]:


relu(x)


# <div class='question_container'>
#     <h2> Pregunta 5 </h2>
#     <p> Calcule "a mano" el gradiente de la siguiente función: </p>
#     $$
#     f(x, y, z) = 3x^4 - 6x\sin(yz) + 3y\cos(z).
#     $$
# </div>

# **Respuesta:** El gradiente que se solicita viene dado por lo siguiente: (lo he calculado a mano).
# 
# $$
# \begin{align*}
#     &\frac{\partial f}{\partial x} = 12x^3 - 6\sin(yz) \\\\
#     &\frac{\partial f}{\partial y} = -6xz\cos(yz) + 3\cos(z) \\\\
#     &\frac{\partial f}{\partial z} = -6xy\cos(yz) -3y\sin(z).
# \end{align*}
# $$
# 
# Por lo tanto
# 
# $$
#     \nabla f(x,y,z) = (12x^3 - 6\sin(yz), -6xz\cos(yz) + 3\cos(z), -6xy\cos(yz) -3y\sin(z) )
# $$

# <div class='question_container'>
#     <h2> Pregunta 6 </h2>
#     <p> Usando el algoritmo del Descenso de Gradiente encuentre "a mano" (puede utilizar <code>excel</code>) un mínimo local de la siguiente función en el intervalo $[1, 4]$, use el punto de partida que considere adecuados. Luego grafique y verifique los resultados con código <code>Python</code>.</p>
#     $$
#     f(x) = 3x^4 - 16x^3 + 18x^2.
#     $$
# </div>

# **Respuesta:**
# 
# Note que $f'(x) = 12x^3 - 48x^2 + 36x$, y en cada paso lo que tenemos que hacer es calcular
# 
# $$
# x_{i+1} = x_i - \eta f'(x_i) \Rightarrow x_{i+1} = x_i - \eta (12x_i^3 - 48x_i^2 + 36x_i)
# $$
# 
# Iniciaremos con $x_0 = 2.5$ (la mitad del intervalo) para ver qué sucede. Utilizaremos un $\eta$ de $0.005$ 

# Los cálculos están en la hoja *Pregunta6* del documento en excel: <code>Tarea11_Jimmy_Calvo_Calculos.xlsx</code> adjunto.

# Ahora en código <code>Python</code>

# In[33]:


figsize = (6,4)
dpi = 150

def f(x):
    return 3*x**4 - 16*x**3 + 18*x**2

def df(x):
    return 12*x**3 - 48*x**2 + 36*x

eta = 0.005
x = [2.5]
y = [f(x[0])]
for i in range(20):
    x.append(x[i] - eta * df(x[i]))
    y.append(f(x[i] - eta * df(x[i])))
    
dt = pd.DataFrame({"X": x, "Y": y})
print(dt)

t1 = np.arange(-1.0, 4.0, 0.1)

fig, ax = plt.subplots(1,1, figsize = figsize, dpi = dpi)
ax.set_xlim(-5, 5)
ax.set_ylim(-45, 45)
ax.plot(t1, f(t1))
ax.plot(x,y,'ro')


# Con lo que vemos que hemos llegado al mínimo en $x=3$. He utilizado este $\eta$ ya que después de algunos intentos, éste fue el que me permitió llegar a un mínimo que estuviera en el intervalo. Como se observa en la figura anterior, hay otro mínimo de esta función en $x=0$, y con $\eta$ más grandes, estaba llegando a ese mínimo (ya vimos que el algoritmo del descenso del gradiente puede converger a mínimos locales, eso era lo que estaba pasando en este caso).

# <div class='question_container'>
#     <h2> Pregunta 7 </h2>
#     <p> Usando el algoritmo del Descenso de Gradiente encuentre "a mano" (puede utilizar <code>excel</code>) el mínimo global de la siguiente función, use un punto de partida que considere adecuado. Luego grafique y verifique los resultados con código <code>Python</code>.</p>
#     $$
#     f(x,y) = x^4 + y^4 -2x^2 + 4xy - 2y^2.
#     $$
# </div>

# En este caso, nuestro gradiente es igual a 
# 
# $$
# \begin{align*}
# &\frac{\partial f}{\partial x} = 4x^3 -4x + 4y\\
# &\frac{\partial f}{\partial y} = 4y^3 + 4x -4y.\\
# \end{align*}
# $$
# 
# Vamos a iniciar con el punto $(x_0,y_0)=(0.5,1.5)$, y utilizaremos un $\eta = 0.01$. Los resultados del cálculo en excel se encuentran en la hoja *Pregunta7* del documento <code>Tarea11_Jimmy_Calvo_Calculos.xlsx</code> adjunto.
# 
# Ahora, procedemos a revisar estos cálculos con <code>Python</code>.

# In[80]:


import plotly.graph_objs as go
import plotly.express as px

# Import the necessaries libraries
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()


# In[81]:


def f7(x,y):
    return x**4 + y**4 -2*x**2 + 4*x*y - 2*y**2

def gradiente_f7(x,y):
    return np.array([4*x**3 - 4*x + 4*y, 4*y**3 + 4*x - 4*y])
    
x = 0.5
y = 1.5
eta = 0.01
xi = np.array([x,y])

x = []
y = []
z = []

for i in range(60):
    xi = xi - eta * gradiente_f7(xi[0],xi[1])
    x.append(xi[0])
    y.append(xi[1])
    z.append(f7(xi[0],xi[1]))
    
    
dataF = pd.DataFrame({"X":x, "Y": y, "Z": z})
print(dataF)

#Gráfico
fig = px.scatter_3d(dataF, x='X', y='Y', z='Z',color_discrete_sequence=["red"])

xdata = np.arange(-2, 2, 0.1)
ydata = np.arange(-2, 2, 0.1)
X,Y = np.meshgrid(xdata, ydata)
Z = X**4 + Y**4 -2*X**2 + 4*X*Y - 2*Y**2
fig.add_trace(go.Surface(
    x = X,
    y = Y,
    z = Z,
    opacity = .7, showscale = False,
    colorscale='Viridis'
))

fig.show()


# Vemos que hay una convergencia hacia el punto $(-\sqrt{2},\sqrt{2})$ que es uno de los mínimos locales de esta función.

# <div class='question_container'>
#     <h2> Pregunta 8 </h2>
#     <p> Usando el algoritmo del Descenso de Gradiente encuentre "a mano" (puede utilizar <code>excel</code>) un mínimo local de la siguiente función, use el punto de partida que considere adecuados. Luego grafique y verifique los resultados con código <code>Python</code>.</p>
#     $$
#     f(x,y) = x^3 + 3y - y^3 - 3x
#     $$
# </div>

# En este caso, nuestro gradiente es igual a 
# 
# $$
# \begin{align*}
# &\frac{\partial f}{\partial x} = 3x^2 - 3 \\
# &\frac{\partial f}{\partial y} = -3y^2 + 3.\\
# \end{align*}
# $$
# 
# Vamos a iniciar con el punto $(x_0,y_0)=(0.8,0.2)$, y utilizaremos un $\eta = 0.01$. Los resultados del cálculo en excel se encuentran en la hoja *Pregunta8* del documento <code>Tarea11_Jimmy_Calvo_Calculos.xlsx</code> adjunto.
# 
# Seleccioné este punto después de ver el gráfico de la función y adivinar que sería un punto en el cual el descenso del gradiente podría funcionar bien.
# 
# Ahora, procedemos a revisar estos cálculos con <code>Python</code>.

# In[ ]:


def f8(x,y):
    return x**3 + 3*y - y**3 - 3*x

def gradiente_f8(x,y):
    return np.array([3*x**2 - 3, -3*y**2 + 3])
    
x = 0.8
y = 0.2
eta = 0.01
xi = np.array([x,y])

x = []
y = []
z = []

for i in range(60):
    xi = xi - eta * gradiente_f8(xi[0],xi[1])
    x.append(xi[0])
    y.append(xi[1])
    z.append(f8(xi[0],xi[1]))
    
    
dataF = pd.DataFrame({"X":x, "Y": y, "Z": z})
print(dataF)

#Gráfico
fig = px.scatter_3d(dataF, x='X', y='Y', z='Z',color_discrete_sequence=["red"])

xdata = np.arange(-2, 2, 0.1)
ydata = np.arange(-2, 2, 0.1)
X,Y = np.meshgrid(xdata, ydata)
Z = X**3 + 3*Y - Y**3 - 3*X
fig.add_trace(go.Surface(
    x = X,
    y = Y,
    z = Z,
    opacity = .7, showscale = False,
    colorscale='Viridis'
))

fig.show()


# Con estos cálculos y los realizados en excel, parece que la convergencia es hacia el mínimo que se encuentra en el punto $(1,-1)$.

# <div class='question_container'>
#     <h2> Pregunta 9 </h2>
#     <p> ¿Qué pasa si en el código visto en clase para optimizar una Función de Costo (<code>gradient_descent</code>) se usa un momentum (impulso) más pequeño a $0.7$? ¿Qué pasa si en el código visto en clase para optimizar una Función de Costo (gradient descent) se usa un $\eta$ (<code>learning_rate</code>) de $10^{-9}$?</p>
# </div>

# In[65]:


def gradient_descent(max_iterations,threshold,w_init,
                     obj_func,grad_func,extra_param = [],
                     learning_rate=0.05,momentum=0.8): 
    w = w_init
    w_history = w
    f_history = obj_func(w,extra_param)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while  i<max_iterations and diff>threshold:
        delta_w = -learning_rate*grad_func(w,extra_param) + momentum*delta_w
        w = w+delta_w
        
        # Almacena la historia of w y f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,obj_func(w,extra_param)))
        
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
    
    return w_history,f_history



# w es un vector de pesos y xy el vetor fila de datos (train_data, target)
def grad_mse(w,xy):
    (x,y) = xy
    (rows,cols) = x.shape
    
    # Calcula la salida
    o = np.sum(x*w,axis=1)
    diff = y-o
    diff = diff.reshape((rows,1))    
    diff = np.tile(diff, (1, cols))
    grad = diff*x
    grad = -np.sum(grad,axis=0)
    return grad


#  w es un vector de pesos y xy el vector fila de datos (train_data, target)
def mse(w,xy):
    (x,y) = xy
    
    # Calcula la salida usando el mse
    o = np.sum(x*w,axis=1)
    mse = np.sum((y-o)*(y-o))
    mse = mse/2
    return mse


# Vamos a leer los datos utilizando la biblioteca de `datasets` de `sklearn`.

# In[69]:


from sklearn import datasets
digits = datasets.load_digits()
datos = pd.DataFrame(digits['data'])
datos.columns = digits['feature_names']
datos['digito'] = digits['target']
datos = datos[datos['digito'].isin([0,1])]
print(datos.shape)
datos.head(5)


# Ahora vamos a aplicar la función con los parámetros por defecto, como vimos en clase.

# In[70]:


Y = datos["digito"].ravel()
X = datos.drop(columns=["digito"]).to_numpy()

# Separa train y test set
x_train, x_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.2, random_state=10)

# Agrega la columna de unos de al regresión (bias) en train and test
x_train = np.hstack((np.ones((y_train.size,1)),x_train))
x_test  = np.hstack((np.ones((y_test.size,1)),x_test))

# Inicializa los pesos y llama a gradient_descent
rand = np.random.RandomState(19)
w_init = rand.uniform(-1,1,x_train.shape[1])*.000001
w_history,mse_history = gradient_descent(100,0.1,w_init,
                              mse,grad_mse,(x_train,y_train),
                             learning_rate=1e-6,momentum=0.7)

# Grafica el MSE
fig, ax = plt.subplots(1,1, figsize = figsize, dpi = dpi)
ax.plot(np.arange(mse_history.size),mse_history)
ax.set_xlabel('Número de Iteración')
ax.set_ylabel('Error Cuadrático Medio')
ax.set_title('Descenso del gradiente para la función de costo - Ejemplo de Dígitos')


# Esto es lo que obtuvimos en clase. Pero ahora qué pasa si tomamos $\eta <0.7$, por ejemplo $0.2$. Veamos.

# In[73]:


w_history,mse_history = gradient_descent(100,0.1,w_init,
                              mse,grad_mse,(x_train,y_train),
                             learning_rate=1e-6,momentum=0.2)

# Grafica el MSE
fig, ax = plt.subplots(1,1, figsize = figsize, dpi = dpi)
ax.plot(np.arange(mse_history.size),mse_history)
ax.set_xlabel('Número de Iteración')
ax.set_ylabel('Error Cuadrático Medio')
ax.set_title('Descenso del gradiente para la función de costo - Ejemplo de Dígitos')


# En este caso vemos que hay una convergencia un poco más rápida hacia un error cuadrático medio de $\epsilon=0$. Si hacemos el learning rate más pequeño, sucederá lo que sigue.

# In[77]:


w_history,mse_history = gradient_descent(100,0.1,w_init,
                              mse,grad_mse,(x_train,y_train),
                             learning_rate=1e-8,momentum=0.7)

# Grafica el MSE
fig, ax = plt.subplots(1,1, figsize = figsize, dpi = dpi)
ax.plot(np.arange(mse_history.size),mse_history)
ax.set_xlabel('Número de Iteración')
ax.set_ylabel('Error Cuadrático Medio')
ax.set_title('Descenso del gradiente para la función de costo - Ejemplo de Dígitos')


# Al disminuir mucho la taza de aprendizaje, se puede ver que se necesitará una cantidad mayor de iteraciones para disminuir el error cuadrádico medio lo suficiente. Y esto tiene sentido, ya que si nuestra taza de aprendizaje es muy pequeña, entonces no habrá mucha diferencia de un paso a otro en el algoritmo del descenso del gradiente.
# 
# En este repositorio de github, Lili Jiang creó una aplicación en C++ que permite crear animaciones para visualizar el proceso del descenso del gradiente: [gradient_descent_viz](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c).
