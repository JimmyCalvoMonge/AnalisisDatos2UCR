#!/usr/bin/env python
# coding: utf-8

# # Análisis de Datos II
# ## Profesor: Oldemar Rodríguez 
# ### Estudiante: Jimmy Calvo Monge
# ### Carné: B31281
# #### Fecha de entrega: 28 de Agosto de 2022
# -------------------------------------------------------------

# Importamos los módulos necesarios para resolver esta tarea.

# In[18]:


import pandas as pd
import numpy as np
import math
import sys
print(sys.version)


# ### Pregunta 1
# 
# ¿Cuál es el resultado?
# - a) s = $|(44^3 - 7!) \cdot (25/88)|$
# - b) s = $\sqrt{99} \cdot \pi^2$
# - c) s = $\log_2(38)$
# - d) s = $\sin(e)^{0.5}$

# In[19]:


abs((44**3 - math.factorial(7)) * (25/88))


# In[20]:


math.sqrt(99)*(math.pi**2)


# In[21]:


math.log2(38)


# In[22]:


math.sin(math.e)**(0.5)


# ### Pregunta 2
# 
# Dada una cadena de texto indique si la cantidad de caracteres en la cadena es un número par
# o impar. Realice una prueba para cada uno de los siguientes valores: "abracadabra", "casa",
# "sol".

# In[187]:


def paridad_char(char):
    if type(char)==str:
        
        n=len(char)
        
        if n%2==0:
            return "Tenemos un número par de caracteres"
        else:
            return "Tenemos un número impar de caracteres"
    else:
        raise Exception("Por favor introduzca un caracter como input")


# In[24]:


paridad_char("abracadabra")


# In[25]:


paridad_char("casa")


# In[26]:


paridad_char("sol")


# In[188]:


paridad_char(5)


# ### Pregunta 3
# 
# Dado un valor numérico indique a que día de la semana pertenece, siendo 1 = domingo y 7
# = sábado. En caso que se digite un número fuera del rango imprima un mensaje indicando el
# error. Realice una prueba para cada uno de los siguientes valores: 4, 2, 9.

# In[28]:


def dia_semana(num):
    
    if type(num)== int and num in range(1,8):
        
        day_list=["domingo","lunes","martes","miércoles","jueves","viernes","sábado"]
        return day_list[num-1]
        
    else:
        raise Exception("Por favor introduzca un número entero entre 1 y 7.")    


# In[29]:


dia_semana(4)


# In[30]:


dia_semana(2)


# In[31]:


dia_semana(9)


# ### Pregunta 4
# 
# Suponga que tienen los valores a, b y c de una ecuación cuadrática $ax^2 +bx+c = 0$. Determine
# si la ecuación es degenerada (a = 0) o en caso contrario determine cuántas soluciones tiene (si
# el discriminante es mayor a cero tiene 2 soluciones, si es igual a cero tiene 1 solución, y si es
# menor a cero tiene 0 soluciones reales). Debe imprimir la cantidad de soluciones. Realice una
# prueba para cada uno de los siguientes casos: (a = 9, b = 0, c = 2), (a = 0, b = -2, c = 3),
# (a = 2, b = -4, c = 2).

# In[32]:


def analisis_cuadratica(a,b,c):
    
    if a==0:
        return "Ecuación cuadrática es degenerada."
    else:
        try:
            
            disc_sqrt = math.sqrt( b**2 -4*a*c )
            if disc_sqrt==0:
                sol= [-1*b/(2*a)]
            else:
                sol = [ (-1*b + disc_sqrt)/(2*a) , (-1*b - disc_sqrt)/(2*a) ]
            
            sol=", ".join(str(s) for s in sol) 
            
            print(f" Número de soluciones: {len(sol)}")
            print(f" Éste es el conjunto solución: {{ {sol} }}. ")
            
        except Exception as e:
            raise Exception(f"Error: {e}")


# In[33]:


analisis_cuadratica(9,0,2)


# In[34]:


analisis_cuadratica(0,-2,3)


# In[35]:


analisis_cuadratica(2,-4,2)


# ### Pregunta 5
# 
# Dado x = (15, 34, 72, 23, 91, 4, 201, 68, 56, 78) realice lo siguiente:
# 
# - Calcule la media, la varianza y la desviación estándar.
# - Extraiga los primeros 3 valores.
# - Indique el valor más pequeño del vector.
# - Obtenga la sumatoria de todos los valores del vector.
# - Obtenga la lista x invertida.

# In[36]:


x = [15, 34, 72, 23, 91, 4, 201, 68, 56, 78]


# In[37]:


print(f"""
media: {np.mean(x)}
varianza: {np.var(x)}
desviación estándar: {np.std(x)}
""")


# In[38]:


x[0:3] ### Primeros tres valores


# In[39]:


min(x) ### Valor más pequeño


# In[40]:


sum(x) ### Sumatoria de los elementos del vector


# In[41]:


x_reversed=x[::-1]
x_reversed


# In[72]:


### Note que el metodo .reversed() hace el cambio sobre el objeto
x.reverse()
x


# ### Pregunta 6
# 
# Realice la siguiente operación entre matrices:
# 
# $$
# A= \begin{pmatrix}
# -17 & 11 & 61 \\
# 4 & 29 & 33
# \end{pmatrix} -12 \cdot \begin{pmatrix}
# -8 & 6  \\
# -15 & 25  \\
# 5 & -13
# \end{pmatrix} ^t
# $$

# In[77]:


A1 = np.array([
    [-17,11,61],
    [4,29,33]
])
A2 = np.array([
    [-8,6],
    [-15,25],
    [5,-13]
]).transpose()

A= A1 - 12*A2
A


# Esto quiere decir que el resultado es
# 
# $$
# A= \begin{pmatrix}
# 79 & 191 & 1 \\
# -68 & -271 & 189
# \end{pmatrix}.
# $$

# ### Pregunta 7
# 
# Dada la matriz cuadrada A que se presenta abajo, calcule, la suma de los elementos que
# conforman la diagonal. Es decir, $6 + (-8) + 4 = 2$.
# 
# $$
# A= \begin{pmatrix}
# 6 & 7 & -5 \\
# 1 & -8 & -6 \\
# 10 & 13 & 4
# \end{pmatrix}
# $$

# In[79]:


A=np.array([
    [6,7,-5],
    [1,-8,-6],
    [10,13,4]
])
traza = sum([ A[i][i] for i in range(0,3)])
traza


# ### Pregunta 8
# 
# Genere, sin utilizar archivos, un DataFrame de la siguiente tabla de datos:
# 
# | Nombre | Matematicas | Ciencias | Español | Historia | EdFisica | Genero |
# | ---    | ---         | ---      | ---     | ---      | ---      | ---    |        
# | Lucia  | 7.0         | 6.5      | 9.2     | 8.6      | 8.0      | F      |
# | Pedro  | 7.5         | 9.4      | 7.3     | 7.0      | 7.0      | M      |
# | Ines   | 7.6         | 9.2      | 8.0     | 8.0      | 7.5      | F      |
# | Luis   | 5.0         | 6.5      | 6.5     | 7.0      | 9.0      | M      |
# | Andres | 6.0         | 6.0      | 7.8     | 8.9      | 7.3      | M      |
# | Ana    | 7.8         | 9.6      | 7.7     | 8.0      | 6.5      | F      |

# In[152]:


estudiantes = pd.DataFrame({
    "Matematicas": [7.0,7.5,7.6,5.0,6.0,7.8],
    "Ciencias" : [6.5,9.4,9.2,6.5,6.0,9.6],
    "Español" : [9.2,7.3,8.0,6.5,7.8,7.7],
    "Historia" : [8.6,7.0,8.0,7.0,8.9,8.0],
    "EdFisica" : [8.0,7.0,7.5,9.0,7.3,6.5],
    "Genero" : ["F","M","F","M","M","F"]
})

estudiantes.index= ["Lucia","Pedro","Ines","Luis","Andres","Ana"]
estudiantes


# ### Pregunta 9
# 
# Utilizando la tabla creada en el punto anterior realice lo siguiente:
# - Ejecute un `info()` de los datos.
# - Muestre el resumen estadístico básico, uno incluyendo solo los tipos "numéricos" y otro excluyendo los tipos numéricos.
# - Despliegue las primeras 3 columnas de la tabla de datos (Usando solamente []).
# - Despliegue las primeras 3 columnas de la tabla de datos (Usando solamente `iloc`).
# - Despliegue las primeras 3 columnas de la tabla de datos (Usando solamente `loc`).
# - Calcule la correlación entre Historia y Matematicas con la función `corrcoef()` de la biblioteca `numpy`.
# - Construya un diccionario llamado `resumen` que tenga 4 campos Media, Mediana, Máximo y mínimo que tienen la media, la mediana, el máximo y el mínimo respectivamente de la variable `Ciencias`.

# In[84]:


### Ejecutamos un info
estudiantes.info()


# In[85]:


### Estadisticas sobre los numericos ###
estudiantes_numerico = estudiantes.select_dtypes(include=np.number)
estudiantes_numerico 


# In[86]:


estudiantes_numerico.describe()


# In[94]:


estudiantes_char=estudiantes.select_dtypes(exclude=np.number)
estudiantes_char


# In[95]:


estudiantes_char.describe()


# In[104]:


### Primeras 3 columnas usando []
### Se me ocurrió hacerlo con .columns
estudiantes[[col for col in estudiantes.columns[0:3]]]


# In[105]:


### Primeras 3 columnas usando iloc
estudiantes.iloc[:,[0,1,2]]


# In[107]:


### Primeras 3 columnas usando loc
### También con .columns
estudiantes.loc[:,[col for col in estudiantes.columns[0:3]]]


# In[179]:


np.corrcoef(estudiantes["Matematicas"],estudiantes["Ciencias"])[1][0] ### Correlacion matematica y ciencias


# In[180]:


### Diccionario resumen para ciencias:

resumen={
    "Media" : np.mean(estudiantes["Ciencias"]),
    "Mediana" : np.median(estudiantes["Ciencias"]),
    "Máximo" : max(estudiantes["Ciencias"]),
    "Mínimo" : min(estudiantes["Ciencias"])
}
resumen


# ### Pregunta 10
# 
# Cargue la tabla de datos que está en el archivo students.csv haga lo siguiente:
# - Calcule la dimensión de la Tabla de Datos.
# - Calcule el resumen numérico de la tabla.
# - Calcule la suma de las columnas con variables cuantitativas (numéricas).
# - Calcule la moda de las columnas con variables cualitativas (categóricas).

# In[112]:


students=pd.read_csv("students.csv", delimiter = ';', decimal = ",", header = 0, index_col = 0)
students


# In[113]:


### Dimensión Tabla de Datos
students.shape


# In[116]:


students.describe() ### Resumen numerico de la tabla


# In[121]:


students_numerico = students.select_dtypes(include=np.number)
students_numerico.sum(axis=0) ### Suma de cada una de las columnas numéricas del dataframe.


# In[122]:


students_no_numerico = students.select_dtypes(exclude=np.number)
students_no_numerico.mode(axis=0) ### Moda de cada una de las columnas categóricas del dataframe.


# In[125]:


#### Verificamos para una columna.
from statistics import mode
mode(students["guardian"])


# ### Pregunta 11
# 
# Usando `for(...)` en Python muestre los números del 1 al 100 que terminan en 8, debe mostrarlos
# en orden inverso, es decir, de mayor a menor.

# In[128]:


for i in range(100,0,-1):
    if str(i).endswith("8"):
        print(i)


# ### Pregunta 12
# 
# Mediante un ciclo, calcule la sumatoria de los números enteros múltiplos de 17, comprendidos
# entre el 1 y el 300.

# In[133]:


sum_d17=0
for i in range(1,301):
    if i%17==0:
        sum_d17=sum_d17+i
sum_d17


# In[134]:


### En una sola línea:
sum_d17= sum([ i for i in range(1,301) if i%17==0])
sum_d17


# ### Pregunta 13
# 
# Mediante un ciclo, guarde en una lista todos los números pares desde 10 hasta el 25.

# In[135]:


lista_pares=[]
for i in range(10,26):
    if i%2==0:
        lista_pares.append(i)
lista_pares


# In[136]:


### En una sola línea:
lista_pares=[i for i in range(10,26) if i%2==0]
lista_pares


# ### Pregunta 14
# 
# Programe una función que recibe tres objetos A, B, y C, si el tipo de dichos objetos es `str`
# retorna la unión de dichos caracteres separados por un espacio. En caso de que alguno de los
# tres objetos no sea str retorna `None`, use `isinstance` para determinar si el objeto es de tipo
# `str`.

# In[141]:


def preg_14(A,B,C):
    tipos=[isinstance(A,str),isinstance(B,str),isinstance(C,str)]
    if False in tipos:
        print("Alguno no es str")
        return None
    else:
        return " ".join([A,B,C])


# In[138]:


preg_14("la","pregunta","catorce")


# In[142]:


preg_14("a",5,"b")


# ### Pregunta 15
# 
# Programe una función que reciba un número y retorne `True` en caso de ser un número primo,
# de lo contrario retorna `False`. La definición de **número primo** dice que *Un número entero
# mayor que 1 se denomina número primo si y sólo si tiene como divisores positivos (factores)
# únicamente a sí mismo y a la unidad 1*. Por ejemplo, son números primos: 2, 3, 5, 7, 11, 13,
# 17.

# In[68]:


def es_primo(n):
    """
    Una proposición de teoría de números dice que un número n es primo si y sólo si
    ningún entero menor que la raíz cuadrada de n divide a n.
    Esto es más eficiente que verificar todos los números antes que n.
    """
    sqrt_n_ent=math.floor(math.sqrt(n))
    resp=True
    
    for i in range(2,sqrt_n_ent+1):
        if n%i==0:
            resp=False
            break
    return resp


# In[69]:


es_primo(13)


# In[70]:


es_primo(19)


# In[71]:


es_primo(20)


# In[72]:


es_primo(49)


# In[73]:


import primesieve  ### Ejemplo para el diez-milésimo primo :)
primo_muy_grande= primesieve.nth_prime(10000)
primo_muy_grande


# In[55]:


es_primo(primo_muy_grande)


# ### Pregunta 16
# 
# Programe una función que recibe una lista numérica y retorna el número primo más pequeño
# que está en la lista. Puede utilizar la función creada en el punto anterior.

# In[52]:


def primo_mas_pequeno(lista):
    primos_en_lista=[i for i in lista if es_primo(i)]
    
    if len(primos_en_lista)>0:
        return min(primos_en_lista)
    else:
        print("No hay primos en esta lista")


# In[75]:


primo_mas_pequeno([32,19,22,193,1163])


# In[74]:


primo_mas_pequeno([10,4,6,9])


# ### Pregunta 17
# 
# Programe una función que reciba un objeto C tipo `str` (cadena de caracteres) y retorne la
# cantidad de vocales que posee la cadena de caracteres C. Por ejemplo: `"Hola Mundo"` $\rightarrow$ `4`.

# In[78]:


def cantidad_vocales(C):
    if isinstance(C,str):
        vocales=["a","e","i","o","u"]
        c_lower=C.lower()
        cant_vocales= len([a for a in c_lower if a in vocales])
        return cant_vocales
    else:
        raise Exception("No es una cadena de caracteres!")


# In[79]:


cantidad_vocales("Hola Mundo")


# ### Pregunta 18
# 
# Programe una función que reciba un objeto C tipo `str` (cadena de caracteres) y una letra l
# y retorne la cantidad de veces que aparece la letra dentro de la cadena de caracteres C. Por
# ejemplo: `"abracadabra"` y `"a"` $\rightarrow$ `5`.

# In[82]:


def cantidad_veces(C,l):
    if isinstance(C,str) and isinstance(l,str):
        cant_veces= len([a for a in C if a == l])
        return cant_veces
    else:
        raise Exception("No es una cadena de caracteres!")


# In[83]:


cantidad_veces("abracadabra","a")


# ### Pregunta 19
# 
# Programe en Python una función que recibe tres valores A, B, y C y retorna el menor.

# In[84]:


def el_menor(A,B,C):
    try:
        return min([A,B,C])
    except:
        print("Introduzca tres valores numéricos")

el_menor(98,87,18)


# ### Pregunta 20
# 
# Programe en Python una función que recibe un número n y retorna la sumatoria de los números
# enteros comprendidos entre el 1 y el n.

# In[86]:


def sumatoria(n):
    try:
        parte_entera=math.floor(n)
        return sum([i for i in range(1,parte_entera+1)])
    except:
        print("Introduzca un numero positivo")


# In[87]:


sumatoria(50)


# In[89]:


sumatoria(100.22)


# ### Pregunta 21
# 
# Programe una función en Python que recibe un número n y realice la sumatoria de los números
# enteros múltiplos de 5, comprendidos entre el 1 y el n.

# In[90]:


def sumatoria_mult5(n):
    try:
        parte_entera=math.floor(n)
        return sum([i for i in range(1,parte_entera+1) if i%5==0])
    except:
        print("Introduzca un numero positivo")


# In[96]:


sumatoria_mult5(27.45)


# In[97]:


5+10+15+20+25


# ### Pregunta 22
# 
# Programe en Python una función que genera 50 números al azar entre 1 y 400. Luego de esos
# 50 números la función calcula y retorna qué porcentaje son pares.

# In[107]:


def preg_22():
    enteros_random=np.random.randint(size=50, low=10, high=401)
    cantidad_pares=len([i for i in enteros_random if i%2==0])
    return cantidad_pares/50*100


# In[111]:


### Varias simulaciones
for i in range(10):
    print(f"{int(preg_22())}%")


# ### Pregunta 23
# 
# Programe en Python una función que genera 100 números al azar entre 1 y 500 y luego calcula
# cuántos están entre el 50 y 450, ambos inclusive.

# In[113]:


def preg_23():
    enteros_random=np.random.randint(size=100, low=1, high=501)
    cantidad_rango=len([i for i in enteros_random if i>=50 and i<=450])
    return cantidad_rango


# In[114]:


### Varias simulaciones
for i in range(10):
    print(f"{int(preg_23())} enteros en [50,450]")


# ### Pregunta 24
# 
# Desarrolle una función en Python que calcula el costo de una llamada telefónica que ha durado $t$
# minutos sabiendo que si $t < 1$ el costo es de $0.4$ dólares, mientras que para duraciones superiores
# el costo es de $0.4 + (t - 1)/4$ dólares, la función debe recibir el valor de $t$.

# In[115]:


def precio_llamada(t):
    if t<1:
        costo=0.4
    elif t>0:
        costo = 0.4 + (t-1)/4
    return costo


# In[116]:


precio_llamada(0.56)


# In[117]:


precio_llamada(t=1.9)


# ### Pregunta 25
# 
# Desarrolle una función en Python que reciba un vector de números reales y un número real x,
# tal que retorne el porcentaje de elementos menores o iguales a un valor x.

# In[118]:


def preg_25(vector,x):
    if len(vector)>0:
        num_menor_x = [i for i in vector if i<x]
        return len(num_menor_x)/len(vector)*100


# In[119]:


preg_25(vector=[23,288,245,199,12,890,1220,223,900,17,14,290,1000,1001,1002,1003],x=56)


# ### Pregunta 26
# 
# Desarrolle una función en Python que recibe 4 números `(x1, x2, y1, y2)`. La función debe calcular
# la distancia entre dos puntos y mostrar mediante texto el resultado, para ello use la fórmula:
# $$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}.$$

# In[122]:


def distancia_euclidea(x1,x2,y1,y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )

distancia_euclidea(2,3,4,5)


# ### Pregunta 27
# 
# Desarrolle una función que recibe la cantidad de entradas que una persona desea comprar
# para un espectáculo y el precio de las mismas (todas valen igual), luego debe calcular el pago
# a realizar por la(s) entrada(s) tomando en cuenta que se pueden comprar sólo hasta cuatro
# entradas, que al costo de dos entradas se les descuenta el 15%, que al de tres entrada el 20%
# y que a la compra de cuatro entradas se le descuenta el 25%. Realice además una función de
# validación que evite que una persona compre más de cuatro entradas.

# In[138]:


def preg_27(cant_entradas,precio):
    
    if cant_entradas>4:
        raise Exception("Se pueden comprar solo hasta cuatro entradas")
    else:
        costo_total= cant_entradas*precio
        
        if cant_entradas==2:
            costo_total= costo_total*0.85
        elif cant_entradas==3:
            costo_total= costo_total*0.8
        elif cant_entradas==4:
            costo_total= costo_total*0.75
        return costo_total


# In[140]:


preg_27(3,99)


# In[141]:


preg_27(1,99)


# In[142]:


preg_27(5,99)


# ### Pregunta 28
# 
# Desarrolle una función que reciba un número natural $n$ (suponiendo que $n > 1$) y que construya
# y retorne un vector $v$ de tamaño $n$ tal que $v_k = v_{k-1}/3 + 0.5$ para $k = 1,\cdots, n$ y siendo que
# $v_0 = 1$.

# In[128]:


def preg_28(n):
    v=[1]
    vk_1=1
    for i in range(1,n+1):
        vk=vk_1/3 + 0.5
        v.append(vk)
        vk_1=vk
    return v
preg_28(10)


# ### Pregunta 29
# 
# Desarrolle una función que construye y retorna una matriz $A$ de tamaño $m \times n$ cuya entrada
# génerica es $i^2 - j$, es decir $a_{ij} = i^2 - j$.

# In[135]:


def preg_29(m,n):
    lista_filas=[]
    for i in range(m):
        lista_filas.append([(i+1)**2-(j+1)**2 for j in range(n)])
    return np.array(lista_filas)
preg_29(3,4)


# ### Pregunta 30
# 
# Desarrolle una función que recibe una matriz cuadrada $A$ de tamaño $n \times n$ y calcula su traza
# (utilizando `for`), es decir, la suma de los elementos de la diagonal. Por ejemplo, la traza de la
# siguiente matriz:
# $$
# B=\begin{pmatrix}
# 8 & 3 & 24 \\
# 12 & 13 & -11 \\
# 14 & 12 & -6
# \end{pmatrix}
# $$
# es $15$.

# In[125]:


def traza(A):
    try:
        
        ### Se supone que la matriz será cuadrada.
        n=len(A)
        m=len(A[0])
        
        if n==m:
            traza= sum([A[i][i] for i in range(n)])
            return traza
        else:
            raise Exception("La matriz no es cuadrada")
        
    except Exception as e:
        print(f"Error: {e}")


# In[126]:


B=np.array([
    [8,3,24],
    [12,13,-11],
    [14,12,-6]
])

traza(B)


# ### Pregunta 31
# 
# Desarrolle una función que reciba dos números enteros a y b, tal que, en un diccionario retorne
# el Máximo Común Divisor (MCD) y el mínimo común múltiplo (mcm). Para calcular el MCD
# puede utilizar la función `gcd` del paquete `math`. La fórmula para calcular el mcm es la siguiente:
# $$mcm(a, b) = \frac{a \cdot b}{MCD(a, b)}$$

# In[124]:


def mcd_mcm(a,b):
    try:
        
        mcd=math.gcd(a,b)
        mcm= int((a*b)/mcd)
        
        return {
            "mcd":mcd,
            "mcm":mcm
        }
        
    except Exception as e:
        print(f"Error: {e}")
        
mcd_mcm(5824,8612)


# ### Pregunta 32
# 
# Desarrolle una función que recibe una matriz cuadrada $A$ de tamaño $n \times n$ y retorna su
# transpuesta (utilizando for). Por ejemplo, la transpuesta de la matriz $A$:
# 
# $$
# A= \begin{pmatrix}
# 46 & 30 & 6 \\
# 4 & 2 & 54 \\
# -7 & -5 & -11
# \end{pmatrix}
# $$
# 
# es la siguiente matriz $A^t$:
# 
# $$
# A^t= \begin{pmatrix}
# 46 & 4 & -7 \\
# 30 & 2 & -5 \\
# 6 & 54 & -11
# \end{pmatrix}
# $$
# 

# In[150]:


def transpuesta(A):
    
    ### Recibe y regresa un objeto de numpy
    n=len(A)
    m=len(A[0])
    
    At=[]
    
    for j in range(n):
        
        ### Obtenemos la columna j esima
        col = [A[i][j] for i in range(m)]
        ### La agregamos como fila
        At.append(col)
        
    At=np.array(At)
    return At

A=np.array([
    [46,30,6],
    [4,2,54],
    [-7,-5,-11]
])    

transpuesta(A)


# ### Pregunta 33
# 
# Desarrolle una función en Python que recibe una lista de edades y retorna que porcentaje son
# mayores de 18 años y que porcentaje tienen 18 o menos años.

# In[148]:


def preg_33(lista_edades):
    
    ### Si la lista tiene elementos y todos son enteros.
    if len(lista_edades)>0 and not False in [isinstance(i,int) for i in lista_edades] and min(lista_edades)>0:
        
        porc_menor_18 = len([i for i in lista_edades if i<=18])/len(lista_edades)*100
        porc_mayor_18 = 100 - porc_menor_18

        return {
            "porcentaje mayor 18" : porc_mayor_18,
            "porcentaje menor o igual a 18" : porc_menor_18
        }
    else:
        raise Exception("ingrese una lista de enteros positivos")


# In[149]:


preg_33(lista_edades=[18,19,20,13,12,90,67,12,3,5,56])


# ### Pregunta 34
# 
# Desarrolle una función que recibe un DataFrame y un nombre de columna y que retorne
# un diccionario con los valores de la media, mínimo, máximo y varianza de dicha columna del
# dataframe. Verifique la correctitud de esta función usando la tabla `celulares.csv` y la columna
# `battery_power`.

# In[172]:


def preg_34(data,columna):
    
    if str(data.dtypes[columna])!='object':
        ### si la columna no es de caracteres
        
        return {
            "media": np.mean(data[columna]),
            "minimo": min(data[columna]),
            "maximo": max(data[columna]),
            "varianza": np.var(data[columna])
        }
    
    else:
        raise Exception("La columna debe ser numerica")


# In[173]:


celulares= pd.read_csv("celulares.csv", delimiter = ',', decimal = ".", header = 0)
celulares


# In[174]:


preg_34(data=celulares,columna="battery_power")


# ### Pregunta 35
# 
# Desarrolle una función que recibe un DataFrame y dos números de columna y que retorna
# la correlación entre esas dos variables. Verifique la correctitud de esta función usando la tabla
# `celulares.csv` y las columnas 5 y 6, que hacen referencia a las variables `four_g` y `int_memory`.

# In[184]:


def preg_35(data,num1,num2):
    try:
        corr= np.corrcoef(data.iloc[:,num1],data.iloc[:,num2])[1][0]
        return corr
    except:
        print("Error")


# In[186]:


### Para aplicarla, encontramos los indices de esas dos columnas 
num1=celulares.columns.tolist().index("four_g")
num2=celulares.columns.tolist().index("int_memory")

preg_35(celulares,num1,num2)

