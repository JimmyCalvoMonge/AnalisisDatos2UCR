#!/usr/bin/env python
# coding: utf-8

# # Análisis de Datos II
# ## Profesor: Oldemar Rodríguez 
# ### Estudiante: Jimmy Calvo Monge
# ### Carné: B31281
# #### TAREA 2
# #### Fecha de entrega: 28 de Agosto de 2022
# -------------------------------------------------------------

# Importamos los módulos necesarios para resolver esta tarea.

# In[115]:


import pandas as pd
import numpy as np
import math
import statistics as stat
import sys
print(sys.version)


# ### Pregunta 1
# 
# Desarrolle en `Python` las clases `Triangulo` y `Rectangulo`, luego para cada una de ellas programe
# los métodos calcular area y calcular perimetro:
# 1. Clase `Triangulo`: Tiene como atributos el lado, la base y altura del triángulo.
# 2. Clase `Rectangulo`: Tiene como atributos la base y altura del rectángulo.

# In[337]:


class Triangulo():
    
    def __init__(self,base,altura,lado):
        
        self.__base=base
        self.__altura=altura
        
        ### Dada una base y una altura, no podemos poner cualesquiera valores de los lados.
        ### Uno de los lados puede ser arbitrario, pero el otro está restringido por el valor de la base
        ### y de la altura.
        
        ### Asumimos que lado es una lista de 3 lados.
        
        
        if isinstance(lado, list) and len(lado)==3:
            
            if self.__base in lado:
                ### La base deberia pertenecer a la lista de lados.
                
                
                ### Lo siguiente verifica que los lados proporcionados, junto con la base y la altura
                ### De verdad den un triangulo que exista.
                
                ### Si L1 Y L2 son los otros dos lados que no son la base b, entonces la base se puede 
                ### dividir en 2 partes b1 y b2 de manera que se cumplan 2 triangulos rectangulos:
                ### b1^2 + h^2 = L1^2  y b2^2 + h^2 = L2^2. Si L2 no cumple esta segunda ecuacion
                ### entonces las medidas proporcionadas no pueden ser las de un triángulo de verdad.
                
    
                otros_lados = lado.copy()
                otros_lados.remove(self.__base)
                
                l1 = otros_lados[0]
                l2 = otros_lados[1]
                
                b1 = math.sqrt(l1**2 - self.__altura**2)
                b2= self.__base - b1
                
                if l2**2 == b2**2 + self.__altura**2:
                    print("Felicidades ha hecho un triangulo existosamente")
                    self.__lado = lado
                else:
                    raise Exception("Estas medidas NO son compatibles con la definición de un Triángulo.")
            
            else:
                raise Exception("Dimensiones invalidas para construir un triángulo")
                
        else:
            
            raise Exception("Lado debe ser una lista de 3 medidas")
            
        if self.__base in lado:
            self.__lado = lado
        
    def set_area(self):
        area = (self.__base * self.__altura)/2
        self.__area = area
    
    def get_area(self):
        if self.__area:
            return self.__area
        else:
            print("Ejecute .set_area() primero")
            
    def set_perimetro(self):
        ### Asumimos que lado es una lista de tres lados
        # Si no entonces es un solo lado y el triángulo es equilátero.
        self.__perimetro = sum(self.__lado)
            
    def get_perimetro(self):
        if self.__perimetro:
            return self.__perimetro
        else:
            print("Ejecute .set_perimetro() primero")
    
    def __str__(self):
        return f"Triángulo de lados {self.__lado}, altura {self.__altura} y base {self.__base}."
    
    
class Rectangulo():
    
    def __init__(self,base, altura):
        self.__base=base
        self.__altura=altura
        
    def set_area(self):
        area = (self.__base * self.__altura)
        self.__area = area
    
    def get_area(self):
        if self.__area:
            return self.__area
        else:
            print("Ejecute .set_area() primero")
    
    def set_perimetro(self):
        
        self.__perimetro = 2*self.__base + 2*self.__altura
            
    def get_perimetro(self):
        if self.__perimetro:
            return self.__perimetro
        else:
            print("Ejecute .set_perimetro() primero")
    
    def __str__(self):
        return f"Rectángulo de altura {self.__altura} y base {self.__base}."


# In[338]:


nuevo_triangulo = Triangulo(4,3,[4,3,5])


# In[339]:


nuevo_triangulo.set_perimetro()
nuevo_triangulo.get_perimetro()


# In[340]:


nuevo_triangulo.set_area()
nuevo_triangulo.get_area()


# In[341]:


nuevo_triangulo.__str__()


# In[342]:


nuevo_triangulo = Triangulo(4,3,[4,3,10])


# ### Pregunta 2
# 
# Desarrolle en `python` la clase `Operación`, que tiene como atributos dos vectores numéricos $U$ y $V$, luego programe los métodos:
# 
# - Sumar: Suma ambos vectores y devuelve el resultado.
# - Restar: Resta al primer vector el segundo y devuelve el resultado.
# - Multiplicar: Calcula el producto punto entre ambos vectores y devuelve el resultado.
# - Correlacion: Devuelve la correlación entre los dos vectores.
# - Covarianza: Devuelve la covarianza entre los dos vectores.

# In[343]:


class Operacion():
    
    def __init__(self, U, V):
        
        if isinstance(U,np.ndarray) and isinstance(V,np.ndarray):
            self.__U= U
            self.__V= V
            print("Iniciemos con las operaciones")
        else:
            ### Asi iniciamos la clase con dos nmumpy array's y los metodos seran mas faciles.
            raise Exception("Los atributos de inicialización deben ser dos numpy array's")
            
    def sumar(self):
        return self.__U + self.__V
    
    def restar(self):
        return self.__U - self.__V
    
    def multiplicar(self):
        return np.inner(self.__U,self.__V)
    
    def correlacion(self):
        return np.corrcoef(self.__U,self.__V)[0][1]
    
    def covarianza(self):
        return np.cov(self.__U,self.__V)[0][1] ### np.cov regresa una matriz de covarianza simetrica.
    
    def __str__(self):
        return f"""
        Esta es la clase Operacion con los siguientes vectores:
        U: {list(self.__U)}
        V: {list(self.__V)}
        ====================================================================="""


# In[344]:


nueva_op = Operacion(np.array([9,2,2]),np.array([1,1,2]))


# In[345]:


nueva_op.sumar()


# In[346]:


nueva_op.restar()


# In[347]:


nueva_op.multiplicar()


# In[348]:


nueva_op.correlacion()


# In[349]:


nueva_op.covarianza()


# In[350]:


print(nueva_op.__str__())


# ### Pregunta 3
# 
# Programe una clase en Python (estilo pythónico) denominada `Jugadores` que tiene como atributo un dataframe de pandas que permite almacenar una tabla de datos como la incluida en el archivo `Players1.csv`. Además, programe los siguientes métodos:
# 
# - `actualizar_position`: Recibe el nombre del jugador y el nombre de la nueva posición, dicho método actualiza la posición del jugador y la muestra.
# - `resumen_jugador`: Recibe el nombre del jugador y retorna toda su información en formato de texto (str).
# - `resumen_columna`: Recibe el nombre de una columna y si es numérica retorna su promedio, en caso de ser categórica (object) retorna la moda.
# - `cantidad_equipos`: Retorna la cantidad de jugadores que existen en el dataframe por equipo, ejemplo Algeria: 10 jugadores.
# - `agregar_jugador`: Recibe el nombre del jugador y los valores de cada columna del dataframe, añade el jugador a la tabla de datos y retorna un mensaje donde se indica que se agregó correctamente. En el siguiente link se muestran algunas maneras de agregar filas a un dataframe: agregar filas.
# - `eliminar_jugador`: Recibe el nombre del jugador, lo elimina de la tabla de datos y retorna un mensaje donde se indica que se eliminó correctamente. Debe validar que el jugador exista dentro de la tabla de datos.

# In[351]:


class Jugadores():
    
    def __init__(self,df):
        
        if isinstance(df, pd.DataFrame):
            ### El atributo de esta clase es un dataframe
            self.__df = df
        else:
            raise Exception("El objeto para inicializar debe ser un DataFrame de pandas")
            
    def actualizar_posicion(self,jugador,posicion):
        
        try:
            self.__df.loc[self.__df["surname"] == jugador, "position"] = posicion
        except Exception as e:
            raise Exception(f"Error al actualizar el jugador: {e}.")
            
    def resumen_jugador(self,jugador):
        
        jug = self.__df.loc[self.__df["surname"]==jugador,:]
        jug_dict = {}
        for col in jug.columns:
            jug_dict[col]= jug[col].tolist()[0]
        return str(jug_dict)
    
    def resumen_columna(self,columna):
        
        if columna in self.__df.columns:
            
            if self.__df[columna].dtype.kind in 'biufc':
                ### Columna es numerica
                return {
                    f"media de {columna}": np.mean(self.__df[columna])
                }
            else:
                return {
                    f"moda de {columna}": stat.mode(self.__df[columna])
                }
            
        else:
            raise Exception("La columna no pertenece al dataframe!")
            
    def cantidad_equipos(self):
        return self.__df["team"].value_counts()
    
    def agregar_jugador(self, jugador, equipo, posicion, minutos, shots, passes, tackles, saves):
        
        df_agregar= pd.DataFrame({
            "surname":[jugador],
            "team": [equipo],
            "position": [posicion],
            "minutes": [minutos],
            "shots":[shots],
            "passes":[passes],
            "tackles":[tackles],
            "saves":[saves]
        })
        self.__df = self.__df.append(df_agregar, ignore_index = True)
        
    def eliminar_jugador(self,jugador):
        
        if jugador in self.__df["surname"].tolist():
            self.__df = self.__df[self.__df["surname"]!=jugador]
            print(f"El jugador {jugador} fue eliminado exitosamente de la tabla de datos.")
        else:
            raise Exception("El jugador NO está en la tabla de datos")

    def get_df(self):
        return self.__df
    
    def __str__(self):
        return f"""
        Esta es la clase Jugadores.
        - El dataframe tiene las siguientes columnas:
        {self.__df.columns.tolist()}
        - Hay {self.__df.shape[0]} jugadores de {len(self.__df['team'].unique().tolist())} paises.
        - Así se ven las primeras filas del dataframe de Jugadores:
        
        {self.__df.head(5)}
          
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        


# In[352]:


data = pd.read_csv("Players1.csv")
data.head(5)


# In[353]:


jugadores = Jugadores(data)


# In[354]:


jugadores.actualizar_posicion(jugador="Belhadj",posicion="midfielder")


# In[355]:


jugadores.get_df().head(5)


# In[356]:


jugadores.resumen_jugador(jugador="Chaouchi")


# In[357]:


jugadores.resumen_columna("minutes")


# In[358]:


jugadores.resumen_columna("team")


# In[359]:


jugadores.cantidad_equipos()


# In[360]:


jugadores.agregar_jugador(jugador="Jimmy", equipo="CR", posicion="defender",
                          minutos=0, shots=0, passes=0, tackles=0, saves=0)


# In[361]:


jugadores.get_df().tail()


# In[362]:


jugadores.eliminar_jugador("Jimmy")


# In[363]:


jugadores.get_df().tail()


# In[364]:


print(jugadores.__str__())


# ### Pregunta 4
# 
# Desarrolle una clase denominada `Análisis` que tiene como atributo una matriz numérica tipo `numpy` y defina los siguientes métodos: 
# 
# - `as_data_frame`: retorna la matriz convertida en DataFrame de pandas.
# - `desviacion_estandar`: retorna un diccionario con la Desviación Estándar de cada columna.
# - `varianza`: retorna un diccionario con la Varianza de cada columna.
# - `moda`: retorna un diccionario con la Moda de cada columna.
# - `maximo`: retorna el máximo de toda la matriz.
# - `buscar`: recibe un número y busca el valor en la matriz, retorna los índices del primer valor encontrado o `None` en caso de no encontrar el valor.

# In[365]:


class Analsis():
    
    def __init__(self,matriz):
        if isinstance(matriz,np.matrix):
            self.__matriz = matriz
        else:
            raise Exception("Debe inicializar con una matriz de numpy")
            
    def as_data_frame(self, **kwargs):
        
        ### Hacemos un dataframe y nombramos las columnas por defecto.
        ### Podemos agregar la opcion de especificar columnas.
        
        nombres_columnas = [f'X{i}' for i in range(self.__matriz.shape[1])]
        
        if "columns" in list(kwargs.keys()) and isinstance(kwargs["columns"], list):
            ### Si hay columnas, son lista y tienen la longitud igual a la de las columnas de la matriz
            ### Guardamos los nombres.
            if len(kwargs["columns"])==self.__matriz.shape[1]:
                nombres_columnas = kwargs["columns"]
        
        dataframe = pd.DataFrame(self.__matriz, columns = nombres_columnas)
        
        ### Al crear un dataframe de una matriz pueden haber problemas con los data types.
        
        
        self.__dataframe = dataframe 
        
        return dataframe
    
    ### El resto de metodos utilizan el dataframe almacenado como objeto de la clase. Por lo que hay que correr
    ### as_data_frame() primero.
    
    def desviacion_estandar(self):
        
        desv_est_dict={}
        
        for col in self.__dataframe.columns.tolist():
            try:
                desv_est_dict[col] = np.std(self.__dataframe[col])
            except:
                continue
                
        return desv_est_dict
    
    def varianza(self):
        
        var_dict=self.desviacion_estandar()
        for key in list(var_dict.keys()):
            var_dict[key] = var_dict[key]**2
        return var_dict
    
    def moda(self):
        
        moda_dict={}
        
        for col in self.__dataframe.columns.tolist():
            try:
                moda_dict[col] = stat.mode(self.__dataframe[col])
            except:
                continue
                
        return moda_dict
    
    def maximo(self):
        ### una matriz tiene una lista de matrices de 1 fila.
        ### Cada fila la convertimos a un array y le sacamos su maximo.
        ### Luego sacamos el maximo de todo eso.
        return max([max(np.array(A[i])[0]) for i in range(len(A))]) 
        
    
    def buscar(self,numero):
        
        indice= None
        m= self.__matriz.shape[0] ## Numero de filas
        n= self.__matriz.shape[1] ## Numero de columnas
        
        encontrado = False
        for i in range(m):
            for j in range(n):
                if A[i,j]==numero and not encontrado:
                    encontrado=True
                    indice = (i,j)

        return indice  ### Regresará None si no se encontró, y regresará el primer indice que encuentre.        
    
    def __str__(self):
        print(f"""
        Esta es la clase Analisis.
        Con la siguiente matriz de numpy:
        {self.__matriz}
        >>>>>>>>>>>>>><<<<<<<<<<<<<<
        """)


# In[366]:


A=np.matrix([[2022, 1, 34.5, 42.23], 
             [2021, 2, 33.22, 39.78],
             [2022, 3, 12.5, 22.90], 
             [2021, 4, 7.47, 12.5],
            ])
anali = Analsis(A)


# In[367]:


A


# In[368]:


df=anali.as_data_frame(columns=["Year","Season","TempMin","TempMax"])
df


# In[369]:


anali.desviacion_estandar()


# In[370]:


anali.varianza()


# In[371]:


anali.moda()


# In[372]:


anali.maximo()


# In[373]:


anali.buscar(12.5)


# In[374]:


anali.buscar(2023)


# ### Pregunta 5
# 
# Una Cadena de Cine desea implementar un sistema para emitir boletos de cine con los siguientes
# datos:
# 
# - Los Boletos de Cine tienen: `NombreCliente`, `DirecciónCliente`, `TeléfonoCliente`, `ValorBoleto`, `ImpuestoVenta`, `PrecioPagar` = `ValorBoleto` + `ImpuestoVenta`, `asiento`, `hora`, `fecha`, además para cada uno se almacena también la información de la película: `título`, `duración` y `productor`.
# - Existen también Boletos de Cine para Clientes Frecuentes que tienen además un descuento, de modo tal que `PrecioPagar` = `ValorBoleto` + `ImpuestoVenta` - `descuento * ValorBoleto`.
# - Los Boletos de Cine para Ejecutivos tienen también una lista de Alimentos Extras, donde cada alimento extra tiene `Código`, `Descripción` y `Precio`, así para los Boletos de Cine para Ejecutivos se tiene que el `TotalAlimentos` = suma de cada uno de los precios de la lista de alimentos, y el precio a pagar del boleto se calcula como sigue: `PrecioPagar` = `ValorBoleto` + `ImpuestoVenta` + `TotalAlimentos`.
# 
# El diseño UML del problema anterior se muestra en la figura siguiente. Programe en `Python` esta jerarquía de clases (diseño en documento de la tarea).

# In[568]:


class BoletoCine():
    
    def __init__(self, cliente, pelicula, valorBoleto, impuestoVenta, asiento, hora, fecha):
        
        self.__cliente = cliente
        self.__pelicula = pelicula
        
        self.__valorBoleto= valorBoleto 
        self.__impuestoVenta= impuestoVenta
        self.__asiento= asiento 
        self.__hora= hora 
        self.__fecha= fecha
        
    def get_precioPagar(self):
        return self.__valorBoleto + self.__impuestoVenta  ### Precio a Pagar por defecto.
        
    def __str__(self):
        return f"""
        === Boleto de Cine ===
        
        Cliente: {self.__cliente.__str__()}
        Pelicula: {self.__pelicula.__str__()}
        
        Valor Boleto: {self.__valorBoleto}
        ImpuestoVenta: {self.__impuestoVenta}
        Asiento: {self.__asiento}
        Hora: {self.__hora}
        Fecha: {self.__fecha}
        
        ---- Boleto Base ----
        
        Total por Pagar: {self.get_precioPagar()}
        
        >>>>> Gracias por su compra! <<<<<
        """
    
class BoletoEjecutivo(BoletoCine):
    
    def __init__(self, cliente, pelicula, valorBoleto, impuestoVenta, asiento, hora, fecha):
        super().__init__(cliente, pelicula, valorBoleto, impuestoVenta, asiento, hora, fecha)
        self.__listaAlimentos =[]
        
    def agregar_alimento(self,alimento):
        self.__listaAlimentos.append(alimento)
        
    def get_totalAlimento(self):
        ### Cambié las referencias aquí para poder accesar a estos atributos con herencia.
        return sum([alimento._AlimentoExtra__precio for alimento in self._BoletoEjecutivo__listaAlimentos])
    
    def get_precioPagar(self):
        precioPagar_base = super().get_precioPagar()
        return precioPagar_base + self.get_totalAlimento()
    
    def __str__(self):
        
        mensaje=super().__str__().split("---- Boleto Base ----")[0]
        
        ### Quitamos la parte de Boleto Base para sustituirla con el nuevo precio por Pagar ###
        
        mensaje=mensaje+f"""
        ---- Boleto Ejecutivo ----
        Alimentos Extra adicionados:"""
        
        for al in self.__listaAlimentos:
            mensaje=mensaje+f"""
            - {al.__str__()}"""
            
        mensaje=mensaje+f"""
        
        Total por Pagar: {self.get_precioPagar()}
        
        >>>>> Gracias por su compra! <<<<<
        """
        
        return mensaje
        
class BoletoClienteFrecuente(BoletoCine):
    
    def __init__(self, cliente, pelicula, valorBoleto, impuestoVenta, asiento, hora, fecha, descuento):
        super().__init__(cliente, pelicula, valorBoleto, impuestoVenta, asiento, hora, fecha)
        self.__descuento= descuento
        
    def get_precioPagar(self):
        precioPagar_base = super().get_precioPagar()
        return precioPagar_base - self.__descuento*self._BoletoCine__valorBoleto ### Cambié esto para poder accesar a este atributo
    
    def __str__(self):
        mensaje=super().__str__().split("---- Boleto Base ----")[0]
        
        ### Quitamos la parte de Boleto Base para sustituirla con el nuevo precio por Pagar ###
        
        mensaje=mensaje+f"""
        ---- Boleto Cliente Frecuente ----
        Descuento: {self.__descuento*100} %
        
        Total por Pagar: {self.get_precioPagar()}
        
        >>>>> Gracias por su compra! <<<<<
        """
        return mensaje
        
class AlimentoExtra():
    
    def __init__(self, codigo, descripcion, precio):
        self.__codigo=codigo
        self.__descripcion=descripcion
        self.__precio=precio
        
    def __str__(self):
        return f"Alimento: código: {self.__codigo}, descripción: {self.__descripcion}, precio: {self.__precio}"
        
class Cliente():
    
    def __init__(self, nombre, direccion, telefono):
        self.__nombre = nombre
        self.__direccion = direccion
        self.__telefono = telefono
        
    def __str__(self):
        return f"{self.__nombre}, dirección: {self.__direccion}, teléfono: {self.__telefono}."
        
class Pelicula():
    
    def __init__(self, titulo, duracion, director):
        self.__titulo= titulo
        self.__duracion= duracion
        self.__director= director
        
    def __str__(self):
        return f"{self.__titulo}, duración: {self.__duracion}, director: {self.__director}."
        


# Fui al cine a ver 'The Shinning' con Maria y Juan.
# 
# - Compré un tiquete simple.
# - María pidió unas papas tostadas.
# - Juan es cliente frecuente del cine.

# In[569]:


cliente1 = Cliente("Jimmy","Costa Rica","99999999")
cliente2 = Cliente("Maria","Indonesia","77777777")
cliente3 = Cliente("Juan", "Australia","88888888")
shinning = Pelicula("The Shinning","2h 26min","Stanley Kubrick")


# In[570]:


boleto_jimmy = BoletoCine(cliente=cliente1,
                          pelicula=shinning,
                          valorBoleto=3670,
                          impuestoVenta=0.13,
                          asiento="2D",
                          hora="8:00pm",
                          fecha="03/09/2022")


# In[571]:


print(boleto_jimmy.__str__())


# In[572]:


alimento_maria = AlimentoExtra(codigo="AS234", descripcion="papitas tostadas", precio=700)


# In[573]:


boleto_maria = BoletoEjecutivo(cliente=cliente2,
                               pelicula=shinning,
                               valorBoleto=3670,
                               impuestoVenta=0.13,
                               asiento="2E",
                               hora="8:00pm",
                               fecha="03/09/2022")


# In[574]:


boleto_maria.agregar_alimento(alimento_maria)


# In[575]:


print(boleto_maria.__str__())


# In[576]:


3670+0.13+700


# In[577]:


boleto_juan = BoletoClienteFrecuente(cliente=cliente3,
                                     pelicula=shinning,
                                     valorBoleto=3670,
                                     impuestoVenta=0.13,
                                     asiento="2E",
                                     hora="8:00pm",
                                     fecha="03/09/2022",
                                     descuento=0.25)


# In[578]:


print(boleto_juan.__str__())


# In[579]:


3670+0.13 - 3670*0.25


# ### Pregunta 6
# 
# Una mecánica desea implementar un sistema para gestionar sus repuestos con los siguientes
# datos:
# 
# a) Se venden repuestos los cuales tienen al menos un código, un nombre y un precio base. Hay repuestos originales y no originales. Los repuestos originales tienen un plazo de garantía dado en años, una fecha de fabricación (día, mes y año) y su precio de venta se calcula sumando al precio base un 25% de utilidad. Mientras que los repuestos no originales tienen un precio de venta que se calcula sumando al precio base un 10% de utilidad.
# 
# b) Los repuestos no originales se dividen en repuestos usados y repuestos nuevos. Los repuestos no originales nuevos tienen también un número de años de garantía, un fabricante y además su precio de venta se calcula sumando al precio un 5% de utilidad adicional por cada año de garantía.
# 
# c) Los repuestos usados tienen una lista de proveedores, donde para cada proveedor se almacena el código, nombre, el país de origen y un índice que indica el nivel de calidad del repuesto, además los repuestos usados tienen una fecha de facturación, una fecha de test (en la cual se probó el repuesto que tiene además un plazo máximo dado de días que establece para el periodo de garantía).
# 
# El diseño UML del problema anterior se muestra en la figura siguiente. Programe en Python
# esta jerarquía de clases. (Figura está en el enunciado de la Tarea).

# *** Observaciones ***
# 
# El diagrama presentado en la tarea no parece concordar con las instrucciones. Por ejemplo, en el mismo es imposible obtener un repuesto No original y nuevo. (La clase Nuevo debería heredar de la clase No_original para poder obtener esto, o en otras palabras la clase No_original debería tener dos subclases: Nuevo y Usado.
# 
# Lo que hice fue programar el diagrama de clases tal y como se muestra en la figura de la tarea.

# In[696]:


class Repuesto():
    
    def __init__(self,codigo,nombre,precio):
        self.__codigo= codigo
        self.__nombre= nombre
        self.__precio= precio
        
    def __str__(self):
        return f"""
        - Repuesto Base. Código: {self.__codigo}, Nombre: {self.__nombre}, Precio: {self.__precio}
        =====================================================================================
        """
    
class No_Original(Repuesto):
    
    def __init__(self,codigo,nombre,precio,paisFabricacion):
        super().__init__(codigo,nombre,precio)
        self.__paisFabricacion = paisFabricacion
        
    def calcular_precio(self):
        return self._Repuesto__precio + 0.10*self._Repuesto__precio ### Repuestos No originales suman un 10% de Utilidad
    
    def __str__(self):
        return super().__str__() + f"""
        - Repuesto No original: Pais Fabricacion: {self._No_Original__paisFabricacion}.
        Precio: {self.calcular_precio()}.
        =====================================================================================
        """
    
class Nuevo(Repuesto):
    def __init__(self,codigo,nombre,precio,aniosGarantia):
        super().__init__(codigo,nombre,precio)
        self.__aniosGarantia = aniosGarantia
        
    def calcular_precio(self):
        return self._Repuesto__precio
        
    def __str__(self):
        return super().__str__()+f"""
        - Repuesto Nuevo: 
        Años Garantía: {self._Nuevo__aniosGarantia}
        Precio: {self.calcular_precio()}
        ====================================================================================
        """
class Usados(No_Original):
    def __init__(self,codigo,nombre,precio,paisFabricacion,fecha,fechatest,lista_proovedores):
        super().__init__(codigo,nombre,precio,paisFabricacion)
        self.__fecha= fecha
        self.__fechatest= fechatest
        self.__listaProovedores = lista_proovedores
        
    def calcular_precio(self):
        return super().calcular_precio() + 0.5*super().calcular_precio()
    
    ### Repuestos No originales usados suman un 5% de Utilidad
    """
    En el enunciado dice que son los no originales nuevos, pero no es posible tener estos deacuerdo al diagrama 
    de clases UML dado en la tarea
    """
    
    def __str__(self):
        mensaje =  super().__str__()+f"""
        - Repuesto Usado: 
        Precio: {self.calcular_precio()}
        Fecha: {self._Usados__fecha.__str__()}
        FechaTest: {self._Usados__fechatest.__str__()}
        Proovedores:""" 
        for proov in self._Usados__listaProovedores:
            mensaje= mensaje+f"""
            - {proov.__str__()}"""
            
        mensaje=mensaje+"""
        =======================================================================================
        """
        return mensaje
    
class No_Nuevo(No_Original):
    
    """
    El diagrama señala a No_Nuevo como subclase de Nuevo (????)
    Solo lo hice como subclase de No_Original
    """
    def __init__(self,codigo,nombre,precio, fabricante):
        super().__init__(codigo,nombre,precio,paisFabricacion)
        self.__Fabricante = fabricante
        
    def calcular_precio(self):
        return super()._Repuesto__precio
        
    def __str__(self):
        super().__str__()+f"""
        - Repuesto No Nuevo: 
        Precio: {self._calcular_precio()}
        Fabricante: {self._No_Nuevo__Fabricante.__str__()}
        ===================================================
        """

class Original(Nuevo):
    def __init__(self,codigo,nombre,precio,aniosGarantia, fabricanteOriginal, fecha):
        super().__init__(codigo,nombre,precio,aniosGarantia)
        self.__fabricanteOriginal = fabricanteOriginal  ### Objeto de la clase FabricanteOriginal
        self.__fecha = fecha
        
    def calcular_precio(self):
        return super().calcular_precio() + 0.5*super().calcular_precio()
    
    ### Repuestos No originales usados suman un 5% de Utilidad
    """
    En el enunciado dice que son los no originales nuevos, pero no es posible tener estos deacuerdo al diagrama 
    de clases UML dado en la tarea
    """
    
    def __str__(self):
        return super().__str__()+f"""
        - Repuesto Original: 
        Precio: {self.calcular_precio()}
        Fabricante Original: {self._Original__fabricanteOriginal.__str__()}
        Fecha: {self._Original__fecha.__str__()}
        """
    
class Fecha():
    
    def __init__(self,dia,mes,anno):
        self.__dia=dia
        self.__mes=mes
        self.__anno=anno
        
    def __str__(self):
        return f"{self.__dia}/{self.__mes}/{self.__anno}"
    
class FechaTest():
    
    def __init__(self,numerodias):
        self.__numerodias=numerodias
    
    def __str__(self):
        return f"Número de días: {self.__numerodias}."
    
class Fabricante():
    
    def __init__(self,nombre,pais):
        self.__nombre=nombre
        self.__pais=pais
        
    def __str__(self):
        return f"Nombre fabricante: {self.__nombre}, país: {self.__pais}."
    
class Proveedor(Fabricante):
    
    def __init__(self,nombre,pais,codigo,indice):
        super().__init__(nombre,pais)
        self.__codigo=codigo
        self.__indice=indice
        
    def __str__(self):
        return super().__str__()+f"""
        Proovedor: Código {self.__codigo}, Índice {self.__indice}
        """
    
class Fabricante_Original(Fabricante):
    
    def __init__(self,nombre,pais,direccion,telefono,email):
        super().__init__(nombre,pais)
        self.__direccion=direccion
        self.__telefono=telefono
        self.__email=email
        
    def __str__(self):
        return super().__str__()+f"""
        Fabricante Original: dirección: {self.__direccion}, teléfono: {self.__telefono} email: {self.__email}
        """


# #### Ejemplo
# Creamos un repuesto No Original Usado

# In[697]:


### Repuesto Base
repuesto_base = Repuesto(codigo="A123",nombre="ValvulaA123",precio=200)
print(repuesto_base.__str__())


# In[698]:


### Repuesto No Original
repuesto_no_original = No_Original(codigo="A123",nombre="ValvulaA123",precio=200,paisFabricacion="China")
print(repuesto_no_original.__str__())


# In[699]:


### Repuesto No Original Usado ###
repuesto_no_original_usado= Usados(codigo="A123",
                           nombre="ValvulaA123",
                           precio=200,
                           paisFabricacion="China",
                           fecha=Fecha(dia="12",mes="12",anno="1997"),
                           fechatest=FechaTest(numerodias=1900),
                           lista_proovedores=[Proveedor(nombre="JGH",pais="Japón",codigo="1777",indice="A98")])
print(repuesto_no_original_usado.__str__())


# ### Otro ejemplo: un repuesto Original y Nuevo

# In[700]:


repuesto_original_nuevo = Original(codigo="DF455",
                                   nombre="Tornillo",
                                   precio=15,
                                   aniosGarantia=1,
                                   fabricanteOriginal=Fabricante_Original(nombre="FFF",
                                                                          pais="Austria",
                                                                          direccion="123GeduldigkeitStrasse",
                                                                          telefono=29292929,
                                                                          email="ichkommeausoesterreich@gmail.com"),
                                   fecha=Fecha(dia="12",
                                               mes="11",
                                               anno="2007")
                                  )


# In[701]:


print(repuesto_original_nuevo.__str__())


# In[695]:


15+15*0.5

