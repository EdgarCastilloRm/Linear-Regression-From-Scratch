#Edgar Castillo Ramírez A00827826
#28 de agosto del 2022
#Modelo de Regresión lineal sin librerías de Machine Learning (VERSION 1.0)

#Librerías
import pandas as pd #Para manejar dataset
import matplotlib.pyplot as plt #Para graficar

#Se importa el dataset
df = pd.read_csv("Fish.csv")

#Se imprimen los primeros 5 valores para ver los datos
print(df.head())

#Se despliega una gráfica que muestra los puntos(X,Y) a utilizar en el modelo.
plt.scatter(df.Length1, df.Weight)
plt.show()

#Se llama esta función para revisar que las columnas a utilizar tengan todos sus valores.
print(df.info())

#Como bien se sabe, es necesario usar el gradiente descendiente para minimizar la función de error.

#Recordemos que para regresión lineal se tiene y = mx + b y lo único con lo que se puede jugar para
#tener una mejor función es con m y b.

def gradiente_descendiente(m, b, valores, L):
    gradienteM = 0
    gradienteB = 0

    valoresCont = len(valores) #Total de puntos que se tienen

    for i in range(valoresCont):
        x = valores.iloc[i].Length1
        y = valores.iloc[i].Weight
        #Estas formulas provienen de derivar el MSE respecto a "m" y "b"
        gradienteM += -(2/valoresCont) * x * (y - (m*x + b))
        gradienteB += -(2/valoresCont) * (y - (m*x + b))
    
    #Al terminar la sumatoria se cambia de dirección ya que no se quiere ir a la distancia más lejana
    mFinal = m - gradienteM * L
    bFinal = b - gradienteB * L
    return mFinal, bFinal


m = 0
b = 0
L = 0.001 #Learning rate
epochs = 1000 #epochs (ir y venir)

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradiente_descendiente(m,b,df,L)

print(m,b)

#Se imprime la comparación de los datos
plt.scatter(df.Length1, df.Weight) #originales
plt.scatter(df.Length1, [m * df.Length1 + b], color="red") #predecidos
plt.show()

#He observado que algunas combinaciones de columnas no son permitidas, ya que necesitan de librerías de alta matemática.