import numpy as np
import math

class ArbolDecision:
    def __init__(self, profundidadMaxima=None, minMuestrasDivision=2):
        self.profundidadMaxima = profundidadMaxima
        self.minMuestrasDivision = minMuestrasDivision
        self.arbol = None

    def calcularEntropia(self, y):
        cuentaClases = np.bincount(y)
        probabilidades = cuentaClases / len(y)
        # Calcula la entropía, que mide la incertidumbre en la distribución de las clases.
        return -np.sum([p * math.log2(p) for p in probabilidades if p > 0])

    def gananciaInformacion(self, X, y, indiceCaracteristica, umbral):
        entropiaPadre = self.calcularEntropia(y)
        
        # Dividir los datos en base al umbral dado.
        mascaraIzquierda = X[:, indiceCaracteristica] < umbral
        mascaraDerecha = ~mascaraIzquierda
        yIzquierda = y[mascaraIzquierda]
        yDerecha = y[mascaraDerecha]

        # Calcular la entropía de los nodos hijos y la ganancia de información.
        n = len(y)
        nIzquierda, nDerecha = len(yIzquierda), len(yDerecha)
        if nIzquierda == 0 or nDerecha == 0:
            return 0

        entropiaHijos = (nIzquierda / n) * self.calcularEntropia(yIzquierda) + (nDerecha / n) * self.calcularEntropia(yDerecha)
        ganancia = entropiaPadre - entropiaHijos
        return ganancia

    def mejorDivision(self, X, y):
        mejorGanancia = 0
        mejorIndiceCaracteristica = None
        mejorUmbral = None

        # Recorre todas las características y sus valores únicos para encontrar la mejor división.
        for indiceCaracteristica in range(X.shape[1]):
            umbrales = np.unique(X[:, indiceCaracteristica])
            for umbral in umbrales:
                ganancia = self.gananciaInformacion(X, y, indiceCaracteristica, umbral)
                # Si se encuentra una ganancia mejor, se actualiza el mejor umbral y característica.
                if ganancia > mejorGanancia:
                    mejorGanancia = ganancia
                    mejorIndiceCaracteristica = indiceCaracteristica
                    mejorUmbral = umbral
        
        return mejorIndiceCaracteristica, mejorUmbral

    def construirArbol(self, X, y, profundidad=0):
        nMuestras, nCaracteristicas = X.shape
        nEtiquetas = len(np.unique(y))

        # Condiciones de parada: profundidad máxima, una sola clase o pocas muestras.
        if profundidad >= self.profundidadMaxima or nEtiquetas == 1 or nMuestras < self.minMuestrasDivision:
            valorHoja = self.etiquetaMasComun(y)
            return valorHoja

        # Seleccionar la mejor división para el nodo actual.
        mejorIndiceCaracteristica, mejorUmbral = self.mejorDivision(X, y)
        if mejorIndiceCaracteristica is None:
            return self.etiquetaMasComun(y)

        # Dividir el conjunto de datos en dos ramas y construir recursivamente los subárboles.
        indicesIzquierda = X[:, mejorIndiceCaracteristica] < mejorUmbral
        indicesDerecha = ~indicesIzquierda

        arbolIzquierdo = self.construirArbol(X[indicesIzquierda], y[indicesIzquierda], profundidad + 1)
        arbolDerecho = self.construirArbol(X[indicesDerecha], y[indicesDerecha], profundidad + 1)

        return (mejorIndiceCaracteristica, mejorUmbral, arbolIzquierdo, arbolDerecho)

    def etiquetaMasComun(self, y):
        # Encuentra y retorna la etiqueta más común en el conjunto de datos.
        return np.bincount(y).argmax()

    def entrenar(self, X, y):
        # Construye el árbol y lo guarda en self.arbol.
        self.arbol = self.construirArbol(X, y)

    def predecirMuestra(self, x, arbol):
        if not isinstance(arbol, tuple):
            return arbol

        # Desempaqueta la información del nodo y decide la dirección en base al umbral.
        indiceCaracteristica, umbral, arbolIzquierdo, arbolDerecho = arbol
        if x[indiceCaracteristica] < umbral:
            return self.predecirMuestra(x, arbolIzquierdo)
        else:
            return self.predecirMuestra(x, arbolDerecho)

    def predecir(self, X):
        # Aplica la predicción para cada muestra en el conjunto de datos.
        return np.array([self.predecirMuestra(x, self.arbol) for x in X])
