import pandas as pd
import numpy as np
import pickle
from model import ArbolDecision

# Función para calcular la matriz de confusión
def matrizConfusion(yVerdadero, yPredicho):
    tp = np.sum((yVerdadero == 1) & (yPredicho == 1))
    tn = np.sum((yVerdadero == 0) & (yPredicho == 0))
    fp = np.sum((yVerdadero == 0) & (yPredicho == 1))
    fn = np.sum((yVerdadero == 1) & (yPredicho == 0))
    return np.array([[tn, fp], [fn, tp]])

# Función para calcular la precisión
def precision(matrizConf):
    tp = matrizConf[1, 1]
    fp = matrizConf[0, 1]
    return tp / (tp + fp)

# Función para calcular el recall
def recall(matrizConf):
    tp = matrizConf[1, 1]
    fn = matrizConf[1, 0]
    return tp / (tp + fn)

# Función para calcular el F1-Score
def f1Score(prec, rec):
    return 2 * (prec * rec) / (prec + rec)

# Cargar el dataset categorizado
df = pd.read_csv('./data/train.csv')

# Convertir la columna price_range a una categoría de precio (0: bajo, 1: alto)
df['categoriaPrecio'] = df['price_range'].apply(lambda x: 1 if x >= 2 else 0)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['price_range', 'categoriaPrecio']).values
y = df['categoriaPrecio'].values

# División del dataset en conjuntos de entrenamiento, validación y prueba
def dividirDataset(X, y, tamanoEntrenamiento=0.7, tamanoValidacion=0.15):
    finEntrenamiento = int(tamanoEntrenamiento * len(X))
    finValidacion = finEntrenamiento + int(tamanoValidacion * len(X))
    
    XEntrenamiento, XValidacion, XPrueba = X[:finEntrenamiento], X[finEntrenamiento:finValidacion], X[finValidacion:]
    yEntrenamiento, yValidacion, yPrueba = y[:finEntrenamiento], y[finEntrenamiento:finValidacion], y[finValidacion:]
    
    return XEntrenamiento, XValidacion, XPrueba, yEntrenamiento, yValidacion, yPrueba

# Dividir el dataset
XEntrenamiento, XValidacion, XPrueba, yEntrenamiento, yValidacion, yPrueba = dividirDataset(X, y)

# Crear y entrenar el árbol de decisión
arbol = ArbolDecision(profundidadMaxima=5, minMuestrasDivision=2)
arbol.entrenar(XEntrenamiento, yEntrenamiento)

# Hacer predicciones en el conjunto de validación
yPredValidacion = arbol.predecir(XValidacion)

# Evaluar el modelo en el conjunto de validación
matrizConfValidacion = matrizConfusion(yValidacion, yPredValidacion)
precValidacion = precision(matrizConfValidacion)
recValidacion = recall(matrizConfValidacion)
f1Validacion = f1Score(precValidacion, recValidacion)

print(f"Matriz de Confusión (Validación):\n{matrizConfValidacion}")
print(f"Precisión (Validación): {precValidacion}")
print(f"Recall (Validación): {recValidacion}")
print(f"F1-Score (Validación): {f1Validacion}")

# Hacer predicciones en el conjunto de prueba
yPredPrueba = arbol.predecir(XPrueba)

# Evaluar el modelo en el conjunto de prueba
matrizConfPrueba = matrizConfusion(yPrueba, yPredPrueba)
precPrueba = precision(matrizConfPrueba)
recPrueba = recall(matrizConfPrueba)
f1Prueba = f1Score(precPrueba, recPrueba)

print(f"Matriz de Confusión (Prueba):\n{matrizConfPrueba}")
print(f"Precisión (Prueba): {precPrueba}")
print(f"Recall (Prueba): {recPrueba}")
print(f"F1-Score (Prueba): {f1Prueba}")

# Guardar el modelo entrenado
with open('./model/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(arbol, f)