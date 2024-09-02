import numpy as np
import pandas as pd
import pickle

# Cargar el modelo entrenado
with open('./model/decision_tree_model.pkl', 'rb') as f:
    tree = pickle.load(f)

dfTest = pd.read_csv('./data/test.csv')

Xtest = dfTest.values

# Hacer predicciones con el modelo
YpredTest = tree.predecir(Xtest)

# Mostrar las predicciones
for i, pred in enumerate(YpredTest):
    if pred > 0:
        print(f"Celular {i+1}: Precio alto")
    else:
        print(f"Celular {i+1}: Precio bajo")
