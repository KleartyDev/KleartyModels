import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle


# SELECCION DE VARIABLES

# Selecciona las n variables mas importantes
def select_features(data, target_column, n_features):

    # Separando las características y el objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Creando el modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Ajustando el modelo a los datos
    model.fit(X, y)

    # Obteniendo la importancia de las características
    importances = model.feature_importances_

    # Creando un DataFrame para visualizar la importancia
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    
    # Ordenando las características por importancia y seleccionando las 'n_features' más importantes
    selected_features = feature_importance.sort_values(by='Importance', ascending=False).head(n_features)['Feature']

    # Retornando el DataFrame con las características seleccionadas
    return data[selected_features.tolist() + [target_column]]





# Prueba 2 modelos y selecciona el que de mayor acccuracy, luego lo exporta. Retorna nombre de archivo y accuracy
def test_and_export_best_model(data, target_column, file_path):

    # Separando las características y el objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print("columnas X:",X.columns)
    #print("columnas y:",y.columns)

    # Dividiendo los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializando los modelos
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Entrenando y evaluando Random Forest
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Entrenando y evaluando XGBoost
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)

    # Seleccionando el mejor modelo
    if rf_accuracy > xgb_accuracy:
        best_model = rf_model
        best_accuracy = rf_accuracy
        model_name = 'RandomForest'
    else:
        best_model = xgb_model
        best_accuracy = xgb_accuracy
        model_name = 'XGBoost'

    # Exportando el mejor modelo como un archivo pickle
    pickle.dump(best_model, open(file_path + model_name + '.pkl', 'wb'))

    return model_name, best_accuracy