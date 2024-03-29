import numpy as np 
import pandas as pd
import json
from datetime import datetime
import time

from parametros import *

from modelos import *
from eda import *


loggeo_acumulador = ""


def logg(texto):
    global loggeo_acumulador
    # Agrega el texto a la variable global con dos saltos de línea
    loggeo_acumulador += texto + "\n\n"


def read_file(file_name,file_path):
    
    if file_name.split('.')[-1]=='csv':
        df = pd.read_csv(file_path)
        print("el archivo se pudo leer con exito")
        logg("el archivo se pudo leer con exito")
        return df
    else:
        print("la extencion del archivo no es valida")
        logg("la extencion del archivo no es valida")

def set_target(df,target):
    df.rename(columns = {target:'y'}, inplace = True)
    return df


def eda(df,gpt =False):
        
    shape = df.shape
    print("Cantidad de filas:",shape[0],"\nCantidad de columnas:",shape[1])
    print("Nombres de columnas:", list(df.columns))
    total_null_value = df.isnull().sum().sum()
    pct_null_value = total_null_value / (shape[0]*shape[1])
    print("Cantidad de valores nulos",total_null_value)
    print("Porcentaje de valores nulos",pct_null_value)
    print(df.info() )

    #status = status(df)
    #print("status",status)

    # Limpieza y adecuacion de datos

    df = standarizar_categorias(df)
    df = convert_object_columns_to_numeric(df)

    df = eliminate_nulls(df)

    df = drop_columns_with_id(df)

    dicc_eda = eda_type_describe_value_counts(df)

    #status2 = status(df)
    #print("nuevo status",status2)



    print("Primera Adecuacion terminada \n Cantidad de filas:",shape[0],"\nCantidad de columnas:",shape[1])
    return df ,dicc_eda 


## Bloque principal




def algoritmo_principal(df,target_name,gpt, filename):


    fecha_hoy = datetime.today().strftime('%Y-%m-%d')
    print("La fecha de hoy es:", fecha_hoy)
    inicio = time.time()

    df = set_target(df,target_name)

    variable_names = list(df.columns)
    shape_inicial = df.shape
    

    df,dicc_eda = eda(df,False)

    shape_sin_nulos = df.shape

    type_variables = {col: str(df[col].dtype) for col in df.columns}

    with open('informacion/dicc_eda.json', 'w') as archivo_json:
        json.dump(dicc_eda, archivo_json)


    # OUTLIERS
    df_without_outliers, df_only_outliers = find_and_remove_outliers(df, dicc_eda, [])
    print("Primera Adecuacion terminada \n Cantidad de filas:",df.shape[0],"\nCantidad de columnas:",df.shape[1])  

    # TRANSFORMACION DE DATOS
    df,category_mappings = transform_categorical_columns(df)



    #df.to_csv('tabla_procesada.csv')  


    # MODELOS


    #data = df.drop(columns=["Unnamed: 0", "Churn_No"])
    #data = df.drop(columns=["Churn_No"])


    selected_data = select_features(df, 'y', 10)

    print("Variables selecccionadas: ",selected_data.columns)
    model_name, best_accuracy = test_and_export_best_model(selected_data, 'y', "modelos_output/modelo_1_")
    print (model_name, best_accuracy)


    real_selected_variables = generate_selected_variables(list(selected_data.columns),variable_names)
    
    variables_info = { "variables_names" : real_selected_variables,
                        'type_variables': type_variables,
                        "category_mappings": category_mappings,
                        "model_name":model_name}

    with open('informacion/variables_info.json', 'w') as archivo_json:
        json.dump(variables_info, archivo_json)

    

    # Calcular el tiempo transcurrido
    fin = time.time()
    tiempo_transcurrido = fin - inicio    


    nueva_fila = {
    'fecha': fecha_hoy,
    'tiempo': tiempo_transcurrido,
    'nombre dataset': filename,
    'cantidad filas': shape_inicial[0],
    'cantidad columnas': shape_inicial[1],
    'accuracy': best_accuracy,
    'modelo': model_name,
    'shape outilers': df_only_outliers.shape,
    'shape luego de la limpieza': shape_sin_nulos
    
        }
    
    insertar_fila_y_guardar_excel("macro informacion modelos.xlsx","Hoja 1",nueva_fila)

    return best_accuracy, selected_data.columns





# MODIFICACIONES

# La variable target tiene que ser un dato de entrada y modificar el nombre con un nombre generico
#  Cuando transformo variables Target no deberia transformarse 

# Revisar los embeddings , Revisar los outilers con otra tabla

# nlp standarizacion para datos categoricos . Argentina vs argentina .eliminacion de espacios; Similitud de palabras
# ver Distancia de Levenshtein


#OPEN AI adecuacion datos - > Argentina vs argentina .eliminacion de espacios; Similitud de palabras
# Preguntas

# Podemos usar github para tener un control de versiones 
# donde podrian correr este modelo mediante flask
# Armamos una interface grafica con flask que se pueda subir tabla y esperar resultado

# Seleccion de variables. Puedo seleccionar variables luego de la transformacion. No puedo asegurar cantidad minima de variables

# Empezar a guardar la informacion. Guardar los logs de Fallo.

# Definir parametria (Podria ser un archivo con MACROS) (Cantidad de variables, % de nulos, etc)



