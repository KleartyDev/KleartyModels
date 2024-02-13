import pandas as pd
import numpy as np



# convierte el input del usuario en df con las variables correspondientes
def list_to_df(valores, nombres_columnas):

    if len(valores) != len(nombres_columnas):
        raise ValueError("La longitud de 'valores' y 'nombres_columnas' debe ser la misma.")
    
    # Crear el DataFrame
    df = pd.DataFrame([valores], columns=nombres_columnas)
    return df

# como los datos ingresados por el usuario son un str se tiene que castear cada valor usando la informacion del eda
def convertir_tipos(df, tipos_dict):

    # Filtramos el diccionario de tipos para quedarnos solo con las columnas que existen en el DataFrame
    tipos_necesarios = {clave: tipos_dict[clave] for clave in df.columns if clave in tipos_dict}
    print(tipos_necesarios)
    for columna, tipo in tipos_necesarios.items():
        try:
            if tipo == 'float':
                df[columna] = pd.to_numeric(df[columna], errors='coerce')
            elif tipo == 'int':
                # Primero convertimos a float y luego a int para manejar cadenas vacías y NaNs
                df[columna] = pd.to_numeric(df[columna], errors='coerce').fillna(0).astype(int)
            elif tipo == 'bool':
                # Convertimos valores no booleanos a booleanos según necesidad
                df[columna] = df[columna].apply(lambda x: str(x).lower() in ['true', 'yes', 'si', 'verdadero'])
            else:
                # Para otros tipos, usamos astype directamente ya que no deberían causar problemas
                df[columna] = df[columna].astype(tipo)
        except ValueError as e:
            print(f"Error al convertir columna {columna} a {tipo}: {e}")
            df[columna] = np.nan

    return df

# realiza  un mappeo a todas las variables categoricas convirtiendolas en un numero con la informacion creadad en el modelado del dataset
def mapear_input_df_to_numeric(df, mapping_dict):

    df_mapped = df.copy()
    for column, map_dict in mapping_dict.items():
        if column in df_mapped.columns:
            # Convierte la columna a string si no lo es, y luego aplica el mapeo
            df_mapped[column] = df_mapped[column].astype(str).str.lower().map(map_dict)

    return df_mapped


# df_to_predict es un df con una sola fila (los datos ingresados por el usuario ya casteados y mappeados)
def predict(df_to_predict,model):

    prediction = model.predict(df_to_predict)
    probabilities = model.predict_proba(df_to_predict)

    return (prediction, probabilities)


# determino el nombre correspondinte de las prediccion que realice.
def find_key(mapping, value):
    inverted_mapping = {v: k for k, v in mapping.items()}
    return inverted_mapping.get(value)




