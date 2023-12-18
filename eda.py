
import numpy as np 
import pandas as pd
import json

import unicodedata # Trabajo con acentos para standarizar categorias






def status(data):

    data2=data

    # total de rows
    tot_rows=len(data2)
    
    # total de nan
    d2=data2.isnull().sum().reset_index()
    d2.columns=['variable', 'q_nan']
    
    # percentage of nan
    d2[['p_nan']]=d2[['q_nan']]/tot_rows
    
    # num of zeros
    d2['q_zeros']=(data2==0).sum().values

    # perc of zeros
    d2['p_zeros']=d2[['q_zeros']]/tot_rows

    # total unique values
    d2['unique']=data2.nunique().values
    
    # get data types per column
    d2['type']=[str(x) for x in data2.dtypes.values]
    
    return(d2)







# Casteo 

def convert_object_columns_to_numeric(df):
    for column in df.select_dtypes(include='object').columns:
        # Compruebo si la columna contiene dígitos y no contiene letras
        contains_digits = df[column].str.contains(r'\d').any()
        contains_no_letters = ~df[column].str.contains(r'[a-zA-Z]').any()

        if contains_digits and contains_no_letters:
            print(f"Convirtiendo la columna '{column}' a numérica")
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df




# Eliminacion de nulos


def eliminate_nulls_coll(df, pct_null):
    if pct_null < 0 or pct_null > 100:
        raise ValueError("pct_null debe estar en el rango de 0 a 100")

    margin = len(df) * pct_null / 100
    columns_to_drop = [col for col in df.columns if df[col].isnull().sum() > margin]
    if columns_to_drop:
        print(f"Se eliminan las columnas: {columns_to_drop}")
    else:
        print("No se eliminan columnas")
    df.drop(columns=columns_to_drop, inplace=True)
    return df


def eliminate_nulls_row(df, pct_coll_nan):
    if pct_coll_nan < 0 or pct_coll_nan > 100:
        raise ValueError("pct_coll_nan debe estar en el rango de 0 a 100")
        
    margin = len(df.columns) * pct_coll_nan / 100
    rows_to_drop = df.index[df.apply(lambda row: row.isnull().sum() > margin, axis=1)]
    num_rows_to_drop = len(rows_to_drop)
    if num_rows_to_drop:
        print(f"Se eliminan {num_rows_to_drop} filas")
    else:
        print("No se eliminan filas")
    df.drop(rows_to_drop, inplace=True)
    return df


def eliminate_nulls(df,p_nulls=50, p_coll_nulls=10):
    print("tamaño antes de eliminar nulos: ", df.shape )

    # Eliminamos las filas con un margen de nulos mayor a p_nulls
    df = eliminate_nulls_row(df, p_nulls)

    # Eliminamos las columnas
    df = eliminate_nulls_coll(df, p_coll_nulls)

    #Eliminamos los nulos restantes y o le aplicamos una funcion para que los complete
    #df = df.fillna(0)
    df = df.dropna()

    print("tamaño despues de eliminar nulos: ", df.shape )
    return df





## elimino columnas cuando todas sus filas son iguales


def drop_columns_with_id(df):
    print("Se esta validando que columnas tienen todos sus elementos distintos")

    columns_to_drop = []
    
    for col in df.columns:
        # Comprobamos si todos los valores son únicos en la columna
        if len(df[col].unique()) == len(df[col]):
            columns_to_drop.append(col)
    if len(columns_to_drop):       
        print("las columnas eliminadas son:",columns_to_drop )
    else: 
        print("No se encontraron columnas para eliminar")
    df_dropped = df.drop(columns=columns_to_drop)
    
    return df_dropped





#FUNCIONES QUE GENERAN INFORMACION

# Determina un eda para las variables categoricas y numericas. Para numericas hace un describe ( determina min, max, mediana, promedio, etc)
# y para las categoricas determina cantidad de valores distintos ( cantidad de categorias) y luego en un sub diccionario determina cuantos 
# valores tiene cada una de las categorias. Esto lo imprime en pantalla y ademas lo guarda en un dicccionario
def eda_type_describe_value_counts(df):
    result_dict = {}
    
    for col in df.columns:
        print(f"--- Descripción de la columna {col} ---")
        
        # Si la columna es numérica
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            desc = df[col].describe()
            unique_count = df[col].nunique()
            print(desc)
            print(f"Valores únicos: {unique_count}")
            result_dict[col] = {
                'tipo_de_dato': 'numérico',
                'unique_count': unique_count,
                'describe': {
                    'count': desc['count'],
                    'mean': desc['mean'],
                    'std': desc['std'],
                    'min': desc['min'],
                    '25%': desc['25%'],
                    '50%': desc['50%'],
                    '75%': desc['75%'],
                    'max': desc['max']
                }
            }
            
        # Si la columna es categórica 
        elif pd.api.types.is_object_dtype(df[col].dtype):
            unique_count = df[col].nunique()
            value_counts = df[col].value_counts(normalize=True).to_dict() # modifico para que traiga pct
            print(f"Valores únicos: {unique_count}")
            print(f"Conteo de valores: {value_counts}")
            
            result_dict[col] = {
                'tipo_de_dato': 'categórico',
                'unique_count': unique_count,
                'value_counts': value_counts
            }
    
    return result_dict




# OUTLIERS

def find_outliers_IQR_method(serie):
    # Calcula el primer y tercer cuartiles
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    
    # Calcula el rango intercuartil
    IQR = Q3 - Q1
    
    # Define límites para los outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Encuentra los outliers
    mask = (serie < lower_bound) | (serie > upper_bound)
    
    return mask


def find_and_remove_outliers(df, dicc_eda, exclude_columns=[]):
    outliers_masks = {}
    outliers_per_row = {}
    
    original_shape = df.shape

    for col, eda_info in dicc_eda.items():
        if col in df.columns and col not in exclude_columns:
            is_numeric = eda_info['tipo_de_dato'] == 'numérico'
            is_many_unique = eda_info['unique_count'] > 10
            value_counts = eda_info.get('value_counts', {})
            
            if is_numeric and is_many_unique:
                outliers_mask = find_outliers_IQR_method(df[col])
                if outliers_mask.shape[0] == 0:
                    print(f"Skipping {col} due to empty mask.")
                    continue
            else:
                rare_categories = {k for k, v in value_counts.items() if v < 0.001}
                outliers_mask = df[col].isin(rare_categories)
            
            outliers_masks[col] = outliers_mask

            # Identificar las filas que tienen outliers en esta columna
            row_indices = np.where(outliers_mask)[0]
            for idx in row_indices:
                if idx in outliers_per_row:
                    outliers_per_row[idx].append(col)
                else:
                    outliers_per_row[idx] = [col]

    # Combinar las máscaras de cada columna
    combined_mask = np.column_stack(list(outliers_masks.values())).any(axis=1)

    # Crear DataFrames con y sin outliers
    df_without_outliers = df[~combined_mask]
    df_only_outliers = df[combined_mask]

    print(f"Tamaño antes de eliminar outliers: {original_shape}")
    print(f"Tamaño después de eliminar outliers: {df_without_outliers.shape}")
    print(f"Cantidad de outliers: {df_only_outliers.shape}")

    # Imprimir las filas y las columnas donde se encontraron outliers
    #for row_idx, cols_with_outliers in outliers_per_row.items():
    #    print(f"Outliers encontrados en fila {row_idx} en las columnas {cols_with_outliers}")

    return df_without_outliers, df_only_outliers


#df_without_outliers, df_only_outliers = find_and_remove_outliers(df, dicc_eda, [])



# TRANSFORMACION DE DATOS

# Transforma los datos categoricos en numericos
def transform_categorical_columns(df):
    processed_columns = []
    category_mappings = {}

    for col in df.columns:
        # Si la columna es de tipo object, asumimos que es categórica
        if df[col].dtype == 'object':
            unique_categories = df[col].unique()
            print("Cantidad de valores únicos en '{}':".format(col), len(unique_categories))

            # Crear mapeo para la columna actual
            col_mapping = {category: idx for idx, category in enumerate(unique_categories)}
            category_mappings[col] = col_mapping

            # Aplicoi el mapeo a la columna
            encoded_col = df[col].map(col_mapping)
            processed_columns.append(encoded_col)
        else:
            # Si no es categórica, se añade la columna tal cual
            processed_columns.append(df[col])

    return pd.concat(processed_columns, axis=1), category_mappings






def eliminar_acentos(texto):

    texto_sin_acentos = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    return texto_sin_acentos

def standarizar_categorias(df):
        # Elimina acentos ; convierto a minuscula, elimino espacios
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].apply(eliminar_acentos).str.lower().str.strip()
    print("columnas standarizadas")
    return df    
