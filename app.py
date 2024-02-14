from flask import Flask, render_template, request, session
import pandas as pd
from algoritmo_principal import *
from predict import *
import json

app = Flask(__name__)
app.secret_key = 'clave_secreta'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        target = request.form['target']
        pro_mode = 'pro_mode' in request.form

        parametros = {"filename": filename, "target": target, "pro_mode": pro_mode}

        # Proceso archivo
        df = pd.read_csv(f)
        columnas = df.columns.tolist()

        # Guardo csv en datos
        df.to_csv("datos_input/" + filename)
        print("archivo guardado")

        metric, variables_principales = algoritmo_principal(df, target, False,filename)
        cant_variables_principales = len(variables_principales)
        print(variables_principales)
        print ("  cantidad:  ") 
        print(cant_variables_principales)
        session['cant_variables_principales'] = cant_variables_principales

        return render_template('resultados.html', parametros=parametros, columnas=columnas,
                               variables_principales=variables_principales, cant_variables_principales=cant_variables_principales,
                               metric=metric)

    return render_template('index.html')

@app.route('/resultados', methods=['GET', 'POST'])
def resultados():

    f = request.files['file']
    filename = f.filename
    target = request.form['target']
    pro_mode = 'pro_mode' in request.form

    parametros = {"filename": filename, "target": target, "pro_mode": pro_mode}

    #if request.method == 'POST':
    #   return render_template('predecir.html', parametros=parametros)
    return render_template('resultados.html', parametros=parametros)


@app.route('/predecir', methods=['POST'])

def predecir():

    # obtengo nombre de variables principales, type de cada variables y mapeo de cada variable
    #{ "variables_names" : real_selected_variables,'type_variables' : type_variables, "category_mappings": category_mappings
    with open("informacion/variables_info.json", 'r', encoding='utf-8') as file:
        variables_info = json.load(file)

    variables_principales = variables_info['variables_names']
    cant_variables = len (variables_principales) #session.pop('cant_variables_principales', None)

    return render_template('predecir.html', variables=variables_principales, cant_variables=cant_variables)

@app.route('/realizar_prediccion', methods=['POST'])
def realizar_predeciccion():

    #Vamos a guardar el input que ingrese el usuario
    variables_input=[]

    #abrimos el archivo con informacion para hacer los distintos pasos para adecuar la data y predecir
    #['variables_names', 'type_variables', 'category_mappings', 'model_name']
    with open("informacion/variables_info.json", 'r', encoding='utf-8') as file:
        predict_info = json.load(file)


    variables_principales = predict_info['variables_names']
    cant_variables = len(variables_principales)
    model_path =  predict_info['model_name']   

    #abrimos el modelo
    with open(model_path, 'rb') as file:
        model = pickle.load(file)    

    # Recuperamos los datos ingresados por el usuario
    for i in range(cant_variables):
        variable = request.form.get(f'variable{i}')
        variables_input.append(variable)
    print("datos ingresados por el usuario:",variables_input)

    # Vamos a hacer el casteo correspondiente para cada input realizado por el usuario
    # Desde el front vienen todas como un str

    # transfomamos en un df 
    input_df = list_to_df(variables_input, variables_principales) 
    print("input_df:",input_df)

    # le asignamos el tipo correspondiente a cada variable
    df_casteado = convertir_tipos(input_df,predict_info['type_variables'])
    print(df_casteado.info())

    
    # transformamos las variables categoricas usando el mappeo que construimos para modelar nuestro dataset
    final_input_df = mapear_input_df_to_numeric(df_casteado,predict_info['category_mappings'])    
    print("final_input:",final_input_df)
    final_input_df = final_input_df.fillna(0)

    print("Nombre del modelo",predict_info['model_name'])
    # aplicamos el modelo que retorna la prediccion y la probabilidad de que ocurra cada evento
    # array([[0.82, 0.18]]), 
    (prediction, probabilities) = predict(final_input_df,model)


    # Generamos el formato correspondiete para poder mostrar al usuario
    prob = probabilities[0][prediction[0]]
    prob=round(prob*100,2)
    y = find_key(predict_info['category_mappings']['y'],prediction[0])
    resultado_prediccion = str(y) + " con una probabilidad del " + str(prob)+ "%."


    return render_template('predecir.html', variables=variables_principales, cant_variables=cant_variables,resultado_prediccion=resultado_prediccion)
     

if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == '__main__':
    ##app.run(debug=True)
#    app.run(host="0.0.0.0", port=5000)