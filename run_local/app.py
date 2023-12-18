from flask import Flask, render_template, request
import pandas as pd

from algoritmo_principal import *


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        target = request.form['target']
        pro_mode = 'pro_mode' in request.form

        parametros = {"filename":filename,"target":target,"pro_mode":pro_mode}

        # Proceso archivo
        df = pd.read_csv(f)
        columnas = df.columns.tolist()

        # Guardo csv en datos
        df.to_csv("datos_input/"+filename)
        print( "archivo guardado")

        metric, variables_principales =algoritmo_principal(df,target,gpt=False)



        return render_template('resultados.html', parametros=parametros, columnas=columnas, variables_principales=variables_principales,metric=metric)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
