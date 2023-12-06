from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


server_name = app.config['SERVER_NAME']

if server_name and ':' in server_name:
    host, port = server_name.split(":")
    port = int(port)
else:
    port = 5000
    host = "localhost"
#app.run(host=host, port=port)

# Supongamos que esta es tu lista de depósitos inicial
depositos = ["DEP01", "DEP02", "DEP03", "DEP04"]

@app.route('/')
def index():
    return render_template('index.html', depositos=depositos)
    #return 'Web App with Python Flask!'

@app.route('/confirmar', methods=['POST'])
def confirmar():
    # Aquí manejarías los depósitos seleccionados que se reciben del frontend
    depositos_seleccionados = request.form.getlist('depositos_agregados[]')
    print("depositos seleccionados: ",depositos_seleccionados)
    # Hacer algo con los depósitos seleccionados
    #return 'Depósitos confirmados!'
    return (str(depositos_seleccionados))

if __name__ == '__main__':
    app.run(debug=True)
#app.run(host=host, port=port)
