from flask import Flask, request, jsonify
import author_inference
import author_identification_grapher as grapher
import requests
from flask import Flask, request, render_template


import re
app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('layouts/index.html',
                           input1="Paste text sample here",
                           output="Prediction")


@app.route('/submit', methods=['POST'])
def my_form_post():
    input1 = request.form['input1']

    result = author_inference.predict(input1)


    return grapher.make_graph(input1, result)
    '''
    return render_template('layouts/index.html',
                           input1=input1,
                           output=response)
    '''

@app.route('/test', methods=['POST', 'GET'])
def test_get():
    input1 = request.values.get('input1')
    result = author_inference.predict(input1)
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
