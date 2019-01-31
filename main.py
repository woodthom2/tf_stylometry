# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging

from flask import Flask

from flask import Flask, request, jsonify
import author_inference
import author_identification_grapher as grapher
import requests
from flask import Flask, request, render_template, send_from_directory


import re

app = Flask(__name__)



@app.route('/')
def my_form():
    return render_template('layouts/index.html',
                           input1=None,
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

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
