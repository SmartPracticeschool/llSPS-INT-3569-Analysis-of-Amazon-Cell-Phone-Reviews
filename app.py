# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:53:18 2020

@author: AISHWARYA
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask("_Assignment_.py")
model = pickle.load(open('aishwaryaModel.hfile','rb')) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict', methods = ['POST'])
def y_predict():
    X_test = [[int (X) for X in request.form.values()]]
    prediction = model.predict(X_test)
    print(prediction)
    output = prediction[0][0]
    return render_template('index.html', prediction_text = 'Overall Ratings and Reviews {}' .format(output))

if app == "__main__":
    app.run(debug=True)
    