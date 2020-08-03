# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:03:20 2020

@author: Arpita
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App

@app.route('/')
def home():
    #pic=os.path.join(app.config['UPLOAD_FOLDER'],'4.png')
    return render_template('index.html')


def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1,6) 
    loaded_model = pickle.load(open("model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Winner'
        else: 
            prediction ='Not Winner'            
        return render_template("result.html", prediction = prediction) 

if __name__ == "__main__":
    app.run(debug=True)


