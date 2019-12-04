from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import warnings
import cv2
import base64
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input , decode_predictions
    from tensorflow.keras.models import load_model,model_from_json
    from tensorflow.keras.preprocessing import image

    from flask import Flask, redirect, url_for, request, render_template
    from werkzeug.utils import secure_filename
    from gevent.pywsgi import WSGIServer

app = Flask(__name__)

with open('models/CNN.json','r') as f:
    model=model_from_json(f.read())
model.load_weights('models/CNN.h5')


def model_predict(img_path, model):
    img=cv2.imread(img_path,0)
    img = np.invert(img)
    img=cv2.resize(img,(28,28))
    img=img.reshape(-1,28,28,1)
    
    preds = model.predict(img)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
   if request.method == 'POST':
     
        f = request.files['file']

      
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
      
        result = str(np.argmax(preds))              
        return result

@app.route('/predict1/', methods=['GET', 'POST'])
def Predict():
    imgData = request.get_data()
    imgData=str((imgData), 'utf-8')
    imgstr = re.search(r'base64,(.*)',imgData).group(1)

    with open('output.png','wb') as output:
        output.write(np.fromstring(base64.b64decode(imgstr), np.uint8))
    file_path='output.png'
    print(file_path)
    result = model_predict(file_path, model)
    response=str(np.argmax(result))  
    return response


       



if __name__ == "__main__":
    app.run(debug=True)
