import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask, app,request,render_template
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.models import load_model

#Loading the model
model=load_model(r"crime.h5",compile=False)

app=Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = 'test images'


#default home page or route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def prediction():
    return render_template('predict.html')


@app.route('/result',methods=["GET","POST"])
def result():
    if request.method=="POST":
        f = request.files['image']

        #basepath=os.path.dirname(__file__) #getting the current path i.e where app.py is present
        #print("current path",basepath)
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],f.filename) #from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        #print("upload folder is",filepath)
        f.save(filepath)

        img = image.load_img(filepath,target_size=(64,64)) # Reading image
        x = image.img_to_array(img) # Converting image into array
        x = np.expand_dims(x,axis=0) # expanding Dimensions
        pred = np.argmax(model.predict(x)) # Predicting the higher probablity index
        op = ['Fighting','Arrest','Vandalism','Assault','Stealing','Arson','NormalVideos','Burglary','Explosion','Robbery','Abuse','Shooting','Shoplifting','RoadAccidents'] # Creating list
        op[pred]
        result = op[pred]
        result = 'The predicted output is {}'.format(str(result))
        #print(result)
        #result=str(op[ pred[0].tolist().op(1)])
    return render_template('result.html',text = result)
        



""" Running our application """
if __name__ == "__main__":
    app.run(debug=True)