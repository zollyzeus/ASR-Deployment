# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:50:04 2020

@author: pramod.singh
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
import tensorflow as tf

app=Flask(__name__)
Swagger(app)

model=load_model("ASR_LJSpeech_Model_Final.h5")
def get_spec_inference(filepath):

  audio_binary=tf.io.read_file(filepath)
  waveform=decode_audio(audio_binary)
  waveform=tf.cast(waveform,tf.float32)

  spectrogram=tf.signal.stft(
      waveform,frame_length=FRAME_LENGTH,frame_step=FRAME_STEP)

  spectrogram=tf.abs(spectrogram)

  return tf.expand_dims(spectrogram,axis=-1)
 
def decode_audio(audio_binary):
    audio,_=tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio,axis=-1)

@app.route('/predict',methods=["Get"])
def predict_class():
    
    """Predict if Customer would buy the product or not .
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: new_user
        in: query
        type: number
        required: true
      - name: total_pages_visited
        in: query
        type: number
        required: true
      
    responses:
        500:
            description: Prediction
        
    """
    age=int(request.args.get("age"))
    new_user=int(request.args.get("new_user"))
    total_pages_visited=int(request.args.get("total_pages_visited"))
    prediction=model.predict([[age,new_user,total_pages_visited]])
    print(prediction[0])
    return "Model prediction is"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def prediction_test_file():
    """Prediction on multiple input test file .
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        500:
            description: Test file Prediction
        
    """
    spectrogram=get_spec_inference("file"))
    out=tf.argmax(model.predict(tf.expand_dims(spectrogram,axis=0))[0],axis=1)
    
    return str(decode(tf.expand_dims(out,axis=0)))

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
    
    
