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
import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import io
import os
import re
import string
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
  Embedding,Input,LSTM,Input,Conv2D,Softmax,Dropout,Dense,GRU,
  MaxPooling2D,LayerNormalization,Reshape,BatchNormalization,Bidirectional)
from tensorflow.keras.optimizers import Adam

app=Flask(__name__)
Swagger(app)

BATCH_SIZE=8
LR=1e-4
FRAME_LENGTH=255
FRAME_STEP=128
N_EPOCHS=100

vocabulary=[""]+[chr(i) for i in range(97,97+26)]+[".",",","?"," "]
    
def decode(y_pred):
  batch_size=tf.shape(y_pred)[0]
  print(tf.shape(y_pred))

  pred_length=tf.shape(y_pred)[1]
  pred_length*=tf.ones([batch_size,],dtype=tf.int32)

  y_pred=tf.one_hot(y_pred,len(vocabulary)+1)
  output=tf.keras.backend.ctc_decode(y_pred,input_length=pred_length,greedy=True)[0][0]

  out=[vocabulary[i] for i in output[0]]
  return ''.join(out)
  
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

def ctc_loss(y_true,y_pred):
  batch_size=tf.shape(y_pred)[0]
  pred_length=tf.shape(y_pred)[1]
  true_length=tf.shape(y_true)[1]

  pred_length=pred_length*tf.ones([batch_size,1],dtype=tf.int32)
  true_length=true_length*tf.ones([batch_size,1],dtype=tf.int32)

  return tf.keras.backend.ctc_batch_cost(y_true,y_pred,pred_length,true_length)
  
def get_model():

    normalization=tf.keras.layers.Normalization()
    input_spectrogram=Input((None,129,1), name="input")

    x=normalization(input_spectrogram)
    x=Conv2D(32,kernel_size=[11,41],strides=[2,2],padding='same',activation='relu')(x)
    x=LayerNormalization()(x)
    x=Conv2D(64,kernel_size=[11,21],strides=[1,2],padding='same',activation='relu')(x)
    x=LayerNormalization()(x)

    x=Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    x=Bidirectional(GRU(128,return_sequences=True))(x)
    x=Bidirectional(GRU(128,return_sequences=True))(x)
    x=Bidirectional(GRU(128,return_sequences=True))(x)


    output=Dense(len(vocabulary)+1, activation="softmax")(x)

    model = tf.keras.Model(input_spectrogram, output, name="DeepSpeech_2_Inspired")
    
    model.compile(loss=ctc_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),)
    
    model.load_weights('/usr/ML/app/ASR_LJSpeech_Model_Final.h5')
    return model   


#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['wav', 'mp3'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

@app.route('/predict_file',methods=["POST"])
def prediction_test_file():
    """Prediction on input test file .
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
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join('/usr/uploads', filename)
    file.save(file_path)
            
    audio_binary=tf.io.read_file(file_path)
    waveform=decode_audio(audio_binary)
    waveform=tf.cast(waveform,tf.float32)

    spectrogram=tf.signal.stft(
      waveform,frame_length=FRAME_LENGTH,frame_step=FRAME_STEP)

    spectrogram=tf.abs(spectrogram)

    spectrogram=tf.expand_dims(spectrogram,axis=-1) 
    
    model=get_model()
    out=tf.argmax(model.predict(tf.expand_dims(spectrogram,axis=0))[0],axis=1)
            
    return str(decode(tf.expand_dims(out,axis=0)))        

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
    
    
