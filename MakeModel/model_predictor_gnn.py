import tensorflow as tf
import tensorflow.keras as keras
import uproot
import pandas as pd
import awkward as ak
import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout,Embedding, GlobalAveragePooling1D

import tensorflow.keras.layers as tfkl
from Layers import RaggedGlobalAverage, GravNet_ragged, RaggedMeanPooling, RaggedMinPooling, RaggedMaxPooling

   
# Model to predict a 5D vector from a ragged tensor of 4D vectors
class Predictor(tf.keras.Model):
    def __init__(self, nNodeParameters=4, nPredictions=5 ):
        super(Predictor, self).__init__()
        self.nNodeParameters = nNodeParameters
        self.nPredictions    = nPredictions
        self.output_names = ['output']
        self.m_input = Input(shape=(None,self.nNodeParameters,),name='input1',ragged=True)
        
        
        self.normalizer = tfkl.Normalization(name='normalizer')
        
        self.globalE0 = RaggedGlobalAverage(name='GE_0')
        self.globalE1 = RaggedGlobalAverage(name='GE_1')
        self.Concatenate = Concatenate(axis=-1,name="concat")
        self.dense0 = tfkl.Dense(64, activation='relu', name='Dense0',input_shape=(None,self.nNodeParameters))
        self.dense1 = tfkl.Dense(32, activation='relu', name='Dense1',input_shape=(None,self.nNodeParameters))
        self.dense2 = tfkl.Dense(32, activation='relu', name='Dense2',input_shape=(None,self.nNodeParameters))
        self.pool0 = RaggedMeanPooling()
        # self.pool1 = RaggedMinPooling()
        self.pool2 = RaggedMaxPooling()

        self.dense4 = tfkl.Dense(64, activation='relu', name='Dense4')
        self.dense5 = tfkl.Dense(64, activation='relu', name='Dense5')
        self.output_layer = tfkl.Dense(nPredictions, activation='linear', name='output')
        

    def adapt(self, input):
        print("adapting",input.shape)
        flattened_input = input.merge_dims(outer_axis=0, inner_axis=-2)
        print(flattened_input.shape)
        self.normalizer.adapt(flattened_input)        

    def call(self, inputs):

        # Normalize the input
        flattened_input = inputs.merge_dims(outer_axis=1, inner_axis=-2).to_tensor(default_value=0)
        normalized_tensor = self.normalizer(flattened_input)        
        normalized_ragged = tf.RaggedTensor.from_tensor(normalized_tensor, lengths=inputs.row_lengths())
        

        v = self.dense0(normalized_ragged)
        globalE = self.globalE0(v)
        # globalE = RaggedGlobalAverage(name='GE_0')(m_input)  
        # v = Concatenate(axis=-1,name="concat")([input, globalE])
        # print(v.shape)
        v       = self.dense1(globalE)
        # globalE = self.globalE1(v)
        # v       = Concatenate(axis=-1,name="concat6")([normalized_ragged, globalE])
        # v       = self.dense2(v)
        v0 = self.pool0(v)
        # v1 = self.pool1(v)
        # v2 = self.pool2(v)
        # v = Concatenate()([v0, v1, v2])
        # v = Concatenate()([v0, v2])
        # v = self.dense4(v)
        v = self.dense5(v0)
        predicted = self.output_layer(v)

        return predicted