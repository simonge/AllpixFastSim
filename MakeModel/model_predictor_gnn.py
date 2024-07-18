import tensorflow as tf
import tensorflow.keras as keras
import uproot
import pandas as pd
import awkward as ak
import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout,Embedding, GlobalAveragePooling1D

import tensorflow.keras.layers as tfkl
from Layers import RaggedGlobalAverage, GravNet_ragged, RaggedMeanPooling, RaggedMinPooling, RaggedMaxPooling

def make_model(nNodeParameters=4, nPredictions=5):
    m_input = Input(shape=(None,nNodeParameters,),name='input1',ragged=True)
    # tf.print(m_input.shape)
    # print(m_input.shape)
    mean = tf.reduce_mean(m_input, axis=1)
    mean_expanded = tf.expand_dims(mean, axis=1)
    zeros_like_x = tf.zeros_like(m_input)
    broadcasted_mean = zeros_like_x + mean_expanded
    # globalE = RaggedGlobalAverage(name='GE_0')(m_input)  
    v = Concatenate(axis=-1,name="concat")([m_input, broadcasted_mean])
    print(v.shape)
    v       = Dense(32, activation='relu',name='Dense0')(v) 
    tf.print(v.shape)
    print(v.shape)
    v       = RaggedMeanPooling(name="Pool")(v)
    tf.print(v.shape)
    v       = Dense(128, activation='relu',name='Dense4')(v)
    predicted = Dense(nPredictions, activation='linear',name='output')(v)

    model = keras.Model(inputs=m_input, outputs=predicted)
    
    return model

   
# Model to predict a 5D vector from a ragged tensor of 4D vectors
class Predictor(tf.keras.Model):
    def __init__(self, nNodeParameters=4, nPredictions=5 ):
        super(Predictor, self).__init__()
        self.nNodeParameters = nNodeParameters
        self.nPredictions    = nPredictions
        self.output_names = ['output']
        self.m_input = Input(shape=(None,self.nNodeParameters,),name='input1',ragged=True)
        
        
        self.normalizer = tfkl.Normalization(name='normalizer')
        
        self.globalE = RaggedGlobalAverage(name='GE_0')
        self.Concatenate = Concatenate(axis=-1,name="concat")
        self.dense0 = tfkl.Dense(32, activation='relu', name='Dense0',input_shape=(None,self.nNodeParameters))
        self.dense1 = tfkl.Dense(32, activation='relu', name='Dense1',input_shape=(None,self.nNodeParameters))
        self.dense2 = tfkl.Dense(32, activation='relu', name='Dense2',input_shape=(None,self.nNodeParameters))
        self.pool0 = RaggedMeanPooling()
        self.pool1 = RaggedMinPooling()
        self.pool2 = RaggedMaxPooling()

        self.dense4 = tfkl.Dense(128, activation='relu', name='Dense4')
        self.dense5 = tfkl.Dense(128, activation='relu', name='Dense5')
        self.output_layer = tfkl.Dense(nPredictions, activation='linear', name='output')
        

    def adapt(self, input):
        print("adapting",input.shape)
        flattened_input = input.merge_dims(outer_axis=0, inner_axis=-2)
        print(flattened_input.shape)
        self.normalizer.adapt(flattened_input)        

# #         #Define sequencial model
        
# #         self.predictor = tf.keras.Sequential([
# #             self.m_input,
# #             RaggedGlobalExchange(name='GE_0'),
# #             tfkl.Dense(32, activation='relu',name='Dense0'),
# #             tfkl.GlobalAveragePooling1D(),
# #             tfkl.Dense(128, activation='relu',name='Dense4'),
# #             tfkl.Dense(self.nPredictions, activation='relu',name='output')
# #         ])
# #         # self.predictor = tf.keras.Sequential([
# #         #     # self.m_input,
# #         #     # self.globalE,
# #         #     self.dense0,
# #         #     self.pool,
# #         #     self.dense4,
# #         #     self.output_layer
# #         # ])

    def call(self, inputs):
        # x = self.m_input(inputs)
        # row_lengths = inputs.row_lengths()
        # x = self.globalE(inputs)
        # Flatten ragged dimension
        # flattened_x = x.merge_dims(outer_axis=0, inner_axis=-2)
        # y = self.dense0(x)
        # Unflatten ragged by indices and values
        # x = tf.RaggedTensor.from_row_lengths(y, row_lengths)
        # x = self.pool(x)
        # x = self.dense4(x)
        # predicted = self.output_layer(x)
        # return predicted
        # Predict the output for the input tensor
        #inputs = self.m_input(inputs)
        # globalE = RaggedGlobalAverage(name='GE_0')(inputs)      
        # v       = tfkl.Dense(32, activation='relu',name='Dense0')(globalE) 
        # v       = tfkl.GlobalMaxPooling1D()(v)
        # v       = tfkl.Dense(128, activation='relu',name='Dense4')(v)
        # predicted = tfkl.Dense(self.nPredictions, activation='relu',name='output')(v)

        # m_input = Input(shape=(None,nNodeParameters,),name='input1',ragged=True)
        # tf.print(m_input.shape)
        # print(m_input.shape)
        # mean = tf.reduce_mean(input, axis=1)
        # mean_expanded = tf.expand_dims(mean, axis=1)
        # zeros_like_x = tf.zeros_like(input)
        # broadcasted_mean = zeros_like_x + mean_expanded
        # globalE = self.Concatenate([input, broadcasted_mean])
        #Apply normalization on the node parameters
        flattened_input = inputs.merge_dims(outer_axis=1, inner_axis=-2).to_tensor(default_value=0)
        
        normalized_tensor = self.normalizer(flattened_input)
        
        normalized_ragged = tf.RaggedTensor.from_tensor(normalized_tensor, lengths=inputs.row_lengths())
        
        v = self.dense0(normalized_ragged)
        globalE = self.globalE(v)
        # globalE = RaggedGlobalAverage(name='GE_0')(m_input)  
        # v = Concatenate(axis=-1,name="concat")([input, globalE])
        # print(v.shape)
        v      = self.dense1(globalE)
        v      = self.dense2(v)
        v0 = self.pool0(v)
        v1 = self.pool1(v)
        v2 = self.pool2(v)
        v = Concatenate()([v0, v1, v2])
        v = self.dense4(v)
        v = self.dense5(v)
        predicted = self.output_layer(v)

        # # v       = Dense(32, activation='relu',name='Dense0')(globalE) 
        # tf.print(v.shape)
        # print(v.shape)
        # v       = RaggedMeanPooling(name="Pool")(v)
        # tf.print(v.shape)
        # v       = Dense(128, activation='relu',name='Dense4')(v)
        # predicted = Dense(self.nPredictions, activation='linear',name='output')(v)


#         # inputs = inputs.to_tensor(default_value=0)
#         predicted = self.predictor(inputs)



        return predicted