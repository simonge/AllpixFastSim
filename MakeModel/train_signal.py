import numpy as np
import uproot
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tf2onnx
import onnx

# train_signal.py
from model_generator import VAE
from model_generator import Generator
from model_generator import LatentSpace
from model_generator import Preprocessor

epochs = 200
batch_size = 5000
model_dir  = '/scratch/EIC/models/Allpix/'
model_name = 'model_electron'
model_path = model_dir+model_name
data_grid_size = 9
data_shape = (-1, data_grid_size, data_grid_size, 2)

condition_columns = ['x', 'y', 'px', 'py', 'start_time']
#condition_columns = ['x', 'y']
nconditions = len(condition_columns)

nInput = nconditions + data_grid_size*data_grid_size*2

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time2.root'

#vae = create_model()
vae = VAE(latent_dim=10,nconditions=nconditions,grid_size=data_grid_size)

#vae.compile(optimizer=Adam())
vae.compile(r_optimizer=Adam(),a_optimizer=Adam())

# Preprocessor
preprocessor = Preprocessor(isSquare=True)

# Assuming the ROOT file structure: MCParticles and PixelHits trees
with uproot.open(file_path) as file:
    tree = file['events']

    # Extracting data from the ROOT file
    df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'charge', 'time'], entry_stop=1000000)
    
    # Shape data into 2 channel image
    target_data = np.stack([df['charge'].to_numpy(), df['time'].to_numpy()],axis=2)
    image_tensor = target_data.reshape(data_shape)
    
    # Create input tensors
    conditions_tensor = np.stack([df[name].to_numpy() for name in condition_columns], axis=1)
    
    # Preprocess the input tensors
    transformed_conditions_tensor, transformed_image_tensor = preprocessor((conditions_tensor, image_tensor))
    
    # Create vectors of train and test indices
    total_samples    = transformed_conditions_tensor.shape[0]
    train_size       = int(0.75 * total_samples)
    indices          = tf.range(start=0, limit=total_samples, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    
    # Use TensorFlow's indexing to split the dataset
    train_indices = shuffled_indices[:train_size]
    val_indices   = shuffled_indices[train_size:]

    conditions_train = tf.gather(transformed_conditions_tensor, train_indices)
    conditions_val   = tf.gather(transformed_conditions_tensor, val_indices)
    image_train      = tf.gather(transformed_image_tensor, train_indices)
    image_val        = tf.gather(transformed_image_tensor, val_indices)
    
    # Now you can use these variables in the fit function
    vae.fit(conditions_train, image_train, validation_data=(conditions_val, image_val), epochs=epochs, batch_size=batch_size)
    
    # model_name = model_path+'.keras'
    # vae.save(model_name)
    
    # decoder = Generator(vae)

    # outTest = decoder(conditions_tensors[:1])
    # #outTest = decoder.predict(conditions_tensors[:1])
    # print(outTest)

    # input_signature = [tf.TensorSpec([None,nconditions], tf.float32, name='x')]
    
    # # Convert the model
    # onnx_model, _ = tf2onnx.convert.from_keras(decoder,input_signature, opset=13)
    # onnx.save(onnx_model, model_path+".onnx")

    # latent_encoder  = LatentSpace(vae)
    # outTest_latent = latent_encoder(input_tensors[:1])
    # print(outTest_latent)


    # input_signature_latent = [tf.TensorSpec([None,nInput], tf.float32, name='x')]
    
    # onnx_model_latent, _ = tf2onnx.convert.from_keras(latent_encoder,input_signature_latent, opset=13)
    # onnx.save(onnx_model_latent, model_path+"_latent.onnx")