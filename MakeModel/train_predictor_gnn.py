import numpy as np
import uproot
import onnx
import torch
from torch_geometric.data import Data

from model_predictor_gnn import Predictor

epochs = 200
batch_size = 5000
model_dir  = '/scratch/EIC/models/Allpix/'
model_name = 'predictor_electron_gnn'
model_path = model_dir+model_name

node_columns   = ['pixel_x', 'pixel_y', 'charge', 'time']
target_columns = ['x', 'y', 'px', 'py', 'start_time']
nnodeparams    = len(node_columns)
ntargets       = len(target_columns)

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time_GNN.root'

predictor = Predictor(nNodeParameters=nnodeparams, nPredictions=ntargets)



# Assuming the ROOT file structure: MCParticles and PixelHits trees
with uproot.open(file_path) as file:
    tree = file['events']

    # Extracting data from the ROOT file
    df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'pixel_x', 'pixel_y', 'charge', 'time'], entry_stop=10)
    
    # Shape data into graph with nodes containing the node_columns as parameters
    node_data = np.stack([df[name].to_numpy() for name in node_columns], axis=1)
    print(node_data)

    # Create input tensors
    target_tensor = np.stack([df[name].to_numpy() for name in target_columns], axis=1)
    print(target_tensor)
    
    # Create vectors of train and test indices
    # total_samples    = transformed_conditions_tensor.shape[0]
    # train_size       = int(0.75 * total_samples)
    # indices          = tf.range(start=0, limit=total_samples, dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices)
    
    # # Use TensorFlow's indexing to split the dataset
    # train_indices = shuffled_indices[:train_size]
    # val_indices   = shuffled_indices[train_size:]

    # conditions_train = tf.gather(transformed_conditions_tensor, train_indices)
    # conditions_val   = tf.gather(transformed_conditions_tensor, val_indices)
    # image_train      = tf.gather(transformed_image_tensor, train_indices)
    # image_val        = tf.gather(transformed_image_tensor, val_indices)
    
    # # Now you can use these variables in the fit function
    # vae.fit(conditions_train, image_train, validation_data=(conditions_val, image_val), epochs=epochs, batch_size=batch_size)
    
    # generator = Generator(vae,preprocessor)

    # outTest = generator(conditions_tensor[:1])
    
    # input_signature = [tf.TensorSpec([None,nconditions], tf.float32, name='x')]
    
    # # Convert the model
    # onnx_model, _ = tf2onnx.convert.from_keras(generator,input_signature, opset=13)
    # onnx.save(onnx_model, model_path+".onnx")
