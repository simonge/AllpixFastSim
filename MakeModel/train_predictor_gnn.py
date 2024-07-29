import numpy as np
import tensorflow as tf
import uproot
import onnx
from tensorflow.keras.optimizers import Adam
from data_preparation import awkward_to_ragged
from data_preparation import awkward_to_tensor
import tf2onnx
import awkward as ak

from model_predictor_gnn import Predictor
from model_predictor_gnn import make_model

epochs = 400
batch_size = 10000
model_dir  = '/scratch/EIC/models/Allpix/'
model_name = 'predictor_gnn_full_det'
model_path = model_dir+model_name

node_columns   = ['pixel_x', 'pixel_y', 'charge', 'time']
target_columns = ['x', 'y', 'px', 'py', 'start_time']
nnodeparams    = len(node_columns)
ntargets       = len(target_columns)

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_full_det.root'

# predictor = make_model(nNodeParameters=nnodeparams, nPredictions=ntargets)
predictor = Predictor(nNodeParameters=nnodeparams, nPredictions=ntargets)
# predictor.build(input_shape=(None,None,nnodeparams))
predictor.compile(optimizer=Adam(), loss='mean_squared_error')
# predictor.summary()

# Assuming the ROOT file structure: MCParticles and PixelHits trees
with uproot.open(file_path) as file:
    tree = file['events']

    # Extracting data from the ROOT file
    df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'pixel_x', 'pixel_y', 'charge', 'time'], entry_stop=1000000)
    
    print('Number of events:', ak.num(df['pixel_x']))
    # Filter out events with 0 pixel hits
    df = df[ak.num(df['pixel_x']) > 0]
    # Print number of events
    print('Number of events:', len(df))


    node_tensor   = awkward_to_ragged(df[node_columns])
    target_tensor = awkward_to_tensor(df[target_columns])

    predictor.adapt(node_tensor)

    # Create vectors of train and test indices
    total_samples    = node_tensor.shape[0]
    train_size       = int(0.75 * total_samples)
    indices          = tf.range(start=0, limit=total_samples, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    # Use TensorFlow's indexing to split the dataset
    train_indices = shuffled_indices[:train_size]
    val_indices   = shuffled_indices[train_size:]
    
    node_train   = tf.gather(node_tensor, train_indices)
    node_val     = tf.gather(node_tensor, val_indices)
    target_train = tf.gather(target_tensor, train_indices)
    target_val   = tf.gather(target_tensor, val_indices)

    print(node_train.shape)
    print(node_val.shape)
    print(target_train.shape)
    print(target_val.shape)

    predictor.fit(node_train, target_train, epochs=epochs, batch_size=batch_size, validation_data=(node_val, target_val))

    test = predictor.predict(node_val[:4])
    print(test)
    print(target_val[:4])

    # Save the model as onnx
    input_signature = [tf.RaggedTensorSpec([None,None,nnodeparams], tf.float32, ragged_rank=1)]
    
    model_proto, _ = tf2onnx.convert.from_keras(predictor, input_signature=input_signature)
    onnx.save_model(model_proto, model_path+'.onnx')