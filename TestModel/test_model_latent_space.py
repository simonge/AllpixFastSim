import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import onnxruntime as ort
import uproot
import pandas as pd
import tensorflow as tf

output_dir = 'plots/'
model_dir = '/scratch/EIC/models/Allpix/'
model_base = "model_electron2_latent"
model_name = model_dir+model_base+".onnx"
# Load the ONNX model
sess = ort.InferenceSession(model_name)

condition_columns    = ['x', 'y', 'px', 'py', 'start_time']
condition_ranges     = [[0, 0.5], [0, 0.5], [-0.2, 0.2], [-0.2, 0.2], [0, 25]]
nConditions = len(condition_columns)

conditions_name = sess.get_inputs()[0].name
image_name = sess.get_inputs()[1].name

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time2.root'
output_dir = 'plots/'

# Assuming the ROOT file structure: MCParticles and PixelHits trees
infile = uproot.open(file_path)
tree  = infile['events']

# Extracting data from the ROOT file
df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'charge', 'time'], entry_stop=100000)

data_grid_size = 9
data_shape = (-1, data_grid_size, data_grid_size, 2)

target_data = np.stack([df['charge'].to_numpy(), df['time'].to_numpy()],axis=2).astype(np.float32)
image_tensor = target_data.reshape(data_shape)
          
conditions_tensor = np.stack([df[name].to_numpy() for name in condition_columns], axis=1).astype(np.float32)

# Predict the output for the input tensor
output = sess.run(None, {conditions_name: conditions_tensor, image_name: image_tensor})

nOutputs = output[0].shape[-1]

# Plot 2D scatter plots showing the correlation between the first 2 latent dimensions
plt.figure()
plt.scatter(output[0][:,0], output[0][:,1])
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.xlabel('Latent dimension 1')
plt.ylabel('Latent dimension 2')
plt.savefig(output_dir + model_base + '_latent_space-01.png')

#Plot a histogram of each latent dimension on a grid
fig, axs = plt.subplots(int(nOutputs/5),5, figsize=(20, 10))
for i in range(nOutputs):
    row=int(i//5)
    col=int(i%5)
    axs[row,col].hist(output[0][:,i], bins=400, range=(-5,5))
    #axs[row,col].set_yscale('log')
    axs[row,col].set_title('Latent dimension '+str(i))



plt.savefig(output_dir + model_base + '_latent_histograms.png')

# Plot 2D scatter plots showing the correlation between the conditions and output dimensions
# A matrix of images

for i in range(nConditions):
    out_tag = model_base + '_latent_hist_'+condition_columns[i]
    nRows = 5
    fig, axs = plt.subplots(nRows, nOutputs//nRows, figsize=(20, 10))
    for j in range(nOutputs):
        col = int(j//nRows)
        row = int(j%nRows)
        axs[row,col].hist2d(conditions_tensor[:,i], output[0][:,j], bins=(200, 200), cmap=plt.cm.jet, range=[condition_ranges[i], [-5, 5]])#, norm=colors.LogNorm())
        axs[row,col].set_xlabel(condition_columns[i])
        axs[row,col].set_ylabel('Output '+str(j))
    plt.savefig(output_dir + out_tag + '.png')
