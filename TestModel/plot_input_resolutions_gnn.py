import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# Construct the path to the directory from which you want to import
make_model_dir = os.path.abspath(os.path.join(current_dir, '../MakeModel'))

# Add this directory to sys.path
if make_model_dir not in sys.path:
    sys.path.append(make_model_dir)

import matplotlib.pyplot as plt
import numpy as np
import uproot
from tensorflow import sparse, stack
import onnxruntime as ort
import time
import awkward as ak
from data_preparation import awkward_to_ragged, awkward_to_tensor

tag = '2'
output_dir = 'plots/'
model_dir = '/scratch/EIC/models/Allpix/'
predictor_base  = 'predictor_gnn_full_det3'
predictor_name = model_dir+predictor_base+".onnx"

predictor_session = ort.InferenceSession(predictor_name)

input_name0 = predictor_session.get_inputs()[0].name
input_name1 = predictor_session.get_inputs()[1].name
#print input names
print("Input name: ", input_name0)


# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_full_det.root'

# Assuming the ROOT file structure: MCParticles and PixelHits trees
infile = uproot.open(file_path)
tree  = infile['events']

# Extracting data from the ROOT file
node_columns   = ['pixel_x', 'pixel_y', 'charge', 'time']
target_columns = ['x', 'y', 'px', 'py', 'start_time']

df = tree.arrays(['x', 'y', 'px', 'py', 'start_time','pixel_x', 'pixel_y', 'charge', 'time'], entry_stop=10000000)

nHits = ak.num(df['pixel_x'])

#########################################################################################################################
# Calculate and plot the resolutions from a simple mean squared charge calculation
#########################################################################################################################
# Calculate sqrt of charge weighted pixel_x and pixel_y and take difference between input x and y to get resolution
df['sqrtcharge'] = np.sqrt(df['charge'])
df['mean_squared_charge_x'] = ak.sum(df['sqrtcharge']*(499.5-df['pixel_x']), axis=1)/ak.sum(df['sqrtcharge'], axis=1)
df['mean_squared_charge_y'] = ak.sum(df['sqrtcharge']*(df['pixel_y']-499.5), axis=1)/ak.sum(df['sqrtcharge'], axis=1)   

df['diff_mean_squared_charge_x'] = df['mean_squared_charge_x'] - df['x']
df['diff_mean_squared_charge_y'] = df['mean_squared_charge_y'] - df['y']

# Convert awkward array to numpy array
diff_mean_squared_charge_x = ak.to_numpy(df['diff_mean_squared_charge_x'])
diff_mean_squared_charge_y = ak.to_numpy(df['diff_mean_squared_charge_y'])

#Plot the 1D and 2D x, y resolutions
plt.figure()
plt.hist(diff_mean_squared_charge_x, bins=100, range=(-4.0, 4.0))
plt.xlabel('Difference between mean squared charge and input x')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'charge_weighted_x_diff'+tag+'.png')

plt.figure()
plt.hist(diff_mean_squared_charge_y, bins=100, range=(-4.0, 4.0))
plt.xlabel('Difference between mean squared charge and input y')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'charge_weighted_y_diff'+tag+'.png')

plt.figure()
plt.hist2d(diff_mean_squared_charge_x, diff_mean_squared_charge_y, bins=(100, 100), range=[[-4.0, 4.0], [-4.0, 4.0]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.savefig(output_dir + 'charge_weighted_x_y_diff'+tag+'.png')

# Plot x-y resolution as a function of pixel hit
maxHits = 10
fig, axs = plt.subplots(maxHits, 4, figsize=(40, 20))

for i in range(maxHits):

    hitFilter = nHits==i+1
    
    #Filter arrays by number of nHits
    diff_mean_squared_charge_x_nHits = diff_mean_squared_charge_x[hitFilter]
    diff_mean_squared_charge_y_nHits = diff_mean_squared_charge_y[hitFilter]

    # Plot the difference between mean squared charge and input x
    axs[i,0].hist(diff_mean_squared_charge_x_nHits, bins=100, range=(-4.0, 4.0))
    axs[i,0].set_xlabel('Difference between mean squared charge and input x')
    axs[i,0].set_ylabel('Number of entries')
    axs[i,0].set_title(f'Number of hits: {i+1}')

    # Plot the difference between mean squared charge and input y
    axs[i,1].hist(diff_mean_squared_charge_y_nHits, bins=100, range=(-4.0, 4.0))
    axs[i,1].set_xlabel('Difference between mean squared charge and input y')
    axs[i,1].set_ylabel('Number of entries')
    axs[i,1].set_title(f'Number of hits: {i+1}')

    # Plot 2D histogram x-y plot of mean squared charge
    axs[i,2].hist2d(diff_mean_squared_charge_x_nHits, diff_mean_squared_charge_y_nHits, bins=(100, 100), range=[[-4.0, 4.0], [-4.0, 4.0]], cmap='viridis')
    axs[i,2].set_xlabel('x [pixel pitch]')
    axs[i,2].set_ylabel('y [pixel pitch]')
    axs[i,2].set_title(f'Number of hits: {i+1}')

plt.savefig(output_dir + 'charge_weighted_resolution'+tag+'.png')



#########################################################################################################################
# Calculate and plot the resolutions from the GNN model
#########################################################################################################################
input_tensor   = awkward_to_ragged(df[node_columns])
target_tensor  = awkward_to_tensor(df[target_columns]).numpy()


values = input_tensor.flat_values.numpy().astype(np.float32)
row_splits = input_tensor.row_splits.numpy().astype(np.int64)

output_tensor = predictor_session.run(None, {input_name0: values, input_name1: row_splits})

output_tensor = output_tensor[0]
mean_index_x = output_tensor[:,0]
mean_index_y = output_tensor[:,1]
time_pred = output_tensor[:,2]

#Plot predicted x-y distribution in a 3x3 range
plt.figure()
plt.hist2d(mean_index_x, mean_index_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.grid(True)
plt.savefig(output_dir + 'mean_index_x_y_distribution_gnn'+tag+'.png')


#Plot mean index x distribution against input x
plt.figure()
plt.hist2d(target_tensor[:,0], mean_index_x, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('Mean index x')
plt.savefig(output_dir + 'mean_index_x_distribution_gnn'+tag+'.png')

#Plot mean index y distribution against input y
plt.figure()
plt.hist2d(target_tensor[:,1], mean_index_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('y [pixel pitch]')
plt.ylabel('Mean index y')
plt.savefig(output_dir + 'mean_index_y_distribution_gnn'+tag+'.png')

#Plot time against time
plt.figure()
plt.hist2d(target_tensor[:,4], time_pred, bins=(1000, 1000), range=[[-1, 30], [-1, 30]], cmap='viridis')
plt.xlabel('time')
plt.ylabel('Predicted time')
plt.savefig(output_dir + 'time_distribution_gnn'+tag+'.png')


print(f"Mean index x: {mean_index_x}")
print(f"Mean index y: {mean_index_y}")

# difference between mean_squared_charge_x and input x
diff_mean_squared_charge_x = mean_index_x - target_tensor[:,0]
# difference between mean_squared_charge_y and input y
diff_mean_squared_charge_y = mean_index_y - target_tensor[:,1]
# Time difference
diff_time = time_pred - target_tensor[:,4]

# Plot the difference between mean squared charge and input x
plt.figure()
plt.hist(diff_mean_squared_charge_x, bins=100, range=(-4.0, 4.0))
plt.xlabel('Difference between mean squared charge and input x')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'diff_mean_squared_charge_x_gnn'+tag+'.png')

# Plot the difference between mean squared charge and input y
plt.figure()
plt.hist(diff_mean_squared_charge_y, bins=100, range=( -4.0, 4.0))
plt.xlabel('Difference between mean squared charge and input y')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'diff_mean_squared_charge_y_gnn'+tag+'.png')

# Plot the difference between time and input time
plt.figure()
plt.hist(diff_time, bins=100, range=(-4.0, 4.0))
plt.xlabel('Difference between time and input time')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'diff_time_gnn'+tag+'.png')

# Plot difference between time and input time against time
plt.figure()
plt.hist2d(diff_time, target_tensor[:,4],  bins=(100, 2000), range=[[-1, 1],[0, 25]], cmap='viridis')
plt.xlabel('time')
plt.ylabel('Difference between time and input time')
plt.savefig(output_dir + 'diff_time_vs_time_gnn'+tag+'.png')

# Plot 2D histogram x-y plot of mean squared charge
plt.figure()
plt.hist2d(diff_mean_squared_charge_x, diff_mean_squared_charge_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.savefig(output_dir + 'mean_squared_charge_x_y_distribution_gnn'+tag+'.png')

# Plot x-y resolution as a function of pixel hit
maxHits = 10
fig, axs = plt.subplots(maxHits, 4, figsize=(40, 20))

for i in range(maxHits):

    hitFilter = nHits==i+1

    #Filter arrays by number of nHits
    diff_mean_squared_charge_x_nHits = diff_mean_squared_charge_x[hitFilter]
    diff_mean_squared_charge_y_nHits = diff_mean_squared_charge_y[hitFilter]
    diff_time_nHits = diff_time[hitFilter]
    target_times_nHits = target_tensor[:,4][hitFilter]

    # Plot the difference between mean squared charge and input x
    axs[i,0].hist(diff_mean_squared_charge_x_nHits, bins=100, range=(-4.0, 4.0))
    axs[i,0].set_xlabel('Difference between mean squared charge and input x')
    axs[i,0].set_ylabel('Number of entries')
    axs[i,0].set_title(f'Number of hits: {i+1}')

    # Plot the difference between mean squared charge and input y
    axs[i,1].hist(diff_mean_squared_charge_y_nHits, bins=100, range=( -4.0, 4.0))
    axs[i,1].set_xlabel('Difference between mean squared charge and input y')
    axs[i,1].set_ylabel('Number of entries')
    axs[i,1].set_title(f'Number of hits: {i+1}')

    # Plot 2D histogram x-y plot of mean squared charge
    axs[i,2].hist2d(diff_mean_squared_charge_x_nHits, diff_mean_squared_charge_y_nHits, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
    axs[i,2].set_xlabel('x [pixel pitch]')
    axs[i,2].set_ylabel('y [pixel pitch]')
    axs[i,2].set_title(f'Number of hits: {i+1}')

    # Plot time difference
    axs[i,3].hist(diff_time_nHits, bins=100, range=(-4.0, 4.0))
    axs[i,3].set_xlabel('Difference between time and input time')
    axs[i,3].set_ylabel('Number of entries')
    axs[i,3].set_title(f'Number of hits: {i+1}')

    # Plot difference between time and input time against time
    # plt.figure()
    # plt.hist2d(diff_time_nHits, target_times_nHits,  bins=(100, 2000), range=[[-1, 1],[0, 25]], cmap='viridis')
    # plt.xlabel('time')
    # plt.ylabel('Difference between time and input time')
    # plt.savefig(output_dir + 'diff_time_vs_time_gnn'+tag+'_'+str(i)+'.png')



plt.savefig(output_dir + 'resolution_gnn'+tag+'.png')
