import matplotlib.pyplot as plt
import numpy as np
import uproot
from tensorflow import sparse, stack
import onnxruntime as ort
import time

output_dir = 'plots/'
model_dir = '/scratch/EIC/models/Allpix/'
predictor_base  = 'predictor_electron_gnn'
predictor_name = model_dir+predictor_base+".onnx"
model_base = "model_electron_5"
model_name = model_dir+model_base+".onnx"

# Load the ONNX model
predictor_session = ort.InferenceSession(predictor_name)
sess = ort.InferenceSession(model_name)

input_name = sess.get_inputs()[0].name

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time2.root'

data_grid_size = 5

# Assuming the ROOT file structure: MCParticles and PixelHits trees
infile = uproot.open(file_path)
tree  = infile['events']

# Extracting data from the ROOT file
df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'charge', 'time'], library='pd')#, entry_stop=10000)

input_data = df[['x', 'y', 'px', 'py', 'start_time']].values.astype(np.float32)

# Predict the output for the input tensor
start_time = time.time()
output = sess.run(None, {input_name: input_data})
# End timing
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference Time: {inference_time} seconds")

output = output[0]

#round_output = np.ceil(output-0.2)
charges = output[:,:,:,0]
round_charges = np.ceil(charges-0.2)

# Squared charge
squared_charges = round_charges
# Total square charge
total_squared_charge = np.sum(squared_charges, axis=(1,2))
# Square charge distribution across x
charge_x = np.sum(squared_charges, axis=2)
# Square charge distribution across y
charge_y = np.sum(squared_charges, axis=1)

# Mean index across x position of square charge
mean_index_x = 2-np.sum(charge_x*np.arange(data_grid_size), axis=1)/np.sum(charge_x, axis=1)#float(data_grid_size)/2
# Mean index across y position of square charge
mean_index_y = np.sum(charge_y*np.arange(data_grid_size), axis=1)/np.sum(charge_y, axis=1)-2#float(data_grid_size)/2

#Plot predicted x-y distribution in a 3x3 range
plt.figure()
plt.hist2d(mean_index_x, mean_index_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.grid(True)
plt.savefig(output_dir + model_base + '_mean_index_x_y_distribution.png')


#Plot mean index x distribution against input x
plt.figure()
plt.hist2d(input_data[:,0], mean_index_x, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('Mean index x')
plt.savefig(output_dir + model_base + '_mean_index_x_distribution.png')

#Plot mean index y distribution against input y
plt.figure()
plt.hist2d(input_data[:,1], mean_index_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('y [pixel pitch]')
plt.ylabel('Mean index y')
plt.savefig(output_dir + model_base + '_mean_index_y_distribution.png')


print(f"Mean index x: {mean_index_x}")
print(f"Mean index y: {mean_index_y}")

# difference between mean_squared_charge_x and input x
diff_mean_squared_charge_x = mean_index_x - input_data[:,0]
# difference between mean_squared_charge_y and input y
diff_mean_squared_charge_y = mean_index_y - input_data[:,1]

# Plot the difference between mean squared charge and input x
plt.figure()
plt.hist(diff_mean_squared_charge_x.flatten(), bins=100, range=(-1.0, 1.0))
plt.xlabel('Difference between mean squared charge and input x')
plt.ylabel('Number of entries')
plt.savefig(output_dir + model_base + '_diff_mean_squared_charge_x.png')

# Plot the difference between mean squared charge and input y
plt.figure()
plt.hist(diff_mean_squared_charge_y.flatten(), bins=100, range=( -1.0, 1.0))
plt.xlabel('Difference between mean squared charge and input y')
plt.ylabel('Number of entries')
plt.savefig(output_dir + model_base + '_diff_mean_squared_charge_y.png')

# Plot 2D histogram x-y plot of mean squared charge
plt.figure()
plt.hist2d(diff_mean_squared_charge_x.flatten(), diff_mean_squared_charge_y.flatten(), bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.savefig(output_dir + model_base + '_mean_squared_charge_x_y_distribution.png')

# Plot x-y resolution as a function of pixel hit
maxHits = 10
fig, axs = plt.subplots(maxHits, 3, figsize=(40, 20))
nHits = np.sum(round_charges>0, axis=(1,2))

for i in range(maxHits):
    #Filter arrays by number of nHits
    diff_mean_squared_charge_x_nHits = diff_mean_squared_charge_x[nHits==i]
    diff_mean_squared_charge_y_nHits = diff_mean_squared_charge_y[nHits==i]

    # Plot the difference between mean squared charge and input x
    axs[i,0].hist(diff_mean_squared_charge_x_nHits.flatten(), bins=100, range=(-1.0, 1.0))
    axs[i,0].set_xlabel('Difference between mean squared charge and input x')
    axs[i,0].set_ylabel('Number of entries')
    axs[i,0].set_title(f'Number of hits: {i}')

    # Plot the difference between mean squared charge and input y
    axs[i,1].hist(diff_mean_squared_charge_y_nHits.flatten(), bins=100, range=( -1.0, 1.0))
    axs[i,1].set_xlabel('Difference between mean squared charge and input y')
    axs[i,1].set_ylabel('Number of entries')
    axs[i,1].set_title(f'Number of hits: {i}')

    # Plot 2D histogram x-y plot of mean squared charge
    axs[i,2].hist2d(diff_mean_squared_charge_x_nHits.flatten(), diff_mean_squared_charge_y_nHits.flatten(), bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
    axs[i,2].set_xlabel('x [pixel pitch]')
    axs[i,2].set_ylabel('y [pixel pitch]')
    axs[i,2].set_title(f'Number of hits: {i}')


plt.savefig(output_dir + model_base + '_resolution.png')
