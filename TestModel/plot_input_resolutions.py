import matplotlib.pyplot as plt
import numpy as np
import uproot
from tensorflow import sparse, stack
import onnxruntime as ort
import time

output_dir = 'plots/'

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time2.root'

data_grid_size = 9
data_shape = (-1, data_grid_size, data_grid_size, 2)

# Assuming the ROOT file structure: MCParticles and PixelHits trees
infile = uproot.open(file_path)
tree  = infile['events']

# Extracting data from the ROOT file
df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'charge', 'time'], library='pd')#, entry_stop=10000)

input_data = df[['x', 'y', 'px', 'py', 'start_time']].values.astype(np.float32)

target_data = np.stack([df['charge'].to_numpy(), df['time'].to_numpy()],axis=2)
image_tensor = target_data.reshape(data_shape)

charges = image_tensor[:,:,:,0]

#round_output = np.ceil(output-0.2)
round_charges = charges

# Squared charge
squared_charges = round_charges
# Total square charge
total_squared_charge = np.sum(squared_charges, axis=(1,2))
# Square charge distribution across x
charge_x = np.sum(squared_charges, axis=2)
# Square charge distribution across y
charge_y = np.sum(squared_charges, axis=1)

# Mean index across x position of square charge
mean_index_x = 4-np.sum(charge_x*np.arange(data_grid_size), axis=1)/total_squared_charge#float(data_grid_size)/2
# Mean index across y position of square charge
mean_index_y = np.sum(charge_y*np.arange(data_grid_size), axis=1)/total_squared_charge-4#float(data_grid_size)/2

#Plot predicted x-y distribution in a 3x3 range
plt.figure()
plt.hist2d(mean_index_x, mean_index_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.grid(True)
plt.savefig(output_dir + 'mean_index_x_y_distribution.png')


#Plot mean index x distribution against input x
plt.figure()
plt.hist2d(input_data[:,0], mean_index_x, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('Mean index x')
plt.savefig(output_dir + 'mean_index_x_distribution.png')

#Plot mean index y distribution against input y
plt.figure()
plt.hist2d(input_data[:,1], mean_index_y, bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('y [pixel pitch]')
plt.ylabel('Mean index y')
plt.savefig(output_dir + 'mean_index_y_distribution.png')


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
plt.savefig(output_dir + 'diff_mean_squared_charge_x.png')

# Plot the difference between mean squared charge and input y
plt.figure()
plt.hist(diff_mean_squared_charge_y.flatten(), bins=100, range=( -1.0, 1.0))
plt.xlabel('Difference between mean squared charge and input y')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'diff_mean_squared_charge_y.png')

# Plot 2D histogram x-y plot of mean squared charge
plt.figure()
plt.hist2d(diff_mean_squared_charge_x.flatten(), diff_mean_squared_charge_y.flatten(), bins=(100, 100), range=[[-4.5, 4.5], [-4.5, 4.5]], cmap='viridis')
plt.xlabel('x [pixel pitch]')
plt.ylabel('y [pixel pitch]')
plt.savefig(output_dir + 'mean_squared_charge_x_y_distribution.png')

# Plot x-y resolution as a function of pixel hit
maxHits = 10
fig, axs = plt.subplots(maxHits, 3, figsize=(40, 20))
nHits = np.sum(charges>0, axis=(1,2))

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


plt.savefig(output_dir + 'resolution.png')
