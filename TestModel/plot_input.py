import matplotlib.pyplot as plt
import numpy as np
import uproot
from tensorflow import sparse, stack

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time2.root'

num_plots = 10
sensor_thickness = 300.0/55.0 # Thickness in pixel dimensions
data_grid_size = 9
data_shape = (-1, data_grid_size, data_grid_size, 2)

# Assuming the ROOT file structure: MCParticles and PixelHits trees
infile = uproot.open(file_path)
tree  = infile['events']

# Extracting data from the ROOT file
df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'charge', 'time'], entry_start=0, entry_stop=num_plots*num_plots)

# Shape data into 2 channel image
target_data = np.stack([df['charge'].to_numpy(), df['time'].to_numpy()],axis=2)
image_tensor = target_data.reshape(data_shape)

# Initialize the figure
fig, axs = plt.subplots(num_plots, num_plots * 2, figsize=(40, 20))

# Flatten the axes
axs = axs.flatten()
for ax in axs:
    # Plot data with a color scale from 0 to 4
    #im = ax.imshow(data, vmin=0, vmax=4)
    
    # Set x and y axes limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

# Plot the data for the entries in the dataframe on a grid
for i, tensor in enumerate(image_tensor):

    charge = tensor[:,:,0]
    time = tensor[:,:,1]

    #print(tensor[:,:,0])
    # Plot the charge data
    im_charge = axs[i * 2].imshow(charge, cmap='viridis', extent=[0, data_grid_size, 0, data_grid_size], vmin=0, vmax=4)
    axs[i * 2].set_title('Charge')
    fig.colorbar(im_charge, ax=axs[i * 2], orientation='vertical')

    # Plot the time data
    im_time = axs[i * 2 + 1].imshow(time, cmap='viridis', extent=[0, data_grid_size, 0, data_grid_size], vmin=0, vmax=60)
    axs[i * 2 + 1].set_title('Time')
    fig.colorbar(im_time, ax=axs[i * 2 + 1], orientation='vertical')

# Show the plot
plt.show()

# Save the plot
fig.savefig('data_plot.png')