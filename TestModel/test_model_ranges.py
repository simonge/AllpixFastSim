import matplotlib.pyplot as plt
import numpy as np
import uproot
from tensorflow import sparse, stack
import onnxruntime as ort
import time

output_dir = 'plots/'
model_dir = '/scratch/EIC/models/Allpix/'
model_base = "model_electron"
model_name = model_dir+model_base+".onnx"
# Load the ONNX model
sess = ort.InferenceSession(model_name)

input_name = sess.get_inputs()[0].name

# Load data from the ROOT file
file_path = '/scratch/EIC/Events/Allpix2/Convert_time2.root'

num_plots = 3
data_grid_size = 9

# Assuming the ROOT file structure: MCParticles and PixelHits trees
infile = uproot.open(file_path)
tree  = infile['events']

# Extracting data from the ROOT file
df = tree.arrays(['x', 'y', 'px', 'py', 'start_time', 'charge', 'time'], library='pd')#, entry_stop=10000)

input_data = df[['x', 'y', 'px', 'py', 'start_time']].values.astype(np.float32)
#input_data = df[['x', 'y']].values.astype(np.float32)

# Predict the output for the input tensor
start_time = time.time()
output = sess.run(None, {input_name: input_data})
# End timing
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference Time: {inference_time} seconds")

output = output[0]
#output = output.reshape((len(input_data), data_grid_size, data_grid_size,2))

#output = np.transpose(output, (0, 2, 3, 1))


#round_output = np.ceil(output-0.2)
charges = output[:,:,:,0]
round_charges = np.ceil(charges-0.2)

times = output[:,:,:,1]
round_times = np.where(round_charges>0,np.ceil(times-0.2),0)
# round_output[:,:,:,0] = np.ceil(output[:,:,:,0])
# round_output[:,:,:,1] = np.round(output[:,:,:,1])

# Calculate the number of pixels with hits > 0 for each entry
output_nhits = np.sum(round_charges > 0, axis=(1,2))
# Plot the number of pixels with a hit for predicted entry
plt.figure()
plt.hist(output_nhits, bins=12, range=(0, 12))
plt.xlabel('Number of hit pixels')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'num_hit_pred.png')

# Plot the charge distribution
plt.figure()
plt.hist(round_charges[round_charges > 0], bins=6, range=(0, 6))
plt.xlabel('Charge')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'charge_distribution_pred.png')


# Plot the time distribution
plt.figure()
plt.hist(round_times[round_times > 0], bins=30, range=(0, 30))
plt.xlabel('Time')
plt.ylabel('Number of entries')
plt.savefig(output_dir + 'time_distribution_pred.png')

input_tensors = np.array([[[0.5, 0.5, 0.0, 0.0, 0.0]],[[0.25, 0.25, 0.0, 0.0, 5.0]],[[0.0, 0.0,-0.05,-0.05,0.1]],[[0.5, 0.1, 0.0, 0.0, 10.0]],[[0.0, 0.5, 0.05, 0.05, 14.0]],[[0.25, 0.5, 0.05, 0.05, 24.5]]], dtype=np.float32)

#input_tensors = np.array([[0.5, 0.5, 0.0, 0.0],[0.25, 0.25, 0.0, 0.0],[0.0, 0.0,-0.05,-0.05],[0.5, 0.1,0.0,0.0],[0.0, 0.5,0.05,0.05],[0.25, 0.5,0.05,0.05]], dtype=np.float32)

input_range = np.array([0.05,0.05,0.01,0.01,5.0])


for j, input_tensor in enumerate(input_tensors[:,0,0:5]):
    print(input_tensor)

    output_extension = 'x-' + str(input_tensor[0]) + '_y-' + str(input_tensor[1]) + '_px-' + str(input_tensor[2]) + '_py-' + str(input_tensor[3]) + '.png'

    print(input_tensor)
    print(len(df))
    # Filter the df by +/- the input range on x, y, px and py
    df2 = df[(df['x'] > input_tensor[0] - input_range[0]) & (df['x'] < input_tensor[0] + input_range[0])]
    print(len(df2))
    df2 = df2[(df2['y'] > input_tensor[1] - input_range[1]) & (df2['y'] < input_tensor[1] + input_range[1])]
    print(len(df2))
    df2 = df2[(df2['px'] > input_tensor[2] - input_range[2]) & (df2['px'] < input_tensor[2] + input_range[2])]
    print(len(df2))
    df2 = df2[(df2['py'] > input_tensor[3] - input_range[3]) & (df2['py'] < input_tensor[3] + input_range[3])]
    print(len(df2))
    df2 = df2[(df2['start_time'] > input_tensor[4] - input_range[4]) & (df2['start_time'] < input_tensor[4] + input_range[4])]
    print(len(df2))
    print('')

    input_indeces = df2.index.values

    # Plot the length of pixel_x for each entry
    plt.figure()
    plt.hist(np.sum(round_times[input_indeces] > 0, axis=(1,2)), bins=12, range=(0, 12))
    plt.xlabel('Number of hit pixels')
    plt.ylabel('Number of entries')
    plt.savefig(output_dir + 'num_hit_pred_'+output_extension)
    #print number of entries