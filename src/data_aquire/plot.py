import numpy as np
import matplotlib.pyplot as plt

# Load the data from the .npy file
output_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\rowdata\averaged_data.npy'
data = np.load(output_path)

# Iterate and plot
for i in range(2):  # Adjust the range as needed
    # plt.plot(data[i, :, 1])  # Plot the second column (averaged data) over the 300 samples
    plt.plot(data[i, :, 0]) 
    plt.title(f'Plot {i+1}')
    plt.xlabel('Sample Index')
    plt.ylabel('Averaged Data')
    plt.show()

