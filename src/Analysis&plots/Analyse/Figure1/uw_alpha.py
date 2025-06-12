import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from util import data_path

# Load the image with alpha channel
img = Image.open(data_path('Figure1/uwlogo.png')).convert('RGBA')  # Adjust path as needed

img_array = np.array(img)

# Extract RGB and alpha channels
rgb_array = img_array[..., :3]
alpha_array = img_array[..., 3]


# Save RGB and Alpha arrays to NPY files
np.save(data_path('Figure1/rgb.npy'), rgb_array)
np.save(data_path('Figure1/alpha.npy'), alpha_array)