import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from util import data_path



# Load the image
img = Image.open(data_path('Figure1/uwlogo.png')).convert('RGB')  # Adjust path as needed

img_array = np.array(img)

# Convert RGB image to hyperspectral representation
height, width, _ = img_array.shape

# Create wavelength range (visible spectrum: 380-750nm)
wavelengths = np.linspace(381, 780, 400)

# Initialize hyperspectral cube
hyperspectral = np.zeros((height, width, len(wavelengths)))

# Convert RGB to approximate spectral data
# This is a simplified conversion - each RGB channel contributes to different wavelength ranges
for i in range(height):
    for j in range(width):
        r, g, b = img_array[i, j]

        # Create gaussian distributions centered at typical wavelengths for R,G,B
        red_contribution = np.exp(-((wavelengths - 650)**2)/(2*50**2)) * (r/255)
        green_contribution = np.exp(-((wavelengths - 550)**2)/(2*50**2)) * (g/255)
        blue_contribution = np.exp(-((wavelengths - 450)**2)/(2*50**2)) * (b/255)
        
        # Combine contributions
        hyperspectral[i,j,:] = red_contribution + green_contribution + blue_contribution

np.save(data_path('Figure1/hyperspectral.npy'), hyperspectral)

# Visualize a slice of the hyperspectral cube at a specific wavelength
wavelength_idx = 50  # Choose wavelength index to visualize
plt.figure(figsize=(10,8))
plt.imshow(hyperspectral[:,:,wavelength_idx], cmap='viridis')
plt.colorbar(label='Relative Intensity')
plt.title(f'Hyperspectral slice at {wavelengths[wavelength_idx]:.0f}nm')
plt.savefig(data_path('Figure1/hyperspectral_slice.svg'))
plt.show()

# Plot spectral profile for a specific pixel
pixel_x, pixel_y = width//2, height//2  # Center pixel
plt.figure(figsize=(10,6))
plt.plot(wavelengths, hyperspectral[pixel_y, pixel_x, :])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Intensity')
plt.title(f'Spectral Profile at pixel ({pixel_x}, {pixel_y})')
plt.savefig(data_path('Figure1/spectral_profile.svg'))
plt.show()
