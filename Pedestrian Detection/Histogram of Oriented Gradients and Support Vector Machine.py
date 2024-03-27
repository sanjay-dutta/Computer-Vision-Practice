from skimage.feature import hog
from skimage import exposure
from skimage.io import imread, imshow, show
import matplotlib.pyplot as plt
# Assuming you've already installed scikit-image, sklearn, and matplotlib

# Example for feature extraction on a single image
image_path = r'D:\Github_Desktop\Computer-vision-practice\Pedestrian Detection\pedestrian-crossing.jpg'
image = imread(image_path)
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

# Adjust contrast of the output image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display the original image and the HOG image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original Image')

# Display the HOG image
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
