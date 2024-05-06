import os
import numpy as np
import json
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
tf.random.set_seed(221) # for reproducible results

INPUT_SIZE = (1920, 1080)
IMG_SIZE = 512 # image size for the network
N = 512
path = ''
image_path = os.path.join(path, '/impacs/sad64/SLURM/Football Player Segmentation/dataset/images')
mask_path = os.path.join(path, '/impacs/sad64/SLURM/Football Player Segmentation/dataset/annotations')
with open('/impacs/sad64/SLURM/Football Player Segmentation/dataset/annotations/instances_default.json') as f:
    annotations = json.load(f)

image_id_dict = {image['id']: image['file_name'] for image in annotations['images']}

images = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for img_id, img_filename in image_id_dict.items():
    img = Image.open(os.path.join(image_path, img_filename))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    images[img_id - 1] = img

# show first 9 images
fig = plt.figure(figsize=(12, 6))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i]/255)
    plt.axis('off')

fig.tight_layout()

# Example of one annotation mask
annote = annotations['annotations'][0]
print(annote['image_id'])
mask = Image.new('1', INPUT_SIZE) # create new image in INPUT_SIZE filled with black (default)
mask_draw = ImageDraw.Draw(mask, '1') # so we can draw on the mask image
mask_draw.polygon(annote['segmentation'][0], fill=1) # draw a player in white
mask = mask.resize((IMG_SIZE, IMG_SIZE))
plt.imshow(mask)

masks = np.zeros((N, IMG_SIZE, IMG_SIZE), dtype=bool)

# iterate through all annotations
for annotation in annotations['annotations']:
    # get image id of the annotation
    img_id = annotation['image_id']
    mask = Image.new('1', INPUT_SIZE)
    mask_draw = ImageDraw.ImageDraw(mask, '1')
    segmentation = annotation['segmentation'][0]
    mask_draw.polygon(segmentation, fill=1)
    bool_array = np.array(mask.resize((IMG_SIZE, IMG_SIZE))) > 0
    masks[img_id - 1] = masks[img_id - 1] | bool_array

masks = masks.reshape(N, IMG_SIZE, IMG_SIZE, 1) # add channel dimension

plt.imshow(masks[0])

# masks applied on top of the images
# fig = plt.figure(figsize=(12, 6))

# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(images[i]/255)
#     plt.imshow(masks[i], alpha=0.5)
#     plt.axis('off')

# fig.tight_layout()
# Show first 9 images and save the figure
fig = plt.figure(figsize=(12, 6))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i] / 255)
    plt.axis('off')
fig.tight_layout()
plt.savefig('/impacs/sad64/SLURM/Football Player Segmentation/output/Inital mask')  # Specify a path here
plt.close(fig)  # Close the figure to free up memory



images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.1, random_state=42)
print(f"Train images shape: {images_train.shape}, Train masks shape: {masks_train.shape}")
print(f"Test images shape: {images_test.shape}, Test masks shape: {masks_test.shape}")

def jaccard_index(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def conv_block(inputs, n_filters, maxpooling=True):
    """
    Convolution block of U-Net. Two convolutional layers, followed by Batch Norm
    and ReLU activation.

    Inputs:
        inputs - input tensor to the block
        n_filters - number of filter for the conv layers
    Returns:
        out - output from the block
        skip - input to the decoder network
    """
    x = tfl.Conv2D(filters=n_filters, kernel_size=3, padding='same')(inputs)
    x = tfl.BatchNormalization()(x)
    x = tfl.Activation('relu')(x)
    x = tfl.Conv2D(filters=n_filters, kernel_size=3, padding='same')(x)
    x = tfl.BatchNormalization()(x)
    skip = tfl.Activation('relu')(x)
    if maxpooling == True:
        out = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(skip)
    else:
        out = skip

    return out, skip

def upsampling_block(expansive_input, contractive_input, n_filters):
    """
    Upsampling block

    Inputs:
        expansive_input - input from the previous layer of the expansive path
        contractive_input - input from the corresponding encoder block
    """
    # upsample and perform convolution
    up = tfl.Conv2DTranspose(n_filters, kernel_size=2, strides=2, padding='same')(expansive_input)
    # concatenate the inputs on the channel axis
    input = tfl.concatenate([up, contractive_input], axis=3)
    out, _ = conv_block(input, n_filters, False)

    return out

def unet_model(input_size=(512, 512, 3), n_filters=64):
    """
    U-Net model

    Inputs:
        input_size - size of the input image
        n_filters - base number of filters

    Returns:
        model - U-Net model
    """
    # Contracting path
    inputs = tfl.Input(input_size)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8)
    # Bridge
    cblock5 = conv_block(cblock4[0], n_filters*16, maxpooling=False)

    # Expansive path
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters*8)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    out = tfl.Conv2D(1, 1, padding='same', activation='sigmoid')(ublock9)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model

unet = unet_model()

unet.summary()

unet.compile(optimizer=tf.keras.optimizers.Adam(),
             loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=[jaccard_index, 'accuracy'])
unet.fit(images_train, masks_train, epochs=10, batch_size=4, validation_split=0.2)

unet.evaluate(images_test, masks_test, batch_size=4)

predicted_mask = unet.predict(images_test, batch_size=4)
predicted_mask2 = (predicted_mask > 0.5).astype(np.uint8)

# predicted masks
# fig, ax = plt.subplots(5, 3, figsize=(12, 10))

# for i in range(5):
#     ax[i, 0].imshow(images_test[i])
#     ax[i, 0].axis('off')
#     ax[i, 1].imshow(masks_test[i])
#     ax[i, 1].axis('off')
#     ax[i, 2].imshow(predicted_mask2[i])
#     ax[i, 2].axis('off')

# ax[0, 0].set_title('Original image')
# ax[0, 1].set_title('True mask')
# ax[0, 2].set_title('Predicted mask')

# fig.tight_layout()
# Predicted masks display and save the figure
fig, ax = plt.subplots(5, 3, figsize=(12, 10))
for i in range(5):
    ax[i, 0].imshow(images_test[i] / 255)
    ax[i, 1].imshow(masks_test[i], alpha=0.5)  # Adjust alpha if needed
    ax[i, 2].imshow(predicted_mask2[i], alpha=0.5)  # Adjust alpha if needed
    ax[i, 0].axis('off')
    ax[i, 1].axis('off')
    ax[i, 2].axis('off')
ax[0, 0].set_title('Original image')
ax[0, 1].set_title('True mask')
ax[0, 2].set_title('Predicted mask')
fig.tight_layout()
plt.savefig('/impacs/sad64/SLURM/Football Player Segmentation/output/Inital mask/predicted masks')  # Specify a path here
plt.close(fig)  # Close the figure to free up memory


cr = classification_report(masks_test.flatten(), predicted_mask2.flatten())
print(cr)
