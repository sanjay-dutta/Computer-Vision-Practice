
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * K).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the input image", default='jemma.png')
args = vars(ap.parse_args())

# Define kernels
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
sharpen = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")
laplacian = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="int")
sobelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="int")
sobelY = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype="int")
emboss = np.array(([-2, -1, 0], [-1, 1, 1], [0, 1, 2]), dtype="int")
prewittX = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]), dtype="int")
prewittY = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss),
    ("prewitt_x", prewittX),
    ("prewitt_y", prewittY)
)

# Load image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply kernels
for (kernelName, K) in kernelBank:
    print(f"[INFO] applying {kernelName} kernel")
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    # Display the original, convolved output, and OpenCV output images
    cv2.imshow("Original", gray)
    cv2.imshow(f"{kernelName} - convolve", convolveOutput)
    cv2.imshow(f"{kernelName} - opencv", opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Canny Edge Detection
cannyOutput = cv2.Canny(gray, 100, 200)
cv2.imshow("Canny Edge Detection", cannyOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
