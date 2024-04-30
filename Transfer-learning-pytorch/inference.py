# USAGE
# python inference.py --model output/warmup_model.pth
# python inference.py --model output/finetune_model.pth

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import argparse
import torch

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

# build our data pre-processing pipeline
testTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our de-normalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# initialize our test dataset and data loader
print("[INFO] loading the dataset...")
(testDS, testLoader) = create_dataloaders.get_dataloader(config.VAL,
	transforms=testTransform, batchSize=config.PRED_BATCH_SIZE,
	shuffle=True)

# check if we have a GPU available, if so, define the map location
# accordingly
if torch.cuda.is_available():
	map_location = lambda storage, loc: storage.cuda()

# otherwise, we will be using CPU to run our model
else:
	map_location = "cpu"

# load the model
print("[INFO] loading the model...")
model = torch.load(args["model"], map_location=map_location)

# move the model to the device and set it in evaluation mode
model.to(config.DEVICE)
model.eval()

# grab a batch of test data
batch = next(iter(testLoader))
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10))

# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)

	# make the predictions
	print("[INFO] performing inference...")
	preds = model(images)

	# loop over all the batch
	for i in range(0, config.PRED_BATCH_SIZE):
		# initalize a subplot
		ax = plt.subplot(config.PRED_BATCH_SIZE, 1, i + 1)

		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[i]
		image = deNormalize(image).cpu().numpy()
		image = (image * 255).astype("uint8")
		image = image.transpose((1, 2, 0))

		# grab the ground truth label
		idx = labels[i].cpu().numpy()
		gtLabel = testDS.classes[idx]

		# grab the predicted label
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDS.classes[pred]

		# add the results and image to the plot
		info = "Ground Truth: {}, Predicted: {}".format(gtLabel,
			predLabel)
		plt.imshow(image)
		plt.title(info)
		plt.axis("off")

	# show the plot
	plt.tight_layout()
	plt.show()