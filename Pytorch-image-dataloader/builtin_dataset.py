# USAGE
# python builtin_dataset.py

# import necessary packages
from pyimagesearch import config
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure("{} batch".format(dataset_type),
	figsize=(config.BATCH_SIZE, config.BATCH_SIZE))

	# loop over the batch size
	for i in range(0, config.BATCH_SIZE):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)

		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")

		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]

		# show the image along with the label
		plt.imshow(image[..., 0], cmap="gray")
		plt.title(label)
		plt.axis("off")

	# show the plot
	plt.tight_layout()
	plt.show()

# define the transform
transform = transforms.Compose([transforms.ToTensor()])

# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = MNIST(root=config.MNIST_DATASET_PATH, train=True,
	download=True, transform=transform)
valDataset = MNIST(root=config.MNIST_DATASET_PATH, train=False,
	download=True, transform=transform)

# create training and validation set dataloaders
print("[INFO] creating training and validation set dataloaders...")
trainDataLoader = DataLoader(trainDataset, 
	batch_size=config.BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE)

# grab a batch from both training and validation dataloader
trainBatch = next(iter(trainDataLoader))
valBatch = next(iter(valDataLoader))

# visualize the training set batch
print("[INFO] visualizing training batch...")
visualize_batch(trainBatch, trainDataset.classes, "train")

# visualize the validation set batch
print("[INFO] visualizing validation batch...")
visualize_batch(valBatch, valDataset.classes, "val")
