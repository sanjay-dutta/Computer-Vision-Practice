# import the necessary packages
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid

class ObjectDetector(Module):
	def __init__(self, baseModel, numClasses):
		super(ObjectDetector, self).__init__()

		# initialize the base model and the number of classes
		self.baseModel = baseModel
		self.numClasses = numClasses

		# build the regressor head for outputting the bounding box
		# coordinates
		self.regressor = Sequential(
			Linear(baseModel.fc.in_features, 128),
			ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4),
			Sigmoid()
		)

		# build the classifier head to predict the class labels
		self.classifier = Sequential(
			Linear(baseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, self.numClasses)
		)

		# set the classifier of our base model to produce outputs
		# from the last convolution block
		self.baseModel.fc = Identity()

	def forward(self, x):
		# pass the inputs through the base model and then obtain
		# predictions from two different branches of the network
		features = self.baseModel(x)
		bboxes = self.regressor(features)
		classLogits = self.classifier(features)

		# return the outputs as a tuple
		return (bboxes, classLogits)