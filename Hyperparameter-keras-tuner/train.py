# USAGE
# python train.py --tuner hyperband --plot output/hyperband_plot.png
# python train.py --tuner random --plot output/random_plot.png
# python train.py --tuner bayesian --plot output/bayesian_plot.png

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.model import build_model
from pyimagesearch import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import kerastuner as kt
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tuner", required=True, type=str,
	choices=["hyperband", "random", "bayesian"],
	help="type of hyperparameter tuner we'll be using")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# load the Fashion MNIST dataset
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# add a channel dimension to the dataset
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

# initialize an early stopping callback to prevent the model from
# overfitting/spending too much time training with minimal gains
es = EarlyStopping(
	monitor="val_loss",
	patience=config.EARLY_STOPPING_PATIENCE,
	restore_best_weights=True)

# check if we will be using the hyperband tuner
if args["tuner"] == "hyperband":
	# instantiate the hyperband tuner object
	print("[INFO] instantiating a hyperband tuner object...")
	tuner = kt.Hyperband(
		build_model,
		objective="val_accuracy",
		max_epochs=config.EPOCHS,
		factor=3,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name=args["tuner"])

# check if we will be using the random search tuner
elif args["tuner"] == "random":
	# instantiate the random search tuner object
	print("[INFO] instantiating a random search tuner object...")
	tuner = kt.RandomSearch(
		build_model,
		objective="val_accuracy",
		max_trials=10,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name=args["tuner"])

# otherwise, we will be using the bayesian optimization tuner
else:
	# instantiate the bayesian optimization tuner object
	print("[INFO] instantiating a bayesian optimization tuner object...")
	tuner = kt.BayesianOptimization(
		build_model,
		objective="val_accuracy",
		max_trials=10,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name=args["tuner"])

# perform the hyperparameter search
print("[INFO] performing hyperparameter search...")
tuner.search(
	x=trainX, y=trainY,
	validation_data=(testX, testY),
	batch_size=config.BS,
	callbacks=[es],
	epochs=config.EPOCHS
)

# grab the best hyperparameters
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal number of filters in conv_1 layer: {}".format(
	bestHP.get("conv_1")))
print("[INFO] optimal number of filters in conv_2 layer: {}".format(
	bestHP.get("conv_2")))
print("[INFO] optimal number of units in dense layer: {}".format(
	bestHP.get("dense_units")))
print("[INFO] optimal learning rate: {:.4f}".format(
	bestHP.get("learning_rate")))

# build the best model and train it
print("[INFO] training the best model...")
model = tuner.hypermodel.build(bestHP)
H = model.fit(x=trainX, y=trainY,
	validation_data=(testX, testY), batch_size=config.BS,
	epochs=config.EPOCHS, callbacks=[es], verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# generate the training loss/accuracy plot
utils.save_plot(H, args["plot"])