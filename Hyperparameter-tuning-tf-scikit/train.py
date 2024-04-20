# USAGE
# python train.py

# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch.mlp import get_mlp_model
from tensorflow.keras.datasets import mnist

# load the MNIST dataset
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# initialize our model with the default hyperparameter values
print("[INFO] initializing model...")
model = get_mlp_model()

# train the network (i.e., no hyperparameter tuning)
print("[INFO] training model...")
H = model.fit(x=trainData, y=trainLabels,
	validation_data=(testData, testLabels),
	batch_size=8,
	epochs=20)

# make predictions on the test set and evaluate it
print("[INFO] evaluating network...")
accuracy = model.evaluate(testData, testLabels)[1]
print("accuracy: {:.2f}%".format(accuracy * 100))